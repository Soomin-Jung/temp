# Hybrid Attention + Mamba/GDN State Migration Deep Dive

업데이트: 2026-09-10 KST

## 1. 목적

Qwen3.5+, Kimi K3 같은 최신 모델은 모든 layer가 standard full-attention KV cache를 쓰는 모델로 취급할 수 없다.

실제 serving state는 개념적으로 다음처럼 복합적이다.

```text
Transformer-like attention layers
  -> token-indexed K/V cache

Mamba / GDN / KDA-like recurrent layers
  -> recurrent state / convolution state / checkpointed state
```

따라서 long-context serving, prefix caching, speculative decoding, P/D disaggregation에서는:

```text
"KV cache가 맞는가?"
```

보다:

```text
"각 layer/state group이 어떤 상태를 어느 boundary에 저장하고,
어떻게 재사용/전송/복원하는가?"
```

가 핵심 질문이다.

---

## 2. Hybrid model의 mental model

Standard attention은 이전 token의 K/V를 block 단위로 저장한다.

대략:

```text
position 0..N
 -> K[0..N], V[0..N]
```

이므로 prefix가 일치하면 특정 block boundary까지의 cache를 재사용하기 쉽다.

반면 recurrent state 계열은 현재 state가 과거 전체 입력의 누적 함수다.

```text
S_t = f(S_{t-1}, x_t)
```

따라서 token `t`의 state를 재사용하려면 해당 boundary의 **정확한 recurrent checkpoint**가 필요하다.

이 차이가 Mamba prefix cache의 checkpoint/replay 구조를 만든다.

---

## 3. Qwen3.5+가 왜 중요 사례인가

PR #50210은 Qwen3.5 text-only dense/MoE를 등록하면서 causal-LM base를 hybrid로 mark하고 GDN state의:

- dtype
- shape
- copy hook

을 제공한다.

Source:
- https://github.com/vllm-project/vllm/pull/50210

즉 Qwen3.5+ 계열의 serving state는 단순 Qwen3 dense transformer와 동일하다고 볼 수 없다.

Platform에서는 모델 load 직후 최소 다음을 introspect한다.

```text
model_config.is_hybrid
attention layer types
Mamba/GDN state spec
number of KV cache groups
block/page sizes
state dtype
prefix caching support
```

Qwen3.6/3.8처럼 이름이 비슷한 후속 checkpoint도 반드시 다시 본다.

---

## 4. Kimi K3

Kimi K3는 hybrid state를 운영 관점에서 가장 적극적으로 노출시키는 모델 중 하나다.

중요 구성:

```text
KDA / recurrent state
MLA-like attention
MoE
DSpark
DCP
prefix caching
P/D
```

따라서 K3에서 recurrent state cache correctness가 깨지면 단순 cache miss가 아니라 **정상 hash 아래 잘못된 recurrent state가 복원되어 출력 자체가 망가질 수 있다.**

Mooncake PR #51358이 실제로 이런 종류의 failure를 수정했다.

---

## 5. v0.26 Mamba cache baseline

v0.26 `CacheConfig`에는 이미:

```text
mamba_block_size
mamba_cache_dtype
mamba_ssm_cache_dtype
mamba_cache_mode
mamba_page_size_padded
```

가 있다.

Source:
- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/cache.py

v0.26 설명상 `mamba_cache_mode`:

```text
none
  prefix caching disabled

all
  block boundary마다 state cache
  지원 모델에서 prefix caching enabled 시 기본 동작

align
  scheduler step의 마지막 token과 block boundary를 정렬해 cache
```

중요한 점은 v0.26의 hybrid prefix caching이 **지원은 하지만 EngineArgs default에서는 hybrid에 대해 opt-in**이라는 것이다.

v0.26 `EngineArgs`:

```python
default_prefix_caching = (
    model_config.is_prefix_caching_supported
    and not model_config.is_hybrid
)
```

즉 config class field의 `enable_prefix_caching=True`와 실제 EngineArgs resolved default를 혼동하면 안 된다.

---

## 6. v0.28: align 중심 cache semantics

v0.28 `CacheConfig` 설명은 prefix caching enabled 시 `align`을 default mode로 설명한다.

Source:
- https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/cache.py

이 변화의 의미는 recurrent state를 모든 possible boundary에 저장하는 것보다, scheduler와 실제 replay 가능한 boundary에 맞춰 state checkpoint를 관리하는 방향으로 옮겨간다는 것이다.

장점:

- state memory 감소
- block lifetime 관리 단순화
- replay semantics 명확화

비용:

- cache hit boundary와 exact request prefix boundary가 다를 수 있음
- 일부 token은 checkpoint 이후 replay가 필요

---

## 7. ReplaySSM

v0.28부터:

```text
use_replayssm
replayssm_buffer_len = 16
```

이 들어온다.

개념적으로 standard decode에서 매 token마다 큰 recurrent state를 HBM에 완전히 write하는 대신 최근 SSM input을 작은 ring/history buffer에 유지하고 일정 시점에 checkpoint를 flush하는 optimization이다.

대략:

```text
normal
step 1 -> state write
step 2 -> state write
step 3 -> state write
...

ReplaySSM
step 1 -> input history
step 2 -> input history
...
step B -> checkpoint flush
```

따라서 decode memory bandwidth를 줄일 여지가 있다.

하지만 migration 초기에는 끄는 것이 좋다.

왜냐하면 동시에:

- MRV1 -> MRV2
- prefix cache policy
- speculative decoding
- P/D state transfer

가 변하기 때문이다.

ReplaySSM은 최종 optimization phase에서 별도 A/B한다.

---

## 8. v0.29 `prefix_cache_retention_interval`

v0.29은 Mamba/Sliding Window checkpoint retention을 별도 runtime policy로 분리한다.

Source:
- PR #52216: https://github.com/vllm-project/vllm/pull/52216
- `CacheConfig`: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/cache.py

Semantics:

### `0`

semantic checkpoint만 유지한다.

예:

- latest replay boundary
- shared-prefix junction

### positive `N`

semantic checkpoint에 더해 N-token 간격 checkpoint를 유지한다.

### `None`

dense retention.

즉 이 값은 단순 cache-memory knob가 아니다.

```text
retention density
 -> nearest reusable state boundary
 -> replay distance
 -> TTFT
 -> memory
 -> external cache persistence volume
```

을 모두 바꾼다.

---

## 9. Internal prefill checkpoint: PR #52789

기존 Mamba prefix caching은 prefill을 마지막 state-cache block boundary에서 실제 model forward까지 나눌 수 있었다.

PR 설명의 8K 예:

### Before

```text
Model forward 7680
 -> FlashKDA 7680
 -> save checkpoint
 -> Model forward 320
 -> FlashKDA 320
```

### After

```text
Model forward 8000
 -> FlashKDA 7680
 -> save checkpoint internally
 -> FlashKDA 320
```

즉 attention/MoE/router/TP collective를 포함한 **두 번째 full-model forward를 제거**한다.

Source:
- PR #52789: https://github.com/vllm-project/vllm/pull/52789

PR benchmark에서는 Kimi K3 8K input에서 TTFT 약 9.4~25.5% 개선이 보고됐다.

### 중요한 해석

0.26 -> 0.29 Kimi/Qwen hybrid TTFT 개선을 전부 kernel speedup으로 해석하면 안 된다.

일부는:

```text
full model pass count 자체 감소
```

에서 올 수 있다.

---

## 10. Prefix cache hit의 실제 cost

Attention-only model의 prefix cache hit은 대략:

```text
prefix KV already present
 -> suffix만 prefill
```

로 생각하기 쉽다.

Hybrid recurrent model에서는:

```text
usable recurrent checkpoint
 -> checkpoint 이후 필요한 state replay
 -> attention suffix computation
```

이 포함될 수 있다.

따라서 metric도:

```text
prefix_cache_hit_tokens
```

하나만으로 충분하지 않다.

추천 추가 관점:

- recurrent checkpoint position
- replay tokens
- checkpoint retention mode
- suffix tokens actually forwarded
- internal checkpoint hit 여부

---

## 11. Speculative decoding과 recurrent state

MTP/EAGLE/DSpark는 target sequence보다 앞선 draft token을 temporary state로 계산한다.

Attention KV에서는 speculative block을 commit/drop하는 문제로 볼 수 있지만 recurrent state에서는 더 까다롭다.

```text
committed state
speculative scratch state
accepted prefix
rejected tail
```

의 ownership/lifetime을 정확히 관리해야 한다.

잘못된 speculative recurrent state를 persistent prefix cache에 저장하면 다음 요청까지 오염될 수 있다.

이 때문에 v0.29의 hybrid cache logic에는 speculative method에 따라 dense retention을 복구하는 등의 별도 path가 존재한다.

---

## 12. Mooncake PR #51358이 보여주는 실제 state corruption

이 PR은 Mooncake Store가 Mamba `align` mode에서 잘못된 state를 persistence할 수 있는 correctness bug를 수정한다.

Source:
- https://github.com/vllm-project/vllm/pull/51358

### Root cause

Attention block table은 대체로 append-only 성격이 강하지만 Mamba align table은 sparse/mutable하다.

- superseded state -> NULL block으로 대체
- physical block free/reuse
- speculative scratch block relocation
- connector mirror는 in-place mutation을 완전히 반영하지 못함

기존 positional save가:

```text
logical position i
 -> worker mirror의 block i
```

처럼 source block을 추정하면 이미 free/reassigned/speculative인 block을 읽을 수 있었다.

문제는 key/hash는 올바른 prefix를 가리킬 수 있다는 점이다.

결과:

```text
valid prefix hash
 -> poisoned recurrent state
 -> later external prefix hit
 -> garbled / runaway output
```

이 된다.

### Fix philosophy

Core가 exact:

```text
(group_id, block_id, boundary_tokens)
```

를 connector에 넘긴다.

그리고 asynchronous persistence가 완료될 때까지 physical block을 pin한다.

이 PR은 recurrent cache를 외부에 저장할 때 **hash correctness + physical state lifetime correctness 둘 다 필요**함을 잘 보여준다.

---

## 13. NULL block 문제

PR #51362도 sparse Mamba block table에서 `NULL_BLOCK_ID`를 physical block 0으로 오인해 잘못된 data를 upload할 수 있는 문제를 막는다.

Source:
- https://github.com/vllm-project/vllm/pull/51362

이런 bug는 cache-transfer backend가 standard attention block-table invariant를 hybrid state에 그대로 적용할 때 발생한다.

신규 connector/backend를 평가할 때 반드시:

```text
NULL / sparse slot semantics
mutable block table
copy-on-write
speculative ownership
```

지원 여부를 본다.

---

## 14. P/D에서 N-1 truncation이 필요한 이유

Mamba 계열 remote decode/prefill path는 마지막 token/state boundary 처리 때문에 producer-side prompt를 N-1 형태로 다루는 path가 있다.

0.29 근처에서 Mooncake와 NIXL 모두 **truncation ordering bug**를 별도로 수정했다.

Mooncake:
- PR #53663: https://github.com/vllm-project/vllm/pull/53663

NIXL:
- PR #53523: https://github.com/vllm-project/vllm/pull/53523

문제는 truncation을 local prefix-cache lookup **후**에 수행하면 request length와 cached-token count가 불일치해 scheduled new token이 0이 되는 assertion/correctness issue가 생길 수 있다는 것이다.

Fix:

```text
on_new_request
  -> prompt truncation
  -> local prefix cache matching
  -> external lookup
  -> scheduling
```

순으로 mutation을 앞당긴다.

### Platform lesson

P/D connector hook ordering은 단순 data transport detail이 아니다.

```text
request mutation
cache lookup
scheduler admission
transfer
```

순서가 semantic correctness를 결정할 수 있다.

---

## 15. Block size와 Mamba page alignment

Hybrid cache에서는 `--block-size 256` 같은 scheduler block size 하나만 보면 안 된다.

관련 개념:

```text
scheduler block size
attention physical page
Mamba page/block size
prefix match unit
DCP world size alignment
external transfer granularity
```

Mamba page가 attention page와 물리적으로 동일한 byte/token granularity를 갖지 않을 수 있으므로 vLLM은 padded Mamba page를 사용해 group alignment를 맞추는 path를 갖는다.

따라서 `block-size=256`을 migration initial value로 KEEP하더라도, **실제 resolved group block/page size를 log**해야 한다.

---

## 16. FP8 KV와 recurrent state dtype을 분리해서 본다

`--kv-cache-dtype fp8`은 attention KV의 storage dtype에 대한 중요한 knob다.

하지만 recurrent state는 별도의:

```text
mamba_cache_dtype
mamba_ssm_cache_dtype
model-specific GDN/KDA state dtype
```

를 가질 수 있다.

따라서:

```text
KV cache = FP8이니 전체 context state가 FP8
```

이라고 계산하면 틀릴 수 있다.

Hybrid model memory model은:

```text
attention KV bytes
+ recurrent state bytes
+ conv state
+ checkpoint metadata
+ padding/alignment
```

으로 나눈다.

---

## 17. Long-context capacity 계산

Standard transformer에서 대략:

```text
capacity ~= KV bytes / bytes_per_token
```

관점이 유효하지만 hybrid에서는 group-aware capacity가 필요하다.

v0.26/v0.29 `CacheConfig`도 `kv_cache_size_tokens`가 group-aware semantics임을 명시한다.

즉 실제 request 하나가 여러 cache group을 동시에 소비한다.

Long-context concurrency estimate에서:

```text
num_gpu_blocks * scheduler_block_size
```

만 사용하는 것은 부정확할 수 있다.

반드시 vLLM이 출력하는 group-aware token capacity / maximum concurrency와 실제 request test를 같이 본다.

---

## 18. Hybrid migration test matrix

### Cache mode

```text
prefix off
prefix on / retention 0
prefix on / periodic retention
prefix on / dense retention where supported
```

### Prompt pattern

```text
cold unique prompt
100% repeated prefix
95% repeated + short suffix
shared system prompt + unique user tail
partial physical-block hit
```

### Sequence length

```text
8K
32K
128K
170K
256K boundary test
```

### Spec

```text
none
MTP
DSpark
EAGLE where supported
```

### Execution

```text
single engine
P/D NIXL
P/D Mooncake
MRV1 where supported
MRV2
```

---

## 19. 반드시 볼 metric

```text
KV/state capacity tokens
prefix hit tokens
external prefix hit tokens
TTFT
prefill tokens/s
replay tokens if observable
Mamba state block count
cache group count
preemption
transfer bytes
transfer descriptors
spec acceptance
peak GPU memory
```

추가로 debugging build에서는 exact cache-group/block table dump 기능이 있으면 매우 유용하다.

---

## 20. Acceptance criteria

Hybrid model은 다음이 모두 통과해야 한다.

### Correctness

- repeated prefix와 cold prompt output equivalent
- P/D와 single-engine output equivalent
- speculative on/off semantic correctness
- no garbled/runaway response

### Memory

- repeated prefix에서 expected cache reuse
- no monotonic block leak
- no connector pin leak
- graph capture 이후 headroom 설명 가능

### Performance

- prefix hit가 TTFT 개선으로 실제 연결
- internal checkpoint/replay cost 확인
- long context에서 preemption/OOM regression 없음

---

## 21. 운영 권장 profile

Compatibility phase:

```text
prefix caching: explicit
retention: explicit for hybrid benchmark
ReplaySSM: off
Mamba backend: baseline-equivalent explicit
SD: off -> method-by-method
P/D: off -> backend-by-backend
MRV2: resolved value recorded
```

Optimization phase:

```text
internal Mamba checkpoint benefit
retention density
ReplaySSM
FlashInfer SSU algorithm
MTP/DSpark
DCP/P-D
```

을 순차적으로 채택한다.

---

## 22. 최종 판단

Hybrid 모델에서는 KV cache와 recurrent state를 하나의 `context cache`라는 추상화로 보되, 구현 검증에서는 반드시 분리해야 한다.

v0.29는 이 영역에서 0.26보다 훨씬 성숙했지만 그만큼:

- default prefix-cache policy
- checkpoint retention
- MRV2
- speculative state
- connector handoff

가 복잡해졌다.

따라서 **Qwen3.5+/Kimi K3 migration의 중심은 모델 load 성공이 아니라 state-lifecycle correctness 인증**이다.

---

## Sources

- v0.26 CacheConfig: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/cache.py
- v0.28 CacheConfig: https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/cache.py
- v0.29 CacheConfig: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/cache.py
- v0.26 MambaConfig: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/mamba.py
- v0.29 MambaConfig: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/mamba.py
- Qwen3.5 hybrid support PR #50210: https://github.com/vllm-project/vllm/pull/50210
- retention PR #52216: https://github.com/vllm-project/vllm/pull/52216
- internal checkpoints PR #52789: https://github.com/vllm-project/vllm/pull/52789
- Mooncake exact-state save PR #51358: https://github.com/vllm-project/vllm/pull/51358
- Mooncake sparse NULL fix PR #51362: https://github.com/vllm-project/vllm/pull/51362
- Mooncake truncation-order PR #53663: https://github.com/vllm-project/vllm/pull/53663
- NIXL truncation-order PR #53523: https://github.com/vllm-project/vllm/pull/53523
