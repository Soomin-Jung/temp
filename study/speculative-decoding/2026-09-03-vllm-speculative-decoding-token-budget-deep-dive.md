# vLLM Speculative Decoding Token Budget Deep Dive

작성일: 2026-09-03  
분류: `study/speculative-decoding`  
범위: vLLM `v0.26.0 ~ v0.28.0`, `max_num_batched_tokens`, `max_num_scheduled_tokens`, `max_num_seqs`, speculative drafting slot reservation, DSpark/DFlash/P-EAGLE/PARD/MTP/draft-model/n-gram

> **핵심 결론**
>
> Speculative decoding을 켠 vLLM에서 `max_num_batched_tokens`는 단순히 "scheduler가 이번 step에 사용자 token을 몇 개 배정할 수 있는가"가 아니다.
> 일부 drafter는 scheduler가 batch를 만든 **뒤에 추가 input slot을 삽입**한다. 따라서 vLLM은
>
> 1. scheduler가 직접 issue하는 token budget과
> 2. drafter가 추가할 수 있는 input-slot headroom
>
> 을 분리해서 관리해야 한다.
>
> v0.26/v0.27은 이 headroom을 `max_num_seqs` 기준으로 **최악의 경우를 정적 선점**했고, v0.28은 실제 scheduled request마다 **동적으로 차감**하도록 바뀌었다.

---

# 1. 왜 이 개념이 중요한가

일반 autoregressive decode만 생각하면 다음 직관이 자연스럽다.

```text
decode request 1개
→ 이번 step에서 next-token query 1개
→ token budget 1개 사용
```

그래서 흔히 다음처럼 이해한다.

```text
max_num_batched_tokens = 4096
→ 한 iteration에서 scheduler가 최대 4096 token까지 처리
```

하지만 parallel speculative drafter가 붙으면 이 설명은 불완전하다.

예를 들어 DSpark `K=7`은 scheduler가 이미 확보한 query position 외에 draft 단계에서 추가 input positions를 필요로 한다.

```text
scheduler-visible query
[Q]

DSpark drafting용 input shape
[Q][M1][M2][M3][M4][M5][M6]
    └──────── additional slots ────────┘
```

v0.28의 `SpeculativeConfig.max_num_new_slots_for_drafting` accounting에서 DSpark K=7의 additional slots는 6이다.

즉 scheduler가 request 하나에 1 token을 issue했다고 해서 model-runner 전체 input footprint도 반드시 1인 것은 아니다.

이 차이를 무시하면 다음 문제가 생긴다.

- `max_num_batched_tokens`가 충분해 보이는데 startup validation이 실패한다.
- long-context chunked prefill budget이 예상보다 크게 줄어든다.
- `max_num_seqs`를 크게 잡았을 뿐인데 TTFT가 악화된다.
- SD 방법을 바꾸었더니 같은 K에서도 MBT 요구량이 달라진다.
- v0.26/0.27에서 정상적인 tuning이 v0.28에서는 과도하게 보수적인 tuning이 된다.
- `draft token`, `scheduled token`, `input slot`, `accepted token`, KV lookahead를 같은 "token"으로 취급해 잘못된 결론을 낸다.

따라서 SD tuning에서는 **token의 의미를 먼저 분해해야 한다.**

---

# 2. 네 개의 서로 다른 token/slot 개념

## 2.1 Scheduler-issued token

scheduler가 request의 진행을 위해 이번 iteration에 직접 배정한 token 수다.

vLLM V1 scheduler에서는 개념적으로:

```python
token_budget = self.max_num_scheduled_tokens
...
token_budget -= num_new_tokens
```

형태로 관리한다.

여기에는 workload와 request state에 따라 다음이 포함될 수 있다.

- prefill chunk
- ordinary decode query
- 이전 cycle에서 만들어진 speculative tokens의 target-side verification work
- resumed/recomputed work

따라서 SD가 켜진 steady-state에서 `num_new_tokens`가 항상 1이라고 가정하면 안 된다.

---

## 2.2 Drafter-added input slot

scheduler가 직접 issue한 work 이외에 drafter가 batch input shape를 확장하기 위해 추가하는 slot이다.

vLLM의 canonical 이름은:

```text
SpeculativeConfig.max_num_new_slots_for_drafting
```

이다.

이 값이 바로 이 문서의 핵심이다.

> 이것은 "최종적으로 accept될 speculative token 수"가 아니다.  
> **drafting 단계에서 batch에 추가될 수 있는 input positions의 상한**이다.

---

## 2.3 Accepted/committed output token

target verification을 통과해 실제 output history에 commit되는 token이다.

draft slot을 6개 예약했다고 해서 6개가 반드시 accept되는 것은 아니다.

```text
draft slots reserved = 6
draft candidates      = up to K
accepted candidates   = workload/model/confidence에 따라 달라짐
committed output      = acceptance 결과에 따라 결정
```

MBT accounting과 acceptance rate는 관련은 있지만 같은 양이 아니다.

---

## 2.4 KV lookahead / cache capacity

parallel drafter가 future position에 KV/state를 기록해야 하는 경우 별도의 lookahead block/slot allocation이 필요할 수 있다.

이것은 `max_num_batched_tokens` reservation과 **다른 자원 차원**이다.

```text
MBT / input slot budget
    ≠ KV cache token capacity
    ≠ lookahead KV block reservation
    ≠ accepted output tokens
    ≠ GPU FLOP budget
```

특히 DFlash 계열은 bonus query/future position 때문에 KV lookahead와 input-buffer sizing 이슈가 별도로 존재해 왔다. MBT 식 하나로 모든 memory/correctness constraint를 설명할 수는 없다.

---

# 3. 세 가지 핵심 설정의 정확한 역할

## 3.1 `max_num_batched_tokens` — MBT

`SchedulerConfig`의 정의는 "single iteration에서 처리할 수 있는 최대 token 수"다.

SD가 없는 경우에는 scheduler budget과 거의 같은 의미로 볼 수 있다.

하지만 parallel drafting이 있는 경우 더 정확한 mental model은 다음이다.

> **engine/model-runner가 한 iteration에서 수용할 input-slot envelope**

즉 scheduler-issued work와 drafting expansion이 모두 이 envelope 안에 들어와야 한다.

기호로:

```text
B = max_num_batched_tokens
```

라고 하자.

---

## 3.2 `max_num_scheduled_tokens`

vLLM config의 설명은 다음 의미다.

> scheduler가 한 iteration에서 issue할 수 있는 최대 token 수.  
> 보통 MBT와 같지만 model이 batch에 token을 append할 수 있는 경우 speculative decoding처럼 더 작을 수 있다.

기호로:

```text
T = max_num_scheduled_tokens
```

라고 하자.

중요한 점:

- public `EngineArgs`에 존재한다.
- CLI `--max-num-scheduled-tokens`도 v0.26.0, v0.27.x, v0.28.0에 존재한다.
- 따라서 완전한 private field는 아니다.
- 그러나 일반 운영에서는 MBT/MNS를 사용자가 정하고 T는 vLLM이 계산하도록 두는 성격이 강한 **expert scheduler knob**다.

### CLI에 존재하지만 일반 사용자 주 설정으로 보기 어려운 이유

MBT는 worker/input capacity와 직접 연결되고 MNS는 concurrency cap과 직접 연결된다.

반면 T는 그 둘과 speculative configuration에서 파생되는 scheduler 내부 admission cap이다.

특히 v0.26/v0.27에서 T를 임의로 크게 설정해 static reservation을 우회하려 해도 validation이 다시 MBT와 reserve의 관계를 검사한다.

---

## 3.3 `max_num_seqs` — MNS

```text
M = max_num_seqs
```

라고 하자.

이 값은 원래 한 iteration에서 처리할 sequence 수의 상한이다.

하지만 v0.26/v0.27의 parallel SD에서는 한 가지 역할이 더 생겼다.

> **worst-case drafter reservation multiplier**

즉 실제 active request 수와 무관하게:

```text
extra_slots_per_request × max_num_seqs
```

를 통째로 예약했다.

이 때문에 v0.26/v0.27에서 `max_num_seqs`는 단순 concurrency knob가 아니었다.

---

# 4. `SchedulerConfig.DEFAULT_*`와 실제 `vllm serve` 기본값을 혼동하지 말 것

`SchedulerConfig` class 자체에는 테스트/편의를 위한 기본값이 있다.

예:

```text
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
DEFAULT_MAX_NUM_SEQS = 128
```

하지만 source comment도 명시하듯 **실제 usage에서는 `EngineArgs.create_engine_config` 경로에서 platform/usage-context에 따라 다시 설정**한다.

v0.26/v0.27의 대표적인 GPU default:

| GPU 조건 | UsageContext | default MBT | default MNS |
|---|---|---:|---:|
| >=70 GiB, A100 제외 (H100/H200 계열) | OPENAI_API_SERVER | 8192 | 1024 |
| >=70 GiB, A100 제외 | LLM_CLASS | 16384 | 1024 |
| 그 외 일반 GPU | OPENAI_API_SERVER | 2048 | 256 |
| 그 외 일반 GPU | LLM_CLASS | 8192 | 256 |

v0.28에서는 >=160 GiB GPU에 별도 branch가 추가되어 B200/B300급에서 online/offline 모두 MBT 16384, MNS 1024가 기본이 된다.

또한 사용자가 MBT만 명시하고 MNS를 생략하면 실제 MNS는 platform default에서 출발한 뒤:

```python
max_num_seqs = min(default_max_num_seqs, max_num_batched_tokens)
```

형태로 제한된다.

따라서 H100/H200급 online server에서:

```text
--max-num-batched-tokens 4096
--max-num-seqs <unset>
```

이면 일반적으로 MNS는 1024다.

### throughput mode 주의

`performance_mode == "throughput"`이면 **사용자가 명시하지 않은** MBT/MNS default가 2배가 될 수 있다.

즉 MBT는 명시하고 MNS만 생략한 경우 MNS만 2배가 되는 조합도 가능하므로 static reservation 버전에서는 특히 주의해야 한다.

---

# 5. SD 방법론별 additional drafting slot

기호:

```text
K = num_speculative_tokens
S = max_num_new_slots_for_drafting
```

## 5.1 v0.26 / v0.27의 generic 계산

두 버전의 핵심 코드는 사실상 다음 규칙이다.

```python
S = 0

if parallel_drafting:
    S = K - 1

if uses_draft_model():
    S += 1
```

이 generic rule의 결과:

| 방식 | method | parallel_drafting | v0.26/0.27 S |
|---|---|---:|---:|
| EAGLE3 serial | eagle3 | false | 0 |
| P-EAGLE | eagle3 | true | K-1 |
| DFlash | dflash | true | K-1 |
| DSpark | dspark | true | K-1 |
| MTP | mtp | false | 0 |
| n-gram | ngram | false | 0 |
| classic draft model | draft_model | false | 1 |
| PARD | draft_model | true | K |

DFlash와 DSpark는 config resolution 중 자동으로:

```python
parallel_drafting = True
```

가 된다.

### 중요한 역사적 결함: DFlash

v0.26/v0.27의 generic `K-1` 계산은 DFlash에는 정확하지 않았다.

DFlash는:

```text
1 bonus query + K mask queries
```

를 사용하므로 scheduler-existing query 대비 실제 net expansion은 K다.

이 문제는 upstream PR #51256에서 수정되었고 v0.28 source에는 method-specific branch로 반영되어 있다.

---

## 5.2 v0.28의 method-specific 계산

v0.28은 이 로직을 명시적으로 방법론별로 구분한다.

| Algorithm | method | Parallel | v0.28 additional slots S | K=7 예 |
|---|---|---:|---:|---:|
| EAGLE3 | eagle3 | No | 0 | 0 |
| P-EAGLE | eagle3 | Yes | K-1 | 6 |
| DFlash | dflash | Yes | K | 7 |
| DSpark | dspark | Yes | K-1 | 6 |
| MTP | mtp | No | 0 | 0 |
| N-gram | ngram | No | 0 | 0 |
| Draft model | draft_model | No | 1 | 1 |
| PARD | draft_model | Yes | K | 7 |

이 표는 v0.28 `SpeculativeConfig.max_num_new_slots_for_drafting`의 docstring과 unit test에 그대로 encode되어 있다.

---

# 6. 방법론별 slot topology를 이해한다

숫자를 외우는 것보다 **왜 S가 그 값인지**가 중요하다.

## 6.1 EAGLE3 serial: S = 0

serial EAGLE3은 scheduler 이후 parallel masked query를 batch에 추가하는 방식이 아니다.

따라서 이 특정 MBT-accounting 차원에서 추가 slot은 0이다.

주의:

> S=0은 "EAGLE가 추가 compute를 전혀 사용하지 않는다"는 뜻이 아니다.  
> **scheduler 이후 batch-shape expansion reserve가 0**이라는 뜻이다.

---

## 6.2 P-EAGLE: S = K-1

parallel EAGLE은 scheduler가 이미 제공한 query 하나를 재사용하고 나머지 masked positions만 추가한다.

```text
scheduler: [Q]

parallel drafter:
[Q][M1][M2] ... [M(K-1)]
```

따라서:

```text
S = K - 1
```

---

## 6.3 DSpark: S = K-1

DSpark 역시 v0.28 accounting에서 existing query를 재사용하고 parallel positions를 붙이는 방식으로 처리한다.

K=7:

```text
scheduler-existing position = 1
additional DSpark positions = 6
accounted query width       = 7
```

따라서:

```text
S = 6
```

이것이 DeepSeek-V4-Flash-0731 + DSpark K=7에서 MBT 계산의 핵심이다.

---

## 6.4 DFlash: S = K

DFlash는 DSpark와 같은 `parallel_drafting=True` 계열이라고 해서 같은 slot 식을 쓰면 안 된다.

DFlash query topology는:

```text
[bonus query][mask1][mask2] ... [maskK]
```

즉 총 K+1 query positions이고, scheduler가 이미 고려하는 하나를 제외한 net expansion은 K다.

따라서 v0.28:

```text
S = K
```

v0.26/v0.27의 generic K-1은 한 slot/request 부족한 accounting이었다.

---

## 6.5 MTP: S = 0

serial MTP는 speculative draft 계산 자체는 존재하지만 이 MBT accounting에서 post-scheduler parallel slot expansion을 추가하지 않는다.

즉:

```text
S = 0
```

이다.

다시 강조하면:

```text
S = 0
≠ draft compute = 0
≠ speculative verification cost = 0
```

이다.

---

## 6.6 N-gram: S = 0

model-free proposer이므로 별도 neural drafter input expansion reserve가 없다.

proposal 후보가 여러 token이라고 해서 `max_num_new_slots_for_drafting`가 자동으로 K가 되는 것이 아니다.

---

## 6.7 Classic draft model: S = 1

autoregressive draft-model path는 draft input을 slice하지 않아 request당 unsliced token 하나가 추가로 남는다.

따라서:

```text
S = 1
```

K가 커져도 이 specific expansion reserve는 1이다.

draft model이 K개의 candidate를 순차 생성하는 compute cost와는 다른 accounting이다.

---

## 6.8 PARD: S = K

parallel draft model은 기존 input을 shift/reuse하는 방식이 아니어서 K query positions 전체가 additional slots가 된다.

따라서:

```text
S = K
```

---

# 7. v0.26 / v0.27: static worst-case reservation

이 버전의 핵심 식은 매우 단순하다.

```text
B = max_num_batched_tokens
M = max_num_seqs
S = additional drafting slots / request
T = max_num_scheduled_tokens
```

사용자가 T를 지정하지 않으면:

```math
T = B - S M
```

이다.

즉 `max_num_seqs`개의 request가 한꺼번에 모두 drafting expansion을 일으킨다고 가정하고 **engine 시작 시점에 통째로 headroom을 떼어낸다.**

## 7.1 startup 가능 조건

auto T인 경우:

```math
B - SM > 0
```

즉:

```math
B > SM
```

이어야 한다.

T가 0 이하이면 startup에서 ValueError가 발생한다.

### 8192 warning은 startup hard minimum이 아니다

코드는 T가 8192 미만이면 suboptimal-performance warning을 내지만:

```text
T < 8192
```

자체가 fatal condition은 아니다.

fatal condition은:

```text
T <= 0
```

이다.

---

## 7.2 사용자가 T를 직접 지정한 경우

v0.26/v0.27은 추가 검증을 수행한다.

개념적으로:

```math
B \ge T + SM
```

이어야 한다.

즉:

> `--max-num-scheduled-tokens`를 직접 크게 줘서 static reservation을 우회할 수 없다.

---

## 7.3 scheduler runtime이 실제로 보는 budget

scheduler는:

```python
token_budget = self.max_num_scheduled_tokens
```

로 시작한다.

즉 static reservation에서 떼어낸 `SM`은 실제로 매 step마다 전부 사용되는 것이 아니더라도 scheduler가 손댈 수 없다.

예:

```text
B = 8192
M = 1024
DSpark K = 7
S = 6

static reserve = 6 * 1024 = 6144
T = 8192 - 6144 = 2048
```

실제 active DSpark request가 1개뿐이어도 scheduler는 여전히 2048에서 시작한다.

필요한 dynamic headroom은 훨씬 작을 수 있는데 6144를 이미 떼어놓았으므로 대부분이 idle headroom으로 남을 수 있다.

이것이 low-concurrency long-prefill/TTFT에 큰 손실을 만든 이유다.

---

# 8. v0.28: adaptive dual-budget accounting

v0.28에서 upstream PR #51725가 static reservation을 제거했다.

초기 상태:

```text
token_budget = max_num_scheduled_tokens
input_budget = max_num_batched_tokens
draft_slots  = max_num_new_slots_for_drafting
```

T를 사용자가 지정하지 않으면:

```text
T = B
```

로 둔다.

그리고 request를 실제로 schedule할 때마다 두 budget을 별도로 줄인다.

개념적인 runtime code:

```python
num_new_tokens = min(
    requested_work,
    token_budget,
    input_budget - draft_slots,
)

token_budget -= num_new_tokens
input_budget -= num_new_tokens + draft_slots
```

따라서 N개의 request가 이번 step에 실제 schedule되었고 각 request에 scheduler가 `n_i` token을 issue했다면:

```math
\sum_i n_i \le T
```

동시에 physical/input envelope는:

```math
\sum_i (n_i + S) \le B
```

를 만족한다.

이를 풀어 쓰면:

```math
input\_budget_{remaining}
=
B - \sum_i n_i - NS
```

이다.

## 8.1 무엇이 adaptive인가

v0.26/v0.27:

```text
reserve = S * max_num_seqs
```

v0.28:

```text
reserve가 실제 schedule된 request가 늘어날 때마다 S씩 증가
```

한다.

즉 MNS=1024라도 실제 이번 step에 request 4개만 admission되었다면 1024개분을 미리 떼지 않는다.

---

## 8.2 PR #51725의 예

PR은:

```text
max_num_seqs = 1024
max_num_batched_tokens = 8192
S = 6
```

인 parallel drafting 사례를 다음처럼 비교한다.

| actual request 수 | old static scheduled-token capacity | adaptive capacity |
|---:|---:|---:|
| 1 | 2048 | 8186 |
| 32 | 2048 | 8000 |
| 128 | 2048 | 7424 |
| 1024 | 2048 | 2048 |

즉 first-order headroom 관점에서:

```math
8192 - 6N
```

형태로 실제 request 수에 따라 공간을 남긴다.

PR benchmark에서는 Kimi-K3 DSpark, 8K input / 1K output, concurrency 1/4/16에서 TTFT가 크게 개선됐다.

이 결과의 핵심은 DSpark 자체가 더 정확해진 것이 아니라:

> **사용하지 않는 speculative headroom을 prefill/scheduler work가 다시 사용할 수 있게 된 것**

이다.

---

# 9. adaptive budget과 DSpark adaptive verification은 다른 기능이다

이 둘은 이름 때문에 매우 쉽게 섞인다.

## vLLM PR #51725 adaptive budget

목적:

```text
MBT 안에서 scheduler work와 drafter expansion headroom을
실제 scheduled request 수에 맞춰 accounting
```

이다.

즉 **capacity accounting / scheduler admission** 문제다.

## DSpark confidence-scheduled adaptive verification

목적:

```text
request별 confidence와 load/cost를 이용해
몇 개의 speculative token을 verify할 가치가 있는지 결정
```

이다.

즉 **verification-length optimization** 문제다.

둘은 직교한다.

```text
fixed-K DSpark
+ v0.28 adaptive MBT accounting
```

도 가능하고 의미가 있다.

---

# 10. DeepSeek-V4-Flash-0731 + DSpark K=7 사례

이제 실제 문제를 계산한다.

가정:

```text
model                    = DeepSeek-V4-Flash-0731
method                   = dspark
num_speculative_tokens K = 7
max_num_batched_tokens B = 4096
max_num_seqs             = unset
usage                    = vllm serve / OPENAI_API_SERVER
GPU class                = H100/H200급 >=70 GiB, non-A100
```

DSpark:

```text
S = K - 1 = 6
```

v0.26/v0.27 H100/H200 online default:

```text
M = 1024
```

따라서:

```math
SM = 6 \times 1024 = 6144
```

static scheduler budget:

```math
T = 4096 - 6144 = -2048
```

결과:

```text
max_num_scheduled_tokens <= 0
→ startup ValueError
```

즉 4K 실패는 모델 weight size나 KV cache 부족보다 **scheduler/drafter input-slot accounting validation 단계**에서 설명된다.

---

# 11. v0.26/v0.27에서 DSpark K=7의 MBT 하한

M=1024, S=6이면 startup hard condition:

```math
B > 6144
```

이다.

수학적 최소 integer:

```text
B_min = 6145
```

하지만 6145를 실제 tuning 값으로 쓰면:

```text
T = 1
```

이므로 실용적이지 않다.

## 11.1 대표값

| MBT B | static reserve | scheduler budget T |
|---:|---:|---:|
| 4096 | 6144 | -2048 → fail |
| 6144 | 6144 | 0 → fail |
| 6145 | 6144 | 1 |
| 8192 | 6144 | 2048 |
| 12288 | 6144 | 6144 |
| 14336 | 6144 | 8192 |
| 16384 | 6144 | 10240 |

즉 scheduler-issued work를 최소 8K 정도 유지하고 싶으면:

```math
B \ge 8192 + 6144 = 14336
```

가 된다.

실무적으로 16K가 자연스러운 후보가 된다.

---

# 12. MNS를 명시하면 왜 결과가 크게 달라지는가

DSpark K=7, B=4096에서:

```text
S = 6
T = 4096 - 6M
```

이다.

| MNS M | static reserve | T |
|---:|---:|---:|
| 64 | 384 | 3712 |
| 100 | 600 | 3496 |
| 128 | 768 | 3328 |
| 256 | 1536 | 2560 |
| 512 | 3072 | 1024 |
| 1024 | 6144 | -2048 fail |

이 표가 보여주는 사실:

> v0.26/v0.27에서 parallel SD를 사용할 때 MNS는 단순히 "최대 동시 sequence 수"가 아니라 **MBT를 정적으로 잠그는 multiplier**다.

따라서 실제 engine당 concurrency가 100 수준인데 default 1024를 그대로 두면 scheduler capacity를 불필요하게 희생한다.

---

# 13. 같은 DSpark K=7 + MBT 4K를 v0.28에서 보면

v0.28 config 단계:

```text
S = 6
B = 4096
T default = 4096
```

startup 검증은 본질적으로:

```text
B > S
```

를 요구한다.

4096 > 6이므로 **v0.26/v0.27의 static-reserve 이유로는 실패하지 않는다.**

runtime에서는 실제 request를 하나씩 schedule하면서:

```text
input_budget -= num_new_tokens + 6
```

한다.

따라서:

- request 수가 적으면 거의 4K를 scheduler/prefill이 활용할 수 있다.
- request 수가 늘수록 request당 6 slot의 headroom이 실제로 차감된다.
- MNS=1024라고 해서 처음부터 6144를 떼지 않는다.
- 결국 physical MBT가 부족하면 input_budget이 admission을 멈춘다.

### 중요한 정밀화

"decode request 하나가 항상 MBT 7개를 먹는다"로 단순화하면 안 된다.

정확한 식은:

```text
scheduler-issued n_i
+
DSpark additional slots 6
```

이다.

ordinary decode-like 상황에서 `n_i=1`이면 7처럼 보이지만, speculative verification/prefill/resume 상태에 따라 `n_i`가 1보다 클 수 있다.

---

# 14. DSpark가 예약한 6 slot은 정확히 무엇인가

가장 중요한 질문이다.

> "그 6개를 빼고 남은 것이 실제 iteration scheduler 예산인가?"

## v0.26/v0.27에서는 그렇다 — 하지만 정적으로 과도하게 빼놓는다

```text
MBT
├─ scheduler usable budget T
└─ worst-case DSpark headroom S*M
```

형태다.

DSpark headroom은 scheduler가 일반 prefill/decode work에 다시 사용할 수 없다.

그러나 실제로 그 step에서 M개의 request가 모두 drafting하지 않으면 일부 headroom은 사용되지 않고 남는다.

즉:

> **reserve는 capacity protection이지, 매 step 실제 computation을 보장하는 사용량이 아니다.**

---

## v0.28에서는 실제 scheduled request별로만 보호한다

```text
MBT input_budget
  ↓ request #1 schedule
  - n_1
  - S

  ↓ request #2 schedule
  - n_2
  - S
...
```

따라서 unused headroom이 scheduler에 더 오래 남는다.

---

## DSpark 전용 token이라고 부르면 생기는 오해

6개는 다음이 아니다.

- 사용자가 보게 될 output token 6개
- 무조건 accept될 token 6개
- KV cache capacity 6개를 영구 소모
- GPU FLOPs를 6 token만큼 별도 quota로 예약
- scheduler가 직접 issue한 token 6개

정확한 표현:

> **DSpark parallel drafting이 scheduler 이후 batch input을 확장할 수 있도록 보호하는 additional input slots**

이다.

---

# 15. Chunked Prefill과 왜 강하게 연결되는가

v0.26/v0.27 static 방식의 가장 큰 부작용 중 하나다.

가정:

```text
B = 8192
M = 1024
DSpark K=7
S=6
```

scheduler는 처음부터:

```text
T = 2048
```

만 받는다.

실제 decode concurrency가 1이어도 long prefill request는 한 iteration에 scheduler budget 2048 이상을 가져갈 수 없다.

즉 speculative headroom 6144가 대부분 비어 있음에도 prefill chunk가 그 공간을 쓸 수 없다.

결과:

```text
long prompt
→ 더 많은 prefill chunks
→ TTFT 증가
```

v0.28은 이 headroom을 request가 실제 admission될 때만 차감하므로 low-concurrency에서 훨씬 큰 prefill chunk를 허용할 수 있다.

이것이 PR #51725의 TTFT 개선을 이해하는 핵심이다.

---

# 16. Mixed prefill + decode를 보는 올바른 식

v0.28에서 한 step에 N개의 request가 schedule되고 각 scheduler-issued token 수가 `n_i`이면:

```math
\sum_i n_i \le T
```

그리고:

```math
\sum_i (n_i + S) \le B
```

이다.

예를 들어 DSpark K=7, B=4096에서:

```text
decode/spec requests가 먼저 headroom을 차지
+
남은 input_budget을 prefill chunk가 사용
```

하는 식으로 생각하는 것이 맞다.

따라서 tuning은 단순:

```text
MBT / (K+1)
```

하나로 끝나지 않는다.

실제 workload mix가 중요하다.

---

# 17. v0.26/v0.27과 v0.28의 본질적인 차이

| 항목 | v0.26 / v0.27 | v0.28 |
|---|---|---|
| T default | B - S*M | B |
| drafter headroom | startup 시 worst-case static reserve | runtime per-scheduled-request reserve |
| MNS 영향 | reserve multiplier로 직접 MBT 잠금 | concurrency cap; static reserve multiplier 아님 |
| low concurrency | unused reserve가 큼 | unused capacity 재사용 |
| chunked prefill | static T에 강하게 제한 | actual input_budget 범위에서 더 크게 활용 |
| config fail condition | B-SM <= 0 가능 | B <= S이면 fail |
| runtime physical cap | reduced T로 보수적으로 보호 | 별도 input_budget으로 직접 보호 |
| DFlash slots | generic K-1 | method-specific K |
| DSpark K=7 | S=6 | S=6 |

---

# 18. `max_num_scheduled_tokens`를 직접 만져야 하는가

## 일반 권장

대부분의 serving tuning에서는:

```text
MBT
MNS
K / speculative method
chunked-prefill policy
```

를 먼저 정하고 T는 auto로 두는 편이 안전하다.

## 직접 설정할 수 있는 이유

T는 scheduler-level throttling/실험에 의미가 있다.

예:

- physical input capacity B는 크게 유지
- scheduler-issued work만 더 낮게 제한
- 특정 latency/shape 실험
- custom scheduling policy

## v0.28에서 T > B를 줘도 물리 capacity가 늘지 않는다

runtime에는 여전히:

```text
input_budget = B
```

가 있으므로 T를 B보다 크게 준다고 MBT envelope를 넘어갈 수 있는 것은 아니다.

따라서 T는 capacity expansion knob가 아니다.

---

# 19. MBT와 CUDA Graph capture size를 혼동하지 말 것

speculative decoding은 query width가 증가하므로 cudagraph capture 후보 크기에도 영향을 준다.

vLLM은 `decode_query_len = 1 + num_speculative_tokens` 같은 정보를 이용해 graph capture size 후보를 만든다.

하지만:

```text
max_num_batched_tokens
max_num_scheduled_tokens
max_cudagraph_capture_size
cudagraph_capture_sizes
```

는 서로 다른 제약이다.

MBT가 통과해도 graph shape/capture/backend 제약으로 실패할 수 있고, 반대도 가능하다.

특히 새로운 parallel drafting method에서는:

- input buffer sizing
- attention metadata
- lookahead KV
- CUDA graph range
- MBT reservation

을 별도로 확인해야 한다.

보다 넓은 serving tuning 문맥은 [Scheduler Budget, Speculative Decoding & CUDA Graph](../inference-serving-optimization/01-scheduler-budget-spec-decode-cudagraph.md)를 참고한다.

---

# 20. DFlash의 버전 경계가 주는 교훈

v0.26/v0.27 generic code는:

```text
parallel_drafting → K-1
```

로 처리했다.

하지만 DFlash는 실제로:

```text
bonus query + K masks
```

라서 K가 필요했다.

upstream PR #51256:

**Reserve the bonus query slot in DFlash scheduling budget**

에서 이 문제를 명시적으로 수정했다.

이 사례가 중요한 이유:

> "parallel drafting"이라는 큰 분류만으로 MBT 식을 일반화하면 안 된다.  
> 실제 drafter가 scheduler-existing query를 어떻게 reuse/shift하고 몇 개의 query positions를 구성하는지까지 봐야 한다.

DSpark와 DFlash가 둘 다 parallel block drafter여도 S가 다른 이유가 바로 이것이다.

---

# 21. 버전별 tuning 공식

## 21.1 v0.26/v0.27

목표 scheduler budget을 `T_target`이라고 하면:

```math
B \ge T_{target} + SM
```

을 시작점으로 잡는다.

DSpark K=7:

```math
B \ge T_{target} + 6M
```

### 운영 순서

1. 실제 필요한 engine당 sequence concurrency M을 먼저 정한다.
2. speculative method와 K에서 S를 계산한다.
3. 원하는 prefill/decode scheduler work budget `T_target`을 정한다.
4. `B >= T_target + SM`을 만족하도록 MBT를 잡는다.
5. startup log에서 실제 resolved MNS/T를 확인한다.
6. TTFT/TPOT/throughput을 workload mix별로 검증한다.

### 하지 말아야 할 것

```text
MNS는 안 건드리고
MBT만 4K/8K/16K로 찍어보는 tuning
```

parallel SD에서는 MNS가 식에 직접 들어가므로 원인 해석이 어려워진다.

---

## 21.2 v0.28

static reserve가 없으므로 기본식은 runtime envelope다.

```math
\sum_i (n_i + S) \le B
```

### 운영 순서

1. MNS는 실제 concurrency/admission cap 기준으로 정한다.
2. method/K에서 S를 계산한다.
3. workload에서 step당 scheduled request 수와 prefill/decode mix를 본다.
4. B가 `scheduler work + actual drafting expansion`을 수용하도록 정한다.
5. T는 특별한 scheduler throttling 목적이 없으면 auto를 우선한다.
6. low/high concurrency를 분리해서 benchmark한다.

v0.28에서는 MNS를 크게 잡았다는 이유만으로 B가 startup 시 static하게 잠기지는 않는다.

---

# 22. 관측해야 할 값

SD MBT 문제를 튜닝할 때 최소한 다음 값을 같이 기록한다.

## Resolved config

```text
vLLM version
usage context
GPU class
max_num_batched_tokens B
max_num_scheduled_tokens T
max_num_seqs M
speculative method
num_speculative_tokens K
parallel_drafting
max_num_new_slots_for_drafting S
chunked prefill
performance mode
```

## Runtime step

```text
scheduled request count N
sum(num_scheduled_tokens)
prefill vs decode/spec mix
drafted token count
accepted token count
acceptance length
```

## Performance

```text
TTFT p50/p90/p99
TPOT / ITL
request throughput
aggregate output tok/s
per-user output tok/s
GPU utilization
HBM usage
KV cache usage / preemption
```

특히:

```text
low concurrency
medium concurrency
saturation
```

을 분리해야 한다.

static reserve 문제는 low concurrency에서 가장 크게 드러난다.

---

# 23. 진단 체크리스트

## startup에서 `max_num_scheduled_tokens <= 0`

v0.26/v0.27이면 우선 계산:

```text
S = method별 extra slots
reserve = S * resolved max_num_seqs
T = MBT - reserve
```

을 확인한다.

---

## MBT를 늘렸는데 TTFT만 개선

가능성:

```text
static scheduler budget 증가
→ chunked prefill chunk 증가
→ prefill iteration 수 감소
```

즉 SD acceptance가 좋아진 것이 아닐 수 있다.

---

## MNS를 줄였더니 v0.26에서 갑자기 빨라짐

가능성:

```text
SM static reserve 감소
→ T 증가
```

다.

단순히 "동시성이 낮아져서"라고만 해석하면 안 된다.

---

## v0.28로 올렸더니 같은 MBT에서 low-concurrency TTFT 개선

가능성:

```text
static reserve 제거
→ actual-request adaptive headroom
→ prefill/scheduler capacity 회수
```

다.

---

## DFlash와 DSpark에 같은 K/MBT를 줬는데 behavior가 다름

v0.28 기준:

```text
DSpark S = K-1
DFlash S = K
```

부터 다르다.

또한 KV lookahead/input buffer/cudagraph 구현 차이가 있으므로 MBT 식만으로 동일 behavior를 기대하면 안 된다.

---

# 24. Source code map

## v0.26.0

### Speculative slot calculation

- `vllm/config/speculative.py`
- `SpeculativeConfig.max_num_new_slots_for_drafting`

https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/speculative.py

### Static scheduled-token reservation

- `vllm/config/vllm.py`
- `VllmConfig._set_max_num_scheduled_tokens`

https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/vllm.py

### Runtime scheduler token budget

- `vllm/v1/core/sched/scheduler.py`

https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/sched/scheduler.py

### MBT/MNS platform defaults and CLI

- `vllm/engine/arg_utils.py`

https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/engine/arg_utils.py

---

## v0.27.0

구조는 v0.26과 같은 static reservation 계열이다.

- https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/config/speculative.py
- https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/config/vllm.py
- https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/v1/core/sched/scheduler.py
- https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/engine/arg_utils.py

---

## v0.28.0

### Method-specific drafting slots

- `vllm/config/speculative.py`
- `max_num_new_slots_for_drafting`

https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/speculative.py

### Adaptive config initialization

- `vllm/config/vllm.py`
- `_set_max_num_scheduled_tokens`

https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/vllm.py

### Runtime dual-budget accounting

- `vllm/v1/core/sched/scheduler.py`
- `token_budget`
- `input_budget`
- `draft_slots`

https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/core/sched/scheduler.py

### Method slot unit test

- `tests/test_config.py`
- `test_max_num_new_slots_for_drafting`

https://github.com/vllm-project/vllm/blob/v0.28.0/tests/test_config.py

---

# 25. 주요 upstream 변경 PR

## PR #51725 — Adaptive budget for spec scheduled token

https://github.com/vllm-project/vllm/pull/51725

merged: 2026-08-11  
merge commit: `0914ed2e816fa1987951f22afba1c00b7786f110`

핵심:

```text
old:
T = B - S*M

new:
T = B
runtime input_budget에서 actual scheduled request마다 S 차감
```

---

## PR #51256 — DFlash bonus query slot fix

https://github.com/vllm-project/vllm/pull/51256

merged: 2026-08-13  
merge commit: `2ac1f683f11cf0f526c896e5fa5e0e22ebc6ffa2`

핵심:

```text
generic parallel drafting:
S = K-1

DFlash actual:
S = K
```

로 수정.

---

# 26. 가장 중요한 식만 다시 모으면

## v0.26 / v0.27

```text
S = method-specific additional drafting slots/request

static_reserve = S * max_num_seqs

max_num_scheduled_tokens
= max_num_batched_tokens - static_reserve
```

즉:

```math
T = B - SM
```

---

## v0.28

```text
token_budget = T
input_budget = B

for each actually scheduled request i:
    schedule n_i tokens

    token_budget -= n_i
    input_budget -= n_i + S
```

즉:

```math
\sum_i n_i \le T
```

그리고:

```math
\sum_i (n_i + S) \le B
```

---

# 27. 본질

`max_num_batched_tokens`를 단순 "사용자 token 처리량"으로 보면 speculative decoding scheduler를 이해하기 어렵다.

더 정확한 구조는:

```text
                  ┌─ scheduler-issued work
MBT input envelope┤
                  └─ drafter-added slot headroom
```

이다.

v0.26/v0.27은 두 번째 영역을 `max_num_seqs` 기준으로 미리 잘라냈다.

v0.28은 두 영역을 별도 runtime budget으로 관리하면서 실제 request가 들어올 때만 headroom을 사용한다.

> **Speculative decoding에서 MBT tuning의 본질은 "몇 개의 output token을 만들 것인가"가 아니라, 한 iteration의 제한된 input-slot envelope를 scheduler work와 speculative expansion 사이에서 어떻게 나눌 것인가에 있다.**

---

# 28. 관련 문서

- [Speculative Decoding: 전통 SD, MTP, DFlash, DSpark의 구조적 차이](2026-08-18-speculative-decoding-mtp-dspark.md)
- [Scheduler Budget, Speculative Decoding & CUDA Graph](../inference-serving-optimization/01-scheduler-budget-spec-decode-cudagraph.md)
- [DeepSeek-V4: MTP, DSpark & Speculative Decoding](../../models/deepseek-v4/05-mtp-dspark-and-speculative-decoding.md)
- [DeepSeek-V4 DSpark checkpoint와 vLLM 구현](../../models/deepseek-v4/2026-08-18-dspark-checkpoint-and-vllm-implementation.md)
