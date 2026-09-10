# Frontier Model Family Compatibility Lanes

업데이트: 2026-09-10 KST

## 1. 목적

vLLM version migration은 generic engine benchmark 하나로 인증할 수 없다.

현재 주요 serving target인:

- Qwen3.5+ / Qwen3.6 / Qwen3.8
- Kimi K3
- DeepSeek V4
- GLM 5.x

는 서로 다른 architecture-specific state, attention, MoE, speculative decode, parser, parallelism path를 사용한다.

따라서 공통 engine certification 위에 **model-family별 별도 compatibility lane**을 둔다.

이 문서는 두 종류의 근거를 명확히 구분한다.

1. **v0.29 tag 포함 사항**: 실제 0.29 candidate의 behavior
2. **post-tag/open upstream watchlist**: 0.29에 아직 고쳐지지 않았을 수 있는 알려진 위험. 절대 0.29 기능으로 혼동하지 않는다.

---

## 2. 공통 certification skeleton

모든 frontier model은 최소 다음 순서를 따른다.

```text
checkpoint/config inspection
 -> model registry / architecture resolution
 -> parser/template resolution
 -> plain eager correctness
 -> compiled/CUDA Graph correctness
 -> prefix/cache correctness
 -> speculative decode
 -> TP/DP/EP/DCP
 -> P/D connector
 -> long-context + soak
```

각 단계에서 이전 단계의 변수를 고정한다.

---

# Part A. Qwen3.5+ family

## 3. Qwen3.5 family architecture contract

v0.29 `qwen3_5.py`는 Qwen3.5 series를 명시적으로 hybrid model로 구현한다.

Model class는:

```text
HasInnerState
IsHybrid
SupportsEagle3
SupportsMRoPE
SupportsPP
```

를 구현하고 decoder layer는 `layer_type`에 따라:

```text
linear_attention
 -> QwenGatedDeltaNetAttention

full_attention
 -> Qwen3NextAttention
```

으로 분기한다.

또 GDN state의 dtype/shape는 model-specific calculator로 계산하며 speculative token 수까지 state shape 계산에 반영된다.

Source:
- v0.29 `qwen3_5.py`: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/qwen3_5.py
- initial text-only support PR #50210: https://github.com/vllm-project/vllm/pull/50210

### 의미

Qwen3.5+를 일반 dense attention-only model로 취급하면 다음 계산이 틀릴 수 있다.

```text
context memory
prefix cache
P/D transferable state
MTP scratch state
max concurrency
```

---

## 4. Qwen3.5 Mamba/GDN cache restriction

v0.29 model source는:

```text
mamba_cache_mode == all
```

을 Qwen3.5에서 `NotImplementedError`로 거부하고 `align` 사용을 요구한다.

즉 generic Mamba config에 `all` mode가 존재한다고 해서 Qwen3.5 family가 이를 지원하는 것은 아니다.

### Deployment rule

Qwen3.5+/3.6/3.8 hybrid deployment에서는 effective:

```text
mamba_cache_mode
prefix_cache_retention_interval
mamba state dtype
mamba state shape
```

를 startup manifest에 남긴다.

---

## 5. Qwen3.6: v0.29 tag에서 확인할 핵심

PR #52676은 Qwen3.6의 gated Q/K normalization + partial MRoPE + gate를 fused operator로 확장했다.

Source:
- https://github.com/vllm-project/vllm/pull/52676

PR은 production compiled/CUDA Graph path까지 검증했다.

이 변경의 의미는 단순 kernel optimization보다 넓다.

Qwen3.6은:

```text
per-head [q | gate]
GemmaRMSNorm
partial/interleaved MRoPE
```

같은 architecture-specific attention preprocessing을 사용한다.

따라서 migration correctness test는:

```text
EAGER
vs
compiled + CUDA Graph
```

둘 다 수행한다.

Fallback와 fused path가 동일한 token output을 내는지도 확인한다.

---

## 6. Qwen3.8-27B와 Qwen3.8-Flash-Next를 혼동하지 않는다

Qwen3.8이라는 이름 아래 서로 다른 architecture/checkpoint line이 존재할 수 있다.

### Qwen3.8-27B

Qwen3.5-family hybrid/GDN lane으로 취급해야 하는 checkpoint는 실제 HF config의:

```text
model_type
architectures
layer_types
```

를 기준으로 판정한다.

현재 upstream의 Qwen3.5 MTP test에서도 `qwen3_5` / `qwen3_5_moe` family mapping을 별도로 유지한다.

### Qwen3.8-Flash-Next

v0.29 PR #53896으로 별도 model support가 들어왔다.

Source:
- https://github.com/vllm-project/vllm/pull/53896

PR validation에는:

```text
BF16 / FP8 / NVFP4
GB300 / GB200 / H200
TP2 / TP4 / TEP4
MTP k=3 example
```

가 포함된다.

따라서:

```text
Qwen3.8-27B baseline
```

과:

```text
Qwen3.8-Flash-Next
```

의 benchmark/config를 하나의 family row로 합치지 않는다.

---

## 7. Qwen parser lane

Qwen 계열의 기본 parser pair는 checkpoint/template에 맞춰 명시적으로 검증한다.

대표적으로:

```text
reasoning-parser: qwen3

tool-call-parser:
  qwen3_xml
  qwen3_coder
  model/template에 맞는 variant
```

### Required cases

```text
thinking on
thinking off
reasoning -> content
reasoning -> tool
required tool
auto tool
parallel tools
streaming JSON arguments
Korean/unicode argument
```

Parser option name이 같더라도 vLLM version별 adapter implementation이 바뀔 수 있다.

---

## 8. Qwen speculative decoding lane

Qwen3.5+에서는 다음을 별도 lane으로 본다.

```text
plain
MTP
EAGLE3 where supported
DFlash/DFlash2 where explicitly validated
```

특히 GDN state shape 계산이 `num_speculative_tokens`의 영향을 받기 때문에 SD는 단순 verifier 추가가 아니다.

### Record

```text
method
num_speculative_tokens
draft dtype
draft quantization
acceptance rate
accepted length
max_num_scheduled_tokens
Mamba/GDN cache mode
MRV version
```

---

## 9. Qwen3.5-family v0.29 위험: FP8 KV calibrated scale loading

**중요한 post-tag/open watchlist다. v0.29에 수정된 내용이 아니다.**

Open PR #54624는 Qwen3.5 family의 `load_weights()`가 checkpoint의:

```text
self_attn.k_scale
self_attn.v_scale
```

를 runtime parameter path로 remap하지 않아 `--kv-cache-dtype fp8`에서 scale 1.0으로 동작할 수 있다고 보고한다.

Source:
- open PR #54624: https://github.com/vllm-project/vllm/pull/54624

v0.29 tag의 `Qwen3_5Model.load_weights()`도 현재:

```python
loader = AutoWeightsLoader(self)
return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

형태이며 해당 KV scale remap이 보이지 않는다.

### 왜 심각한가

Checkpoint가 calibrated K/V scale을 제공하는데 runtime이 1.0을 쓰면 FP8 E4M3 dynamic range clipping이 발생할 수 있다.

PR 작성자의 reproducer에서는 Qwen3.8-27B NVFP4 checkpoint가 calibrated scale을 포함하는 사례를 들고 있다.

### 0.29 candidate gate

FP8 KV를 사용하는 Qwen3.5-family 모델은 반드시 startup에서:

```text
checkpoint에 k_scale/v_scale가 있는지
실제로 runtime parameter로 load됐는지
scale 값이 1.0 fallback인지
```

를 확인한다.

이 검증을 통과하기 전에는 단순 `v0.29 + fp8 KV startup success`를 production-ready로 판단하지 않는다.

---

## 10. Qwen post-tag DFlash2 dtype watchlist

역시 v0.29 tag 기능이 아니라 open upstream issue/fix다.

Open PR #55294는 BF16-trained Qwen3.8 DFlash2 draft를 FP16으로 cast할 경우 grouped-convolution residual에서 inf/NaN이 발생해 draft acceptance가 0이 될 수 있다고 보고한다.

Source:
- https://github.com/vllm-project/vllm/pull/55294

Reported reproducer:

```text
FP16 draft -> 1100+ drafts, 0 accepted
BF16 draft -> 37~41% acceptance
```

### Rule

DFlash2를 사용한다면 draft dtype을 implicit target inheritance에 맡기지 않고 explicit 검증한다.

---

## 11. Qwen hybrid+MTP concurrency watchlist

Open PR #55617은:

```text
Hybrid GDN (Qwen3.5/Qwen3.8 27B-class) + MTP
```

조합에서 batch >=4 시 약 3 concurrent sequence 관련 이슈를 추적 중이다.

Source:
- https://github.com/vllm-project/vllm/pull/55617

현재 WIP 수준이므로 원인/해결책으로 단정하지 않는다.

하지만 Qwen3.8 27B + MTP production validation에서 **동시성 1 성공만으로 충분하지 않다는 강한 신호**다.

반드시:

```text
concurrency 1 / 2 / 4 / 8 / 16+
```

를 올리며 cache/state allocation behavior를 확인한다.

---

# Part B. Kimi K3

## 12. K3는 full-stack compatibility target이다

Kimi K3 umbrella PR #50000은 모델 하나를 등록한 수준이 아니다.

분리된 landing 영역:

```text
model/kernels
Python frontend
Rust frontend
FlashInfer / MLA dependencies
DSpark
```

Source:
- https://github.com/vllm-project/vllm/pull/50000

따라서 K3 qualification은 engine-only test와 parser-only test를 둘 다 요구한다.

---

## 13. K3 recurrent-state / prefix-cache lane

K3는 KDA/Mamba-like state를 사용하므로:

```text
mamba_cache_mode
retention interval
internal checkpoint
speculative state
P/D transfer
```

를 모두 본다.

PR #52789의 internal prefill checkpoint는 K3에서 두 번째 full-model forward를 제거해 TTFT를 크게 줄이는 path다.

Source:
- https://github.com/vllm-project/vllm/pull/52789

따라서 0.26/초기 K3와 0.29 K3의 TTFT 차이를 kernel 성능 하나로 해석하지 않는다.

---

## 14. K3 Mamba metadata preparation

v0.29에 포함된 PR #52388은 여러 KV-cache group의 Mamba align metadata를 group별 kernel launch가 아니라 **하나의 multi-group Triton launch**로 만든다.

Source:
- https://github.com/vllm-project/vllm/pull/52388

Reported microbenchmark는 약 6.6~7.6x metadata-kernel speedup이다.

이는 decode small-batch에서 CPU/launch overhead가 중요한 K3 특성에 영향을 준다.

### Test

```text
batch 1 / 4 / 16 / 64
plain decode
DSpark
prefix-cache hit/miss
```

에서 TPOT와 kernel launch profile을 비교한다.

---

## 15. K3 parser lane

v0.29 registry에는:

```text
reasoning-parser = kimi_k3
tool-call-parser = kimi_k3
```

가 들어온다.

Required corpus:

- reasoning_effort max/default/none where supported
- reasoning -> tool
- multiple tool call
- reserved structural marker가 content에 노출되지 않는지
- stream/non-stream parity

K3는 parser와 model architecture가 동시에 빠르게 발전하고 있으므로 parser config를 image default에 맡기지 않는다.

---

## 16. K3 P/D / external cache lane

K3는 Mooncake Mamba-state persistence bug #51358 같은 문제의 직접적인 검증 사례다.

특히:

```text
prefix cache + DSpark/EAGLE + external state
```

는 poisoned recurrent state가 later hit에서 재사용되는지 확인해야 한다.

P/D acceptance에는 반드시 repeated-prefix soak를 넣는다.

---

# Part C. DeepSeek V4

## 17. DSV4는 kernel + sparse MLA + MoE + P/D lane이다

DSV4는 버전별 변화가 매우 크다.

0.29에서 특히 보는 축:

```text
sparse MLA
MTP / DSpark
MoE / EP
shared expert fusion
CUDA Graph regions
P/D
quantization
```

따라서 model version과 vLLM version을 함께 기록한다.

예:

```text
DeepSeek-V4-Flash-0731
vLLM 0.29.0
```

을 하나의 certification tuple로 본다.

---

## 18. DSV4 shared-expert MegaMoE

PR #53040은 NVIDIA SM100 path에서 replicated FP8 shared expert를 persistent MegaMoE kernel에 fuse한다.

Source:
- https://github.com/vllm-project/vllm/pull/53040

기존:

```text
routed MegaMoE
 -> shared gate/up
 -> shared down
 -> separate add
```

새 path:

```text
shared L1
+ routed dispatch/MMA
+ shared L2
+ FP32 accumulation
```

을 한 native scheduling path로 묶는다.

PR benchmark는 여러 workload에서 output throughput/TPOT가 의미 있게 개선된 사례를 보고한다.

### Platform implication

DSV4에서 `moe_backend`, shared-expert fusion, hardware architecture(SM90/SM100)는 hidden variable이 아니다.

Benchmark manifest에 명시한다.

---

## 19. DSV4 P/D

과거 PR #45831이 보여준 것처럼 DSV4는 `SlidingWindowMLASpec` 같은 model-specific cache region이 connector classification에 포함되지 않으면 P/D가 깨질 수 있었다.

따라서 0.29에서도:

```text
P/D supported = connector starts
```

가 아니라:

```text
all cache groups transferable
+ long-context correctness
+ partial hit
+ spec decode
```

를 검증한다.

---

## 20. DSV4 spec lane

0.29 release는 DSV4 sparse MLA를:

```text
plain decode
MTP
DSpark
```

에 걸쳐 강화한다.

각 method는 별도 benchmark row로 남긴다.

DSpark/MTP의 효율은 acceptance만 보지 말고:

```text
scheduled token budget
verification cost
MoE communication
CUDA Graph coverage
```

를 같이 본다.

---

# Part D. GLM 5.x

## 21. GLM 5.x는 parser와 sparse-attention 두 lane이 모두 중요하다

GLM 5.x는 frontend parser뿐 아니라 DeepSeek-V3.2-style sparse MLA/DSA execution path를 공유/응용하는 부분이 있다.

따라서:

```text
output garbling
repeated token
empty content
```

같은 증상을 보면 parser만 혹은 kernel만 먼저 단정하지 않는다.

Engine raw token과 parsed API response를 분리해서 확인한다.

---

## 22. GLM-5.2 dense-MHA dispatch correctness

PR #52512는 short-prefill에서 GLM-5.2 output이 repeated token으로 무너질 수 있는 실제 execution bug를 수정했다.

Source:
- https://github.com/vllm-project/vllm/pull/52512

문제:

```text
generic sparse-MLA metadata
 -> short prefill이면 dense MHA 가능하다고 판단
 -> sparse indexer top-k scoring skip

하지만 actual DeepSeekV32Attention wrapper
 -> dense MHA를 실행하지 않고 MQA만 실행
 -> cleared -1 top-k buffer consume
 -> repeated-token degeneration
```

Fix는 backend capability뿐 아니라 **model-layer가 dense prefill을 실제 실행할 수 있는지**를 조건에 포함한다.

### Lesson

Attention backend capability와 model wrapper capability는 동일하지 않다.

GLM deployment에서 backend 이름만 확인해서는 안 된다.

---

## 23. GLM DCP query replication

PR #50382는 GLM sparse-attention model에서 decode DCP 사용 시 query replication을 기본 활성화하는 path를 추가했다.

Source:
- https://github.com/vllm-project/vllm/pull/50382

Trade-off:

```text
QREP ON
 -> decode query all-gather 제거 가능
 -> query projection weights를 DCP ranks에 replicate
```

PR의 GLM-5.2-NVFP4 TP4/DCP4 사례에서 model-loading memory가 rank당 약 +4.57 GiB 증가한 측정이 있다.

즉 latency 개선과 memory overhead를 함께 평가해야 한다.

### B300/H200 implication

VRAM이 큰 B300에서는 memory trade-off가 상대적으로 받아들이기 쉬울 수 있지만, H100/H200에서도 모델/TP topology에 따라 headroom을 확인해야 한다.

QREP default를 hardware memory만 보고 자동 승인하지 않는다.

---

## 24. GLM parser lane

GLM4.7/5.1/5.2 parser는 Streaming Parser Engine으로 migration됐다.

Source:
- PR #45915: https://github.com/vllm-project/vllm/pull/45915

Required:

```text
thinking true / false
stream true / false
reasoning -> final
reasoning -> tool
strict structured tool
```

Attention correctness benchmark와 parser regression corpus를 서로 독립적으로 통과시킨다.

---

# Part E. Cross-model validation table

## 25. 기능별 우선순위

| Axis | Qwen3.5+/3.6/3.8 | Kimi K3 | DeepSeek V4 | GLM 5.x |
|---|---|---|---|---|
| MRV2 migration | **P0** | **P0** | **P0/P1** | **P0/P1** |
| Hybrid recurrent state | **P0** | **P0** | model-specific | model-specific |
| Prefix retention/replay | **P0** | **P0** | P1 | P1 |
| FP8 KV | **P0** | P1 | **P0/P1** | **P0/P1** |
| MTP | **P0/P1** | P1 | **P0** | model-dependent |
| DSpark | model-dependent | **P0/P1** | **P0/P1** | model-dependent |
| MoE/EP | Qwen3.5 MoE only | **P0** | **P0** | model-dependent |
| DCP/PCP | P1 | **P0/P1** | **P1** | **P0** |
| P/D | **P0** | **P0** | **P0** | P1 |
| reasoning parser | **P0** | **P0** | **P0** | **P0** |
| tool parser | **P0** | **P0** | **P0** | **P0** |
| CUDA Graph | **P0/P1** | **P0/P1** | **P0** | **P0/P1** |
| collective backend | P1 | **P0/P1** | **P0** | **P0/P1** |

---

## 26. Standard benchmark shapes

모델별 특수 workload에 더해 공통 shape를 유지한다.

### Short/decode-heavy

```text
ISL 128 / OSL 2K
ISL 1K  / OSL 1K
```

### Long-context baseline

```text
ISL 32K  / OSL 2K
ISL 128K / OSL 2K
ISL 170K / OSL 2K
```

### Boundary

```text
max-model-len 근처 single request
```

### Concurrency

```text
1
4
8
16
model-capacity boundary
```

Hybrid+MTP model은 반드시 batch/concurrency를 높여 state-capacity regression을 찾는다.

---

## 27. Correctness corpus

### Generation

- greedy deterministic prompts
- long-context retrieval
- Korean/English mixed
- EOS behavior
- 2K output stability

### Reasoning

- thinking on/off
- reasoning boundary
- no reasoning case

### Tool

- required
- auto
- named
- parallel
- nested JSON

### P/D

- cold
- repeated prefix
- partial prefix
- producer/decode restart

### Spec

- plain vs SD output quality
- acceptance
- zero-acceptance detection

---

## 28. Model release manifest

Production catalog에 모델마다 다음을 저장한다.

```yaml
model: qwen3.8-27b
checkpointRevision: <sha>
tokenizerRevision: <sha>
architecture: <resolved>
modelType: <resolved>
vllmVersion: 0.29.0
modelRunner: v2
maxModelLen: 262144
kvCacheDtype: fp8
mambaCacheMode: align
prefixCacheRetentionInterval: 0
reasoningParser: qwen3
toolParser: qwen3_xml
speculative:
  method: mtp
  tokens: 3
parallel:
  tp: 4
  dp: 1
  ep: 1
kvTransfer:
  connector: mooncake
  packageVersion: 0.3.13
```

이 manifest를 model deployment의 source of truth로 사용한다.

---

## 29. v0.29 candidate-specific blocker/watchlist

### Qwen3.5 family + FP8 KV

Open #54624가 v0.29 source에도 존재할 가능성이 높은 calibrated KV-scale loading gap을 지적한다.

**FP8 KV production candidate에서는 P0 validation item.**

### Qwen3.8 DFlash2

Open #55294에 따라 draft dtype을 반드시 검증한다.

### Hybrid GDN + MTP concurrency

Open #55617은 원인이 아직 확정되지 않은 WIP다. 그러나 high-concurrency soak를 의무화하는 이유로 충분하다.

### 다른 모델

0.29 release 직후 새 regression이 발견될 가능성이 높으므로 version audit은 release 시점에 닫는 문서가 아니라 **tag + post-tag watchlist** 방식으로 유지한다.

다만 post-tag item은 반드시 상태를 표시한다.

```text
OPEN / MERGED after tag / NOT IN v0.29
```

---

## 30. 최종 판단

최신 frontier model의 vLLM compatibility는:

```text
"지원 모델 목록에 이름이 있다"
```

가 아니라:

```text
model architecture
+ state/cache semantics
+ model runner
+ parser
+ speculative method
+ parallel topology
+ native kernels
+ KV transfer
+ exact dependency versions
```

의 tuple이다.

특히 Qwen3.5+와 Kimi K3는 hybrid recurrent state 때문에 **state-lifecycle certification**, DSV4와 GLM 5.x는 sparse attention/MoE/DCP 때문에 **backend-path certification**이 핵심이다.

---

## Sources

- Qwen3.5 text-only support #50210: https://github.com/vllm-project/vllm/pull/50210
- Qwen3.5 v0.29 source: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/qwen3_5.py
- Qwen3.6 fused QK/MRoPE/gate #52676: https://github.com/vllm-project/vllm/pull/52676
- Qwen3.8-Flash-Next #53896: https://github.com/vllm-project/vllm/pull/53896
- Kimi K3 umbrella #50000: https://github.com/vllm-project/vllm/pull/50000
- Kimi Mamba metadata #52388: https://github.com/vllm-project/vllm/pull/52388
- Kimi internal checkpoint #52789: https://github.com/vllm-project/vllm/pull/52789
- DSV4 P/D fix #45831: https://github.com/vllm-project/vllm/pull/45831
- DSV4 MegaMoE #53040: https://github.com/vllm-project/vllm/pull/53040
- GLM-5.2 dense-MHA correctness #52512: https://github.com/vllm-project/vllm/pull/52512
- GLM DCP Q replication #50382: https://github.com/vllm-project/vllm/pull/50382
- GLM parser #45915: https://github.com/vllm-project/vllm/pull/45915

### Open/post-tag watchlist — NOT v0.29 fixes

- Qwen3.5-family KV scale loading #54624: https://github.com/vllm-project/vllm/pull/54624
- Qwen3.8 DFlash2 FP16 #55294: https://github.com/vllm-project/vllm/pull/55294
- hybrid GDN + MTP concurrency #55617: https://github.com/vllm-project/vllm/pull/55617
