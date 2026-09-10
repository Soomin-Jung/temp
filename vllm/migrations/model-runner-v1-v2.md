# Model Runner V1 -> V2 Migration Deep Dive

업데이트: 2026-09-10 KST

## 1. 왜 MRV1/MRV2를 별도 migration 축으로 봐야 하는가

vLLM 0.26 -> 0.29 migration에서 `Model Runner V2`는 단순 내부 refactor가 아니다.

Model Runner는 실제로 다음 execution surface를 소유하거나 강하게 결합한다.

```text
input batch preparation
attention/Mamba metadata
CUDA Graph dispatch
KV cache initialization
sampling/logits path
speculative draft/verification path
multimodal encoder path
parallel communication timing
sleep/offload lifecycle
```

따라서 같은 model + 같은 CLI + 같은 GPU여도 MRV1과 MRV2가 다르면 다른 serving implementation을 테스트하는 것으로 봐야 한다.

---

## 2. Rollout timeline

## v0.24 이전/초기

V1이 사실상 안정된 기본 execution path이고 V2는 selective architecture/feature rollout 단계다.

## v0.25

PR #44443이 ordinary dense generate model을 MRV2 default로 넓힌다.

하지만 실제 source는 명시적으로:

```python
if model_config.is_hybrid:
    return False

if model_config.is_attention_free:
    return False
```

를 수행했다.

그리고 일부 known MoE architecture + non-MoE dense model만 V2 default가 됐다.

즉 당시 selection model은:

```text
known/ordinary V2-capable model?
  yes -> V2
  no  -> V1
```

이었다.

Source:
- PR #44443: https://github.com/vllm-project/vllm/pull/44443

## v0.26

MRV2 capability가 계속 늘어나지만 hybrid/Mamba/GDN 계열은 여전히 일반적인 default-V2 대상으로 보면 안 된다.

Qwen3.5+처럼 이후 hybrid model이 많아지는 시점에서 이 historical state가 중요하다.

## v0.27

MRV2가:

- encoder-only attention
- pooling
- multi-layer MTP
- more multimodal paths
- PCP-related execution

으로 확장된다.

그러나 selection philosophy는 여전히 selective rollout에 가깝다.

## v0.28

MRV2가 중요한 missing capability를 채운다.

- E/P/D disaggregation: #38390
- multi-layer MTP KV cache: #50062
- encoder CUDA graphs
- attention-free models: #52374
- weight offloading
- thinking token budget

이 release가 0.29 all-default의 기술적 전제다.

## v0.29

PR #53183이 selection policy를 뒤집는다.

이전:

```text
V2를 쓸 수 있는 모델을 고른다.
```

0.29:

```text
기본은 V2다.
V1이 필요한 예외/unsupported condition을 찾는다.
```

이 차이가 핵심이다.

Source:
- PR #53183: https://github.com/vllm-project/vllm/pull/53183

---

## 3. PR #53183 source-level 의미

실제 patch는 기존:

```text
DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES
```

whitelist를 제거한다.

또한 과거 V2를 강제로 선택하던 special-case도 다수 사라진다.

예:

- PCP
- DSpark
- mixed sliding/full DFlash
- DFlash2
- diffusion

과거에는:

```text
이 기능은 V1에서 안 되므로 강제로 V2
```

가 필요했다.

0.29에서는 기본이 V2라 special forcing이 불필요해진다.

동시에 V1을 명시적으로 쓸 경우 `_get_v1_model_runner_unsupported_features()`를 통해 지원 불가능한 기능을 검사한다.

즉 **fallback behavior도 더 explicit하게 validation되는 방향**으로 바뀐다.

---

## 4. MRV1이 사라진 것은 아니다

`MRV2 default for all models`를 `MRV1 removed`로 해석하면 안 된다.

0.29 source는 ROCm에서:

```text
DeepseekV32ForCausalLM
DeepseekV4ForCausalLM
```

을 default MRV1 architecture 예외로 남긴다.

또 MRV2가 아직 지원하지 않는 feature가 있으면 V2 validation이 실패할 수 있다.

따라서 operationally는 다음 세 상태가 있다.

```text
A. MRV2 default
B. MRV1 explicit/exception
C. requested feature와 runner가 충돌하여 startup validation failure
```

이 세 가지를 observability에 드러내야 한다.

---

## 5. Qwen3.5+에 미치는 영향

Qwen3.5 text-only support PR #50210은 해당 architecture를 hybrid로 등록하고 GDN state shape/dtype/copy hook을 제공한다.

즉 Qwen3.5+ 계열은:

```text
ordinary attention KV
+ recurrent/GDN state
```

를 함께 갖는 hybrid execution path다.

v0.26 baseline에서는 hybrid라는 이유로 MRV1 계열 path를 타는 경우가 있고, v0.29에서는 MRV2가 default다.

따라서 Qwen3.6/3.8 migration에서 다음 변화가 한 번에 섞일 수 있다.

```text
MRV1 -> MRV2
Mamba/GDN metadata implementation
prefix-cache policy
CUDA Graph manager
MTP speculator
KV transfer state handling
```

Qwen benchmark에서 0.26 vs 0.29만 비교하면 이 변수들을 분리할 수 없다.

---

## 6. Kimi K3에 미치는 영향

Kimi K3는 MRV migration의 가장 복잡한 validation target 중 하나다.

K3 execution은 대략:

```text
KDA / recurrent state
+ MLA-like attention path
+ MoE
+ EP communication
+ DSpark
+ prefix caching
+ DCP
+ P/D
```

가 함께 동작한다.

0.29에서 K3 관련 최적화가 대부분 MRV2 기반으로 진행되므로 장기적으로 V2를 production target으로 보는 것이 자연스럽다.

하지만 migration phase에서는 MRV version 자체가 결과 차이의 원인인지 확인할 수 있는 baseline을 남긴다.

---

## 7. DeepSeek V4 / GLM 5.x

DSV4와 GLM 계열은 dense transformer보다 runner migration 영향이 더 넓다.

### DSV4

- sparse MLA
- MTP/DSpark
- MoE
- EP
- graph break regions
- P/D

### GLM 5.x

- sparse attention/MLA
- head padding/special kernel
- DCP query replication
- reasoning/tool frontend

따라서 MRV2 acceptance를 단순 output text 비교로 끝내면 안 된다.

---

## 8. CUDA Graph와 MRV2

0.29 PR #53306은 MRV2의 `profile_cudagraph_memory()`를 실제 구현한다.

Source flow의 중요한 부분:

1. profiling용 KV cache를 임시 초기화
2. KV connector는 NO-OP으로 대체
3. Mamba layer는 profiling 시 cheap warmup path를 사용
4. attention metadata/kernel은 tuning/profile이 가능하도록 준비
5. graph memory estimate를 KV auto-sizing에 반영

Source:
- PR #53306: https://github.com/vllm-project/vllm/pull/53306

### Implication

MRV2 migration과 CUDA Graph migration은 독립적이지 않다.

```text
runner change
 -> graph profiling change
 -> non-KV reserved bytes change
 -> KV capacity change
 -> max concurrency change
```

가 될 수 있다.

그러므로 MRV1/MRV2 test에서 반드시 memory report를 같이 캡처한다.

---

## 9. Sampling / logits memory

v0.29 MRV2에는 batch-sharded sampling이 들어와 TP 환경에서 per-step logits memory를 줄이는 path가 존재한다.

이 역시 `TP size`와 runner implementation이 결합한다.

특히 large vocabulary + high max_num_seqs 환경에서는 logits memory가 non-KV headroom에 유의미할 수 있다.

따라서 decode OOM/headroom 분석에서:

```text
KV cache만 크다/작다
```

로 결론내리지 않는다.

---

## 10. Speculative decoding과 MRV

0.29 source에서 V1 unsupported feature validation에 포함되는 대표 항목:

- DSpark speculative decoding
- DFlash2
- mixed sliding/full DFlash drafts
- PCP
- batch-sharded sampling
- diffusion

즉 최신 spec decode feature는 사실상 MRV2와 강하게 묶인다.

### Migration order

잘못된 순서:

```text
0.26 + MTP
 -> 0.29 + MRV2 + new async scheduler + new MTP + new graph
```

권장 순서:

```text
0.26 baseline

0.29 MRV2 plain decode
 -> correctness

0.29 MRV2 + graph
 -> graph effect

0.29 MRV2 + MTP
 -> spec effect

0.29 MRV2 + MTP + async
 -> scheduler effect
```

---

## 11. KV transfer와 MRV2

v0.28에 MRV2 E/P/D support가 들어간 것은 중요한 경계다.

P/D에서는 model runner가:

- KV cache tensor layout
- cache group metadata
- Mamba state representation
- transfer connector hook timing

에 영향을 준다.

따라서 0.26의 Mooncake/NIXL P/D가 성공했다는 사실은 0.29 MRV2 P/D의 compatibility evidence가 아니다.

같은 connector라도 runner side worker implementation이 달라진다.

---

## 12. MRV migration acceptance matrix

| Test | MRV1 | MRV2 | 필수 관찰 |
|---|---:|---:|---|
| short greedy | O | O | token IDs |
| long 170K prefill | O if supported | O | TTFT, peak memory |
| 2K decode | O | O | TPOT, memory growth |
| prefix cache | O | O | hit boundary, correctness |
| hybrid GDN/Mamba | baseline dependent | O target | state correctness |
| MTP | O/feature dependent | O | acceptance, scheduled tokens |
| DSpark | limited/unsupported | O | startup validation, acceptance |
| CUDA Graph | O | O | capture memory/sizes |
| P/D | baseline | O | transfer correctness |
| DCP/PCP | limited | V2-centric | output + routing |
| TP4/TP8 | O | O | collective backend |

---

## 13. 강제 MRV1 A/B는 언제 유효한가

가능한 모델/feature 조합이라면 0.29 migration 초기에 MRV1을 강제한 비교는 매우 가치가 있다.

목적은 V1을 장기 production target으로 남기는 것이 아니다.

다음 질문에 답하기 위한 diagnostic control이다.

```text
0.29 regression이
- dependency/kernel 때문인가?
- frontend 때문인가?
- runner rewrite 때문인가?
```

MRV1 forced에서 regression이 사라진다면 조사 영역을 크게 줄일 수 있다.

단, 0.29에서 해당 feature가 V1 unsupported라면 억지로 비교하지 않는다.

---

## 14. Runtime log에 반드시 남길 정보

engine startup 시 다음을 한 줄의 structured metadata로 남기는 것을 권장한다.

```json
{
  "vllm_version": "0.29.0",
  "runner": "MRV2",
  "model_arch": "Qwen3_5ForCausalLM",
  "is_hybrid": true,
  "is_attention_free": false,
  "spec_method": "mtp",
  "cudagraph_mode": "FULL_DECODE_ONLY",
  "tp": 4,
  "dp": 1,
  "ep": 1
}
```

이 정보가 없으면 Grafana에서 version별 latency regression을 봐도 실제 runner population을 구분할 수 없다.

---

## 15. CI gate

모델 image smoke test 단계에서:

1. engine config 생성
2. resolved runner 출력
3. expected runner와 비교
4. unsupported feature validation
5. one-token generation

을 수행한다.

예:

```text
Qwen3.6 candidate
expected runner = MRV2

DSV4 ROCm candidate
expected runner = MRV1 or explicitly reviewed exception
```

이런 식으로 architecture-aware expectation을 관리한다.

---

## 16. 최종 판단

v0.29에서 MRV2는 더 이상 실험적 alternate path가 아니라 **primary execution contract**로 보는 것이 맞다.

하지만 0.26 -> 0.29 migration에서는 이를 단순 default 변화로 흡수하지 않는다.

**MRV1 -> MRV2 자체를 하나의 독립 migration으로 인증한 뒤**, graph/spec/P-D optimization을 그 위에 올리는 것이 가장 안전하고 분석 가능하다.

---

## Sources

- v0.25 MRV2 dense default PR #44443: https://github.com/vllm-project/vllm/pull/44443
- v0.29 MRV2 all-model default PR #53183: https://github.com/vllm-project/vllm/pull/53183
- MRV2 E/P/D PR #38390: https://github.com/vllm-project/vllm/pull/38390
- MRV2 CUDA Graph memory profiling PR #53306: https://github.com/vllm-project/vllm/pull/53306
- Qwen3.5 hybrid registration PR #50210: https://github.com/vllm-project/vllm/pull/50210
- v0.29 release: https://github.com/vllm-project/vllm/releases/tag/v0.29.0
