# 04. DeepSeekMoE, Hash Routing, Muon & FP4

작성일: 2026-08-25

## 1. V4는 DeepSeekMoE를 버리지 않았다

V4 technical report는 MoE backbone을 DeepSeek-V3의 DeepSeekMoE 계보로 둔다.

핵심 구조:

```text
Token hidden state
   ├─ routed expert bank → top-k experts
   └─ shared expert      → always-on path
        ↓
weighted sum
```

Flash:

```text
256 routed experts
6 active/token
1 shared expert
expert intermediate = 2048
```

Pro:

```text
384 routed experts
6 active/token
1 shared expert
expert intermediate = 3072
```

---

## 2. 첫 3개 layer는 Hash-MoE

V4의 특이점 중 하나는 첫 `num_hash_layers=3` MoE layer에서 learned router가 expert identity를 고르지 않는다는 점이다.

대신 checkpoint에 저장된 token-id → expert-id mapping을 사용한다.

개념:

```text
input token id
    ↓
tid2eid lookup
    ↓
expert indices
```

여기서 learned gate score가 완전히 사라지는 것은 아니다.

구현에서는 router score를 계산한 후 **어떤 expert를 고를지(index)** 는 hash table이 결정하고, 선택된 expert들의 **weight** 는 learned affinity score에서 gather한다.

즉:

```text
selection = static/hash
weighting = learned
```

이다.

---

## 3. 왜 early layer에 hash routing을 쓰는가

초기 layer는 token identity와 shallow lexical/syntactic features에 강하게 의존한다.

초기 routing을 deterministic하게 만들면:

- early router instability 감소
- expert specialization bootstrap 단순화
- routing entropy 문제 완화
- load behavior 예측 가능성 증가

를 기대할 수 있다.

하지만 이것을 `전체 MoE가 hash-based`라고 해석하면 틀린다.

첫 3개 이후에는 learned top-k MoE로 전환한다.

---

## 4. Router affinity: Sigmoid → sqrt(softplus)

V3 계열의 Sigmoid affinity 대신 V4는:

```math
score(x)=\sqrt{\mathrm{softplus}(x)}
```

를 사용한다.

직관적으로 softplus는 항상 양수이며 큰 positive input에서는 대략 linear하게 증가한다.

그 위에 square root를 씌우면 큰 logit의 dynamic range를 완화하면서도 strictly-positive affinity를 유지한다.

V4가 extreme expert count / long training에서 router score의 saturation과 magnitude distribution을 더 안정적으로 다루려는 변경으로 볼 수 있다.

---

## 5. Auxiliary-loss-free load balancing은 유지

DeepSeek의 중요한 MoE 특징은 routing balance를 위해 main loss에 큰 auxiliary loss를 더하는 대신 expert별 correction bias를 사용하는 방식이다.

개념:

```text
raw affinity score
     +
load-balance correction bias
     ↓
expert selection
```

하지만 실제 mixture weight는 bias가 들어가기 전 original score를 사용한다.

즉 correction bias는 **selection을 바꾸지만 model probability weighting에는 직접 끼지 않는다.**

이렇게 main objective gradient와 load-balancing objective를 분리한다.

---

## 6. V3의 group-limited routing 제약 완화

V3에서는 communication topology 때문에 expert group 제한을 두는 설계가 중요했다.

V4는 router 쪽에서 V3의 `n_group/topk_group` constraint를 제거한다.

이 의미는 단순 코드 삭제가 아니다.

- model-level routing freedom 증가
- system-level expert placement/fused communication이 더 중요해짐
- communication-aware restriction을 architecture router에 강하게 박지 않음

즉 hardware/system layer가 더 많은 책임을 가져간다.

---

## 7. Clipped SwiGLU

V4 routed expert는 SwiGLU를 그대로 쓰되 intermediate pre-activation을 clamp한다.

config:

```text
swiglu_limit = 10.0
```

개념:

```text
gate = clamp(gate, max=10)
up   = clamp(up, -10, 10)
SiLU(gate) * up
```

목표는 extreme activation outlier를 줄여:

- training stability
- low-precision robustness
- FP4 expert quantization

을 돕는 것이다.

---

## 8. Expert weights는 FP4

V4 checkpoint에서 routed expert는 `expert_dtype=fp4`를 사용한다.

나머지 attention/norm/router 계열은 FP8 계열 저장/compute path를 사용한다.

중요한 점:

> V4의 FP4는 배포 후 사후 PTQ만 한 것이 아니라 technical report에서 **quantization-aware training**을 architecture/system co-design 일부로 다룬다.

특히:

- MoE expert weights
- indexer QK path

에 FP4를 활용해 memory와 compute를 줄인다.

---

## 9. 왜 MoE expert가 FP4에 특히 잘 맞는가

V4 total parameter 대부분은 routed expert bank에 있다.

따라서 expert precision을:

```text
8-bit → 4-bit
```

로 낮추면 전체 checkpoint/HBM capacity에서 큰 절감이 생긴다.

반면 attention/router/norm 등 sensitivity가 높은 작은 부분은 더 높은 precision으로 남겨 quality degradation을 제한한다.

이건 전형적인 **heterogeneous precision allocation**이다.

---

## 10. Muon optimizer

V4는 대부분의 matrix parameters에 Muon을 도입한다.

다만 모든 parameter가 Muon은 아니다.

Technical report 기준 Flash에서:

### Muon

- 대부분의 matrix parameters

### AdamW

- embedding
- prediction head
- RMSNorm weights

즉 optimizer도 parameter role에 따라 hybrid다.

Muon은 matrix update를 orthogonalization 계열 변환으로 다뤄 large-scale matrix optimization에서 conditioning과 convergence를 개선하려는 optimizer다.

V4에서는:

- faster convergence
- training stability

를 핵심 이유로 든다.

---

## 11. Flash training setup 핵심 숫자

Technical report 공개값:

```text
pretraining tokens ≈ 32T
batch tokens       → 최대 약 75.5M
warmup             = 2000 steps
peak LR            = 2.7e-4
final LR           = 2.7e-5
Muon momentum      = 0.95
weight decay       = 0.1
Muon update RMS rescale = 0.18
```

숫자를 외우는 것보다 중요한 점은 V4가 architecture innovation만으로 나온 모델이 아니라 **optimizer / precision / data / kernels까지 동시에 재설계한 모델**이라는 것이다.

---

## 12. Fused MoE와 시스템 co-design

V4 report는 MoE에서:

- computation
- communication
- memory access

를 하나의 fused kernel에서 overlap하도록 설계했다고 강조한다.

MoE serving에서 병목은 expert GEMM만이 아니다.

```text
route
→ dispatch/all-to-all
→ expert GEMM
→ gather/reduce
```

가 연결되므로 communication overlap이 없으면 sparse MoE의 이론적 FLOP 절감이 실제 latency로 이어지지 않는다.

---

## 13. TileLang과 deterministic kernels

V4 infrastructure는 TileLang DSL을 사용해 custom kernels의 개발 생산성과 성능을 함께 노린다.

또 batch-invariant / deterministic kernel library를 강조한다.

이것은 대규모 distributed training에서:

- reproducibility
- debugging
- batch shape 변화에 따른 numerical drift 관리

에 중요하다.

---

## 14. Serving engineer 관점

V4 MoE를 배포할 때 확인해야 하는 것은 `EP 몇 개`만이 아니다.

```text
expert HBM placement
FP4 kernel availability
router implementation
all-to-all transport
shared expert overlap
token imbalance
DeepGEMM backend
CUDA architecture compatibility
```

가 모두 중요하다.

특히 H100에서 framework version 변경 시 DeepGEMM/SM90 kernel path가 바뀌어 CUDA IMA가 발생했던 실제 기록은 이 디렉터리의 runtime 문서를 같이 본다.

- [vLLM 0.27.x DeepGEMM SM90 CUDA IMA](2026-08-18-vllm-0.27-deepgemm-sm90-cuda-ima.md)

---

## 15. 출처

- DeepSeek-V4 Technical Report §2.1, §4: https://arxiv.org/abs/2606.19348
- HF DeepSeek-V4 implementation docs: https://huggingface.co/docs/transformers/en/model_doc/deepseek_v4
- Flash official inference implementation: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/inference
