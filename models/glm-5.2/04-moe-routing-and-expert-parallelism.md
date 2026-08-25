# 04. MoE Routing & Expert Parallelism

작성일: 2026-08-25  
기준: GLM-5.2 current config + Transformers `GlmMoeDsaMoE`

## 1. Current MoE geometry

```text
hidden_size           = 6144
first_k_dense_replace = 3
num_hidden_layers     = 78
n_routed_experts      = 256
num_experts_per_tok   = 8
n_shared_experts      = 1
moe_intermediate_size = 2048
scoring_func          = sigmoid
topk_method           = noaux_tc
norm_topk_prob        = true
routed_scaling_factor = 2.5
moe_router_dtype      = float32
```

MLP pattern:

```text
Layer 0  Dense
Layer 1  Dense
Layer 2  Dense
Layer 3  MoE
...
Layer 77 MoE
```

즉 78 backbone layer 중 75개에서 sparse expert routing이 발생한다.

---

## 2. GLM MoE block의 기본 dataflow

현재 Transformers 구현의 `GlmMoeDsaMoE`는 conceptual하게 다음과 같다.

```text
                           ┌─ router ─→ top-8 routed experts ─┐
hidden ────────────────────┤                                 ├─→ add → output
                           └────────→ shared expert ──────────┘
```

routed path와 shared path는 경쟁 관계가 아니라 동시에 존재한다.

```math
y = y_{routed}+y_{shared}
```

### Shared expert의 역할

shared expert는 모든 token에 실행되는 dense/common path다.

expert specialization이 극단적으로 진행되더라도:

- 공통 lexical feature
- universal transformation
- 여러 domain에 반복되는 기본 연산

을 routed expert마다 중복 학습하지 않도록 common capacity를 제공한다.

---

## 3. Router는 softmax가 아니라 sigmoid score를 사용한다

current implementation:

```python
router_logits = linear(hidden, router_weight)  # fp32
scores = sigmoid(router_logits)
```

softmax와 sigmoid의 차이를 구조적으로 보면:

### softmax

expert score가 한 simplex 안에서 직접 경쟁한다.

```math
\sum_i p_i = 1
```

### sigmoid

각 expert affinity를 독립적으로 점수화한 뒤 top-k를 선택한다.

```math
s_i=\sigma(z_i)
```

GLM-5.2는 선택된 top-k weight를 다시 normalize한다.

```text
sigmoid scores
 → expert selection
 → selected 8 scores normalization
 → × routed_scaling_factor(2.5)
```

따라서 routing decision과 final mixture weight normalization이 분리되어 있다.

---

## 4. `e_score_correction_bias`와 noaux routing

router code에는 각 expert에 대한 correction bias가 별도 buffer로 존재한다.

```python
scores_for_choice = scores + e_score_correction_bias
```

중요한 점은 이 bias가 **expert 선택용 score**에 들어가고, 최종 routed weight는 원래 sigmoid score에서 gather한다는 것이다.

```text
selection:
  sigmoid score + correction bias

mixture weight:
  original sigmoid score → normalize → scale
```

이 설계의 의미는 load balancing을 위해 expert 선택 경계는 움직이되, token-expert affinity 자체를 auxiliary loss 때문에 직접 왜곡하는 것을 줄이는 데 있다.

`topk_method=noaux_tc`는 DeepSeek 계보의 auxiliary-loss-free expert balancing과 연결해서 봐야 한다.

---

## 5. Current config에서는 expert group restriction이 사실상 없다

router implementation은 group-based top-k를 지원하지만 current config는:

```text
n_group    = 1
topk_group = 1
```

이다.

즉 현재 GLM-5.2 config에서는 256 experts 전체가 하나의 group이며, group pruning이 실질적인 추가 sparsity를 만들지 않는다.

이 사실은 generic implementation capability와 model-specific setting을 구분할 때 중요하다.

---

## 6. Routed expert의 actual computation

각 expert는 SwiGLU/SiLU-gated FFN 형태다.

```text
h[6144]
  ├─ gate_proj → 2048 ─→ SiLU ─┐
  └─ up_proj   → 2048 ─────────┼─ elementwise multiply
                                │
                                ▼
                             2048
                                │
                           down_proj
                                │
                                ▼
                              6144
```

선택된 top-8 expert output에 router weight를 곱해 합친다.

shared expert도 동일 계열 MLP이며 current config에서 shared expert 수가 1이므로 intermediate width는 2048이다.

---

## 7. 256×top-8이 GPU cluster에서 만드는 문제

model-level sparsity:

```text
256 experts 중 8개 = 3.125% routed expert activation
```

하지만 cluster-level로는 token이 expert가 있는 device로 이동해야 한다.

Expert Parallel(EP) simplified flow:

```text
GPU local tokens
   │
   ▼ routing
expert destination 결정
   │
   ▼ all-to-all dispatch
각 GPU의 local experts에서 GEMM
   │
   ▼ all-to-all combine
원래 token ownership으로 복귀
```

### 병목 1 — all-to-all latency/bandwidth

TP는 큰 tensor를 여러 GPU가 함께 계산하는 반면 EP는 token payload가 expert owner로 shuffle된다.

multi-node에서는 NIC/RDMA/IB topology가 직접 성능에 영향을 준다.

### 병목 2 — load imbalance

특정 batch에서 일부 expert로 token이 몰리면:

```text
GPU A expert queue ████████████
GPU B expert queue ███
GPU C expert queue █████
```

가 되고 collective는 가장 늦은 rank를 기다린다.

### 병목 3 — small expert GEMM

decode batch가 작으면 expert별 token 수가 적어 GEMM utilization이 떨어진다.

### 병목 4 — shared expert overlap

shared expert는 항상 실행된다. routed expert communication과 shared expert GEMM을 overlap할 수 있으면 latency hiding 여지가 생긴다.

실제 inference engine에 fused shared-expert / communication overlap optimization이 등장하는 이유다.

---

## 8. TP와 EP를 어떻게 구분해서 생각할 것인가

### Tensor Parallel

한 layer의 matrix를 여러 GPU로 split한다.

```text
one token
→ all TP ranks participate
```

### Expert Parallel

expert 자체를 rank별로 배치한다.

```text
one token
→ selected expert owners로 이동
```

GLM-5.2에서는 attention/MLA projection과 MoE가 모두 크므로 실제 deployment는 TP/EP 조합을 고민하게 된다.

```text
Attention / dense projection → TP-friendly
MoE expert bank             → EP-friendly
```

따라서 한 가지 parallel dimension으로 모든 kernel을 최적화하기 어렵다.

---

## 9. 왜 H200/B200에서 FP8이 실용 기본안인가

~744B-class weights는 BF16이면 대략 1.5TB 이상의 raw parameter storage가 필요하고 runtime overhead까지 고려하면 single-node H200에 들어가지 않는다.

current vLLM recipe는 GLM-5.2-FP8을 실용적인 NVIDIA default로 제시하며:

- 8×H200/H20: FP8 single-node serving
- 8×B200: FP8 KV cache와 함께 full 1M-context target
- Blackwell: NVFP4 expert quantization option

을 제공한다.

여기서 quantization은 단순 weight-fit 문제가 아니다.

MoE에서는:

- expert weight bandwidth
- expert GEMM throughput
- dispatch/combine overlap

에도 직접 영향을 준다.

---

## 10. MoE profiler에서 반드시 분해할 metric

### Per-layer

- router time
- dispatch time
- expert GEMM time
- shared expert time
- combine time
- communication/computation overlap

### Per-expert

- tokens/expert histogram
- max/mean load ratio
- dropped token 여부
- expert locality

### Cluster

- all-to-all bandwidth
- cross-node traffic ratio
- NVLink vs IB/RDMA path
- rank straggler
- batch size에 따른 expert GEMM occupancy

### End-to-end

- TTFT
- ITL
- output tok/s/GPU
- context length별 throughput
- concurrency별 HBM usage

---

## 11. P/D disaggregation과 MoE의 관계

Prefill과 decode는 MoE에서도 workload shape가 다르다.

### Prefill

- token batch가 큼
- expert별 token aggregation이 상대적으로 잘 됨
- GEMM 효율이 높아지기 쉬움
- attention/indexing cost가 큼

### Decode

- step당 token 수가 작음
- expert GEMM이 작아지기 쉬움
- communication latency가 더 민감
- MTP acceptance가 throughput에 큰 영향

따라서 P/D disaggregation은 GLM-5.2에서도 단순 KV transfer 문제가 아니라 **prefill용 sparse-attention/large-batch MoE와 decode용 low-latency MoE를 다른 resource shape로 최적화할 수 있는 수단**이다.

---

## 12. 한 문장 정리

> **GLM-5.2의 MoE는 744B-class capacity를 약 40B active compute로 바꾸지만, 절약된 dense FLOPs의 대가로 expert routing·all-to-all·load balancing이라는 distributed-systems 문제를 만든다.**

## Sources

- GLM-5 Technical Report: https://arxiv.org/abs/2602.15763
- GLM-5.2 config: https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
- Transformers GLM-MoE-DSA implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
- vLLM GLM-5.2 recipe: https://recipes.vllm.ai/zai-org/GLM-5.2
