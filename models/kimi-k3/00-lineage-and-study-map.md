# 00. Kimi 계보와 K3 학습 지도

작성일: 2026-08-19  
상위 문서: [Kimi K3 Model Architecture & Lineage Study Guide](README.md)

## 1. K3의 '근원'을 어떻게 볼 것인가

Kimi K3를 이해하려면 모델 이름의 출시 순서만 따라가면 부족하다. 서로 다른 문제를 해결하던 연구가 K3에서 합쳐졌기 때문이다.

가장 유용한 계보는 다음 다섯 줄이다.

```text
A. Reasoning / RL
Kimi k1.5 ────────────────┐
                           │
B. Frontier MoE / Agentic  │
Kimi K2 ────────────────┐  │
                        ├──┼──→ Kimi K3
C. Native Multimodality │  │
Kimi-VL → Kimi K2.5 ────┤  │
                        │  │
D. Efficient Sequence   │  │
GDN → Kimi Linear ──────┤  │
                        │  │
E. Depth / Width        │  │
AttnRes + LatentMoE ────┘  │
                           │
Post-training / system ────┘
```

이 중 어느 하나도 K3 전체를 설명하지 않는다.

---

## 2. 연구 계보를 시간순보다 문제순으로 본다

### 2.1 Kimi k1.5 — `추론 compute를 RL로 어떻게 scale할 것인가?`

**Reinforcement Learning(리인포스먼트 러닝, 강화학습; RL)**과 긴 **Chain-of-Thought(체인 오브 쏘트, 사고 연쇄; CoT)**가 Moonshot의 reasoning 계보에서 중요한 출발점이다.

Kimi k1.5의 핵심은 특정 attention architecture가 아니라:

- long-context RL
- partial rollout
- policy optimization
- long-to-short reasoning transfer
- multimodal reasoning RL

을 large-scale로 구현했다는 점이다.

K3의 post-training에서 `reasoning effort`, long-horizon agentic rollout, on-policy distillation을 이해하려면 이 계보를 알아야 한다.

---

### 2.2 Kimi K2 — `1T sparse MoE를 안정적으로 학습하고 agentic foundation으로 만들 수 있는가?`

K2는 K3의 가장 직접적인 large-scale language foundation이다.

핵심:

- 약 1T total / 32B active MoE
- 61 layers
- 384 routed experts, top-8
- MLA
- MuonClip
- 15.5T-token pretraining
- tool-use/agentic data synthesis
- RLVR + self-critique based post-training

K3는 K2의 hidden size 7168, vocabulary 160K, MoE/MLA 설계 경험, Muon lineage, agentic training 철학을 상당 부분 이어받으면서 scale과 architecture를 크게 바꾼다.

---

### 2.3 Kimi-VL / K2.5 — `vision을 language model에 나중에 붙이지 않고 foundation부터 같이 학습할 수 있는가?`

Kimi-VL은 Moonshot의 native-resolution visual encoder **MoonViT(문빗)**와 long-context VLM 연구선이다.

K2.5는 K2-Base 위에서 대규모 text+vision continual pretraining을 수행해 native multimodal agentic model로 확장했다.

K3는 K2.5의 큰 설계 방향을 잇지만 중요한 차이가 있다.

- K2.5: 기존 vision representation을 활용하는 MoonViT 계보
- K3: **MoonViT-V2를 next-token objective로 scratch부터 함께 학습**

즉 K3에서는 vision이 별도 pretrained encoder를 language backbone에 정렬하는 부속물이 아니라 foundation training의 일부가 된다.

---

### 2.4 Kimi Linear — `Full Attention의 long-context 비용을 fixed-state recurrence로 대체해도 frontier 품질을 유지할 수 있는가?`

Kimi Linear는 48B total / 3B active 연구 모델로 KDA의 핵심 가설을 검증했다.

구조:

```text
KDA
KDA
KDA
MLA
→ repeat
```

즉 3:1 hybrid다.

Kimi Linear 연구의 중요한 결론은 `pure linear attention`이 아니라:

> **finite recurrent memory를 대부분의 layer에 쓰고, 주기적인 global MLA로 exact/global memory path를 보완하면 long-context efficiency와 quality를 동시에 노릴 수 있다.**

K3의 69 KDA + 24 MLA는 이 연구선의 대규모 확장이다.

### 공식 Kimi Linear architecture 그림

![Kimi Linear architecture](https://raw.githubusercontent.com/MoonshotAI/Kimi-Linear/master/figures/arch.png)

*Source: MoonshotAI/Kimi-Linear official repository.*

Attention 수식 자체는 다음 문서를 읽는다.

- [Linear/Recurrent Attention](../../study/attention/03-linear-recurrent-attention.md)
- [Kimi-K3 Attention Architecture](../../study/attention/10-kimi-k3.md)

---

### 2.5 Attention Residuals — `93 layer의 depth에서 residual stream은 충분한가?`

Standard PreNorm Transformer는 이전 layer output을 residual stream에 계속 더한다.

깊이가 깊어질수록:

- representation이 모두 하나의 합으로 압축되고
- 특정 과거 depth의 feature를 직접 다시 선택하기 어렵고
- hidden magnitude/dilution 문제가 발생할 수 있다.

**Attention Residuals(어텐션 레지듀얼즈; AttnRes)**는 token 축이 아니라 **depth 축**에서 attention한다.

K3는 full AttnRes의 메모리 비용을 줄인 Block AttnRes를 쓴다.

따라서 K3는:

```text
Sequence axis → KDA / MLA
Depth axis    → AttnRes
Width axis    → Stable LatentMoE
```

처럼 information routing을 서로 다른 축으로 나눈다.

---

### 2.6 LatentMoE → Stable LatentMoE — `896 expert를 어떻게 싸고 안정적으로 운영할 것인가?`

Sparse MoE는 total parameter는 크게 늘리면서 token당 일부 expert만 실행한다.

하지만 expert가 많아질수록:

- expert weight memory
- EP all-to-all communication
- routing imbalance
- expert activation instability

가 커진다.

**LatentMoE(레이턴트 엠오이, 잠재공간 MoE)**는 token을 model hidden width보다 작은 latent dimension으로 내려 expert computation과 communication을 수행한다.

K3에서는 이를 **Stable LatentMoE**로 확장해:

- RMS normalization
- bounded SiTU-GLU
- Quantile Balancing

등을 추가한다.

K3의 896 experts / top-16 설계는 attention만큼 중요한 model-width scaling innovation이다.

---

## 3. K3는 세 종류의 `희소성`을 동시에 사용한다

K3를 이해할 때 sparsity라는 말을 하나로 묶지 않는다.

### 3.1 Expert Sparsity

896 routed experts 중 16개만 실행한다.

```math
\text{expert activation ratio}=\frac{16}{896}\approx1.79\%
```

즉 routed expert bank는 매우 sparse하다.

### 3.2 Attention Layer Sparsity

93 layer 모두 global MLA를 쓰지 않는다.

- 69 KDA
- 24 Gated MLA

즉 expensive token-wise global memory path도 layer-level로 sparse하다.

### 3.3 Depth Source Sparsity/Compression

AttnRes도 모든 93 layer output을 모든 layer에 직접 유지하는 Full AttnRes 대신 12-layer block representation을 source로 사용한다.

K3의 efficiency는 이처럼 **parameter, sequence, depth에서 각각 다른 방식으로 정보 흐름을 제한/압축**한 결과다.

---

## 4. K3의 scale-up은 'hidden size 확대'가 아니다

K2와 K3 모두 hidden size가 7168이다.

그런데 total parameter는 약:

```math
1.04T\rightarrow2.78T
```

active parameter는:

```math
32.6B\rightarrow104.2B
```

로 커졌다.

어디서 커졌는가?

### Depth

```math
61\rightarrow93\text{ layers}
```

### Expert Count

```math
384\rightarrow896
```

### Active Experts

```math
8\rightarrow16
```

### Expert Width

```math
2048\rightarrow3072
```

### Shared Experts

```math
1\rightarrow2
```

### Vision

401M MoonViT-V2 추가

즉 K3는 residual width를 과도하게 키우는 대신 **depth와 sparse expert capacity를 확장**한다.

이 선택은 단순 parameter scaling이 아니라 GPU memory/communication 구조를 함께 고려한 것이다.

---

## 5. Kimi Linear가 해결하지 않았던 것

Kimi Linear는 KDA/MLA hybrid의 효과를 검증했지만 K3가 해결해야 할 frontier-scale 문제는 더 많았다.

| 문제 | Kimi Linear | Kimi K3 |
|---|---|---|
| Sequence efficiency | KDA + MLA | KDA + Gated MLA, lower-bounded decay, optimized kernels |
| Model scale | 48B / A3B | 2.8T / A104B |
| MoE | 연구 모델 MoE | 896-expert Stable LatentMoE |
| Depth mixing | 기본 residual / 연구 단계 | Block AttnRes |
| Vision | text model | native MoonViT-V2 |
| Context | 1M | 1M + large-scale curriculum/system support |
| Optimizer | research setup | Per-Head Muon lineage |
| Quantization | 별도 | post-training QAT integrated |
| Agentic RL | 제한적 | broad coding/agentic/reasoning RL infrastructure |
| Serving | KDA kernel | hybrid cache, prefix state, P/D, fleet scheduling |

따라서 K3를 이해하려면 Kimi Linear 논문 이후의 architecture뿐 아니라 K2/K2.5의 training lineage를 같이 봐야 한다.

---

## 6. 가장 먼저 배워야 하는 기반 개념

K3 문서를 읽기 전에 다음 개념이 익숙해야 한다.

### Model Architecture

- Transformer residual stream
- PreNorm / RMSNorm
- FFN / GLU / SwiGLU
- sparse Mixture-of-Experts
- routing / top-k expert selection
- shared expert vs routed expert
- active parameter vs total parameter

### Optimization

- gradient descent
- optimizer state
- AdamW
- matrix-valued parameter update
- orthogonalization
- Newton–Schulz iteration
- gradient/update norm

### Distributed Training/Serving

- Tensor Parallelism(TP)
- Expert Parallelism(EP)
- Context Parallelism(CP)
- Sequence Parallelism(SP)
- all-reduce / all-gather / all-to-all
- HBM bandwidth
- GEMM
- kernel fusion

### Post-training

- SFT
- Reinforcement Learning(RL)
- RLVR
- on-policy rollout
- reward model
- distillation
- speculative decoding
- MTP

Attention 수학은 `study/attention`을 prerequisite로 본다.

---

## 7. 단계별 학습 목표

### Stage 1 — K2를 이해한다

다음 문장을 설명할 수 있어야 한다.

> K2는 1T parameter지만 token 하나가 모든 1T parameter를 계산하지 않는다. 384개의 routed expert 중 8개만 활성화해 32B 수준의 active compute를 유지한다.

### Stage 2 — Kimi Linear를 이해한다

> Kimi Linear는 long context에서 모든 layer가 token-wise KV를 유지할 필요가 없다는 가설을 3 KDA : 1 MLA hybrid로 검증했다.

### Stage 3 — LatentMoE/AttnRes를 이해한다

> K3는 sequence 축만 최적화한 것이 아니라 expert computation을 latent width로 압축하고 depth history도 learned attention으로 재사용한다.

### Stage 4 — K3 architecture를 재구성한다

`config.json`만 보고:

```text
93 language layers
= 69 KDA + 24 MLA
= 23 × (KDA,KDA,KDA,MLA) + final MLA
```

와 896/top-16 expert structure를 직접 읽을 수 있어야 한다.

### Stage 5 — 시스템을 설명한다

> K3는 KDA recurrent state와 MLA KV cache를 동시에 관리하므로 conventional PagedAttention-only serving보다 prefix caching, preemption, P/D cache transfer가 복잡하다.

까지 연결한다.

---

## 8. 이 계보에서 버리면 안 되는 본질

Kimi 연구선을 하나의 문장으로 줄이면 다음과 같다.

> **Kimi 계열은 model intelligence를 키우면서 증가하는 sequence, depth, width, rollout의 비용을 각각 별도의 구조와 시스템으로 분해해 scale하는 방향으로 발전했다.**

- Sequence → KDA/MLA
- Depth → AttnRes
- Width → Stable LatentMoE
- Modality → MoonViT-V2
- Optimization → Muon lineage
- Reasoning/Agent → large-scale RL
- Serving → architecture-aware cache/kernel/scheduler

K3는 이 각각의 연구선을 하나의 frontier-scale model로 통합한 결과로 보는 것이 가장 정확하다.

---

## 9. 주요 원문

- Kimi k1.5  
  https://arxiv.org/abs/2501.12599
- Kimi K2  
  https://github.com/MoonshotAI/Kimi-K2
- Kimi-VL  
  https://arxiv.org/abs/2504.07491
- Kimi K2.5  
  https://github.com/MoonshotAI/Kimi-K2.5
- Kimi Linear  
  https://github.com/MoonshotAI/Kimi-Linear
- Attention Residuals  
  https://github.com/MoonshotAI/Attention-Residuals
- Kimi K3  
  https://github.com/MoonshotAI/Kimi-K3
