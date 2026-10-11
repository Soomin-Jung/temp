# Core Forge — Root-to-System Learning Roadmap

> 목적: 논문 속 수식을 외우는 수준을 넘어서, **수식이 어떤 정보를 표현·보존·제거·변환하는지 이해하고 그 의미를 Tensor → PyTorch → CUDA/GPU → LLM Serving → 분산 시스템까지 추적할 수 있는 엔지니어**가 되기 위한 길잡이 지도.

이 문서는 차시별 교재가 아니라 **길을 잃었을 때 돌아오는 지도**다. 위로만 진도 나가는 커리큘럼이 아니라, 상위 개념에서 막히면 필요한 근원 개념으로 내려갔다가 다시 올라오는 나선형 학습을 전제로 한다.

- 현재 체크포인트: **2026-10-11 / 14차시 비선형성(Nonlinearity)까지**
- 학습 원칙: 진도보다 메커니즘, 정의보다 필요성, 암기보다 정보 변환
- 최종 연결 축: **Math → CS/System → Deep Learning → Transformer/LLM → Serving → CUDA/GPU → Distributed Inference**

---

## 0. 지도를 읽는 방법

모든 개념은 가능하면 아래 여섯 층에서 다시 본다.

| 층 | 질문 |
|---|---|
| Intuition | 왜 이 개념이 필요한가? 어떤 문제를 해결하는가? |
| Math | 어떤 관계를 기호와 수식으로 표현하는가? |
| Tensor | 실제 데이터의 shape과 값은 어떻게 변하는가? |
| Code | PyTorch/vLLM 코드에서는 어떤 연산·객체로 나타나는가? |
| Kernel/Hardware | CUDA kernel, SM, HBM, fabric에서는 실제로 무엇이 움직이는가? |
| System | latency, throughput, KV cache, scheduler, routing, fault와 어떻게 연결되는가? |

학습 중 막히면 상위 개념을 더 외우는 대신 아래로 내려간다.

~~~text
논문/시스템에서 막힘
        │
        ▼
무슨 수식/연산인지 식별
        │
        ▼
그 연산이 보존/제거/변환하는 정보는?
        │
        ▼
필요한 선수 개념으로 하강
        │
        ▼
작은 숫자로 다시 이해
        │
        ▼
Tensor / Code / GPU로 재상승
~~~

---

# 1. 전체 대지도

~~~mermaid
flowchart TD
    A[수와 연산의 의미] --> B[변수·식·등식]
    B --> C[함수]
    C --> D[좌표·변화량·변화율]
    D --> E[기울기·비선형성]
    E --> F[극한·미분]
    F --> G[편미분·Gradient·Chain Rule]

    D --> H[벡터·공간]
    H --> I[내적·거리·Norm·Projection]
    I --> J[행렬·선형변환]
    J --> K[Basis·Rank·Eigen·SVD]

    A --> P[비율·확률의 기초]
    P --> Q[확률변수·분포]
    Q --> R[기댓값·분산·조건부 확률]
    R --> S[Logit·Softmax·Entropy·Cross Entropy]

    G --> DL[Deep Learning Mechanics]
    K --> DL
    S --> DL

    DL --> T[Transformer Core]
    T --> U[Modern LLM Architecture]
    U --> V[LLM Serving]
    V --> W[CUDA/GPU]
    W --> X[Distributed Inference]
    X --> Y[Platform / Production]

    T --> Z[Paper Reading Loop]
    U --> Z
    V --> Z
    W --> Z
    X --> Z

    Z --> AA[수식 → Tensor → Code → Kernel → Serving → Experiment]
~~~

이 지도에서 중요한 것은 **화살표가 한 방향이 아니라는 점**이다. 예를 들어 Attention에서 내적이 이해되지 않으면 벡터·각도·projection으로 내려가고, RMSNorm에서 막히면 제곱·평균·제곱근으로 내려간다.

---

# 2. Phase A — Math Foundations I: 이미 만든 뿌리

현재까지 학습한 핵심 노드다.

| 차시 | 개념 | 본질 | 이후 연결 |
|---:|---|---|---|
| 1 | Number | 숫자는 무엇을 나타내도록 정의했는가 | token ID, tensor value, metric |
| 2 | Sign / Zero | 기준점과 방향 | gradient 방향, 증감 |
| 3 | Absolute Value | 방향 제거, 크기 보존 | error magnitude, L1 |
| 4 | Square / Square Root | 방향 제거 + 큰 차이 강조 | MSE, variance, RMS, L2 |
| 5 | Division | 기준 1단위당 얼마인가 | tok/s, GB/s, rate |
| 6 | Fraction / Ratio / % | 절대량을 기준에 대한 관계로 표현 | utilization, hit rate |
| 7 | Variable | 값이 아니라 역할에 이름 부여 | B/S/H, parameter |
| 8 | Expression / Equation | 계산 구조 vs 관계 보존 | computation graph |
| 9 | Function | 입력 → 규칙 → 출력 | model/layer/API |
| 10 | Coordinate | 관련 값을 하나의 관측 상태로 묶음 | graphs, tensor samples |
| 11 | Delta | 두 상태 사이 변화 | metric delta |
| 12 | Rate of Change | 입력 변화 1단위당 출력 반응 | throughput, bandwidth |
| 13 | Slope | 변화율의 기하학적 모습 | performance curve |
| 14 | Nonlinearity | 위치에 따라 반응률이 달라짐 | saturation, activation |

현재 핵심 언어:

~~~text
값
 ↓
상태
 ↓
상태 간 변화
 ↓
변화량 Δ
 ↓
변화율 Δy/Δx
 ↓
위치별 반응 차이
 ↓
비선형성
~~~

---

# 3. Phase B — Calculus Foundations: 지금 여기서 얼마나 반응하는가

다음 주요 관문이다.

## B1. Local change

전체 평균과 현재 지점의 반응을 분리한다.

~~~text
넓은 구간 평균 변화율
        │
        ▼
관찰 구간 축소
        │
        ▼
현재 operating point 주변의 반응
~~~

예: 전체 concurrency 10→100 평균 TTFT 증가율과 현재 concurrency 80 근처의 증가율은 다를 수 있다.

## B2. Limit — 극한

핵심 질문:

> 두 점을 같은 점으로 만들어 0/0을 계산하는 대신, 서로 다른 상태를 유지하면서 간격을 계속 줄이면 무엇에 가까워지는가?

여기서 0이다와 0으로 가까워진다를 엄밀히 구분한다.

## B3. Derivative — 미분

목표는 공식을 외우는 것이 아니라 다음 문장이 자연스러워지는 것이다.

> 입력을 아주 조금 바꿨을 때 출력이 어느 방향으로 얼마나 민감하게 반응하는가?

연결:
- latency sensitivity
- loss sensitivity
- local linear approximation

## B4. Partial Derivative / Gradient

입력이 하나가 아니라 여러 개일 때:

~~~text
Loss = f(w1, w2, w3, ...)
          │
          ├─ w1 방향 변화?
          ├─ w2 방향 변화?
          └─ w3 방향 변화?
~~~

각 방향의 변화율 → 편미분, 이를 한 객체로 묶음 → Gradient.

## B5. Chain Rule / Backpropagation

~~~text
x → f → h → g → y → loss
~~~

중간 연산이 연쇄될 때 앞의 작은 변화가 뒤까지 어떻게 전달되는가를 추적한다.

### 심화 분기
- Jacobian: 여러 입력 → 여러 출력의 민감도 지도
- Hessian: gradient 자체가 어떻게 변하는지
- Taylor approximation: 복잡한 함수를 현재 위치 주변에서 단순화

이 세 개는 필요할 때 들어간다. 초기 목표가 아니다.

---

# 4. Phase C — Vector & Geometry: 숫자 하나에서 표현 공간으로

좌표 (x, y)를 고차원으로 확장한다.

~~~text
Scalar
  │
  ▼
(x, y)
  │
  ▼
[x1, x2, x3, ...]
  │
  ▼
Vector
~~~

## C1. Vector
- 여러 숫자를 한 상태로 묶는 이유
- 성분(component)과 축(axis)
- 방향과 크기

LLM 연결:
- embedding
- hidden state
- Q/K/V vector
- gradient vector

## C2. Distance & Norm

예전에 미뤄둔 L1/L2를 여기서 제대로 배운다.

~~~text
Absolute value
      │
      ▼
1차원 거리
      │
      ▼
고차원 거리 / 크기
      │
      ▼
Norm
~~~

연결:
- L1 / L2
- RMS
- RMSNorm
- gradient clipping
- similarity와 scale 분리

## C3. Dot Product

핵심 질문:

> 두 벡터가 얼마나 같은 방향을 보고 있는가?

연결:
- cosine similarity
- projection
- Attention의 QK^T

## C4. Projection

> 여러 정보가 섞인 표현에서 특정 방향 성분만 얼마나 들어 있는가?

Attention/representation을 이해하는 중요한 기하학적 기반.

---

# 5. Phase D — Matrix & Linear Algebra: 여러 방향을 한 번에 바꾸기

1차원의 y = ax + b를 고차원으로 확장한다.

y = Wx + b

## D1. Matrix

행렬을 숫자 표가 아니라 **벡터를 다른 벡터로 바꾸는 규칙**으로 본다.

## D2. Matrix Multiplication

질문:

> 왜 행렬곱은 단순 element-wise multiplication이 아니라 행×열의 내적 조합인가?

연결:
- Linear layer
- Q/K/V projection
- MLP
- GEMM
- Tensor Core

## D3. Linear Transformation

- scale
- rotation
- shear
- projection
- dimension change

입력 표현 공간을 다른 표현 공간으로 바꾼다는 감각을 만든다.

## D4. Basis

> 같은 벡터라도 어떤 축을 기준으로 보느냐에 따라 좌표가 달라질 수 있다.

representation learning을 이해하는 중요한 기반.

## D5. Rank

> 변환이 실제로 몇 개의 독립적인 정보 방향을 보존하는가?

연결:
- redundancy
- compression
- low-rank
- LoRA

## D6. Eigenvalue / Eigenvector

> 변환을 적용해도 방향은 유지되고 크기만 바뀌는 특별한 방향은 무엇인가?

반복 변환, 안정성, 스펙트럼 분석의 기반.

## D7. SVD

~~~text
복잡한 Matrix
    │
    ▼
중요한 방향들을 분해
    │
    ▼
큰 singular component
    │
    ▼
정보를 많이 설명
~~~

연결:
- low-rank approximation
- model compression
- LoRA 직관

---

# 6. Phase E — Probability & Information: LLM이 왜 확률분포를 출력하는가

## E1. Probability foundation
비율 → 가능성 → 확률로 연결.

## E2. Random Variable / Distribution
한 값만 예측하는 대신 가능한 결과 전체에 값이 어떻게 분포하는지 본다.

## E3. Expectation / Variance
- expectation: 분포의 중심
- variance: 얼마나 퍼져 있는가

제곱을 다시 여기서 만난다.

## E4. Conditional Probability

> 어떤 정보가 주어졌을 때 가능성이 어떻게 바뀌는가?

LLM의 next-token prediction과 자연스럽게 연결된다.

## E5. Logit → Softmax → Probability

~~~text
raw score
  │
  ▼
logits
  │
  ▼
exp / normalization
  │
  ▼
softmax distribution
  │
  ▼
sampling
~~~

질문:
- 왜 exponential인가?
- 왜 합을 1로 만드는가?
- score와 probability는 왜 다른가?

## E6. Entropy / Cross Entropy / KL
연결:
- training objective
- uncertainty
- model distribution vs target distribution
- distillation

## E7. Sampling
- temperature
- top-k
- top-p
- greedy
- categorical sampling

---

# 7. Phase F — Deep Learning Mechanics

수학 축들이 처음으로 하나의 모델 안에서 합쳐진다.

~~~text
Input
  │
  ▼
Linear
  │
  ▼
Activation
  │
  ▼
Normalization
  │
  ▼
Residual
  │
  ▼
Loss
  │
  ▼
Backward
  │
  ▼
Optimizer
~~~

## 핵심 노드

### Linear Layer
y = Wx + b

### Activation
- ReLU
- GELU
- SiLU

왜 nonlinearity가 필요한가?

### Loss
모델의 출력과 목표 사이의 차이를 하나의 최적화 대상으로 압축.

### Computational Graph
연산을 node/edge로 보고 backward 가능한 구조 이해.

### Backpropagation
Chain rule의 실제 구현.

### Optimizer
- SGD
- Momentum
- Adam/AdamW

Gradient를 계산하는 것과 parameter를 업데이트하는 것은 별개의 단계.

### Normalization
- mean / variance
- LayerNorm
- RMS
- RMSNorm

예전에 미뤘던 제곱/평균/제곱근이 여기서 돌아온다.

### Residual Connection
왜 x + f(x)가 깊은 모델에서 중요한가?

### Numerical Precision
- FP32
- FP16/BF16
- FP8
- INT8/INT4

수 표현 → 범위/정밀도 → GPU Tensor Core → quantization으로 연결.

---

# 8. Phase G — Transformer Core

## G1. Tokenization

~~~text
Text
 ↓
Token
 ↓
Token ID
~~~

1차시의 ID는 수량이 아니라 식별자라는 내용이 다시 등장한다.

## G2. Embedding

~~~text
Token ID
   │ lookup
   ▼
High-dimensional vector
~~~

좌표/벡터 공간과 연결.

## G3. Position
- positional encoding
- RoPE

순서 정보가 왜 필요한가?

## G4. Q / K / V

~~~text
Hidden state X
   ├─ Wq → Q
   ├─ Wk → K
   └─ Wv → V
~~~

행렬/선형변환이 그대로 구현된다.

## G5. Attention

Attention(Q,K,V) = softmax(QK^T / sqrt(d))V

이 한 줄을 다음처럼 해체할 수 있게 되는 것이 핵심 milestone이다.

~~~text
QKᵀ
→ 내적
→ 관계 score

/ √d
→ scale 조절

softmax
→ 상대적인 weight

× V
→ 정보의 weighted combination
~~~

## G6. MHA / MQA / GQA

Head를 왜 여러 개 두는가?
KV head를 줄이면 무엇을 잃고 무엇을 얻는가?

연결:
- KV cache size
- memory bandwidth
- serving throughput

## G7. MLP / SwiGLU
Attention 뒤의 representation transformation.

## G8. Residual + RMSNorm
Deep Learning 단계의 수학을 Transformer block 안에서 재해석.

---

# 9. Phase H — Modern LLM Architecture

기본 Transformer 이해 후 확장한다.

## Attention 계열
- MHA / MQA / GQA
- MLA
- latent/compressed KV
- local/sliding-window
- sparse attention
- hybrid attention

## Position 계열
- RoPE
- scaling / long-context extension

## FFN 계열
- SwiGLU
- gated MLP

## MoE
- router
- experts
- top-k routing
- expert parallelism

## Recurrent / Linear Attention
- state 기반 attention
- DeltaNet/GDN/KDA 계열
- KV cache 대신 state를 유지하는 구조

여기부터는 모델별 논문을 제품명이 아니라 **정보 흐름의 차이**로 비교한다.

---

# 10. Phase I — LLM Serving

모델의 수학이 실제 request 처리 시스템으로 내려온다.

~~~text
Request
  │
  ▼
Tokenization
  │
  ▼
Scheduler / Queue
  │
  ▼
Batch
  │
  ├──── Prefill ────┐
  │                 │
  │              KV Cache
  │                 │
  └──── Decode ◀────┘
          │
          ▼
       Response
~~~

## I1. Prefill vs Decode
- Prefill: 큰 matrix compute 중심
- Decode: 작은 step 반복 + KV read 중심

이 차이를 arithmetic intensity와 memory traffic으로 연결.

## I2. Continuous Batching
요청 전체가 아니라 iteration/token budget 기준으로 섞이는 이유.

## I3. KV Cache
- K/V가 무엇인지
- 왜 저장하는지
- token/context/heads/dtype가 capacity에 어떻게 반영되는지

## I4. Paged Attention / Block Allocation
연속 메모리 문제를 block/page 관리로 바꾸는 이유.

## I5. Chunked Prefill
긴 prompt를 왜 chunk로 나누는가?
latency/throughput/memory/scheduling tradeoff.

## I6. Prefix Caching
동일 prefix 계산을 왜 재사용할 수 있는가?

## I7. Speculative Decoding
- draft
- verify
- acceptance
- token budget

확률과 serving throughput이 만나는 영역.

## I8. Scheduler
- waiting/running
- max-num-seqs
- max-num-batched-tokens
- preemption
- fairness
- throughput vs latency

## I9. PD Disaggregation
- Prefill/Decode 자원 특성 분리
- KV transfer
- routing
- NIXL/Mooncake
- backpressure/failure

---

# 11. Phase J — CUDA / GPU

PyTorch 연산이 GPU에서 실제로 어떻게 실행되는지 내려간다.

## J1. GPU Physical Hierarchy

~~~text
GPU
 └─ GPC / TPC
     └─ SM
         ├─ CUDA cores
         ├─ Tensor cores
         ├─ registers
         └─ shared memory
~~~

## J2. CUDA Execution Model

~~~text
Kernel
  │
Grid
  │
Block
  │
Warp
  │
Thread
~~~

## J3. Memory Hierarchy

~~~text
Register
  ↓
Shared Memory / L1
  ↓
L2
  ↓
HBM
~~~

latency, capacity, bandwidth의 차이를 구분한다.

## J4. GEMM / Tensor Core
Transformer의 대부분을 차지하는 matrix multiplication이 실제 hardware에서 어떻게 실행되는가.

## J5. Compute-bound vs Memory-bound

핵심 질문:

> 느린 이유가 계산량 때문인가, 데이터를 옮기는 비용 때문인가?

연결:
- Prefill
- Decode
- KV cache
- quantization

## J6. Arithmetic Intensity / Roofline
FLOPs와 memory traffic의 관계로 bottleneck을 분석.

## J7. Kernel Fusion / FlashAttention
중간 tensor를 HBM에 쓰고 읽는 일을 줄이면 왜 빨라지는가?

## J8. CUDA Graph / Compilation
kernel launch 및 runtime overhead를 줄이고 실행 경로를 고정하는 이유.

---

# 12. Phase K — Distributed Inference

한 GPU의 한계를 넘어갈 때 시작된다.

## K1. Tensor Parallel

Matrix를 나눈다는 것이 실제로 무엇을 뜻하는가?

~~~text
XW
 │
 ├─ GPU0: W0
 ├─ GPU1: W1
 └─ ...
      │
      ▼
Collective communication
~~~

## K2. Pipeline Parallel
layer를 stage로 분리.

## K3. Data Parallel / Replica
model replica와 request distribution.

## K4. Expert Parallel
MoE expert를 GPU/node에 분산.

## K5. Collectives
- AllReduce
- AllGather
- ReduceScatter
- AllToAll

어떤 parallelism에서 왜 어떤 collective가 필요한지 연결.

## K6. Topology

~~~text
GPU
 ├─ NVLink
 ├─ NVSwitch
 └─ PCIe
      │
      ▼
     NIC
      │
      ▼
IB / RoCE
      │
      ▼
Remote GPU
~~~

## K7. RDMA / GPUDirect RDMA
CPU bounce 없이 GPU memory 간 data path를 단축하는 이유.

## K8. NCCL
Collective algorithm과 실제 topology/fabric의 매핑.

## K9. Scaling Efficiency

핵심 질문:

> GPU를 2배 썼는데 왜 성능은 2배가 되지 않는가?

~~~text
Compute benefit
     -
Communication
     -
Synchronization
     -
Load imbalance
     -
Memory overhead
     =
Actual scale-out
~~~

---

# 13. Phase L — Platform / Production

최종적으로 모델과 GPU를 서비스 시스템 안에 넣는다.

## Request Lifecycle

~~~text
Client
  ↓
Gateway
  ↓
Proxy / Auth / Policy
  ↓
Model Router
  ↓
Serving Engine
  ↓
GPU
  ↓
Streamed Response
~~~

## 주요 주제
- API semantics / streaming / cancellation
- timeout / retry / idempotency
- request routing / load balancing
- model lifecycle / deployment
- stateful response / recovery
- autoscaling
- GPU reservation / placement
- observability
- failure domains
- capacity planning
- SLO / latency percentile
- cost per token / utilization / service rate

여기서는 모델이 잘 돈다와 플랫폼이 서비스를 보장한다를 분리한다.

---

# 14. Paper → Production 통합 루프

최종적으로 어떤 논문이나 새로운 모델을 만나도 아래 순서로 해체한다.

~~~text
1. Problem
   왜 새로운 방법이 필요한가?
        │
        ▼
2. Information Flow
   입력에서 출력까지 무엇을 보존/압축/버리는가?
        │
        ▼
3. Math
   각 수식의 분자/분모/축/정규화는 무엇인가?
        │
        ▼
4. Tensor Shapes
   [B,S,H]가 각 단계에서 어떻게 변하는가?
        │
        ▼
5. Compute
   GEMM? element-wise? reduction? softmax?
        │
        ▼
6. Memory
   무엇을 저장하고, 몇 번 읽고 쓰는가?
        │
        ▼
7. GPU Kernel
   어떤 kernel/fusion/library가 필요한가?
        │
        ▼
8. Distributed
   TP/EP 시 어떤 communication이 생기는가?
        │
        ▼
9. Serving
   KV/state, TTFT, TPOT, throughput에 어떤 영향인가?
        │
        ▼
10. Experiment
    어떤 metric으로 가설을 검증할 것인가?
~~~

이 루프를 혼자 수행할 수 있게 되는 것이 Core Forge의 최종 목표다.

---

# 15. 개념별 아래로 내려가는 길

상위 개념에서 막힐 때 사용할 복귀 지도다.

| 막힌 질문 | 내려갈 선수 개념 |
|---|---|
| 왜 Attention에 QK^T가 필요한가? | vector → dot product → projection |
| 왜 sqrt(d)로 나누는가? | square → variance → scale → softmax sensitivity |
| Softmax가 왜 필요한가? | ratio → probability → exponential → normalization |
| RMSNorm이 왜 동작하는가? | square → mean → square root → scale |
| 왜 LoRA가 low-rank인가? | matrix → rank → basis → SVD |
| 왜 Decode가 memory-bound인가? | bytes/FLOPs → memory hierarchy → arithmetic intensity |
| 왜 Prefill은 GPU를 잘 채우는가? | matrix multiplication → batching → Tensor Core |
| 왜 TP가 늘면 통신이 커지는가? | matrix partition → collective → topology |
| 왜 MoE에는 AllToAll이 필요한가? | routing → expert placement → distributed data movement |
| 왜 long-context가 어려운가? | attention shape → KV size → compute/memory complexity |
| 왜 speculative decoding이 빨라질 수 있는가? | probability → verification → acceptance rate → decode bottleneck |
| 왜 queue가 어느 순간 폭증하는가? | rate → capacity → saturation → queueing |

---

# 16. 기술별 위로 올라가는 길

~~~text
Division
  └─ rate
      ├─ tokens/s
      ├─ GB/s
      ├─ request/s
      └─ scaling efficiency

Square
  ├─ variance
  ├─ RMS
  ├─ RMSNorm
  └─ L2 norm

Dot Product
  ├─ cosine similarity
  ├─ projection
  └─ attention score

Matrix Multiplication
  ├─ Linear Layer
  ├─ QKV projection
  ├─ MLP
  ├─ GEMM kernel
  └─ Tensor Parallel

Probability
  ├─ Softmax
  ├─ Cross Entropy
  ├─ Sampling
  └─ Speculative Decoding

Derivative
  ├─ Gradient
  ├─ Backpropagation
  ├─ Optimization
  └─ sensitivity / local approximation
~~~

---

# 17. 통과 기준 — 안다의 단계

개념을 배웠다고 판단하는 기준은 정의 암기가 아니다.

| Level | 할 수 있어야 하는 것 |
|---:|---|
| 1 | 왜 필요한지 말할 수 있다 |
| 2 | 작은 숫자 예제를 직접 계산할 수 있다 |
| 3 | 수식의 각 기호가 어떤 역할인지 설명할 수 있다 |
| 4 | Tensor shape / data flow로 그릴 수 있다 |
| 5 | PyTorch/vLLM 코드에서 해당 연산을 찾을 수 있다 |
| 6 | GPU memory/compute 관점으로 비용을 설명할 수 있다 |
| 7 | 분산 시 communication이 어디서 발생하는지 설명할 수 있다 |
| 8 | 실제 metric으로 가설을 검증할 수 있다 |
| 9 | 논문에서 변형된 형태를 만나도 근원 개념으로 환원할 수 있다 |

Core Forge는 궁극적으로 Level 9를 목표로 하지만, 모든 개념을 한 번에 Level 9까지 올리지는 않는다. 여러 번 다시 만나면서 깊어진다.

---

# 18. 현재 위치와 당분간의 진행 방향

현재:

~~~text
Number
  ↓
Expression
  ↓
Function
  ↓
Coordinate
  ↓
Delta
  ↓
Rate
  ↓
Slope
  ↓
Nonlinearity   ← CURRENT
~~~

당분간:

~~~text
Nonlinearity
   ↓
좁은 구간의 변화
   ↓
Limit intuition
   ↓
Derivative
   ↓
Derivative의 부호/크기
   ↓
Gradient의 씨앗
~~~

그 뒤 바로 고급 미적분로 돌진하지 않는다.

~~~text
Derivative 기초
   │
   ├───────────────┐
   ▼               ▼
Vector/Geometry   Probability foundation
   │               │
   ▼               ▼
Matrix           Distribution/Softmax
   └───────┬───────┘
           ▼
      Deep Learning
~~~

---

# 19. 최종 목적지의 모습

논문에서 다음과 같은 식을 본다고 하자.

Y = softmax(QK^T / sqrt(d))V

최종적으로는 이를 Attention 공식이라고 외우는 대신 다음 흐름으로 해체할 수 있어야 한다.

~~~text
X
│
├─ Linear projection → Q
├─ Linear projection → K
└─ Linear projection → V
          │
          ▼
       QKᵀ
          │
          ├─ vector dot product
          ├─ pairwise relation score
          └─ GEMM
          │
          ▼
       / sqrt(d)
          │
          └─ score scale 제어
          │
          ▼
        Softmax
          │
          ├─ relative weighting
          └─ probability-like distribution
          │
          ▼
          × V
          │
          └─ weighted information aggregation
          │
          ▼
      Output representation
          │
          ├─ Tensor shape
          ├─ KV cache
          ├─ GPU memory traffic
          ├─ FlashAttention kernel
          ├─ TP communication
          └─ serving latency
~~~

즉 최종 목표는 **수학을 공부하는 것 자체가 아니라, 정보 변환의 근원을 끝까지 추적하는 능력**이다.

---

## 한 문장 원칙

> **개념을 안다는 것은 공식을 기억하는 것이 아니라, 그 연산이 왜 생겼고 어떤 정보를 보존·제거·변환하며 그 결과가 모델·GPU·시스템에서 어떤 현상으로 나타나는지 연결해서 설명할 수 있다는 뜻이다.**
