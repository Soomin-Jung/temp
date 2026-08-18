# 99. 논문 읽기 순서, 수식 Cheat Sheet, 용어집

작성일: 2026-08-18  
상위 문서: [Attention Architecture Study Guide](README.md)

## 1. 이 문서의 목적

Attention 분야는 논문 이름과 약어가 빠르게 늘어난다. 이 문서는 앞의 상세 장을 다시 읽기 전에 필요한 개념을 빠르게 복구하기 위한 reference다.

구성:

1. 권장 논문 읽기 순서
2. 핵심 수식 cheat sheet
3. 메커니즘 분류표
4. 영문 용어/발음/한국어 의미
5. 논문을 읽을 때 확인할 질문

---

# 2. 권장 논문 읽기 순서

## Level 1 — Transformer와 Full Attention

### 1. Attention Is All You Need

Vaswani et al., 2017

https://arxiv.org/abs/1706.03762

먼저 볼 것:

- scaled dot-product attention
- multi-head attention
- residual + normalization
- causal self-attention

수식:

```math
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

### 2. RoFormer

Su et al., 2021

https://arxiv.org/abs/2104.09864

먼저 볼 것:

- RoPE
- query/key rotation
- relative position이 dot product에 나타나는 이유

### 3. Multi-Query Attention

Shazeer, 2019

https://arxiv.org/abs/1911.02150

먼저 볼 것:

- decode memory bandwidth
- query heads와 shared KV

### 4. Grouped-Query Attention

Ainslie et al., 2023

https://arxiv.org/abs/2305.13245

먼저 볼 것:

- MHA ↔ MQA 사이의 interpolation
- KV head sharing

---

# 3. Level 2 — GPU와 Serving

### 5. FlashAttention

Dao et al., 2022

https://arxiv.org/abs/2205.14135

먼저 볼 것:

- IO complexity
- HBM/SRAM hierarchy
- online softmax
- tiling

### 6. PagedAttention / vLLM

Kwon et al., 2023

https://arxiv.org/abs/2309.06180

먼저 볼 것:

- KV cache fragmentation
- logical block / physical block
- copy-on-write
- continuous batching과 memory utilization

---

# 4. Level 3 — MLA

### 7. DeepSeek-V2

https://arxiv.org/abs/2405.04434

먼저 볼 것:

- KV down-projection
- query compression
- decoupled RoPE
- matrix absorption
- KV cache 비교

이 논문 이후 DeepSeek-V3를 읽으면 MLA가 실제 대규모 MoE에서 어떻게 계승됐는지 보인다.

### 8. DeepSeek-V3

https://arxiv.org/abs/2412.19437

Attention만 볼 경우 MLA chapter를 중심으로 보고 이후 MTP/MoE/system section으로 확장한다.

---

# 5. Level 4 — Linear/Recurrent Attention

### 9. Transformers are RNNs

Katharopoulos et al., 2020

https://arxiv.org/abs/2006.16236

먼저 볼 것:

- kernel feature map
- associativity
- recurrent state

### 10. Mamba

https://arxiv.org/abs/2312.00752

Attention 논문은 아니지만 selective state-space recurrence를 이해하는 데 중요하다.

### 11. Transformers are SSMs / Mamba-2

https://arxiv.org/abs/2405.21060

먼저 볼 것:

- State Space Duality
- semiseparable matrix
- attention/SSM 사이의 수학적 연결

### 12. Parallelizing Linear Transformers with the Delta Rule

https://arxiv.org/abs/2406.06484

먼저 볼 것:

- Delta Rule
- associative memory overwrite
- parallel/chunkwise algorithm

### 13. Gated Delta Networks

https://arxiv.org/abs/2412.06464

먼저 볼 것:

- forgetting gate
- delta rule의 targeted update
- Mamba-2와의 비교

---

# 6. Level 5 — Kimi와 Qwen Hybrid

### 14. Kimi Linear

https://arxiv.org/abs/2510.26692

먼저 볼 것:

- KDA
- finer-grained gate
- KDA/MLA hybrid
- recurrent position interpretation
- long-context cache/throughput experiment

### 15. Kimi-K3

https://arxiv.org/abs/2607.24653

먼저 볼 것:

- 69 KDA + 24 Gated MLA
- lower-bounded decay
- FlashKDA co-design
- Attention Residuals
- Stable LatentMoE
- state-aware serving

### 16. Qwen3-Next

공식 technical blog:

https://qwenlm.github.io/blog/qwen3-next/

먼저 볼 것:

- 3:1 GDN/full hybrid
- output gating
- partial RoPE
- hybrid ratio ablation

### 17. FlashQLA

https://github.com/QwenLM/FlashQLA

먼저 볼 것:

- GDN을 GPU kernel로 어떻게 fuse하는가
- TileLang implementation
- forward/backward kernel organization

---

# 7. Level 6 — Sparse/Compressed Attention

### 18. DeepSeek-V3.2 / DSA

https://arxiv.org/abs/2512.02556

먼저 볼 것:

- Lightning Indexer
- MLA + top-k sparse attention
- long-context efficiency

### 19. DeepSeek-V4

Model/report:

https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro

Transformers architecture documentation:

https://huggingface.co/docs/transformers/model_doc/deepseek_v4

먼저 볼 것:

- CSA
- HCA
- shared K=V MQA
- partial RoPE
- local raw branch
- compression ratio

---

# 8. Level 7 — 다른 Hybrid 사례

### Gemma 3

https://arxiv.org/abs/2503.19786

- 5 Local : 1 Global
- GQA
- QK-Norm

### Mistral 7B

https://arxiv.org/abs/2310.06825

- GQA
- Sliding Window
- rotating buffer cache

### MiniMax-01

https://arxiv.org/abs/2501.08313

- Lightning Attention
- periodic softmax layer

### Griffin / RecurrentGemma

https://arxiv.org/abs/2402.19427

- gated linear recurrence
- local attention hybrid

---

# 9. 핵심 수식 Cheat Sheet

## 9.1 Full Softmax Attention

```math
O=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
```

의미:

> 모든 허용된 query-key pair의 score를 만든 뒤 softmax weight로 value를 섞는다.

---

## 9.2 Conventional KV Cache

```math
M_{KV}=2LTH_{kv}d_hb
```

- $L$: layer 수
- $T$: token 수
- $H_{kv}$: KV head 수
- $d_h$: head dimension
- $b$: bytes/element

---

## 9.3 RoPE

```math
q_t'=R_tq_t,\qquad k_i'=R_ik_i
```

```math
(q_t')^\top k_i'
=q_t^\top R_t^\top R_i k_i
```

의미:

> position에 따른 rotation을 dot-product 안에 넣어 상대 위치 관계를 표현한다.

---

## 9.4 MLA KV Compression

```math
c_t^{KV}=W_{DKV}x_t
```

```math
k_t=W_{UK}c_t^{KV},\qquad
v_t=W_{UV}c_t^{KV}
```

의미:

> full K/V를 cache하지 않고 K/V를 재구성할 수 있는 latent를 cache한다.

---

## 9.5 MLA Matrix Absorption

```math
q^\top W_{UK}c
=(W_{UK}^\top q)^\top c
```

```math
\sum_i a_iW_{UV}c_i
=W_{UV}\sum_i a_ic_i
```

의미:

> up-projection을 모든 cached token에 적용하지 않고 query/output path에 흡수한다.

---

## 9.6 Additive Linear Attention State

```math
S_t=S_{t-1}+k_tv_t^\top
```

```math
o_t=S_t^\top q_t
```

의미:

> 과거 key-value 외적을 fixed matrix memory에 누적한다.

---

## 9.7 Delta Rule

```math
\hat v_t=S_{t-1}^\top k_t
```

```math
e_t=v_t-\hat v_t
```

```math
S_t=S_{t-1}+\beta_tk_te_t^\top
```

의미:

> memory가 이미 아는 것과 새 target의 차이만 현재 key 방향에 기록한다.

---

## 9.8 Delta Rule — Erase + Write

```math
S_t
=(I-\beta_tk_tk_t^\top)S_{t-1}
+\beta_tk_tv_t^\top
```

의미:

> 현재 key와 겹치는 기존 association을 약화하고 새 association을 기록한다.

---

## 9.9 Gated DeltaNet

단순화:

```math
S_t
=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}
+\beta_tk_tv_t^\top
```

의미:

> memory lifetime을 제어하는 forgetting gate를 Delta Rule에 추가한다.

---

## 9.10 KDA

```math
S_t
=(I-\beta_tk_tk_t^\top)
\mathrm{Diag}(\alpha_t)S_{t-1}
+\beta_tk_tv_t^\top
```

의미:

> forgetting을 state channel별로 세분화한다.

---

## 9.11 Sliding Window

```math
\mathcal{N}(t)=\{i:t-W+1\le i\le t\}
```

```math
o_t=\mathrm{Attention}(q_t,K_{\mathcal N(t)},V_{\mathcal N(t)})
```

의미:

> 최근 $W$개 token만 직접 조회한다.

---

## 9.12 Sequence Compression

```math
T_c\approx\frac{T}{m}
```

의미:

> $m$개의 raw token 정보를 더 적은 compressed memory entry로 요약한다.

---

# 10. Attention Mechanism 분류 Cheat Sheet

| Mechanism | token별 memory | context 증가 시 memory | query lookup | 핵심 compression axis |
|---|---|---:|---|---|
| MHA | full K/V | $O(T)$ | all tokens | 없음 |
| GQA | grouped K/V | $O(T)$ | all tokens | KV heads |
| MQA | one/shared KV | $O(T)$ | all tokens | KV heads 극대화 공유 |
| MLA | latent KV | $O(T)$ | all tokens | feature/rank |
| Sliding Window | recent K/V | $O(W)$ 가능 | recent W | token range |
| Sparse | token/compressed K/V | $O(T)$ 계열 | top-k | lookup count |
| CSA | compressed entries + local | $O(T/m)$ 계열 | top-k compressed | sequence + lookup |
| HCA | heavily compressed + local | $O(T/m')$ 계열 | all compressed | sequence |
| GDN | recurrent state | $O(1)$ wrt T | state | history→state |
| KDA | recurrent state | $O(1)$ wrt T | state | history→state + fine gate |

---

# 11. 자주 혼동하는 쌍

## FlashAttention vs Linear Attention

- FlashAttention: exact softmax attention을 빠르게 계산
- Linear Attention: attention formulation/state 자체를 바꿈

## PagedAttention vs Sparse Attention

- PagedAttention: KV physical memory management
- Sparse Attention: query가 조회하는 logical token subset

## GQA vs MLA

- GQA: head sharing
- MLA: low-rank latent compression

## MLA vs KDA

- MLA: token별 compressed memory 유지
- KDA: token history를 fixed recurrent state에 압축

## Sliding Window vs Recurrent State

- SWA: 최근 token들을 개별 KV로 유지
- recurrent: 전체 history가 state에 누적됨

## DSA vs CSA

- DSA: token-level MLA history에서 sparse selection
- CSA: history를 먼저 sequence-compress하고 sparse selection

## CSA vs HCA

- CSA: moderate compression + sparse top-k
- HCA: heavy compression + dense lookup

---

# 12. 주요 영문 용어집

| English | 읽는 법 | 한국어 의미 / 문맥 |
|---|---|---|
| Attention | 어텐션 | 주의집중; query가 memory에서 필요한 정보를 선택해 가져오는 연산 |
| Query | 쿼리 | 질의; 현재 token이 찾는 정보의 표현 |
| Key | 키 | 검색 주소/특징; query와 비교되는 표현 |
| Value | 밸류 | 선택된 key가 전달할 실제 정보 |
| Head | 헤드 | 독립 attention subspace/연산 단위 |
| Multi-Head Attention | 멀티 헤드 어텐션 | 여러 attention head를 병렬로 사용하는 구조 |
| Multi-Query Attention | 멀티 쿼리 어텐션 | 여러 query head가 하나의 KV를 공유 |
| Grouped-Query Attention | 그룹드 쿼리 어텐션 | query head group이 KV head를 공유 |
| Multi-head Latent Attention | 멀티 헤드 레이턴트 어텐션 | token KV를 low-rank latent로 압축하는 attention |
| Latent | 레이턴트 | 직접 관측/표현되는 큰 공간보다 압축된 잠재 표현 |
| Projection | 프로젝션 | learned matrix로 표현 공간/차원을 변환 |
| Low Rank | 로우 랭크 | 적은 독립 basis로 큰 표현을 근사/구성 |
| Matrix Absorption | 매트릭스 업소프션 | projection matrix를 다른 연산과 결합해 runtime reconstruction을 줄임 |
| RoPE | 로프 | Rotary Position Embedding, 회전형 위치 임베딩 |
| Partial RoPE | 파셜 로프 | head dimension 일부에만 RoPE 적용 |
| NoPE | 노프/노 피이 | 명시적 positional encoding 없음 |
| Softmax | 소프트맥스 | score를 합이 1인 양수 weight로 변환 |
| Causal Mask | 코절 마스크 | 미래 token을 보지 못하게 하는 mask |
| KV Cache | 케이브이 캐시 | autoregressive decode용 과거 key/value memory |
| Prefill | 프리필 | prompt token을 처리해 cache/state를 만드는 단계 |
| Decode | 디코드 | 새 output token을 한 step씩 생성하는 단계 |
| Recurrent | 리커런트 | 이전 state를 입력으로 다음 state를 계산하는 순환 구조 |
| Recurrence | 리커런스 | recurrent state update 관계/점화식 |
| State | 스테이트 | 과거 정보를 요약해 다음 step으로 전달하는 상태 |
| Fast Weight | 패스트 웨이트 | sequence 처리 중 빠르게 바뀌는 임시 memory matrix |
| Associative Memory | 어소시에이티브 메모리 | key를 입력하면 연관된 value를 복원하는 기억 구조 |
| Delta Rule | 델타 룰 | 기존 prediction과 target의 차이만큼 memory를 수정하는 규칙 |
| DeltaNet | 델타넷 | Delta Rule을 recurrent linear attention에 사용한 구조 |
| Gated DeltaNet | 게이티드 델타넷 | forgetting gate와 Delta Rule을 결합한 구조 |
| Kimi Delta Attention | 키미 델타 어텐션 | GDN을 channel-wise retention으로 확장한 Kimi attention |
| Retention | 리텐션 | 기존 memory를 유지하는 정도 |
| Decay | 디케이 | memory가 시간이 지나며 감쇠하는 것 |
| Forget Gate | 포겟 게이트 | 과거 state를 얼마나 지울지 제어하는 gate |
| Output Gate | 아웃풋 게이트 | 계산된 attention/state output을 residual stream에 얼마나 보낼지 제어 |
| Outer Product | 아우터 프로덕트 | 두 vector에서 matrix를 만드는 외적 |
| Dot Product | 닷 프로덕트 | 두 vector에서 scalar similarity를 만드는 내적 |
| Rank-1 Update | 랭크 원 업데이트 | vector outer product로 matrix의 한 방향을 갱신 |
| Diagonal Matrix | 다이애거널 매트릭스 | 대각선 외 원소가 0인 matrix |
| Sliding Window Attention | 슬라이딩 윈도우 어텐션 | 최근 fixed window token만 조회 |
| Sparse Attention | 스파스 어텐션 | 전체 token 중 일부만 선택해 attention |
| Indexer | 인덱서 | sparse attention 후보를 빠르게 score/select하는 모듈 |
| Top-k | 탑 케이 | score가 높은 k개 후보 선택 |
| Sequence Compression | 시퀀스 컴프레션 | 여러 token memory를 더 적은 sequence entry로 압축 |
| Hybrid Attention | 하이브리드 어텐션 | 서로 다른 attention/sequence mixer layer를 한 모델에 결합 |
| FlashAttention | 플래시 어텐션 | exact softmax attention의 IO-aware GPU algorithm |
| FlashMLA | 플래시 엠엘에이 | MLA optimized GPU kernel |
| FlashKDA | 플래시 케이디에이 | KDA optimized GPU kernel |
| FlashQLA | 플래시 큐엘에이 | Qwen GDN/linear attention optimized kernel |
| PagedAttention | 페이지드 어텐션 | paged KV memory management algorithm |
| Continuous Batching | 컨티뉴어스 배칭 | request를 running batch에 동적으로 넣고 빼는 serving scheduler 방식 |
| Preemption | 프리엠션 | running request를 중단하고 나중에 복구하는 것 |
| Tensor Parallelism | 텐서 패럴렐리즘 | layer tensor 연산을 여러 GPU에 분할 |
| Context Parallelism | 컨텍스트 패럴렐리즘 | sequence/context token 축을 여러 GPU에 분할 |
| Expert Parallelism | 엑스퍼트 패럴렐리즘 | MoE expert를 여러 GPU에 분산 |
| Arithmetic Intensity | 아리스메틱 인텐시티 | memory byte당 수행 FLOPs |
| Memory-bound | 메모리 바운드 | 연산보다 memory bandwidth가 병목인 상태 |
| Compute-bound | 컴퓨트 바운드 | memory보다 연산 throughput이 병목인 상태 |
| Attention Sink | 어텐션 싱크 | attention probability mass를 흡수하는 token/bias 경로 |
| QK-Norm | 큐케이 놈 | query/key를 attention score 전에 정규화하는 기법 |
| ShortConv | 쇼트 컨볼루션 | 최근 소수 token을 혼합하는 짧은 convolution |

---

# 13. 수학 기호 읽기

| 기호 | 읽는 법 | 의미 |
|---|---|---|
| $x_t$ | 엑스 서브 티 | t번째 token/step의 vector |
| $x^\top$ | 엑스 트랜스포즈 | 전치 |
| $\hat v$ | 브이 햇 | 예측/추정된 v |
| $\alpha$ | 알파 | retention/weight 등에 자주 사용 |
| $\beta$ | 베타 | update strength 등에 자주 사용 |
| $\gamma$ | 감마 | cumulative factor 등에 자주 사용 |
| $\sum$ | 시그마/서메이션 | 합 |
| $\prod$ | 프로덕트 | 곱 |
| $\odot$ | 하다마드 프로덕트 | 원소별 곱 |
| $\|x\|_2$ | 엘투 놈 오브 엑스 | Euclidean vector length |
| $\nabla$ | 나블라 | gradient |
| $\in$ | 이즈 인 | 어떤 집합/공간에 속함 |
| $\mathbb{R}^d$ | 알 디 차원 실수공간 | d개의 실수로 된 vector space |
| $O(T)$ | 빅 오브 티 | T에 선형 비례하는 점근 비용 |
| $O(T^2)$ | 빅 오브 티 스퀘어드 | T 제곱에 비례하는 점근 비용 |

---

# 14. 새 논문을 읽을 때의 질문

Attention architecture paper를 처음 읽으면 abstract의 `efficient`, `linear`, `sparse`라는 단어보다 다음을 먼저 찾는다.

## Memory

- token마다 무엇을 저장하는가?
- state가 fixed-size인가?
- feature rank가 압축되는가?
- sequence가 압축되는가?

## Lookup

- query가 모든 token을 보는가?
- local window인가?
- top-k인가?
- recurrent state 하나를 보는가?

## Position

- RoPE인가?
- partial RoPE인가?
- NoPE layer가 있는가?
- recurrence 자체가 position-sensitive한가?

## Complexity

- prefill complexity는?
- decode의 history-length dependency는?
- indexer/compressor 비용은 식에 포함됐는가?

## GPU

- recurrence를 training에서 어떻게 parallelize하는가?
- chunk/block size는?
- Tensor Core GEMM으로 변환되는가?
- precision/numerical range 제약은?

## Quality

- exact retrieval benchmark는?
- long-context needle retrieval은?
- in-context learning은?
- multi-hop reasoning은?
- pure/full baseline 대비 어떤 품질 trade-off가 있는가?

## Serving

- cache layout은?
- prefix cache가 가능한가?
- state snapshot이 필요한가?
- TP/CP sharding은?
- optimized kernel이 공개됐는가?

---

# 15. 논문의 성능 수치를 읽는 원칙

`6× faster`, `90% KV reduction` 같은 수치는 다음 조건을 함께 읽는다.

- 어떤 baseline인가?
- model size가 같은가?
- context length는?
- batch/concurrency는?
- hardware는?
- dtype은?
- training 또는 inference인가?
- prefill인가 decode인가?
- kernel maturity가 동일한가?

Model architecture의 우수성과 특정 benchmark kernel의 우수성을 분리한다.

---

# 16. 이 Study Guide의 핵심 철학

Attention 이름을 외우는 대신 다음 질문을 기억한다.

> **이 모델은 과거 정보를 어떤 형태로 저장하고, 현재 query가 그 기억의 어느 부분을 어떤 비용으로 읽는가?**

모든 주요 구조는 이 질문으로 다시 표현할 수 있다.

- GQA: token memory는 그대로 두고 KV head를 공유한다.
- MLA: token memory를 latent feature로 압축한다.
- SWA: 최근 token memory만 읽는다.
- DSA: 중요한 token memory만 고른다.
- CSA/HCA: token sequence 자체를 압축한다.
- GDN/KDA: token memory를 fixed recurrent state로 접는다.
- Hybrid: 서로 다른 기억 구조를 layer별로 나눠 사용한다.

이 관점이 새로운 모델과 논문을 계속 따라가기 위한 가장 재사용 가능한 기준이다.
