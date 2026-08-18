# 99. Kimi K3 논문 읽기 순서, 핵심 수식, 용어집

작성일: 2026-08-19  
상위 문서: [Kimi K3 Study Guide](README.md)

## 1. 이 문서의 목적

Kimi K3 technical report를 처음부터 바로 읽으면 KDA, AttnRes, LatentMoE, Muon, MTP, RL infrastructure가 한꺼번에 등장한다.

이 문서는 K3를 이해하기 위한 논문 dependency와 용어를 빠르게 복구하는 reference다.

---

# 2. 가장 추천하는 논문 읽기 순서

## Stage 0 — Transformer / Attention 기반

이미 `study/attention`에서 정리했다.

1. [Attention Foundations](../../study/attention/00-foundations.md)
2. [Full Attention / GQA](../../study/attention/01-full-attention.md)
3. [MLA](../../study/attention/02-mla-low-rank-attention.md)
4. [Linear Attention / DeltaNet / KDA](../../study/attention/03-linear-recurrent-attention.md)

K3 model study에서는 이 내용을 prerequisite로 둔다.

---

# 3. Stage 1 — Moonshot의 RL/Reasoning 뿌리

## Kimi k1.5: Scaling Reinforcement Learning with LLMs

https://arxiv.org/abs/2501.12599

먼저 볼 것:

- long-context RL
- partial rollout
- long-CoT
- policy optimization
- long-to-short
- multimodal RL

왜 먼저 읽는가?

K3의 `reasoning effort`, long-horizon rollout, agentic RL을 architecture와 구분해 이해할 수 있다.

---

# 4. Stage 2 — K2: Large MoE Foundation

## Kimi K2

https://github.com/MoonshotAI/Kimi-K2

먼저 볼 것:

- 1T / 32B active architecture
- 384 routed / top-8
- shared expert
- MLA
- sparsity scaling
- MuonClip
- agentic data synthesis
- RLVR + critique/reward

K3와 Table을 나란히 놓고 보는 것이 좋다.

---

# 5. Stage 3 — Muon

## Muon is Scalable for LLM Training

https://arxiv.org/abs/2502.16982

먼저 볼 것:

- matrix-aware optimizer라는 의미
- Newton–Schulz orthogonalization
- update scaling
- weight decay
- distributed implementation
- AdamW 대비 scaling-law 비교

그 다음 K2의 MuonClip, K3의 Per-Head Muon을 읽는다.

---

# 6. Stage 4 — Vision 계보

## Kimi-VL Technical Report

https://arxiv.org/abs/2504.07491

먼저 볼 것:

- MoonViT
- native resolution
- multimodal long context
- multimodal reasoning SFT/RL

## Kimi K2.5

https://github.com/MoonshotAI/Kimi-K2.5

먼저 볼 것:

- K2-Base continual multimodal pretraining
- mixed text/vision tokens
- zero-vision SFT
- joint text-vision RL
- Agent Swarm

그 뒤 K3 MoonViT-V2를 읽으면 무엇이 계승되고 무엇이 바뀌었는지 보인다.

---

# 7. Stage 5 — Kimi Linear

## Kimi Linear: An Expressive, Efficient Attention Architecture

https://arxiv.org/abs/2510.26692

공식 repo:

https://github.com/MoonshotAI/Kimi-Linear

먼저 볼 것:

- finite-state memory 문제
- KDA fine-grained gating
- 3 KDA : 1 MLA hybrid
- chunkwise KDA
- 1M context evaluation
- controlled full-attention comparison

읽은 뒤 다음 질문에 답한다.

> `K3에서 69 KDA + 24 MLA가 왜 자연스러운가?`

---

# 8. Stage 6 — Attention Residuals

## Attention Residuals

https://arxiv.org/abs/2603.15031

공식 repo:

https://github.com/MoonshotAI/Attention-Residuals

먼저 볼 것:

- PreNorm residual dilution
- Full AttnRes
- learned pseudo-query
- depth-wise softmax
- Block AttnRes
- $O(Ld)\rightarrow O(Nd)$ source memory
- Kimi Linear 48B experiment

KDA와 AttnRes가 서로 다른 axis임을 확실히 구분한다.

---

# 9. Stage 7 — LatentMoE

## LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

https://arxiv.org/abs/2601.18089

먼저 볼 것:

- accuracy per FLOP vs accuracy per parameter
- memory-bound MoE decode
- EP all-to-all communication
- $d\rightarrow\ell$ latent compression
- expert count $N$ 확대
- top-k $K$ 확대
- shared experts full-width 유지

핵심 식:

$$
z=W_{\downarrow}x
$$

$$
y_{route}=W_{\uparrow}\sum_{i\in\mathcal T_K}p_iE_i(z)
$$

K3의 Stable LatentMoE가 왜 `latent`인지 이해한 다음 K3 report의 stability modification을 읽는다.

---

# 10. Stage 8 — Kimi K3

## Kimi K3: Open Frontier Intelligence

Official repo/report:

https://github.com/MoonshotAI/Kimi-K3

ArXiv:

https://arxiv.org/abs/2607.24653

이제 report를 순서대로 읽어도 된다.

권장 집중 순서:

1. Architecture overview / K2 comparison table
2. KDA modifications
3. Attention Residuals
4. Stable LatentMoE
5. Native multimodality
6. Optimizer / pretraining recipe
7. Post-training/RL
8. MTP/speculative decoding
9. System co-design / serving

---

# 11. 핵심 수식 Cheat Sheet

## 11.1 Sparse MoE

$$
\operatorname{MoE}(x)
=
\sum_{i\in\mathcal T_K(x)}p_iE_i(x)
$$

의미:

> 전체 expert 중 top-k만 token별로 실행한다.

---

## 11.2 LatentMoE

$$
z=W_{\downarrow}x,
\qquad z\in\mathbb{R}^{\ell}
$$

$$
y_{route}
=W_{\uparrow}\left(
\sum_{i\in\mathcal T_K}p_iE_i(z)
\right)
$$

의미:

> Routed expert compute와 dispatch를 model hidden $d$보다 작은 latent $\ell$에서 수행한다.

---

## 11.3 Compression Ratio

$$
\alpha=\frac{d}{\ell}
$$

K3:

$$
7168/3584=2
$$

---

## 11.4 Expert Parameter Approximation

GLU expert의 large matrices를 단순화하면:

$$
P_{expert}\approx3\ell m
$$

K3:

$$
3\times3584\times3072
\approx33.0M
$$

---

## 11.5 AttnRes

$$
h_l
=
\sum_{i<l}\alpha_{i\to l}v_i
$$

$$
\alpha_{i\to l}
=
\operatorname{softmax}_i
\left(w_l^\top\operatorname{Norm}(v_i)\right)
$$

의미:

> 현재 layer가 과거 depth representation을 learned pseudo-query로 선택한다.

---

## 11.6 Standard Residual

$$
x_{l+1}=x_l+F_l(x_l)
$$

반복 전개하면 과거 layer contribution의 additive accumulation으로 볼 수 있다.

---

## 11.7 Next-Token Prediction

$$
\mathcal L_{NTP}
=-\sum_t\log p(x_t\mid x_{<t})
$$

K3는 multimodal input에서도 foundation objective의 중심을 next-token prediction에 둔다.

---

## 11.8 Adam-style Update

$$
\Delta w
\propto
\frac{m}{\sqrt v+\epsilon}
$$

Element-wise adaptive scale.

---

## 11.9 Muon Matrix View

Momentum matrix:

$$
M
$$

을 Newton–Schulz polynomial iteration으로 orthogonalized/matrix-sign-like update에 가깝게 변환한다.

개념 목표:

$$
M=U\Sigma V^\top
\quad\rightarrow\quad
UV^\top
$$

정확한 implementation 식은 Muon paper/code를 참조한다.

---

## 11.10 Context Curriculum

K3의 큰 단계:

```text
8K → 64K → 256K → 1M
```

이는 position config 한 줄 변경이 아니라 training/data/system curriculum이다.

---

# 12. K3 숫자 Cheat Sheet

| 항목 | 값 |
|---|---:|
| Total parameters | 2.8T (report precise scale 약 2.78T) |
| Activated parameters | 104B (약 104.2B) |
| Language layers | 93 |
| Dense FFN layers | 1 |
| Hidden size | 7168 |
| KDA layers | 69 |
| Gated MLA layers | 24 |
| Attention heads | 96 |
| KDA head dim | 128 |
| KV latent rank | 512 |
| Q latent rank | 1536 |
| Stable LatentMoE latent dim | 3584 |
| Expert intermediate | 3072 |
| Routed experts | 896 |
| Active routed experts | 16 |
| Shared experts | 2 |
| AttnRes block size | 12 |
| Vision encoder | MoonViT-V2 |
| Vision encoder params | 401M |
| Vision layers | 27 |
| Context | 1,048,576 |
| Vocab | 160K |
| Expert activation | SiTU-GLU |
| Quantized serving target | MXFP4 weights / MXFP8 activations |
| MTP | 1 auxiliary predictive layer/module lineage |

---

# 13. K2 → K3 변화 Cheat Sheet

| 축 | K2 | K3 |
|---|---|---|
| Depth | 61 | 93 + Block AttnRes |
| Sequence | MLA | KDA + MLA hybrid |
| Width | 384/top8 experts | 896/top16 Stable LatentMoE |
| Shared expert | 1 | 2 |
| Expert hidden | 2048 | 3072 |
| Hidden width | 7168 | 7168 |
| Context | 128K | 1M |
| Activation | SwiGLU | SiTU-GLU |
| Optimizer | MuonClip | Per-Head Muon lineage |
| Vision | text foundation | MoonViT-V2 native multimodal |
| Post-training | agentic RL | domain×effort RL + multi-teacher distillation |

---

# 14. 주요 용어집

| English | 읽는 법 | 한국어 의미 / K3 문맥 |
|---|---|---|
| Mixture-of-Experts | 믹스처 오브 엑스퍼츠 | 여러 FFN expert 중 일부만 선택하는 sparse architecture |
| Expert | 엑스퍼트 | MoE에서 선택적으로 실행되는 FFN subnetwork |
| Routed Expert | 라우티드 엑스퍼트 | router top-k로 선택되는 expert |
| Shared Expert | 셰어드 엑스퍼트 | 모든 token이 실행하는 공통 expert |
| Router | 라우터 | token별 expert score와 top-k를 결정하는 module |
| Expert Parallelism | 엑스퍼트 패럴렐리즘 | experts를 여러 GPU에 분산하는 병렬화 |
| All-to-All | 올투올 | 모든 rank가 서로 token data를 교환하는 collective |
| Latent | 레이턴트 | 원 hidden보다 작은 압축 잠재 representation |
| LatentMoE | 레이턴트 엠오이 | routed expert compute/dispatch를 latent space에서 수행하는 MoE |
| Stable LatentMoE | 스테이블 레이턴트 엠오이 | K3의 stabilized LatentMoE: normalization, bounded GLU, balancing 포함 |
| Sparsity | 스파시티 | 전체 parameter/path 중 일부만 활성화되는 정도 |
| Activated Parameters | 액티베이티드 파라미터스 | token forward에서 실제 계산되는 parameter 규모 |
| SiTU-GLU | 시투 글루 | branch activation을 smooth bounded transform으로 제한한 K3 GLU |
| Quantile Balancing | 퀀타일 밸런싱 | expert score distribution의 quantile을 이용해 dispatch load를 맞추는 방법 |
| Attention Residuals | 어텐션 레지듀얼즈 | 과거 depth representations를 softmax로 선택하는 residual mechanism |
| Pseudo-Query | 슈도 쿼리 | AttnRes에서 layer별로 학습하는 depth-selection query vector |
| Depth Mixing | 뎁스 믹싱 | 서로 다른 network layer/depth representation을 결합하는 것 |
| Block AttnRes | 블록 어텐션 레지듀얼즈 | depth sources를 layer가 아닌 block 단위로 압축한 practical AttnRes |
| PreNorm | 프리 놈 | attention/FFN 전에 normalization하는 Transformer 구조 |
| Residual Stream | 레지듀얼 스트림 | layer들을 지나며 information이 누적되는 main hidden representation |
| Muon | 뮤온 | matrix orthogonalization 기반 optimizer family |
| Newton–Schulz | 뉴턴-슐츠 | matrix inverse/sign/orthogonalization을 polynomial iteration으로 근사하는 방법 |
| MuonClip | 뮤온클립 | K2의 Muon stability extension |
| Per-Head Muon | 퍼 헤드 뮤온 | K3에서 Q/K/V projection update를 head별로 orthogonalize하는 optimizer 설계 |
| RMS Matching | 알엠에스 매칭 | parameter/update RMS scale을 맞추는 안정화 관점 |
| QK-Clip | 큐케이 클립 | Q/K-related attention scale instability를 제한하는 K2 optimizer technique |
| Native Multimodal | 네이티브 멀티모달 | foundation training 단계부터 여러 modality를 공동 학습 |
| MoonViT | 문빗 | Moonshot의 native-resolution vision transformer family |
| MoonViT-V2 | 문빗 브이투 | K3의 scratch-trained 401M vision encoder |
| Pixel Shuffle | 픽셀 셔플 | spatial visual tokens를 재배열/병합해 token count를 줄이는 기법 |
| Continual Pretraining | 컨티뉴얼 프리트레이닝 | 기존 base checkpoint에서 pretraining objective를 계속 학습 |
| Supervised Fine-Tuning | 슈퍼바이즈드 파인튜닝 | curated instruction/trajectory로 지도 미세조정 |
| Reinforcement Learning | 리인포스먼트 러닝 | rollout reward를 이용해 policy를 개선하는 강화학습 |
| RLVR | 알엘브이알 | 검증 가능한 reward를 사용하는 RL |
| Partial Rollout | 파셜 롤아웃 | trajectory 전체가 아니라 일부 prefix를 재사용해 continuation을 다시 sampling |
| On-Policy | 온 폴리시 | 현재 policy가 실제 방문하는 trajectory distribution에서 학습 |
| Distillation | 디스틸레이션 | teacher behavior/distribution을 student에 전달 |
| Multi-Teacher On-Policy Distillation | 멀티 티처 온폴리시 디스틸레이션 | 여러 specialized teacher signal을 student의 own rollout에서 통합하는 방법 |
| Generative Reward Model | 제너레이티브 리워드 모델 | scalar만 내지 않고 평가 reasoning/verdict를 생성하는 reward model |
| Agent Swarm | 에이전트 스웜 | task를 여러 sub-agent로 동적 분해해 병렬 실행하는 orchestration system |
| Multi-Token Prediction | 멀티 토큰 프레딕션 | 한 위치에서 여러 future token prediction을 학습하는 auxiliary objective |
| Speculative Decoding | 스페큘러티브 디코딩 | cheap drafter 후보를 target model이 병렬 검증해 decode를 가속하는 방법 |
| EAGLE-3 | 이글 쓰리 | target internal features를 사용하는 speculative draft architecture family |
| Quantization-Aware Training | 퀀타이제이션 어웨어 트레이닝 | quantization effect를 training forward에 반영해 최종 low-precision model을 직접 최적화 |
| MXFP4 | 엠엑스 에프피포 | microscaling 4-bit floating-point 계열 format |
| MXFP8 | 엠엑스 에프피에이트 | microscaling 8-bit floating-point 계열 format |
| Context Parallelism | 컨텍스트 패럴렐리즘 | 긴 sequence를 GPU들에 분할해 처리하는 병렬화 |
| Unified Page Pool | 유니파이드 페이지 풀 | KDA state와 MLA KV를 공통 paged allocator abstraction으로 관리하는 serving 설계 |
| Prefix Cache | 프리픽스 캐시 | 동일 prompt prefix의 이미 계산된 state/cache를 재사용하는 기능 |
| Admission Control | 어드미션 컨트롤 | request cost/resource budget에 따라 실행 유입을 제어하는 scheduler 계층 |

---

# 15. `block`이라는 단어 구분

K3에서 매우 자주 혼동한다.

### Attention Hybrid Group

```text
3 KDA + 1 MLA
```

### AttnRes Block

```text
12-layer depth group
```

### Cache Physical Block/Page

```text
token/state memory allocation unit
```

### Prefix Hash Block

```text
prefix equality/cache lookup granularity
```

### KDA Chunk

```text
parallel recurrence kernel의 sequence tile
```

모두 다른 object다.

---

# 16. `state`라는 단어 구분

### Hidden State

현재 token/layer representation.

### KDA Recurrent State

과거 sequence information을 압축한 fixed matrix state.

### Optimizer State

momentum/moment 등 training optimizer memory.

### Agent Environment State

filesystem/process/tool world의 실행 상태.

### Prefix Cache State

해당 prefix boundary에서 resume하기 위한 KDA/MLA stored state.

문맥에 따라 완전히 다르다.

---

# 17. K3 Technical Report를 읽으며 체크할 질문

## Architecture

- 이 mechanism은 sequence/depth/width 중 어느 axis를 바꾸는가?
- K2 대비 어떤 bottleneck 때문에 추가됐는가?
- model quality와 hardware cost를 어떤 knob로 절충하는가?

## Optimization

- instability가 activation, gradient, routing, optimizer 중 어디에서 오는가?
- stabilization이 model equation인가 optimizer rule인가?

## System

- 이 optimization이 HBM, compute, NVLink/IB 중 무엇을 줄이는가?
- prefill과 decode에서 효과가 같은가?
- TP/EP/CP 중 어떤 parallelism과 연결되는가?

## Post-training

- capability가 pretraining에서 나온 것인가 RL에서 강화된 것인가?
- reward가 verifiable한가 judge-based인가?
- runtime agent system과 checkpoint capability를 구분했는가?

---

# 18. 반드시 기억할 12문장

1. K3는 Kimi Linear를 단순 scale-up한 모델이 아니라 여러 Kimi 연구선의 합류점이다.
2. K2는 1T MoE, MuonClip, agentic data/RL foundation을 제공했다.
3. Kimi Linear는 KDA + MLA hybrid sequence architecture를 large-model regime에서 검증했다.
4. KDA는 sequence axis, AttnRes는 depth axis, Stable LatentMoE는 model-width/expert axis를 담당한다.
5. K3 residual hidden size는 K2와 같은 7168이며 scale-up은 depth/expert capacity 중심이다.
6. Stable LatentMoE는 routed expert computation을 3584 latent width에서 수행한다.
7. 896 routed expert weights가 K3 total parameter의 압도적 대부분을 차지한다.
8. AttnRes는 과거 token이 아니라 과거 layer/block representation을 선택한다.
9. MoonViT-V2는 K3 next-token objective 아래 scratch부터 공동 학습된다.
10. Per-Head Muon은 model의 attention-head structure와 optimizer matrix unit을 정렬한다.
11. K3의 1M capability는 KDA architecture, context curriculum, data, CP/cache system이 함께 있어야 성립한다.
12. K3 serving은 KDA state + MLA KV + EP expert bank를 동시에 관리해야 하므로 conventional KV-only dense model과 근본적으로 다르다.

---

# 19. 최종 학습 목표

이 디렉터리를 다 읽은 뒤 K3를 다음처럼 설명할 수 있어야 한다.

> Kimi K3는 7168 residual width를 유지하면서 93-layer depth와 896-expert sparse width로 capacity를 확장한 2.8T MoE다. Sequence axis에서는 KDA fixed recurrent memory와 periodic NoPE Gated MLA를 결합해 1M context 비용을 제어하고, depth axis에서는 Block AttnRes로 과거 block representation을 선택적으로 재사용한다. Routed expert computation은 3584 latent space로 압축해 top-16 routing과 EP communication을 감당하며, RMS normalization·SiTU-GLU·Quantile Balancing으로 extreme-scale LatentMoE를 안정화한다. K2의 MuonClip/agentic foundation, Kimi Linear의 KDA, Kimi-VL/K2.5의 multimodality, k1.5의 RL scaling 연구가 K3에서 통합되고, 실제 deployment에서는 hybrid KDA/MLA cache, expert parallelism, context parallelism, state-aware prefix caching과 speculative state replay까지 model architecture와 공동 설계된다.

이 문장을 각 절의 수식/shape/system cost로 풀어 설명할 수 있으면 K3의 구조를 상당히 깊게 이해한 상태다.
