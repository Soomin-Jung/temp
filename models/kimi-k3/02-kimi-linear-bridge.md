# 02. Kimi Linear: KDA Hybrid에서 K3로 가는 다리

작성일: 2026-08-19  
상위 문서: [Kimi K3 Study Guide](README.md)

## 1. Kimi Linear의 정확한 위치

Kimi Linear는 Kimi K3 이전의 `작은 K3`가 아니다. 공식 공개 모델은 **48B total / 3B activated** MoE 연구 모델이며, 목적은 long-context architecture에 대한 한 가지 중요한 질문을 실험하는 것이었다.

> **Full Softmax Attention을 대부분의 layer에서 fixed-state linear recurrence로 바꾸면서도 short/long-context quality와 RL scaling을 유지할 수 있는가?**

Kimi Linear가 검증한 핵심 설계는:

- Kimi Delta Attention(KDA)
- 3 KDA : 1 global MLA hybrid
- finite recurrent memory의 fine-grained gating
- long-context용 chunkwise KDA kernel

이다.

K3는 이 sequence-mixing 연구 결과를 2.8T-scale foundation에 채택했지만, MoE·depth·vision·optimizer·post-training은 다른 연구 계보와 결합한다.

---

## 2. Kimi Linear 공개 모델

공식 repository 기준:

| 항목 | Kimi Linear |
|---|---:|
| Total parameters | 48B |
| Activated parameters | 3B |
| Context | 1M |
| Core attention | KDA |
| Hybrid ratio | 3 KDA : 1 global MLA |
| 공개 checkpoint | Base / Instruct |
| 공개 training 규모 | checkpoint 기준 5.7T tokens |

공식 report의 fair architecture comparison은 별도 smaller-scale controlled training 조건에서도 full attention/다른 linear mechanisms와 비교한다.

### 공식 architecture 그림

![Kimi Linear architecture](https://raw.githubusercontent.com/MoonshotAI/Kimi-Linear/master/figures/arch.png)

*Source: MoonshotAI/Kimi-Linear official repository.*

### 공식 long-context 성능/속도 그림

![Kimi Linear performance and speed](https://raw.githubusercontent.com/MoonshotAI/Kimi-Linear/master/figures/perf_speed.png)

*Source: MoonshotAI/Kimi-Linear official repository. 논문의 특정 benchmark/hardware 조건에서 측정된 결과이므로 일반적인 절대 배속으로 해석하지 않는다.*

---

## 3. Kimi Linear 이전에 풀어야 했던 문제

Full attention에서 sequence length $T$가 길어지면:

- token별 KV cache가 $O(T)$로 증가하고
- global prefill attention은 $O(T^2)$ 관계를 가지며
- decode query는 과거 cache를 계속 읽는다.

MLA는 token당 KV representation을 줄이지만 token entry 수 $T$는 유지한다.

Kimi Linear의 질문은 더 급진적이다.

```text
Full / MLA
Token 1 → memory entry
Token 2 → memory entry
...
Token T → memory entry

KDA
Token 1 ┐
Token 2 ├──→ fixed recurrent state S_t
...     │
Token T ┘
```

즉 history 자체를 finite state로 접는다.

KDA 수학은 별도 문서:

- [Linear/Recurrent Attention — DeltaNet, GDN, KDA](../../study/attention/03-linear-recurrent-attention.md)

에서 상세히 다룬다.

---

## 4. 왜 Gated DeltaNet만으로 끝나지 않았는가

**Gated DeltaNet(게이티드 델타넷; GDN)**은 Delta Rule 기반 associative memory에 forgetting gate를 결합한다.

Kimi Linear의 문제의식은 finite state의 memory allocation을 더 세밀하게 제어하는 것이다.

단순 scalar/head-level retention 대신 KDA는 state/key channel마다 다른 retention을 학습한다.

```text
state channel 0 → 오래 유지
state channel 1 → 중간 유지
state channel 2 → 빠르게 교체
...
```

즉 한 head 내부에서도 서로 다른 memory timescale을 만들 수 있다.

이 fine-grained gating이 KDA의 핵심 architecture contribution이다.

---

## 5. KDA의 장점과 근본적인 대가

### 장점

- context length가 늘어도 recurrent state shape 고정
- decode history-length dependence를 크게 낮출 수 있음
- long context에서 KV cache/HBM read 감소
- chunkwise formulation으로 prefill을 matrix operation에 매핑 가능

### 대가

1M token의 raw memory를 고정 크기 state 하나에 압축한다.

따라서 finite-state memory는 원칙적으로 **lossy compression(로시 컴프레션, 손실 압축)**이다.

특히 어려울 수 있는 것:

- exact string copy
- 특정 먼 token span 재조회
- 다수의 유사 key/value association 유지
- long repository에서 precise symbol retrieval

이 문제 때문에 Kimi Linear는 pure KDA가 아니라 **KDA + global MLA hybrid**를 선택한다.

---

## 6. 3:1 Hybrid의 의미

기본 패턴:

```text
KDA
KDA
KDA
MLA
→ repeat
```

각 mechanism 역할:

### KDA

- cheap recurrent memory
- sequence order/recency-sensitive state
- 대부분의 token mixing
- long-context decode 효율

### MLA

- token-wise latent cache
- global softmax lookup
- exact/content-addressable historical memory
- recurrent compression의 정보 손실 보완

즉 architecture 철학은:

> **대부분의 layer는 compressed finite-state memory로 문맥을 전달하고, 일정 간격마다 token-addressable global memory로 다시 교정한다.**

K3에서도 이 철학이 유지된다.

---

## 7. 왜 MLA인가, 일반 GQA Full Attention이 아닌가

Global correction layer도 1M context에서는 cache가 비싸다.

Kimi Linear는 global layer에 MLA를 사용해:

1. global token addressability는 유지하고
2. token당 KV cache representation은 latent rank로 줄인다.

즉 hybrid 구조 자체가 두 단계 compression이다.

```text
Layer axis:
Global attention을 1/4 수준으로 줄임

Global layer 내부:
MLA로 token당 KV representation까지 압축
```

K3가 Qwen-style `GDN + full GQA`가 아니라 `KDA + MLA`를 사용하는 차이 중 하나다.

---

## 8. Kimi Linear의 중요한 과학적 의미

Kimi Linear를 단순 speed paper로 읽으면 놓치는 점이 있다.

### 8.1 Linear Attention을 작은 모델 benchmark에만 두지 않았다

48B total / 3B active MoE를 실제 pretrain해 linear attention을 large-model regime에서 검증했다.

### 8.2 Short Context도 같이 비교했다

Efficient attention은 long context에서는 유리해도 short-context quality/latency에서 손해가 날 수 있다.

Kimi Linear는 short context, long context, RL-style scaling을 함께 비교하며 architecture를 평가한다.

### 8.3 Hybrid가 핵심 결론이다

논문의 메시지는 `KDA가 full attention을 완전히 대체한다`가 아니다.

오히려:

> finite-state KDA와 global MLA의 조합이 practical Pareto point를 만든다.

는 쪽이 중요하다.

---

## 9. Algorithm과 Architecture를 같이 봐야 한다

Recurrent KDA 수식은 token 순서대로 state가 바뀌므로 naïve implementation에서는 GPU가 sequence dimension을 병렬화하기 어렵다.

Kimi Linear는 이를 위한 bespoke chunkwise algorithm을 함께 설계한다.

개념적으로:

```text
sequence
[chunk 0][chunk 1][chunk 2]...

within chunk:
  matrix/tensor operations로 병렬화

between chunks:
  compact recurrent state 전달
```

이 연구가 후속 FlashKDA로 더 발전한다.

따라서 Kimi Linear의 의미는:

```text
KDA equation만 발명
```

이 아니라:

```text
KDA model rule
+
parallel/chunkwise algorithm
+
long-context benchmark
+
large MoE checkpoint
```

을 함께 검증한 것이다.

---

## 10. Kimi Linear에서 K3로 그대로 이어지는 것

### 10.1 Hybrid Philosophy

K3도 대부분 KDA + periodic MLA를 사용한다.

### 10.2 KDA State Semantics

Delta-memory + fine-grained retention이라는 핵심은 유지된다.

### 10.3 1M Context 목표

KDA가 long-context cache/decode 비용을 줄이는 핵심 mechanism이라는 방향이 유지된다.

### 10.4 Hardware Co-design

K3에서는 이 부분이 훨씬 강해져 FlashKDA의 precision/tile 조건이 model parameterization에도 영향을 준다.

---

## 11. K3에서 달라진 KDA/MLA

K3는 Kimi Linear architecture를 복사하지 않는다.

### 11.1 Lower-Bounded KDA Decay

K3 config에는:

```text
gate_lower_bound = -5.0
```

이 존재한다.

Decay range를 제한해 BF16 chunkwise FlashKDA의 numerical stability와 hardware tile 설계를 맞춘다.

이것은 **algorithm-system co-design(알고리즘-시스템 공동설계)**의 대표 사례다.

### 11.2 Full-Rank Output Gating

K3의 KDA와 MLA output path는 input-dependent output gate를 사용한다.

Memory에서 읽은 정보를 residual stream으로 얼마나 보낼지를 별도로 제어한다.

### 11.3 NoPE Gated MLA

K3 config:

```text
mla_use_nope = true
mla_use_output_gate = true
```

KDA recurrent path가 sequence-order 정보를 계속 공급하고 MLA는 content-based global lookup에 집중하는 역할 분담이다.

### 11.4 Scale

Kimi Linear 48B/A3B와 K3 2.8T/A104B 사이에는 architecture 밖의 문제가 폭발적으로 커진다.

- expert bank
- EP communication
- depth
- multimodality
- optimizer
- cache manager
- post-training

따라서 K3에서는 Stable LatentMoE와 AttnRes가 함께 필요하다.

---

## 12. Kimi Linear와 K3의 구조 비교

| 항목 | Kimi Linear | Kimi K3 |
|---|---|---|
| 목적 | KDA hybrid architecture 검증 | frontier-scale unified model |
| Total/Active | 48B / 3B | 2.8T / 104B |
| Context | 1M | 1M |
| Sequence mixer | KDA + MLA | KDA + Gated MLA |
| Hybrid pattern | 3:1 | 69 KDA + 24 MLA, 약 3:1 |
| KDA decay | 초기 KDA formulation | lower-bounded decay |
| Depth routing | 기본 residual/연구 단계 | Block AttnRes |
| MoE | smaller research MoE | Stable LatentMoE, 896/top16 |
| Vision | 없음 | MoonViT-V2 |
| Optimizer/system | research-scale | Per-Head Muon + large-scale distributed system |
| Serving | KDA/FLA deployment | KDA+MLA hybrid cache + state-aware prefix/P-D |

---

## 13. Kimi Linear 이후 별도 연구선: Attention Residuals

AttnRes 논문은 Kimi Linear 규모의 model에 depth-mixing mechanism을 적용해 실험했다.

이는 연구 순서상 중요하다.

```text
Kimi Linear
  ↓ sequence mixer를 효율화

Attention Residuals
  ↓ depth 정보 흐름을 개선

Kimi K3
  ↓ KDA hybrid + AttnRes를 큰 foundation에서 통합
```

즉 Kimi Linear는 K3의 sequence architecture prototype 역할을 했고 AttnRes 연구는 depth architecture prototype 역할을 했다.

---

## 14. Kimi Linear 결과를 읽을 때 주의할 점

공식 repository는 특정 조건에서:

- RULER long-context 성능
- MLA 대비 long-sequence TPOT speedup
- KV cache reduction

등을 보고한다.

이 숫자를 production vLLM 성능으로 그대로 가져오면 안 된다.

실제 배포 성능은:

- GPU generation
- dtype
- TP/EP
- batch/concurrency
- kernel backend
- context distribution
- CUDA Graph
- scheduler

에 따라 달라진다.

논문의 핵심은 절대 배속보다 **context가 길어질수록 KDA hybrid가 더 유리해지는 scaling trend**다.

---

## 15. Kimi Linear를 읽은 뒤 K3 report에서 바로 보이는 것

K3 report의 다음 표현들이 더 이상 낯설지 않아야 한다.

- `KDA layer`
- `Gated MLA layer`
- `KDA recurrent state`
- `chunkwise KDA`
- `lower-bounded decay`
- `FlashKDA`
- `hybrid cache`
- `state-aware prefix caching`

그 다음 K3에서 새로 공부해야 하는 것은:

- Stable LatentMoE
- Block AttnRes
- MoonViT-V2
- Per-Head Muon
- large-scale RL/post-training

이다.

---

## 16. 이 장을 읽고 답할 수 있어야 하는 질문

1. Kimi Linear는 K3의 작은 버전이 아니라 어떤 hypothesis를 검증한 연구 모델인가?
2. KDA가 cache를 크게 줄일 수 있는 근본 이유는 무엇인가?
3. Pure KDA가 아니라 3:1 KDA/MLA hybrid가 필요한 이유는 무엇인가?
4. MLA를 global correction path로 선택하면 일반 full GQA 대비 어떤 추가 cache 이점이 있는가?
5. Recurrent equation만 있어서는 GPU에서 빠르지 않은 이유는 무엇인가?
6. Chunkwise KDA가 training/prefill 병렬성을 어떻게 회복하는가?
7. K3의 lower-bounded decay가 model quality뿐 아니라 BF16/GPU kernel 제약과 연결되는 이유는 무엇인가?
8. Kimi Linear가 해결한 sequence problem과 AttnRes가 해결한 depth problem을 구분할 수 있는가?

---

## 17. 원문

- Kimi Linear official repository  
  https://github.com/MoonshotAI/Kimi-Linear
- Kimi Linear report  
  https://arxiv.org/abs/2510.26692
- Kimi K3 official repository/report  
  https://github.com/MoonshotAI/Kimi-K3
- Gated Delta Networks  
  https://arxiv.org/abs/2412.06464
