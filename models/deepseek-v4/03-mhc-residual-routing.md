# 03. mHC — Manifold-Constrained Hyper-Connections

작성일: 2026-08-25

## 1. mHC는 무엇을 해결하는가

mHC는 attention mechanism이 아니다.

해결하려는 축은 **network depth / residual information routing**이다.

기존 Transformer는 대략:

```text
x_{l+1} = x_l + F_l(x_l)
```

처럼 하나의 residual stream을 이어간다.

DeepSeek-V4는 이 residual stream을 4-copy로 확장한다.

```text
X_l ∈ R^(4 × d)
```

Flash라면 d=4096이므로 residual state는 개념적으로:

```text
4 × 4096
```

의 parallel streams를 갖는다.

---

## 2. Hyper-Connections의 기본 아이디어

일반 HC는 sublayer를 통과할 때 세 종류의 mapping을 둔다.

```text
A_l : 여러 residual streams → 현재 layer input 하나
B_l : 기존 residual streams끼리 mixing
C_l : layer output → residual streams로 재주입
```

개념 수식:

```math
X_{l+1} = B_l X_l + C_l F_l(A_l X_l)
```

이렇게 하면 단일 residual sum보다 layer마다 과거 representation을 더 유연하게 재조합할 수 있다.

하지만 자유로운 B_l은 depth가 깊어질수록 signal norm을 증폭하거나 불안정하게 만들 수 있다.

---

## 3. mHC의 핵심 — residual mapping을 manifold에 제한

DeepSeek-V4는 residual mixing matrix가 arbitrary matrix가 되지 않게 한다.

`Sinkhorn-Knopp` iteration을 사용해 mixing matrix를 **doubly stochastic** 성질에 가까운 manifold로 projection한다.

직관적으로 각 row/column의 mass가 통제된다.

```text
arbitrary mixing
     ↓
Sinkhorn normalization × 20
     ↓
controlled residual mixing
```

Flash/Pro 공통:

```text
hc_mult           = 4
hc_sinkhorn_iters = 20
hc_eps            = 1e-6
```

---

## 4. 왜 doubly-stochastic constraint가 유용한가

residual mixing matrix가 unconstrained라면 반복 multiplication에 의해 특정 direction의 signal이 크게 증폭되거나 사라질 수 있다.

mHC는 각 layer의 residual transport를 제한해:

- extreme amplification 방지
- 특정 stream으로 mass collapse 방지
- deep network에서 stable propagation
- 여러 residual paths의 expressivity 유지

를 동시에 노린다.

즉:

> `더 많은 residual path`를 만들되 `그 path mixing이 폭주하지 못하게 geometry를 제한`한다.

---

## 5. 실제 block에서는 두 번 사용된다

Transformer block에는 보통:

1. attention
2. FFN/MoE

두 sublayer가 있다.

V4에서는 각각 앞뒤로 HyperConnection mapping이 존재한다.

```text
X_l [4 residual streams]
  ↓
attn HC pre-mix
  ↓
RMSNorm
  ↓
Attention
  ↓
attn HC post-mix
  ↓
ffn HC pre-mix
  ↓
RMSNorm
  ↓
MoE
  ↓
ffn HC post-mix
  ↓
X_{l+1}
```

따라서 mHC cost는 모델 전체에 반복된다.

---

## 6. 모델 끝에서는 다시 하나로 collapse한다

중간 block들은 `[B,S,4,D]` 형태의 residual state를 유지하지만 최종 LM head는 하나의 hidden sequence를 필요로 한다.

그래서 마지막에 `HyperHead`가 residual streams를 다시 하나로 합친다.

```text
4 residual streams
       ↓
HyperHead
       ↓
1 hidden stream
       ↓
RMSNorm
       ↓
LM head
```

---

## 7. Kimi K3 Attention Residuals와 비교

둘 다 depth-axis information routing 문제를 다루지만 방식은 다르다.

### Kimi K3 Block AttnRes

과거 block representation을 source로 두고 content-dependent attention으로 다시 읽는다.

```text
depth memories
→ attention over depth
```

### DeepSeek-V4 mHC

동일 depth 위치에서 residual stream 자체를 multiple copies로 확장하고 constrained linear mixing을 반복한다.

```text
parallel residual streams
→ constrained mixing
```

즉:

- AttnRes: **과거 depth를 explicit memory처럼 선택**
- mHC: **현재 residual transport topology 자체를 확장**

이다.

---

## 8. Serving engineer에게 왜 중요한가

mHC는 training-only trick이 아니다. inference forward에도 존재한다.

영향:

- hidden-state memory traffic 증가
- residual representation shape 변화
- fusion opportunities
- CUDA graph capture shape 고정성
- custom mHC fused kernel 필요성

V4 technical report가 `single fused kernel`, TileLang, deterministic kernels를 강조하는 이유 중 하나도 이런 구조적 overhead를 시스템 수준에서 흡수해야 하기 때문이다.

---

## 9. naive memory 계산에 주의

`hc_mult=4`라고 해서 모델 전체 activation memory가 정확히 4배라고 단순 계산하면 안 된다.

이유:

- 각 sublayer actual F(x)는 hidden_size d 하나에 대해 계산됨
- residual stream만 expanded
- fused implementation이 intermediate materialization을 줄일 수 있음
- decode에서는 sequence dimension이 작음

하지만 prefill/training에서는 residual stream bandwidth가 무시할 수 없는 비용이 된다.

---

## 10. 본질

mHC의 핵심은 다음 한 문장으로 볼 수 있다.

> **DeepSeek-V4는 residual connection을 단순한 덧셈 경로가 아니라, 여러 parallel representation을 안정적으로 운반하는 constrained routing network로 바꿨다.**

---

## 11. 출처

- DeepSeek-V4 Technical Report §2.2: https://arxiv.org/abs/2606.19348
- HF DeepSeek-V4 docs — Manifold-Constrained Hyper-Connections: https://huggingface.co/docs/transformers/en/model_doc/deepseek_v4
