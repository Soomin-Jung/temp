# 02. MLA, DSA & Lightning Indexer

작성일: 2026-08-25  
기준: current `zai-org/GLM-5.2` config + Hugging Face `glm_moe_dsa` implementation

## 1. 먼저 세 문제를 분리한다

GLM-5.2 attention을 이해할 때 가장 흔한 혼동은 MLA와 DSA를 같은 sparsity mechanism으로 보는 것이다.

| Mechanism | 줄이려는 것 | 핵심 질문 |
|---|---|---|
| MLA | KV representation / cache | 과거 token의 K/V를 얼마나 작게 저장할 수 있는가? |
| DSA | main attention compute | 긴 history 중 실제로 어떤 token만 읽을 것인가? |
| IndexShare | DSA indexer compute | top-k 검색을 매 layer 다시 해야 하는가? |

이 문서는 앞의 두 개를 다룬다.

---

## 2. Current attention geometry

공식 config:

```text
hidden_size          = 6144
num_attention_heads  = 64
q_lora_rank          = 2048
kv_lora_rank         = 512
qk_nope_head_dim     = 192
qk_rope_head_dim     = 64
qk_head_dim          = 256
v_head_dim           = 256

index_n_heads        = 32
index_head_dim       = 128
index_topk           = 2048
```

메인 attention과 indexer는 별도 projection을 갖는다.

```text
                     ┌──────────── Main MLA ────────────┐
hidden 6144 ─────────┤                                   ├─→ sparse attention
                     └──────── DSA Indexer ─────────────┘
                                      │
                                      └─→ top-k positions
```

indexer가 main Q/K를 그대로 쓰지 않는 것이 중요하다. **검색용 representation과 실제 attention representation을 분리**한다.

---

## 3. Query path: 6144 → 2048 latent → 64×256

Transformers current implementation의 `GlmMoeDsaAttention`은 query를 두 단계 projection한다.

```text
hidden h: [..., 6144]
   │
   ▼ q_a_proj
q_resid: [..., 2048]
   │
   ▼ RMSNorm
normalized q latent
   │
   ▼ q_b_proj
64 heads × 256
   │
   ├─ q_pass : 192 dimensions
   └─ q_rot  : 64 dimensions → RoPE
```

수식으로 개념화하면:

```math
c_t^Q = \mathrm{RMSNorm}(W_{DQ}h_t)
```

```math
q_t = W_{UQ}c_t^Q
```

그리고 각 head의 q를:

```math
q_t=[q_t^{noPE};q_t^{RoPE}]
```

로 분리한다.

### 왜 positional/non-positional subspace를 분리하는가

MLA에서는 compressed latent를 효율적으로 cache해야 한다. RoPE가 모든 latent dimension에 얽혀 버리면 위치에 따라 representation이 달라져 low-rank cache 재사용이 어려워진다.

그래서:

- content path: no-PE
- position-sensitive path: small decoupled RoPE component

로 나눈다.

---

## 4. KV path: 6144 → 512 compressed latent + 64 RoPE key

KV path는 더 중요하다.

```text
hidden h: [..., 6144]
   │
   ▼ kv_a_proj_with_mqa
[ kv_latent 512 | k_rot 64 ]
       │              │
       │              └─ RoPE
       ▼ RMSNorm
compressed KV latent
       │
       ▼ kv_b_proj
64 × (K_noPE 192 + V 256)
```

개념적으로:

```math
[c_t^{KV};k_t^{R}] = W_{DKV}h_t
```

`c_t^{KV}`가 512-dimensional compressed latent다.

이후 필요할 때:

```math
[k_t^{noPE};v_t] = W_{UKV}c_t^{KV}
```

로 expand한다.

### KV cache 관점

MLA-native optimized inference의 핵심은 full `64 × (K,V)`를 매 token 저장하는 대신 compressed latent와 positional component를 중심으로 cache하는 것이다.

다만 generic framework 구현은 backend에 따라 expanded K/V를 cache할 수 있다. 현재 Transformers 구현에도 sparse path에서 expanded K/V를 cache하고 추후 개선할 TODO가 존재한다.

따라서 **architecture-level cache requirement와 특정 runtime의 physical cache layout을 구분**해야 한다.

---

## 5. DSA indexer는 무엇을 계산하는가

GLM-5.2 indexer는 main attention보다 훨씬 작은 retrieval network다.

current code 기준 query:

```text
q_resid 2048
   │
   ▼ wq_b
32 index heads × 128
   │
   ├─ 64 RoPE dimensions
   └─ remaining pass-through dimensions
```

index key:

```text
hidden 6144
   │
   ▼ wk
128
   │
 LayerNorm
   │
 RoPE split/merge
```

즉 index key는 main MLA K/V latent와도 별도의 projection이다.

---

## 6. Index score의 실제 forward flow

Transformers implementation을 simplified pseudocode로 옮기면:

```python
q = index_wq(q_resid)            # [B,S,32,128]
k = norm(index_wk(hidden))       # [B,S,128]

q, k = apply_interleaved_rope(q, k)

scores = relu(q @ k.T / sqrt(128))
# [B,S,32,T]

head_weights = weights_proj(hidden) / sqrt(32)
# [B,S,32]

index_scores = weighted_sum_over_32_heads(scores, head_weights)
# [B,S,T]

apply_causal_mask(index_scores)
topk = topk(index_scores, k=2048)
```

여기서 눈여겨볼 세 가지가 있다.

### 6.1 ReLU score

indexer raw similarity에 ReLU를 적용한다. negative similarity는 retrieval importance에서 제거한다.

### 6.2 Per-token head weighting

32 index head의 중요도를 고정 평균하지 않는다.

```text
hidden state → weights_proj → 32 weights
```

를 통해 current token 내용에 따라 retrieval head combination을 바꾼다.

### 6.3 top-k output은 position index

최종 결과는 attention value가 아니라:

```text
[B, query_len, 2048] int32 positions
```

이다.

optimized sparse attention kernel은 이 index를 받아 해당 KV position만 gather/read한다.

---

## 7. DSA가 main attention을 sparse하게 만드는 과정

full-indexer layer에서는:

```text
hidden
  │
  ├─ Main MLA Q/K/V path ─────────────┐
  │                                    │
  └─ Indexer ─→ top-k position list ──┤
                                       ▼
                              Sparse attention kernel
                              selected K/V only
                                       │
                                       ▼
                                    o_proj
```

generic eager/SDPA 구현은 top-k position을 dense additive mask로 바꿀 수 있다.

```text
selected position → 0
other position    → -inf
```

하지만 이것은 correctness/reference path에 가깝다. 실제 long-context performance를 얻으려면 kernel이 sparse index를 직접 소비해야 한다.

즉:

> **DSA architecture 지원과 DSA-optimized kernel 지원은 별개다.**

모델이 load된다는 것만으로 1M context efficiency가 보장되지 않는다.

---

## 8. Complexity를 어떻게 볼 것인가

Dense attention 본체를 단순화하면 token axis에서 대략:

```math
O(L^2 d)
```

DSA sparse attention은 query당 k개만 선택하므로 core attention이:

```math
O(Lkd)
```

형태로 바뀐다.

GLM-5.2에서는:

```text
L = 1,048,576
k = 2,048
```

이므로 `k/L ≈ 0.195%`다.

그러나 indexer가 top-k를 찾으려면 lightweight dimension이라고 해도 history를 score해야 해서 quadratic component가 남는다.

```text
main attention: sparse O(Lk)
indexer:        lightweight but global scan
```

이것이 IndexShare가 필요한 직접적 이유다.

---

## 9. DSA에서 cache가 하나 더 생긴다

standard attention cache만 생각하면 K/V cache만 보면 된다.

DSA에는 indexer용 key history도 필요하다.

Transformers code의 cache abstraction에서도:

```python
past_key_values.update_indexer(k, layer_idx)
```

라는 별도 path가 존재한다.

따라서 long-context memory accounting에는:

- main MLA/KV state
- positional state
- indexer key/cache
- block/sparse metadata

가 포함될 수 있다.

vLLM Ascend의 DCP design이 `indexer cache는 replicated`, `large SFA KV cache는 sharded`로 나누는 것도 이 두 cache의 성격 차이 때문이다.

---

## 10. RoPE: interleaved layout

GLM-MoE-DSA implementation은 indexer와 main attention에서 interleaved RoPE를 사용한다.

pair layout:

```text
(x0,x1), (x2,x3), ...
```

각 pair를 하나의 angle로 회전한다.

현재 Transformers 코드 주석은 DeepSeek-V3.2의 half-split RoPE와 GLM-MoE-DSA의 interleaved layout 차이를 명시한다.

이것은 checkpoint compatibility와 custom kernel 구현에서 중요하다. Q/K tensor shape만 맞춘다고 동일 모델이 되는 것이 아니라 **rotary dimension layout까지 맞아야 한다.**

---

## 11. Serving engineer 관점에서 확인할 것

GLM-5.2 engine 검증 시 최소한 다음을 분리해서 측정한다.

1. DSA sparse kernel이 실제 enable됐는가?
2. indexer가 FP8/optimized GEMM을 사용하는가?
3. top-k selection이 CPU fallback하지 않는가?
4. MLA cache가 compressed/native layout인가?
5. context가 길어질수록 indexer 비중이 얼마나 커지는가?
6. IndexShare가 실제 적용돼 indexer launch 수가 감소하는가?
7. prefix cache / context parallel과 indexer cache semantics가 맞는가?
8. CUDA graph capture 범위에 sparse-index shape variation이 어떤 영향을 주는가?

`vLLM serve가 뜬다`는 검증의 시작일 뿐이다.

---

## 12. 한 문장 정리

> **MLA는 과거를 작게 저장하고, DSA는 그 과거 중 필요한 2048개 위치만 찾아 읽는다. GLM-5.2 long-context efficiency를 이해하려면 “압축”과 “검색”을 반드시 별개 문제로 봐야 한다.**

## Sources

- GLM-5 technical report: https://arxiv.org/abs/2602.15763
- GLM-5.2 config: https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
- Transformers implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
- DeepSeek-V3.2 / DSA lineage: referenced from GLM-5 technical report
