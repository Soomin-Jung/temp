# 02. Hybrid Attention — SWA, CSA, HCA, DSA

작성일: 2026-08-25

## 1. V4 attention의 출발 질문

1M context에서 모든 layer가 모든 과거 token의 KV를 그대로 저장하고 읽는 것은 너무 비싸다.

DeepSeek-V4의 답은 단일 sparse attention이 아니라 **multi-resolution memory hierarchy**다.

```text
Recent exact memory    → SWA
Medium compressed memory → CSA
Very coarse global memory → HCA
```

그리고 CSA 내부에서 `무엇을 읽을지` 선택하는 mechanism으로 DSA/Lightning Indexer를 사용한다.

---

## 2. 공통 backbone — Shared K=V MQA

V4의 attention은 `num_key_value_heads=1`이다.

즉 many-query-head 구조이지만 KV는 shared single head다.

또한 구현상 K와 V를 별도 projection 두 개로 만들기보다 **shared K=V memory representation**을 사용한다.

핵심 목적:

- KV cache dimension 축소
- long context에서 memory footprint 절감
- compression branch를 단순하게 유지

Q는 여전히 많은 head를 가지므로 query expressivity는 유지한다.

---

## 3. Sliding Window Attention — local fidelity path

모든 compressed attention path만 사용하면 최근 token의 미세한 정보가 compression boundary에서 손실될 수 있다.

그래서 V4는 local exact branch를 유지한다.

```text
window_size = 128
```

query token은 최근 128 raw token을 직접 볼 수 있다.

이 local branch의 역할:

- immediate syntax
- local variable/reference
- recent tool output
- code editing neighborhood
- compression artifact 보완

즉 V4는 long memory를 압축하면서도 **최근 정보는 압축하지 않는다.**

---

## 4. CSA — Compressed Sparse Attention

CSA는 두 단계다.

### 4.1 Compression

Flash 기준 compression ratio `m=4`.

vLLM 설명에 따르면 c4a compressor는 단순 4-token 평균이 아니라 overlapping compression을 사용한다.

개념적으로:

```text
raw KV sequence
  ↓ compression
~1/4 length memory entries
```

1M token이면 대략 250K compressed entries 규모가 남는다.

이것만으로는 attention compute가 여전히 크다.

### 4.2 Sparse retrieval

그래서 Lightning Indexer가 compressed entries를 score하고 top-k만 선택한다.

Flash:

```text
index_topk = 512
```

Pro:

```text
index_topk = 1024
```

따라서 CSA의 본질:

```text
raw 1M history
   ↓
compressed ~250K memory
   ↓
learned indexer
   ↓
top-512 / top-1024
   ↓
main attention
```

이다.

---

## 5. Lightning Indexer와 DSA

Lightning Indexer는 main attention보다 가벼운 retrieval network다.

Flash config:

```text
index_n_heads  = 64
index_head_dim = 128
index_topk     = 512
```

Pro:

```text
index_n_heads  = 64
index_head_dim = 128
index_topk     = 1024
```

Indexer가 해결하는 질문은:

> 현재 query에 대해 compressed long-memory 중 어떤 entry가 실제 attention 계산 가치가 있는가?

이다.

여기서 중요한 점:

> **DSA는 KV memory를 압축하는 mechanism이 아니다. selection mechanism이다.**

Compression과 sparsification을 반드시 분리해서 이해해야 한다.

---

## 6. HCA — Heavily Compressed Attention

HCA는 retrieval 없이 훨씬 강한 compression을 사용한다.

```text
m' = 128
```

즉 약 128 raw tokens를 하나의 coarse global memory entry로 만든다.

1M token이면 약:

```text
1,048,576 / 128 ≈ 8,192 entries
```

정도다.

8K entry는 전체를 attention해도 CSA의 250K 후보보다 훨씬 작다.

따라서 HCA는 Lightning Indexer 없이 모든 available compressed memory를 사용할 수 있다.

---

## 7. 왜 CSA와 HCA 둘 다 필요한가

### CSA

장점:

- compression이 덜 강함
- 세부 정보 보존력 높음
- retrieval로 compute 제한

단점:

- indexer 필요
- candidate pool이 큼
- index cache 추가

### HCA

장점:

- memory가 매우 작음
- global coverage가 cheap
- retrieval indexer 불필요

단점:

- compression이 강함
- fine-grained detail 손실 가능

그래서 V4는 interleave한다.

```text
local exact      → SWA
fine long memory → CSA
coarse global    → HCA
```

이건 Transformer memory를 사실상 multi-level cache처럼 만드는 설계다.

---

## 8. Layer schedule

Flash-0731 config의 `compress_ratios`는 대략:

```text
0, 0,
4,128,4,128, ...,
4,
0,0,0
```

으로 되어 있다.

해석:

- bootstrap/local-only layer
- CSA/HCA alternating core
- 후반 local-only refinement

즉 모든 layer가 같은 memory representation을 읽지 않는다.

---

## 9. Partial RoPE

V4 head dim은 512지만 RoPE가 적용되는 부분은 64 dimension이다.

```text
head_dim         = 512
qk_rope_head_dim = 64
```

즉:

```text
448 dims → content/no-pos path
 64 dims → rotary positional path
```

compressed branch에는 별도의 `compress_rope_theta=160000`도 사용한다.

장거리 compression representation과 local raw representation의 positional geometry를 구분해서 다룬다.

---

## 10. Grouped Output Projection

V4 attention head output은 매우 넓다.

Flash:

```text
64 heads × 512 = 32768
```

이를 곧바로 hidden_size 4096으로 dense projection하면 비용이 크다.

그래서 head를 group으로 나누고 low-rank intermediate를 거친다.

Flash:

```text
o_groups    = 8
o_lora_rank = 1024
```

Pro:

```text
o_groups    = 16
o_lora_rank = 1024
```

이것은 KV compression과 별개의 **attention output projection cost reduction**이다.

---

## 11. Attention Sink

V4는 per-head learnable attention sink를 둔다.

attention sink는 long-context / streaming attention에서 특정 neutral/default destination 역할을 줄 수 있다.

모든 probability mass를 실제 content token에 강제로 배분하지 않고 head가 안정적으로 residual attention mass를 처리할 수 있게 하는 장치로 볼 수 있다.

---

## 12. 1M KV cache에서 실제 의미

vLLM 분석은 V4의 bf16 KV state를 1M sequence에서 약 9.62 GiB 수준으로 추정하며, V3.2-style 61-layer stack의 약 83.9 GiB보다 크게 작다고 설명한다.

production에서는:

- attention cache FP8
- indexer cache FP4

를 활용해 더 줄인다.

즉 V4의 1M은 단순 `max_position_embeddings=1048576` 숫자가 아니다.

```text
MQA
+ local window
+ 4x compression
+ 128x compression
+ sparse top-k
+ low precision cache
```

가 동시에 맞물려야 실제 serving 가능한 context가 된다.

---

## 13. GLM-5.2와의 차이를 다시 정리

### GLM-5.2

```text
MLA compressed latent
→ DSA sparse token selection
→ IndexShare
```

### DeepSeek-V4

```text
raw/local KV
+ c4 compressed memory → DSA retrieval
+ c128 global memory
```

V4는 **multi-resolution memory hierarchy**, GLM-5.2는 **compressed latent + sparse access + cross-layer index reuse**에 더 가깝다.

---

## 14. 출처

- DeepSeek-V4 Technical Report §2.3: https://arxiv.org/abs/2606.19348
- HF Transformers architecture docs: https://huggingface.co/docs/transformers/en/model_doc/deepseek_v4
- vLLM implementation note: https://vllm-project.github.io/2026/04/24/deepseek-v4.html
