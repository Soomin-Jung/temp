# 03. IndexShare, IndexCache & 1M Context

작성일: 2026-08-25  
핵심 원문:
- GLM-5.2 official blog: https://z.ai/blog/glm-5.2
- IndexCache paper: https://arxiv.org/abs/2603.12201

## 1. DSA를 넣었는데 왜 또 최적화가 필요한가

DSA의 main attention은 query마다 전체 history 대신 top-k token만 읽는다.

```text
Dense attention
query ───────────────→ all L tokens

DSA
query → indexer → top-k positions → k tokens only
```

하지만 top-k를 찾는 indexer 자체는 history 전체를 score해야 한다.

따라서 1M context에서 비용은 대략 두 부분으로 갈린다.

```text
A. sparse attention core
   L queries × k selected keys

B. sparse indexer
   L queries × long-history candidate scoring
```

`A`를 줄였더니 `B`의 상대 비중이 커지는 구조다.

이것이 IndexCache/IndexShare의 출발점이다.

---

## 2. IndexCache의 핵심 관찰: layer가 달라도 중요한 token은 비슷하다

IndexCache 논문은 DSA의 adjacent layer 사이에서 selected top-k set의 overlap이 매우 높다는 점을 관찰한다.

논문에서는 상황에 따라 인접 layer top-k overlap이 약 70~100%까지 나타남을 보고한다.

Transformer layer가 하나 바뀌었다고 query representation이 완전히 다른 semantic retrieval target을 요구하는 것이 아니라면, 직전 layer가 고른 중요한 history token 상당수가 다음 layer에서도 중요할 수 있다.

따라서:

```text
매 layer마다
full-history indexing
       ↓
중복 계산
```

을:

```text
Full layer    : index compute
Shared layer  : reuse
Shared layer  : reuse
Shared layer  : reuse
```

로 바꿀 수 있다.

---

## 3. IndexCache와 IndexShare의 관계

용어를 구분한다.

### IndexCache

cross-layer sparse-attention index reuse를 체계화한 연구 아이디어/기법이다.

- Full layer(F): 자체 indexer 계산
- Shared layer(S): 이전 Full layer index 재사용
- multi-layer distillation로 shared group에 적합한 indexer 학습

### IndexShare

GLM-5.2 공식 architecture에서 이 아이디어를 실제 모델 training과 serving에 적용한 형태다.

Z.AI의 설명:

> every 4 transformer layers share a lightweight indexer

즉 한 번 계산한 top-k를 네 layer가 공유하도록 학습한다.

문서에서는 다음처럼 해석한다.

```text
IndexCache = general cross-layer index-reuse research
IndexShare = GLM-5.2에서 채택한 trained architecture/application
```

---

## 4. Current config의 실제 pattern

GLM-5.2 `config.json`에는 `indexer_types`가 layer별로 박혀 있다.

핵심 parameter:

```text
index_topk        = 2048
index_topk_freq   = 4
index_skip_topk_offset = 3
```

그리고 `indexer_types`는 초기 bootstrap 예외 뒤에 사실상 FSSS cadence를 형성한다.

개념적으로:

```text
L0  F
L1  F
L2  F  ─────┐
L3  S       │ reuse L2 top-k
L4  S       │
L5  S  ─────┘
L6  F  ─────┐
L7  S       │
L8  S       │
L9  S  ─────┘
L10 F ...
```

즉 official blog의 “4-layer sharing”을 실제 config에서는 **초기 layer의 별도 Full 처리 + 이후 1 Full / 3 Shared 반복**으로 구현한 것으로 볼 수 있다.

이 차이를 알고 있어야 model diagram과 runtime trace가 어긋나지 않는다.

---

## 5. 실제 source code에서 어떻게 전달되는가

Transformers의 `GlmMoeDsaModel.forward()`에는 매우 직접적인 state가 있다.

```python
topk_indices = None

for decoder_layer in self.layers:
    hidden_states, topk_indices = decoder_layer(
        ...,
        prev_topk_indices=topk_indices,
    )
```

각 attention layer는:

```python
self.skip_topk = config.indexer_types[layer_idx] == "shared"
self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(...)
```

로 구성된다.

### Full layer

```text
indexer != None
 → current hidden으로 top-k 계산
 → topk_indices return
```

### Shared layer

```text
indexer == None
 → prev_topk_indices 사용
 → 같은 indices를 다음 layer로 계속 전달
```

즉 IndexShare는 추상적인 논문 개념이 아니라 **forward loop에서 layer-to-layer로 int32 position tensor가 전달되는 stateful dataflow**다.

---

## 6. 왜 1M에서 효과가 커지는가

indexer가 candidate history 전체를 보는 비용은 sequence가 길수록 커진다.

context가 작을 때:

```text
attention GEMM / MoE / other compute
>>> indexer overhead
```

일 수 있다.

하지만 1M에 가까워지면:

```text
long-history index dot product
+ top-k selection
+ cache access
```

이 큰 비중을 차지한다.

GLM-5.2 official blog는 1M context에서 IndexShare 포함 architecture가 GLM-5.1 대비 **per-token FLOPs 2.9× reduction**을 제공한다고 설명한다.

이 숫자는 단순히 `3/4 indexer를 제거했으니 4×`라는 뜻이 아니다. 전체 model FLOPs에는:

- MoE expert GEMM
- MLA projections
- sparse attention core
- shared expert
- normalization
- output projection
- MTP

등이 포함되므로 end-to-end per-token reduction은 별도다.

---

## 7. IndexShare는 왜 inference-only cache trick이 아닌가

아무 layer의 top-k를 다음 세 layer에 강제로 reuse하면 quality가 떨어질 수 있다.

왜냐하면 layer마다 representation이 변하고 실제 attention distribution도 달라지기 때문이다.

GLM-5.2는 공식적으로 **128K mid-training부터 IndexShare를 포함해 training**한다.

IndexCache 연구도 training-aware multi-layer distillation을 제안한다.

핵심 아이디어:

```text
한 Full indexer가
자기 layer만 잘 맞추는 것이 아니라
자기가 serve할 여러 layer의 attention target을
함께 대표하도록 학습
```

즉 shared index는 우연한 cache hit가 아니라 **cross-layer retrieval representation을 학습한 결과**다.

---

## 8. top-k=2048은 얼마나 sparse한가

1M maximum position 기준:

```math
\text{sparse ratio} = \frac{2048}{1,048,576}
\approx 0.001953 = 0.1953\%
```

반대로 말하면 query 하나는 potential history의 약 **99.8%를 main attention에서 읽지 않는다.**

이 정도 sparsity가 가능한 이유는 모든 정보가 동일하게 필요한 것이 아니기 때문이다.

agent/coding trajectory에는:

- 현재 수정 중인 파일
- 최근 test failure
- 관련 symbol definition
- 직전 tool result
- 오래전 핵심 requirement

처럼 query relevance가 강하게 치우치는 경우가 많다.

DSA는 dense global averaging 대신 **content-based retrieval + exact attention** 형태로 볼 수 있다.

다만 top-k=2048은 fixed semantic memory budget과 유사하게 작동하므로, indexer quality가 곧 long-context recall quality가 된다.

---

## 9. IndexShare의 failure mode

### 9.1 Rapid representation shift

인접 layer 사이 attention target이 크게 바뀌는 구간에서는 stale index가 손해가 될 수 있다.

### 9.2 Rare but critical token omission

1M history에서 한 번만 등장한 중요한 token이 top-k에서 빠지면 main attention은 그것을 볼 수 없다.

### 9.3 Shared index error propagation

Full indexer가 잘못 선택하면 그 결과가 여러 Shared layer에 반복 사용된다.

### 9.4 Kernel/layout mismatch

architecture가 sparse여도 backend가 selected positions를 효율적으로 gather하지 못하면 theoretical FLOP reduction이 실제 latency로 이어지지 않는다.

### 9.5 Context parallelism interaction

sequence가 여러 rank에 shard되어 있을 때 top-k를 global하게 일관되게 고르려면 indexer가 전체 sequence visibility를 가져야 한다.

예를 들어 vLLM Ascend의 GLM-5.2 DCP 설계는:

- indexer cache: replicated
- large sparse-attention KV cache: sharded

로 나눈다.

이것은 IndexShare/DSA가 distributed serving layout에 직접 영향을 준다는 좋은 사례다.

---

## 10. 1M context에서 bottleneck은 attention FLOPs만이 아니다

Z.AI는 GLM-5.2 serving에서 1M으로 확장하면 bottleneck이 다음으로 이동한다고 명시한다.

- KV-cache capacity
- long-context kernel overhead
- CPU-side cache management
- scheduler/runtime overhead
- cache transfer coordination

즉 IndexShare가 index compute를 줄여도:

```text
1M token state를 어디에 저장하는가?
어떤 rank가 어떤 cache를 갖는가?
request preemption 때 cache를 어떻게 이동하는가?
prefix가 얼마나 재사용되는가?
CPU scheduler가 GPU를 굶기지 않는가?
```

가 남는다.

**architecture optimization이 새로운 system bottleneck을 노출한다.**

---

## 11. Kimi K3와 대비하면 더 선명하다

Kimi K3는 long sequence 대부분을 KDA recurrent state로 처리하고 주기적인 MLA로 global path를 보완한다.

GLM-5.2는 모든 history를 recurrent state 하나로 압축하기보다:

```text
history를 보존
   ↓
lightweight indexer로 relevant positions 검색
   ↓
selected token에 exact sparse attention
```

을 택한다.

따라서 두 모델은 1M context 문제를 서로 다른 철학으로 푼다.

```text
Kimi K3: memory를 recurrent state로 적극 압축
GLM-5.2: memory는 유지하되 access를 극도로 sparse하게
```

serving system에서 cache/parallelism 요구사항도 자연히 달라진다.

---

## 12. 실전 profiler에서 볼 항목

GLM-5.2 long-context benchmark를 할 때 다음 kernel/time breakdown을 따로 본다.

```text
prefill
├─ Q/KV projection
├─ indexer projection
├─ indexer QK dot
├─ top-k selection
├─ sparse attention
├─ MoE dispatch
├─ expert GEMM
└─ shared expert

decode
├─ indexer/cache read
├─ sparse decode attention
├─ MoE
├─ MTP draft
└─ verification
```

그리고 IndexShare ON/OFF 또는 full/shared layer trace에서:

- indexer launch count
- top-k time
- sparse attention time
- memory bandwidth
- HBM usage
- TTFT/ITL

을 비교해야 한다.

---

## 13. 한 문장 정리

> **DSA가 ‘읽을 token 수’를 줄였다면, IndexShare는 ‘무엇을 읽을지 결정하는 검색 비용’까지 여러 layer에 걸쳐 amortize한다. 1M context에서 GLM-5.2가 성립하는 핵심은 sparse attention 그 자체보다 이 두 단계의 sparsity를 같이 설계한 데 있다.**

## Sources

- GLM-5.2 official blog: https://z.ai/blog/glm-5.2
- IndexCache: https://arxiv.org/abs/2603.12201
- GLM-5.2 config: https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
- Transformers GLM-MoE-DSA: https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
- vLLM Ascend context parallel design: https://docs.vllm.ai/projects/ascend/en/main/developer_guide/design_doc/context_parallel.html
