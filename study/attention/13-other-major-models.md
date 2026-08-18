# 13. 다른 주요 LLM의 Attention 설계 비교

작성일: 2026-08-18  
상위 문서: [Attention Architecture Study Guide](README.md)

## 1. 이 장의 목적

Kimi, Qwen, DeepSeek만 보면 최근 attention의 방향을 `linear/recurrent` 또는 `compressed attention`으로만 오해하기 쉽다. 실제 frontier/open-weight 모델들은 서로 다른 workload와 training 목표에 따라 다양한 선택을 유지하고 있다.

이 장은 다음 모델 계열을 attention 설계 축으로 비교한다.

- Llama
- Gemma
- Mistral
- MiniMax
- gpt-oss
- RecurrentGemma/Griffin

폐쇄형 frontier 모델은 공식적으로 확인 가능한 architecture 정보가 부족한 경우 추정하지 않는다.

---

## 2. 비교를 위한 공통 축

각 모델은 다음 여섯 질문으로 본다.

1. Full/Local/Recurrent/Compressed 중 어떤 token mixer를 사용하는가?
2. Query/KV head sharing은 MHA/MQA/GQA 중 무엇인가?
3. Position encoding은 RoPE/partial RoPE/NoPE 중 무엇인가?
4. 모든 layer가 같은 attention인가, hybrid인가?
5. KV cache가 context length에 어떻게 증가하는가?
6. long-context 비용을 어떤 방식으로 줄이는가?

---

# 3. Llama 계열

## 3.1 Llama 2 → Llama 3: GQA의 일반화

Meta Llama의 큰 모델들은 conventional softmax Transformer를 유지하면서 **Grouped-Query Attention(그룹드 쿼리 어텐션; GQA)**을 적극적으로 사용했다.

핵심 구조:

```text
RoPE
  +
Global causal softmax attention
  +
GQA
  +
SwiGLU FFN
```

즉 Llama 3 계열은 DeepSeek MLA나 Qwen GDN처럼 attention memory representation 자체를 근본적으로 바꾸기보다 mature한 full attention + GQA를 선택한 대표적인 계열이다.

### 장점

- token-addressable global memory
- exact retrieval에 강함
- mature FlashAttention ecosystem
- conventional paged KV cache

### 비용

- cache는 $O(T)$
- global attention prefill은 $O(T^2)$
- long context decode는 KV read가 $T$에 비례

---

## 3.2 Llama 4와 iRoPE

Meta의 Llama 4 공개 설명에서는 **iRoPE(interleaved RoPE, 아이로프, RoPE layer와 NoPE-like attention layer를 교차 배치하는 positional 설계)**를 주요 long-context innovation 중 하나로 소개한다.

공식 설명의 핵심은:

- 대부분 attention layer는 RoPE를 사용
- 일부 layer는 positional embedding 없이 attention
- interleaving을 통해 very-long-context generalization을 개선
- attention temperature scaling 등을 함께 사용

한다는 것이다.

이를 다음처럼 개념화할 수 있다.

```text
RoPE Attention
RoPE Attention
NoPE / position-free Attention
RoPE Attention
...
```

정확한 layer layout은 checkpoint/config별로 확인해야 한다.

### 왜 position-free layer가 도움이 될 수 있는가

RoPE는 relative position을 score에 직접 주입하지만 training range 밖으로 length extrapolation할 때 rotation frequency/generalization 문제가 생길 수 있다.

일부 layer를 content-only global attention으로 만들면 position encoding에 덜 민감한 semantic retrieval path를 제공할 수 있다.

이 철학은 Kimi-K3의 `KDA가 sequence position을 담당하고 MLA는 NoPE global content lookup`과 완전히 같지는 않지만, **모든 attention layer가 동일한 positional mechanism을 가질 필요는 없다**는 공통 흐름을 보여준다.

---

# 4. Gemma 3: Local과 Global의 5:1 Hybrid

Google의 Gemma 3는 **Local Sliding-Window Attention(로컬 슬라이딩 윈도우 어텐션)**과 global attention을 layer 단위로 섞는다.

대형 Gemma 3 모델에서 공개된 기본 패턴은:

```text
Local
Local
Local
Local
Local
Global
```

즉 5:1이다.

### 4.1 Local Window

Local layer는 대략 1024-token sliding window를 사용한다.

```math
\text{cost}\approx O(TW),\qquad W=1024
```

오래된 KV를 local layer에서 계속 조회하지 않아도 되므로 cache cost를 크게 줄일 수 있다.

### 4.2 Global Layer

매 6번째 layer는 전체 context를 본다.

이는 멀리 떨어진 token 간 direct interaction을 제공한다.

### 4.3 GQA와 QK-Norm

Gemma 3는 GQA와 **QK-Norm(큐케이 놈, attention score 계산 전 query/key 정규화)**를 함께 사용한다.

공식 기술자료는 Gemma 2의 attention logit soft-capping 대신 QK normalization을 채택했다고 설명한다.

### 4.4 Position Design

Gemma 3는 local/global layer에서 RoPE base를 다르게 사용한다.

- local path: 짧은 범위에 적합한 base
- global path: long context에 적합한 큰 base

즉 같은 model 안에서도 layer의 receptive-field 역할에 맞춰 position encoding hyperparameter를 다르게 잡는다.

### 4.5 의미

Gemma 3는 recurrent state를 만들지 않는다.

```text
Local layer  → recent raw KV
Global layer → all raw/global KV
```

따라서 Qwen/Kimi의 recurrent hybrid와 달리 모든 memory는 여전히 token-addressable softmax KV 형태다.

---

# 5. Mistral 7B: GQA + Sliding Window

Mistral 7B는 efficient open LLM attention 설계의 대표적인 초기 사례다.

두 핵심 요소:

1. **Grouped-Query Attention(GQA)**
2. **Sliding Window Attention(SWA)**

### 5.1 4096 Window

Mistral 7B는 각 layer에서 4096-token sliding window를 사용한다.

따라서 새 token은 모든 과거 token이 아니라 최근 window만 직접 조회한다.

### 5.2 Rotating Buffer Cache

Decode에서는 고정 window보다 오래된 KV를 버리거나 rotating buffer 위치에 덮어쓸 수 있다.

```text
physical cache slots: 0 ... W-1
logical position t → slot (t mod W)
```

따라서 local KV cache의 size가 context length와 무관하게 bounded될 수 있다.

### 5.3 Layer Depth와 Effective Context

각 layer가 4096만 직접 보더라도 여러 layer를 통과하면 information이 간접적으로 더 멀리 전파될 수 있다.

하지만 current query가 100K token 전의 exact K/V를 한 번에 직접 찾는 것은 global attention과 다르다.

---

# 6. MiniMax-01 / M1: Lightning Attention Hybrid

MiniMax-01은 million-token context를 위해 **Lightning Attention(라이트닝 어텐션, block-wise linear attention 계열)**과 conventional Softmax Attention을 hybrid로 사용한다.

공식 MiniMax-01 기술보고서의 큰 패턴은:

```text
Lightning Attention × 7
Softmax Attention   × 1
```

이다.

### 6.1 Lightning Attention

Lightning Attention은 linear attention을 block-wise matrix form으로 계산해 긴 sequence에서 sequence-length scaling을 줄이는 계열이다.

핵심 관점은:

```text
long-history interactions
   → linear/block state path
```

이고 periodically softmax attention을 배치해 exact/global interaction을 보완한다.

### 6.2 MiniMax-M1

MiniMax-M1은 이 hybrid architecture와 large-scale RL reasoning을 결합한 공개 reasoning model이다.

공식 논문은 long-context reasoning을 주요 목표로 한다.

---

## 6.3 MiniMax M2의 중요한 반례

MiniMax는 이후 M2 설명에서 hybrid Lightning/full attention을 계속 유지하지 않고 **full attention으로 돌아간 이유**도 공개했다.

공식 engineering 설명은 hybrid linear attention이 일부 complex multi-hop reasoning에서 품질 deficit을 보였다고 설명한다.

이는 매우 중요한 사례다.

> Linear/hybrid attention이 항상 full attention의 상위호환은 아니다.

Training scale, reasoning pattern, exact retrieval 요구, kernel efficiency를 함께 봐야 한다.

Kimi/Qwen이 hybrid에서 좋은 결과를 얻었다고 해서 모든 모델에 동일한 cheap/global ratio가 적용되는 것은 아니다.

---

# 7. gpt-oss: Local Banded + Full Attention

OpenAI의 공개 가중치 모델 **gpt-oss-120b / gpt-oss-20b**는 conventional softmax family 안에서 global attention과 locally banded attention을 교대한다.

공식 model card 기준 gpt-oss-120b의 attention 관련 핵심은:

- 36 Transformer layers
- residual width 2880
- 64 query heads
- head dimension 64
- 8 KV heads
- GQA group size 8
- RoPE/YaRN 계열 long-context position encoding
- 최대 131,072 context
- locally banded attention과 full/dense attention 교대
- local band width 128
- head별 learned softmax denominator bias/sink

### 7.1 GQA

```math
H_q=64,\qquad H_{kv}=8
```

따라서:

```math
g=64/8=8
```

이다.

### 7.2 Local/Full Alternation

개념적으로:

```text
Local banded attention
Full attention
Local banded attention
Full attention
...
```

이다.

### 7.3 Attention Sink 성격의 Learned Bias

공식 model card는 각 head의 softmax denominator에 learned bias를 추가해 head가 특정 token에 attention mass를 강제로 배분하지 않을 수 있게 한다.

개념적으로:

```math
a_i=
\frac{e^{s_i}}
{e^{b_{sink}}+\sum_j e^{s_j}}
```

처럼 생각할 수 있다.

이것은 긴 context에서 attention distribution 안정화와 `none-of-the-above` 경로를 제공하는 하나의 방법이다.

---

# 8. RecurrentGemma / Griffin: Attention과 RNN의 다른 Hybrid

Google의 **Griffin** architecture와 RecurrentGemma는 recurrent sequence model과 local attention을 결합한 흥미로운 사례다.

Griffin block은 크게:

- gated linear recurrence
- local attention

을 혼합한다.

이는 DeltaNet/KDA와 같은 recurrence 수식은 아니지만 `global KV cache에 모든 history를 저장하지 않고 recurrent state로 정보 일부를 전달한다`는 broader design space를 보여준다.

RecurrentGemma는 이런 Griffin architecture의 open model implementation이다.

### 왜 참고할 가치가 있는가

현대 efficient sequence architecture가 반드시:

```text
Transformer full attention
   ↓
Linear attention
```

한 계보로만 발전한 것이 아니라:

- SSM
- gated RNN
- linear attention
- local attention

사이의 경계가 계속 재조합되고 있음을 보여준다.

---

# 9. 주요 모델 비교표

| 모델 | Sequence mixer | Head/KV 구조 | Global exact path | Recurrent fixed state | Local window | 주요 long-context 전략 |
|---|---|---|---|---|---|---|
| Llama 3 | Global full | GQA | 모든 layer | 아니오 | 없음 | GQA + optimized full attention |
| Llama 4 | interleaved positional/full attention | model별 | 있음 | 아니오 | model별 | iRoPE / NoPE interleaving |
| Gemma 3 | Local + Global | GQA | 1/6 layer | 아니오 | 1024 | 5:1 local/global |
| Mistral 7B | Sliding Window | GQA | 없음(직접) | 아니오 | 4096 | bounded local KV |
| MiniMax-01 | Lightning + Softmax | hybrid | 1/8 layer | linear state 계열 | blockwise | 7:1 linear/global |
| gpt-oss | Local banded + Full | GQA | alternating | 아니오 | 128 | local/full alternation |
| Qwen3.6 | GDN + Full | GDN + GQA | 1/4 layer | 예 | full layer는 global | 3:1 recurrent/global |
| Kimi-K3 | KDA + MLA | KDA + latent MLA | 24/93 layer | 예 | ShortConv local | recurrent + latent global |
| DeepSeek-V4 | CSA/HCA + local | Shared K=V MQA | compressed global | 아니오 | 128 | sequence compression + sparse/dense |

---

# 10. 최적화 축으로 다시 분류하기

## 10.1 Head-axis Compression

```text
MHA → GQA → MQA
```

대표:

- Llama
- Gemma
- Mistral
- gpt-oss
- DeepSeek-V4 일부

## 10.2 Feature-axis Compression

```text
full K/V → latent K/V
```

대표:

- DeepSeek MLA
- Kimi MLA

## 10.3 Token/Sequence-axis Restriction

```text
all tokens → window/top-k/compressed entries
```

대표:

- Mistral SWA
- Gemma local
- DeepSeek DSA/CSA/HCA
- gpt-oss banded attention

## 10.4 History-to-State Compression

```text
all token memories → fixed recurrent state
```

대표:

- Qwen GDN
- Kimi KDA
- broader linear/RNN/SSM families

## 10.5 Layer-axis Hybridization

```text
cheap layer × N → expensive/global layer
```

대표:

- Gemma 3: 5:1
- MiniMax-01: 7:1
- Qwen: 3:1
- Kimi: ~3:1
- gpt-oss: alternating
- DeepSeek-V4: CSA/HCA interleave

---

# 11. 왜 모든 Frontier Model이 Linear Attention으로 가지 않는가

Linear recurrent attention은 매우 매력적이다.

- state size independent of context length
- decode history-length independent
- prefill linearizable

하지만 fixed state에는 finite information capacity가 있다.

Full attention의:

```text
query → exact token address
```

능력은 다음 task에서 중요하다.

- exact text retrieval
- repository code reference
- long-horizon tool trace
- multi-hop reasoning
- repeated entity disambiguation
- exact numeric/string copying

따라서 최근 모델들은 크게 세 전략 중 하나를 택한다.

1. full attention을 유지하고 local/GQA로 비용 절감
2. recurrent + full hybrid
3. compressed/sparse token memory로 full-like addressability 유지

모델마다 training recipe와 target workload에 따라 최적점이 다르다.

---

# 12. Closed Frontier Model에 대한 원칙

GPT, Claude, Gemini 계열의 최신 폐쇄 모델은 product API나 고수준 technical report만 공개되고 실제 attention head/cache 구조가 공개되지 않는 경우가 많다.

이 문서에서는 다음을 하지 않는다.

- benchmark 결과로 attention architecture를 추정
- 특허/루머를 실제 production model architecture로 단정
- 이전 세대 공개 논문의 구조를 최신 폐쇄 모델에 자동 적용

Architecture가 공개된 경우에만 해당 release/version을 명시해 기술한다.

OpenAI의 `gpt-oss`는 공개 model card와 weights/config가 있으므로 이 문서에 포함했다.

---

# 13. 새 모델을 만났을 때 이름 대신 확인할 것

새 공개 LLM을 보면 다음 config/paper 항목을 먼저 찾는다.

```text
num_hidden_layers
num_attention_heads
num_key_value_heads
head_dim
sliding_window
layer_types
full_attention_interval
attention_pattern
rope_theta / partial_rotary_factor
linear_attention_config
kv_lora_rank
q_lora_rank
compression_ratio
index_topk
```

그리고 질문한다.

> `이 모델은 context-length cost를 head, feature, token, state, layer 중 어느 축에서 줄였는가?`

이 질문이 모델 이름을 외우는 것보다 훨씬 오래 쓸 수 있다.

---

# 14. 핵심 정리

1. Llama 3는 mature GQA full-attention 설계의 대표 사례다.
2. Llama 4는 RoPE와 position-free attention layer를 interleave하는 iRoPE를 long-context design에 사용한다.
3. Gemma 3는 5 local : 1 global GQA로 KV/cache 비용과 global retrieval을 절충한다.
4. Mistral 7B는 GQA + 4096 Sliding Window로 bounded local cache를 보여준 대표 모델이다.
5. MiniMax-01/M1은 Lightning linear attention + periodic softmax hybrid를 사용했지만, 이후 M2는 complex reasoning 품질 문제를 이유로 full attention을 선택한 사례를 공개했다.
6. gpt-oss는 8-way GQA와 local banded/full attention alternation을 사용한다.
7. RecurrentGemma/Griffin은 gated recurrence와 local attention을 섞는 또 다른 recurrent design space를 보여준다.
8. 최신 모델의 공통점은 `attention 하나가 모든 layer를 지배한다`보다 layer와 memory 역할을 분리하는 hybrid가 많아졌다는 것이다.
9. 그럼에도 full attention의 exact addressability 때문에 pure linear architecture가 universal replacement가 되지는 않았다.

---

# 15. 공식 자료

- Meta, **The Llama 3 Herd of Models**  
  https://arxiv.org/abs/2407.21783
- Meta, **The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation**  
  https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Google, **Gemma 3 Technical Report**  
  https://arxiv.org/abs/2503.19786
- Google Developers, **Introducing Gemma 3**  
  https://developers.googleblog.com/en/introducing-gemma3/
- Mistral AI, **Mistral 7B**  
  https://arxiv.org/abs/2310.06825
- MiniMax, **MiniMax-01: Scaling Foundation Models with Lightning Attention**  
  https://arxiv.org/abs/2501.08313
- MiniMax, **MiniMax-M1**  
  https://arxiv.org/abs/2506.13585
- MiniMax engineering documentation, M2 architecture discussion  
  https://www.minimax.io/news/minimax-m2
- OpenAI, **gpt-oss model card / architecture**  
  https://openai.com/index/introducing-gpt-oss/
- Google, **Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models**  
  https://arxiv.org/abs/2402.19427
