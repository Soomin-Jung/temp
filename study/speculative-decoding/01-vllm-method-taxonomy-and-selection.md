# vLLM Speculative Decoding Method Taxonomy & Selection Guide

작성일: 2026-09-03  
분류: `study/speculative-decoding`  
기준 runtime: vLLM `v0.28.0` 중심, v0.26/v0.27과의 차이는 필요한 곳에 병기

> **이 문서의 목적**
>
> speculative decoding을 "K개 미리 뽑고 target이 검증한다"라는 한 문장으로 이해하면 실제 방법 선택이 불가능하다.
>
> 운영에서 필요한 질문은 다음이다.
>
> 1. 누가 draft를 만드는가?
> 2. 후보 K개는 serial / parallel / multi-head / tree 중 어떤 topology로 생성되는가?
> 3. target hidden state나 native MTP weight를 재사용하는가?
> 4. 별도 draft checkpoint와 KV cache가 필요한가?
> 5. target verification은 어떤 acceptance / rejection scheme을 쓰는가?
> 6. load가 바뀔 때 K 또는 verification length를 동적으로 조절하는가?
> 7. vLLM에서 실제 production path까지 연결돼 있는가?
>
> **SD 방법론 선택은 proposer 하나를 고르는 문제가 아니라 proposer topology + verifier + runtime policy의 조합을 고르는 문제다.**

---

# 1. 가장 먼저: vLLM의 "method"는 하나의 분류축이 아니다

vLLM v0.28.0의 `SpeculativeMethod`에는 다음 계열이 들어간다.

```text
ngram
ngram_gpu
suffix

draft_model
medusa
mlp_speculator

eagle
eagle3
mtp
dflash
dspark

extract_hidden_states
custom_class
```

하지만 이것을 동일한 수준의 알고리즘 목록으로 보면 안 된다.

예:

- `draft_model`, EAGLE, MTP, DFlash, DSpark는 **실제 proposer family**
- `ngram_gpu`는 n-gram 알고리즘의 GPU implementation variant
- `extract_hidden_states`는 inference acceleration method라기보다 **speculator 학습/분석용 plumbing**
- `custom_class`는 **extension interface**
- Dynamic Speculative Decoding은 proposer가 아니라 **K scheduling policy**
- Adaptive Verification은 proposer가 아니라 **verification-budget policy**
- `standard / block / synthetic`은 proposer가 아니라 **rejection/verification scheme**

따라서 다음 4축으로 보는 것이 가장 정확하다.

```text
Axis A. Proposer
  무엇이 후보를 만드는가?

Axis B. Candidate topology
  후보를 serial / parallel / multi-head / tree 중 어떻게 구성하는가?

Axis C. Verification / sampling
  target이 후보를 어떤 규칙으로 accept/reject하는가?

Axis D. Runtime adaptation
  load/confidence에 따라 K 또는 verify length를 어떻게 바꾸는가?
```

---

# 2. 전체 taxonomy

```text
Speculative Decoding
│
├─ A. Model-free proposer
│  ├─ N-gram / Prompt Lookup
│  └─ Suffix Decoding
│
├─ B. Independent model proposer
│  ├─ Classic Draft Model
│  ├─ PARD / Parallel Draft Model
│  └─ TLI cross-vocabulary draft
│
├─ C. Target-coupled lightweight proposer
│  ├─ EAGLE
│  ├─ EAGLE-3
│  ├─ Medusa
│  └─ MLP Speculator
│
├─ D. Target-native predictor
│  └─ MTP
│
├─ E. Parallel block proposer
│  ├─ DFlash
│  └─ DSpark
│
├─ F. Runtime policy overlay
│  ├─ Dynamic Speculative Decoding
│  └─ DSpark Adaptive Verification
│
└─ G. Infrastructure / extension
   ├─ Hidden State Extraction
   └─ Custom Proposer
```

이 분류에서 중요한 것은 **C/D/E가 모두 model-based SD지만 계산 topology가 전혀 다르다**는 점이다.

---

# 3. 모든 SD가 해결하려는 공통 문제

normal autoregressive decoding:

```text
target forward
→ token 1개 확정
→ target forward
→ token 1개 확정
→ ...
```

큰 target model의 decode는 low batch에서 보통 weight/HBM bandwidth의 영향을 강하게 받는다.

SD는 이를:

```text
cheap proposer
→ K개 후보
→ expensive target이 K개 위치를 한 번에 verify
→ 여러 token commit
```

으로 바꾼다.

성능을 단순화하면:

```math
cost per committed token
approx
rac{T_{draft} + T_{verify}}
{E[committed tokens per cycle]}
```

따라서 좋은 SD는 세 가지를 동시에 만족해야 한다.

1. draft가 target보다 충분히 싸야 한다.
2. acceptance가 충분히 높아야 한다.
3. verification이 batch capacity를 지나치게 잠식하지 않아야 한다.

즉 acceptance rate 하나만 높다고 좋은 SD가 아니다.

---

# 4. Classic Draft Model

vLLM:

```json
{
  "method": "draft_model",
  "model": "<smaller-compatible-model>",
  "num_speculative_tokens": 5
}
```

## 4.1 개념

가장 전통적인 speculative decoding이다.

```text
Target LLM: Qwen3-8B
Draft LLM : Qwen3-0.6B

context
  ↓
small draft LM
  → d1
  → d2
  → d3
  → d4
  → d5
  ↓
target verifies d1..d5
```

draft model 자체도 autoregressive이므로 K개 후보를 만들려면 원칙적으로 K회의 sequential draft step이 필요하다.

## 4.2 장점

- 구조가 가장 일반적이다.
- target architecture를 직접 수정할 필요가 없다.
- 적절한 small sibling model이 있으면 바로 적용하기 쉽다.
- target-native MTP나 EAGLE checkpoint가 없어도 시도할 수 있다.

## 4.3 약점

- K가 증가하면 draft latency도 거의 순차적으로 증가한다.
- 별도 draft weights가 필요하다.
- draft model 자체 KV cache와 runtime state가 필요하다.
- 너무 작은 drafter는 acceptance가 낮다.
- 너무 큰 drafter는 draft cost가 커져 SD 이득이 사라진다.

핵심 trade-off:

```text
draft model 작게
→ 빠름
→ target alignment 나쁠 수 있음

draft model 크게
→ acceptance 좋음
→ draft가 비싸짐
```

## 4.4 vLLM operational point

v0.28 MBT accounting에서 classic serial draft model의 additional drafting slot은 request당 1이다.

이 숫자는 K회의 draft compute가 1 token이라는 뜻이 아니다.

> scheduler 이후 batch-input expansion 관점의 추가 slot이 1이라는 뜻이다.

---

# 5. PARD — Parallel Draft Model

vLLM에서는 별도 method string이 아니라:

```json
{
  "method": "draft_model",
  "parallel_drafting": true
}
```

로 표현한다.

## 5.1 핵심 아이디어

classic draft model의 가장 큰 약점은:

```text
K candidates
→ K sequential draft forwards
```

이다.

PARD는 specially-trained draft model이 K future positions를 한 번에 예측한다.

```text
Classic draft
d1 → d2 → d3 → d4

PARD
[d1 d2 d3 d4]
     ↑
single parallel draft pass
```

즉 **draft phase 자체의 autoregressive dependency를 제거/완화**한다.

## 5.2 target independence

PARD의 중요한 설계 목표 중 하나는 target-specific feature drafter보다 높은 재사용성이다.

적절하게 학습된 parallel draft model을 같은 model family의 여러 target에 적용할 가능성이 있다.

이는 EAGLE/DSpark처럼 target hidden-state semantics에 강하게 결합된 drafter와 다른 장점이다.

## 5.3 단점

parallel prediction은 실제 이전 sampled token을 모른 채 미래 위치를 예측하기 때문에 suffix prediction quality가 떨어질 수 있다.

즉:

```text
draft latency 감소
vs
conditional dependency 약화
```

의 trade-off가 있다.

## 5.4 vLLM runtime 비용

v0.28 MBT accounting:

```text
PARD additional slots = K
```

즉 parallelism으로 draft latency는 줄지만 **한 step의 input width는 커진다.**

---

# 6. TLI — Cross-Vocabulary Draft Model

TLI(Token-Level Intersection)는 독립적인 proposer가 아니라 classic draft model에 붙는 compatibility overlay다.

```json
{
  "method": "draft_model",
  "use_heterogeneous_vocab": true
}
```

## 6.1 해결하는 문제

일반 speculative sampling은 draft와 target token ID 의미가 동일해야 다루기 쉽다.

하지만 서로 다른 tokenizer/vocabulary를 가진 모델을 조합하면:

```text
draft token ID 123
≠
target token ID 123
```

이다.

TLI는 초기화 시 token string을 정규화해 vocabulary intersection과 mapping을 만들고, shared token 공간에서만 draft하게 한다.

## 6.2 의미

이 기능은:

> "좋은 small sibling model이 없으면 다른 family small model을 drafter로 쓸 수 있는가?"

라는 선택지를 넓힌다.

다만 v0.28에서는 heterogeneous-vocab path가 greedy draft sampling에 제한되는 등 제약을 반드시 확인해야 한다.

---

# 7. EAGLE

vLLM:

```json
{
  "method": "eagle",
  "model": "<EAGLE-speculator>",
  "num_speculative_tokens": K
}
```

## 7.1 classic draft와 무엇이 다른가

classic draft model은 token sequence를 보고 다음 token distribution을 모방한다.

EAGLE의 핵심 관찰은:

> target model의 **상위 hidden feature 공간을 autoregress**하는 것이 token distribution 전체를 작은 LM으로 다시 모방하는 것보다 효율적일 수 있다.

원 논문은 second-to-top-layer feature를 예측하고, feature uncertainty를 줄이기 위해 one-step-shifted token을 함께 조건으로 사용한다.

개념:

```text
Target hidden feature h_t
+
sampled token x_t
       ↓
small EAGLE drafter
       ↓
future feature approximation
       ↓
LM head / token proposal
```

## 7.2 왜 acceptance가 좋아질 수 있는가

drafter가 target이 이미 계산한 representation을 직접 이용하므로:

```text
small independent LM
보다
target-specific feature predictor
```

가 target continuation에 더 잘 정렬될 수 있다.

## 7.3 비용

- target별 또는 target-family별 trained speculator가 필요하다.
- target hidden states를 전달해야 한다.
- 기본 형태는 draft rollout의 autoregressive dependency가 남는다.
- target/drafter architecture coupling이 강하다.

## 7.4 고려할 상황

- 범용적인 model-based SD가 필요하다.
- target에 맞는 EAGLE checkpoint가 존재한다.
- model-free proposer보다 높은 acceptance가 필요하다.
- 별도 full draft LM보다 더 target-aligned한 lightweight drafter를 원한다.

---

# 8. EAGLE-3

vLLM method:

```text
eagle3
```

EAGLE-3는 이름만 EAGLE의 version upgrade가 아니라 predictor target 자체가 바뀐다.

## 8.1 EAGLE과 핵심 차이

EAGLE:

```text
future feature prediction 중심
```

EAGLE-3:

```text
direct token prediction
+
target의 low/mid/high layer feature fusion
+
training-time test
```

원 논문은 EAGLE의 feature-prediction constraint가 data scaling을 제한한다고 보고, direct token prediction과 multi-layer feature fusion으로 바꾼다.

## 8.2 operational 의미

target의 여러 depth에서 정보를 가져오므로 draft가:

- lexical/local signal
- intermediate semantic signal
- high-level target representation

을 같이 활용할 수 있다.

일반적으로 EAGLE 계열 중 먼저 검토할 후보가 EAGLE-3인 이유다.

## 8.3 P-EAGLE

vLLM의 `parallel_drafting=true`는 EAGLE 계열에도 적용 가능한 별도 topology 축이다.

따라서:

```text
EAGLE-3
= proposer architecture

P-EAGLE
= EAGLE 계열을 parallel candidate generation으로 사용하는 execution topology
```

라고 구분한다.

v0.28 MBT accounting에서 P-EAGLE additional slots:

```text
K - 1
```

---

# 9. Medusa

vLLM method:

```text
medusa
```

## 9.1 핵심 아이디어

큰 target backbone 뒤에 여러 lightweight decoding head를 붙인다.

```text
target hidden state
   ├─ head 1 → x(t+1)
   ├─ head 2 → x(t+2)
   ├─ head 3 → x(t+3)
   └─ head 4 → x(t+4)
```

즉 separate draft Transformer를 돌리지 않고 여러 future offset을 **parallel multi-head prediction**한다.

## 9.2 original Medusa

원 논문에서는 여러 head의 후보를 조합해 candidate tree를 만들고 tree attention으로 target이 여러 continuation을 동시에 검증하는 것이 핵심 설계 중 하나다.

Medusa-1:

- target backbone freeze
- Medusa heads만 학습

Medusa-2:

- backbone과 heads를 같이 fine-tune
- 더 높은 draft quality 가능
- target behavior 보존을 위한 training recipe가 중요

## 9.3 vLLM v0.28 source nuance

v0.28 `MedusaProposer`는 target hidden state를 Medusa model에 넣고 각 head logit의 argmax를 stack해:

```text
[batch_size, num_heads]
```

형태의 draft token tensor를 만든다.

따라서 **논문의 전체 candidate-tree 설계와 vLLM에서 실제 선택된 proposer/runtime path를 구분해서 읽어야 한다.**

## 9.4 장단점

장점:

- full draft LM보다 매우 작은 추가 weight
- future positions를 parallel head로 생성
- target hidden state를 직접 활용

약점:

- 각 future head 사이의 autoregressive conditional dependency가 약하다.
- 후반 offset으로 갈수록 prediction mismatch가 커질 수 있다.
- 전용 head checkpoint/training 필요

---

# 10. MLP Speculator

vLLM config surface:

```text
mlp_speculator
```

## 10.1 알고리즘 개념

IBM 계열 MLP speculator는 target context embedding과 이미 sampled된 token을 함께 conditioning해 lightweight MLP 계층으로 future n-gram을 예측한다.

```text
target context vector
+
sampled token embedding
      ↓
small MLP cascade
      ↓
draft token sequence
```

full Transformer draft model보다 훨씬 싼 predictor를 목표로 한다.

## 10.2 왜 EAGLE/Medusa와 별도인가

- Medusa: offset별 multi-head parallel prediction
- EAGLE: target feature를 autoregressive extrapolation
- MLP speculator: token + embedding context를 사용하는 lightweight MLP predictor

각각 "target representation을 이용한다"는 공통점은 있지만 conditional structure가 다르다.

## 10.3 v0.28 support 주의

v0.28 source에는:

- `SpeculativeMethod`에 `mlp_speculator` 존재
- 공식 docs도 존재
- config class도 존재

하지만 V1 model registry에는:

```text
MLPSpeculatorPreTrainedModel
→ Temporarily disabled
→ Re-enable once supported in V1
```

라는 명시적 주석이 있다.

따라서 v0.28 기준으로는:

> **API/docs 존재와 production-usable V1 runtime 지원을 동일시하면 안 되는 대표 사례**

로 본다.

---

# 11. Native MTP — Multi-Token Prediction

vLLM:

```json
{
  "method": "mtp",
  "num_speculative_tokens": K
}
```

## 11.1 MTP는 먼저 model architecture / training feature다

MTP는 본래 target model이 하나의 next-token loss만 보는 대신 여러 future offset을 예측하도록 학습하는 구조다.

따라서:

```text
MTP training support
≠
MTP speculative decoding enabled
```

이다.

checkpoint에 MTP layers가 있어도 runtime에서 사용하지 않을 수 있다.

## 11.2 SD에서의 MTP

target과 함께 학습된 MTP module을 lightweight drafter로 사용한다.

```text
target hidden state
→ native MTP module
→ future token proposal
→ target verification
```

## 11.3 강점

- 별도 unrelated draft LM이 필요하지 않다.
- target training과 함께 alignment가 만들어진다.
- 추가 weights가 비교적 작다.
- 모델 vendor가 native MTP를 제공한다면 운영상 가장 자연스러운 baseline이 된다.

## 11.4 약점

- 모델이 native MTP architecture/checkpoint를 제공해야 한다.
- 구현은 model-specific하다.
- K를 늘릴 때 동일 MTP layer를 여러 번 reuse하는 구조인지, multi-module MTP인지에 따라 cost와 acceptance가 달라진다.

v0.28의 `MTPModelTypes`에는 DeepSeek, Qwen, GLM, Kimi, MiMo, MiniMax, Gemma 등 다양한 model-specific MTP type이 존재한다.

즉 "MTP" 하나의 보편적인 runtime module이라기보다:

> **여러 model family의 native future-prediction 구조를 vLLM이 하나의 method 아래 묶은 interface**

에 가깝다.

---

# 12. DFlash

vLLM method:

```text
dflash
```

## 12.1 해결하려는 병목

EAGLE나 classic draft의 핵심 한계:

```text
draft 자체가 autoregressive
→ K가 커질수록 draft 단계도 sequential
```

DFlash는 lightweight block-diffusion drafter로 K개 후보를 병렬 생성한다.

```text
target context features
       ↓
block-diffusion drafter
       ↓
[d1 d2 d3 ... dK]
single parallel draft forward
```

## 12.2 핵심 장점

K가 커져도 heavy draft backbone을 K번 serial 실행할 필요가 없다.

따라서:

```text
draft depth ↑
에 따른
T_draft 증가율
```

을 크게 낮출 수 있다.

## 12.3 구조적 약점 — suffix decay

future positions를 동시에 예측하면 뒤 위치가 실제 앞 draft token에 완전히 condition되지 못한다.

즉:

```text
d1 정확
d2 약간 덜 정확
d3 더 불확실
...
```

한 위치가 reject되면 뒤 suffix가 무의미해지는 prefix-acceptance SD에서는 이 현상이 치명적이다.

## 12.4 runtime 특성

- parallel drafting
- non-causal-capable draft attention backend 필요
- target hidden state를 활용
- specialized DFlash checkpoint 필요

v0.28 method-specific MBT accounting:

```text
additional slots = K
```

DFlash는 bonus query + K mask query topology 때문에 DSpark와 slot 식이 다르다.

---

# 13. DSpark

vLLM method:

```text
dspark
```

DSpark는 DFlash의 parallelism을 유지하면서 suffix dependency와 verification waste를 동시에 해결하려는 architecture/system co-design이다.

## 13.1 Draft architecture

```text
target hidden features
       ↓
parallel backbone
       ↓
K base logits
       ↓
lightweight sequential correction
  Markov / recurrent dependency
       ↓
draft sequence
```

즉 heavy computation은 병렬이고, token 간 dependency는 값싼 sequential head로 복원한다.

이를 **semi-autoregressive drafting**으로 볼 수 있다.

## 13.2 왜 DFlash보다 acceptance가 좋아질 수 있는가

DFlash:

```text
future positions mostly parallel
→ suffix conditional mismatch
```

DSpark:

```text
parallel base prediction
+
previous draft token 기반 correction
→ intra-block dependency 회복
```

따라서 parallel drafting의 latency 장점을 유지하면서 accepted prefix length를 늘리는 것이 목표다.

## 13.3 confidence head

DSpark는 각 future position이 accept될 가능성을 추정하는 confidence signal을 제공할 수 있다.

이는 단순 draft-quality metric이 아니라 adaptive verification의 입력이 된다.

## 13.4 vLLM MBT

K=7이면 v0.28 accounting:

```text
additional slots = K - 1 = 6
```

DFlash의 K와 다르다.

## 13.5 언제 특히 강한가

- target-specific DSpark checkpoint가 존재
- low/medium concurrency에서 긴 speculative block을 활용
- load 변화가 크고 adaptive verification까지 사용할 수 있음
- target decode가 memory-bound이고 extra verification compute를 흡수할 여유가 있음

DeepSeek-V4 계열처럼 모델과 serving stack이 함께 설계된 경우 대표적인 후보다.

---

# 14. N-Gram / Prompt Lookup

vLLM:

```text
method = ngram
method = ngram_gpu
```

## 14.1 model-free proposer

현재 sequence의 suffix와 과거 context에서 동일 n-gram을 찾는다.

예:

```text
context:
... function foo(x):
    return x + 1
...
function foo(x):

현재 suffix "function foo(x):" 발견
→ 과거 continuation "return x + 1"을 draft
```

neural model forward가 필요 없다.

## 14.2 잘 맞는 workload

- code editing
- document continuation
- template
- repetitive structured output
- prompt 내용을 많이 복사하는 생성

## 14.3 약점

open-ended reasoning/chat에서 반복 패턴이 없으면 proposal 자체가 나오지 않거나 acceptance가 낮다.

## 14.4 CPU vs GPU

`ngram_gpu`는 새로운 SD 이론이 아니라 동일 lookup proposer의 GPU implementation variant로 보는 것이 맞다.

v0.28 MBT additional drafting slots:

```text
0
```

이다.

model-free라는 점에서 high-QPS 시 model-based drafter compute를 추가하지 않는 장점이 있다.

---

# 15. Suffix Decoding

vLLM:

```text
method = suffix
```

Arctic Inference dependency가 필요하다.

## 15.1 N-gram보다 발전된 model-free 방법

n-gram이 현재 sequence 내 exact pattern match 중심이라면 Suffix Decoding은:

- prompt
- 현재 generation
- 이전 cached responses

에 대한 suffix tree를 유지한다.

```text
최근 token pattern
      ↓
suffix tree match
      ↓
frequency-count continuation
      ↓
variable-length draft
```

## 15.2 핵심 차이

N-gram:

```text
"같은 pattern을 찾았나?"
```

Suffix:

```text
"이 suffix 뒤에 역사적으로 어떤 continuation이 얼마나 자주 왔나?"
```

를 이용한다.

## 15.3 dynamic speculation length

Suffix Decoding은 request/step마다 실제 draft length가 달라질 수 있다.

`num_speculative_tokens`는 실질적으로 maximum cap 역할을 한다.

## 15.4 적합 workload

- code
- agentic loop
- self-reflection / self-consistency 반복
- RL rollout
- 반복적 structured generation

별도 neural drafter가 없으므로 memory footprint가 작고, peak traffic에서 model-based drafter보다 부담이 작을 수 있다.

---

# 16. Dynamic Speculative Decoding — proposer가 아니다

vLLM:

```text
num_speculative_tokens_per_batch_size
```

## 16.1 문제

SD K의 optimum은 batch size에 따라 달라진다.

low batch:

```text
memory-bound
→ K를 늘려도 추가 compute가 비교적 싸다
```

high batch:

```text
compute saturation
→ BS × K verification 폭증
→ SD가 오히려 느려질 수 있다
```

## 16.2 DSD

batch-size range마다 K를 다르게 설정한다.

```text
BS 1~16    → K=7
BS 17~64   → K=4
BS 65~128  → K=2
BS 129+    → K=0
```

즉:

> **같은 proposer를 사용하되 load에 따라 speculation depth를 바꾸는 runtime policy**

다.

## 16.3 v0.28 limitation

공식 문서 기준 테스트된 계열은 EAGLE, EAGLE-3, DFlash 중심이며 다른 방식은 별도 검증이 필요하다.

DP에서는 각 rank가 서로 다른 K를 고르면 collective divergence가 생길 수 있어 vLLM이 dynamic K를 disable하고 static K로 fallback한다.

---

# 17. Adaptive Verification — DSpark 전용 policy

vLLM:

```text
enable_adaptive_verification = true
```

현재 DSpark + confidence head에 한정된다.

## 17.1 DSD와 다른 점

Dynamic SD:

```text
batch size
→ 모든 request의 K 조정
```

Adaptive Verification:

```text
request별 confidence
+
position별 survival probability
+
현재 target cost profile
→ 실제 verify할 prefix 길이 결정
```

즉 훨씬 fine-grained하다.

## 17.2 예

```text
request A confidence:
.99 .95 .92 .85 .80 .70 .60

request B:
.75 .45 .20 .10 ...
```

고정 K=7이면 둘 다 7개를 verify한다.

adaptive verification은:

```text
A → 6~7개 verify
B → 1~2개 verify
```

처럼 verification capacity를 더 가치 있는 position에 배분할 수 있다.

## 17.3 운영 의미

이건 acceptance를 높이는 기술이 아니다.

> **낮은 acceptance 가능성의 token에 target compute를 낭비하지 않는 scheduling 기술**

이다.

v0.28에서는 full CUDA Graph, backend capability 등 별도 제약이 있으므로 feature flag 하나로 끝나는 기능이 아니다.

---

# 18. Verification / Rejection Sampling 축

proposer가 후보를 만들었다고 끝이 아니다.

vLLM v0.28에는:

```text
rejection_sample_method =
  standard
  synthetic
  block
```

이 존재한다.

## 18.1 Standard

일반 speculative rejection sampling이다.

random sampling에서는 draft distribution q와 target distribution p의 관계를 이용해 draft token을 accept하고, rejection 시 corrected distribution에서 recovery token을 뽑는다.

모든 draft가 accept되면 target-only bonus token을 하나 더 commit할 수 있다.

핵심:

> 올바른 rejection sampling은 draft가 틀릴 수 있어도 target distribution을 보존하도록 설계된다.

greedy에서는 target argmax와 draft token 일치 여부가 핵심이 된다.

---

## 18.2 Block verification

vLLM config 설명상 draft tokens를 개별 position 중심으로 처리하는 대신 block 단위로 jointly verify하는 방식이다.

DSpark 같은 block proposer와 결합될 수 있다.

중요한 것은:

```text
block proposer
≠
block verification
```

이다.

candidate generation topology와 acceptance algorithm은 별도 축이다.

---

## 18.3 Synthetic

synthetic acceptance rate/length를 강제로 주는 benchmark/test mode다.

실제 proposer quality를 평가하는 production method로 보면 안 된다.

```text
synthetic
→ scheduler/kernel/system cost를 acceptance profile과 분리해서 측정할 때 유용
```

하다.

---

# 19. Draft sampling 축

vLLM:

```text
draft_sample_method =
  greedy
  probabilistic
```

## Greedy draft

drafter의 argmax를 candidate로 사용한다.

장점:

- 단순함
- draft probability tensor 처리 비용을 줄일 수 있는 path가 있음

## Probabilistic draft

drafter distribution에서 sampling한다.

target-distribution-preserving rejection sampling과 더 자연스럽게 결합되지만 draft probability를 유지/전달해야 한다.

따라서:

```text
temperature / sampling mode
+
draft_sample_method
+
rejection_sample_method
```

를 함께 봐야 한다.

---

# 20. Candidate topology 축

같은 proposer family라도 candidate topology가 달라질 수 있다.

## Serial chain

```text
d1 → d2 → d3 → d4
```

예:

- classic draft model
- default EAGLE
- 일부 MTP rollout

장점: conditional consistency  
단점: draft latency가 K에 따라 증가

---

## Parallel block

```text
[d1 d2 d3 d4]
```

예:

- PARD
- DFlash
- DSpark backbone
- P-EAGLE 계열

장점: draft latency 감소  
단점: future-token dependency 모델링이 어려움

---

## Multi-head

```text
h
├─ head1 → t+1
├─ head2 → t+2
└─ head3 → t+3
```

대표: Medusa

---

## Semi-autoregressive block

```text
parallel base logits
+
cheap sequential correction
```

대표: DSpark

parallelism과 dependency를 절충한다.

---

## Tree

하나의 continuation chain이 아니라 branch들을 같이 검증한다.

```text
       a
     /   \
    b     c
   / \     \
  d   e     f
```

원래 Medusa, EAGLE 계열 연구와 tree verification에서 중요한 topology다.

다만 **논문이 tree를 쓴다는 사실과 특정 vLLM version의 proposer가 동일 tree path를 사용한다는 사실은 별도로 확인해야 한다.**

---

# 21. vLLM v0.28 실제 support surface

| Category | vLLM surface | V1 runtime 관찰 | 운영 해석 |
|---|---|---|---|
| Classic draft model | `draft_model` | active proposer | 범용 baseline |
| PARD | `draft_model + parallel_drafting` | active | 전용 checkpoint 필요 |
| EAGLE | `eagle` | active | target-specific speculator |
| EAGLE-3 | `eagle3` | active | 우선 검토 가치 높음 |
| MTP | `mtp` | active, model-specific paths 다수 | native MTP 모델 baseline |
| DFlash | `dflash` | active DFlash proposer | non-causal draft backend 고려 |
| DSpark | `dspark` | active target-hidden-state family path | confidence/adaptive verify 가능 |
| Medusa | `medusa` | active Medusa proposer | multi-head predictor |
| MLP Speculator | `mlp_speculator` | V1 registry disabled 주석 | version/path 확인 필수 |
| N-gram | `ngram` | active | model-free |
| N-gram GPU | `ngram_gpu` | active implementation | n-gram의 execution variant |
| Suffix | `suffix` | active + Arctic Inference | model-free adaptive pattern |
| Hidden state extraction | `extract_hidden_states` | active utility | acceleration method 아님 |
| Custom proposer | `custom_class` | experimental extension | 직접 proposer 구현 |

이 표는 **method enum 존재 여부만으로 지원을 판단하지 말아야 한다**는 의미가 핵심이다.

---

# 22. 어떤 방법을 언제 고려할 것인가

## Case A — target에 native MTP가 있다

첫 비교군:

```text
Target only
vs
MTP
```

이다.

이유:

- target-native
- 운영 통합이 가장 단순한 편
- 별도 unrelated draft model 불필요

그 뒤 EAGLE-3/DSpark 같은 specialized drafter가 있으면 비교한다.

---

## Case B — target 전용 EAGLE-3 checkpoint가 있다

범용적인 고성능 model-based SD 후보로 매우 강하다.

특히:

- low/medium QPS
- decode 비중 큰 workload
- 충분한 output length

에서 우선 benchmark할 가치가 높다.

---

## Case C — DSpark-aware model/checkpoint가 있다

다음 비교를 권장한다.

```text
target only
MTP baseline
DSpark fixed K
DSpark adaptive verification
```

DSpark는 단순 proposer가 아니라 verification policy까지 포함해 성능 Pareto를 바꾸는 것이 목표이므로 fixed-K만 보고 결론 내리지 않는다.

---

## Case D — 별도 작은 sibling model만 있다

classic draft model부터 시작한다.

```text
draft size
×
K
×
draft TP
```

를 같이 sweep한다.

---

## Case E — parallel-trained draft가 있다

PARD를 고려한다.

특히 serial draft latency가 전체 SD cycle에서 큰 비중이면 효과가 크다.

단 high concurrency에서는 widened query가 target compute를 빠르게 saturation시킬 수 있다.

---

## Case F — code/editing/agent loop처럼 반복이 많다

model-free 방법을 반드시 baseline에 넣는다.

```text
N-gram
Suffix Decoding
```

은 별도 model memory와 draft compute가 거의 없어 의외로 Pareto가 좋을 수 있다.

---

## Case G — 매우 높은 steady concurrency

SD 자체를 끈 baseline을 반드시 유지한다.

```text
Target-only
vs
SD K=1/2
vs
Dynamic SD
```

를 비교해야 한다.

high batch에서는 extra verification이 free compute가 아니라 **실제 사용자 token과 경쟁하는 compute**가 된다.

---

# 23. workload 관점 선택표

| Workload | 우선 후보 | 이유 |
|---|---|---|
| open-ended chat, low QPS | EAGLE-3 / native MTP / DSpark | model-based acceptance 우선 |
| reasoning, 긴 output | EAGLE-3 / MTP / DSpark | decode step 절감 가치 큼 |
| code completion | EAGLE-3 + N-gram/Suffix 비교 | neural + repetition baseline 모두 강함 |
| code editing | Suffix / N-gram / model-based 비교 | 원문 repetition이 많음 |
| agentic repeated loop | Suffix / DSpark / Dynamic SD | history repetition + load 변동 |
| RL rollout | Dynamic SD / Suffix / model-based | batch가 rollout 진행 중 크게 변화 |
| high-QPS saturated service | Target-only / low-K / Dynamic SD | verification waste 경계 중요 |
| model-native MTP 제공 | MTP부터 | 가장 자연스러운 integration baseline |
| 별도 speculator 없음 | N-gram / Suffix / sibling draft | training 없이 시작 가능 |

---

# 24. 비용 구조 비교

| Method | Draft compute | Extra weights | Extra KV/state | K 확장성 | target coupling | 반복 workload 의존 |
|---|---|---|---|---|---|---|
| Draft model | 중~높음 | 높음 | 높음 | 낮음~중간 | 낮음 | 낮음 |
| PARD | 낮음~중간 | 높음 | 있음 | 높음 | 낮음~중간 | 낮음 |
| EAGLE | 낮음~중간 | 작음 | 있음 | 중간 | 높음 | 낮음 |
| EAGLE-3 | 낮음~중간 | 작음 | 있음 | 중간 | 높음 | 낮음 |
| Medusa | 낮음 | 매우 작음 | 작음 | 중간 | 매우 높음 | 낮음 |
| MLP | 낮음 | 작음 | 작음 | 중간 | 높음 | 낮음 |
| MTP | 낮음 | native | native/shared 성격 | model-specific | 매우 높음 | 낮음 |
| DFlash | 낮음 | 중간 | 있음 | 높음 | 높음 | 낮음 |
| DSpark | 낮음 | 중간 | 있음 | 높음 | 높음 | 낮음 |
| N-gram | 매우 낮음 | 없음 | CPU metadata | pattern-dependent | 없음 | 매우 높음 |
| Suffix | 매우 낮음 | 없음 | CPU suffix tree | dynamic | 없음 | 높음 |

"낮음/높음"은 상대적인 architecture 비교이며 실제 runtime cost는 target size, TP, backend, batch에 따라 달라진다.

---

# 25. acceptance만 보면 안 되는 이유

예:

```text
Method A:
accepted length = 5
draft latency = 2.5 ms
verify latency = 4 ms

Method B:
accepted length = 4
draft latency = 0.4 ms
verify latency = 3 ms
```

accepted length는 A가 높아도 end-to-end는 B가 더 빠를 수 있다.

따라서 최소 metric:

```text
draft latency
target verify latency
accepted length
position-wise acceptance
cycle당 committed tokens
ITL / TPOT
aggregate output tok/s
TTFT
queue latency
GPU utilization
HBM bandwidth
scheduler/input budget utilization
```

을 같이 본다.

---

# 26. long-context workload에서는 더 조심한다

SD는 기본적으로 **decode optimization**이다.

입력이 170K이고 출력이 2K라면 전체 request cost 중 prefill/context attention 비중이 매우 클 수 있다.

따라서:

```text
SD speedup on decode
≠
same ratio의 E2E speedup
```

이다.

또한 context가 길어질수록 target verification의 attention/KV read cost도 커질 수 있으므로 low-context에서 얻은 optimal K를 그대로 복사하면 안 된다.

---

# 27. P/D disaggregation에서는 어떻게 볼 것인가

P/D에서는 SD를 Decode role의 policy로 보는 것이 명확하다.

```text
Prefill engine
  → long prompt 처리
  → KV transfer

Decode engine
  → target decode
  → speculative draft
  → verify / accept
```

따라서:

- P의 MBT tuning
- D의 MBT/K/MNS tuning
- SD proposer cost
- KV/state capacity

를 분리해 benchmark한다.

특히 native recurrent/hybrid state model에서는 SD와 별개로 state block/alignment constraint가 있으므로 MBT minimum을 SD 식 하나로 결정하면 안 된다.

---

# 28. selection decision tree

```text
START
 │
 ├─ target에 native MTP가 있는가?
 │     └─ YES → MTP를 첫 model-based baseline
 │
 ├─ target-specific EAGLE-3 / DSpark checkpoint가 있는가?
 │     ├─ EAGLE-3 → 범용 high-acceptance 후보
 │     └─ DSpark → fixed-K + adaptive verification 비교
 │
 ├─ parallel-trained draft checkpoint가 있는가?
 │     └─ YES → PARD / P-EAGLE 계열 비교
 │
 ├─ compatible small sibling model이 있는가?
 │     └─ YES → classic draft model
 │
 ├─ workload repetition이 높은가?
 │     ├─ YES → Suffix
 │     └─ 부분적 → N-gram
 │
 └─ high concurrency인가?
       ├─ YES → K 축소 / Dynamic SD / target-only 반드시 비교
       └─ NO  → 더 큰 K 탐색 가능
```

이 decision tree는 "하나만 선택"이 아니라 benchmark 후보를 줄이는 용도다.

---

# 29. production evaluation matrix

모든 신규 SD method는 최소 다음 matrix를 통과시킨다.

## Algorithm

```text
method
K
parallel_drafting
draft_sample_method
rejection_sample_method
dynamic K
adaptive verification
```

## Workload

```text
short / p50 / p90 / p99 context
short / mean / long output
chat / code / reasoning / tool JSON / agent loop
```

## Load

```text
concurrency 1
low
medium
saturation
burst
```

## Runtime

```text
MBT
MNS
max_num_scheduled_tokens
CUDA Graph mode
attention backend
TP/DP/EP
KV dtype
prefix cache
P/D 여부
```

## Correctness

```text
greedy equality
sampling distribution
tool-call correctness
reasoning parser
streaming
cancellation
long-context
```

---

# 30. 운영자가 가져야 할 핵심 직관

## 직관 1

> **Draft accuracy가 목표가 아니다.**

목표는 target-only 대비 latency-throughput Pareto를 개선하는 것이다.

## 직관 2

> **K가 많다고 좋은 것이 아니다.**

K는 draft cost, acceptance decay, target verification width, MBT, CUDA Graph shape를 동시에 키운다.

## 직관 3

> **parallel drafter가 항상 serial drafter보다 좋은 것이 아니다.**

parallelism으로 draft latency는 줄지만 conditional dependency를 잃어 acceptance가 떨어질 수 있다.

DSpark 같은 방법은 바로 이 문제를 해결하려 한다.

## 직관 4

> **native MTP가 있다고 최적이라는 뜻은 아니다.**

통합은 쉽지만 EAGLE-3/DSpark/PARD가 더 좋은 Pareto를 만들 수도 있다.

## 직관 5

> **model-free SD는 싸구려 fallback이 아니다.**

repetition-heavy workload에서는 neural draft model보다 훨씬 낮은 overhead로 좋은 결과를 낼 수 있다.

## 직관 6

> **high concurrency에서는 SD가 손해가 될 수 있다.**

low batch의 spare compute를 활용하는 최적화가 saturation 이후에는 실제 user-token compute를 뺏을 수 있다.

---

# 31. source map — vLLM v0.28.0

## Config taxonomy

`vllm/config/speculative.py`

- `SpeculativeMethod`
- `MTPModelTypes`
- `parallel_drafting`
- `num_speculative_tokens_per_batch_size`
- `rejection_sample_method`
- `draft_sample_method`
- `enable_adaptive_verification`

https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/speculative.py

## Actual proposer dispatch

`vllm/v1/worker/gpu_model_runner.py`

- NgramProposer
- NgramProposerGPU
- SuffixDecodingProposer
- EagleProposer
- DFlashProposer
- DraftModelProposer
- MedusaProposer
- ExtractHiddenStatesProposer
- model-specific MTP proposers
- custom proposer

https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/worker/gpu_model_runner.py

## Rejection sampling

`vllm/v1/sample/rejection_sampler.py`

https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/sample/rejection_sampler.py

## Official feature docs

https://github.com/vllm-project/vllm/tree/v0.28.0/docs/features/speculative_decoding

---

# 32. primary papers

- Leviathan et al., Fast Inference from Transformers via Speculative Decoding  
  https://arxiv.org/abs/2211.17192
- Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling  
  https://arxiv.org/abs/2302.01318
- EAGLE  
  https://arxiv.org/abs/2401.15077
- EAGLE-3  
  https://arxiv.org/abs/2503.01840
- Medusa  
  https://arxiv.org/abs/2401.10774
- MLP Speculator  
  https://arxiv.org/abs/2404.19124
- PARD  
  https://arxiv.org/abs/2504.18583
- DFlash  
  https://arxiv.org/abs/2602.06036
- DSpark  
  https://arxiv.org/abs/2607.05147
- Suffix Decoding  
  https://arxiv.org/abs/2411.04975

---

# 33. 관련 문서

- [Speculative Decoding: 전통 SD, MTP, DFlash, DSpark의 구조적 차이](00-foundations-and-method-lineage.md)
- [vLLM Speculative Decoding Token Budget Deep Dive](02-vllm-token-budget-and-slot-topology.md)
- [Scheduler Budget, Speculative Decoding & CUDA Graph](../inference-serving-optimization/01-scheduler-budget-spec-decode-cudagraph.md)
- [DeepSeek-V4 MTP / DSpark](../../models/deepseek-v4/05-mtp-dspark-and-speculative-decoding.md)

---

# 34. 본질

Speculative decoding의 발전 흐름은 한 문장으로 보면 다음과 같다.

```text
"더 작은 모델로 미리 추측하자"
        ↓
"target feature를 이용해 더 잘 추측하자"
        ↓
"여러 미래 token을 병렬로 추측하자"
        ↓
"병렬성 때문에 잃은 dependency를 싸게 복원하자"
        ↓
"accept 가능성이 낮은 token은 아예 verify하지 말자"
        ↓
"load에 따라 speculation 자체를 동적으로 바꾸자"
```

> **결국 SD의 본질은 '미래 token을 맞히는 기술'이 아니라, expensive target model의 한 번의 실행에서 얻는 유효 진전량을 최대화하도록 draft cost·acceptance·verification capacity를 함께 설계하는 기술이다.**
