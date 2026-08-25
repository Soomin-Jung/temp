# 05. MTP & Speculative Decoding

작성일: 2026-08-25  
기준: GLM-5 Technical Report + GLM-5.2 official blog + current engine recipes

## 1. 왜 long-horizon agent에서는 decode가 다시 중요해지는가

1M context model이라고 하면 prefill만 생각하기 쉽다. 하지만 coding/EDA agent는 긴 reasoning과 반복 tool interaction 때문에 output token도 많다.

```text
long prompt
 → think
 → tool call
 → tool result
 → think
 → patch
 → test
 → think
 → ...
```

따라서 GLM-5.2는 attention cost뿐 아니라 autoregressive decode의 serial dependency도 줄이려 한다.

MTP(Multi-Token Prediction)가 그 축이다.

---

## 2. MTP의 본질

standard autoregressive LM:

```text
h_t → token t+1
             │
             ▼
h_{t+1} → token t+2
```

future token을 하나씩 serial하게 생성한다.

MTP는 training 단계에서 추가 future position을 예측하도록 학습된 module을 이용해 여러 candidate token을 draft한다.

inference에서는:

```text
MTP draft:       t+1, t+2, t+3, ...
                     │
                     ▼
backbone verification in parallel
                     │
            accepted prefix만 commit
```

하는 speculative decoding으로 연결된다.

핵심 metric은 단순 draft length가 아니라 **acceptance length**다.

---

## 3. GLM-5의 MTP lineage

GLM-5 technical report는 MTP parameter sharing을 중요한 효율화로 설명한다.

여러 future step을 위해 완전히 별도 parameter block을 복제하는 대신 같은 MTP parameters를 반복 적용한다.

```text
MTP step 1 ─┐
MTP step 2 ─┼─ same/shared parameters
MTP step 3 ─┘
```

training memory를 크게 늘리지 않으면서 multi-step predictive capability를 학습하는 방향이다.

GLM-5 report는 동일 sampled setting에서 MTP acceptance 관련 개선도 보고한다.

---

## 4. GLM-5.2에서 새로 생기는 문제: MTP도 DSA를 한다

backbone이 DSA라면 MTP draft step도 sparse attention을 위해 index를 필요로 한다.

naive하게 하면:

```text
MTP step 1 → indexer → top-k
MTP step 2 → indexer → top-k
MTP step 3 → indexer → top-k
...
```

가 되어 draft model 자체가 비싸진다.

GLM-5.2는 backbone에서 사용한 IndexShare 아이디어를 MTP iteration에도 적용한다.

current config:

```text
index_share_for_mtp_iteration = true
```

즉 첫 MTP iteration에서 계산한 sparse index를 뒤 step에서 재사용한다.

---

## 5. KVShare까지 필요한 이유

MTP step마다 input token이 달라지면 index만 공유해도 internal state/cache handling이 달라질 수 있다.

GLM-5.2 official blog는 **IndexShare + KVShare**를 함께 적용한다.

목표는:

1. draft MTP의 index computation 제거
2. 반복 step의 KV-related overhead 제거
3. training/inference의 cache semantics mismatch 완화

이다.

conceptual flow:

```text
MTP step 1
  ├─ compute index
  ├─ establish/share KV state
  └─ draft token
        │
        ▼
MTP step 2..N
  ├─ reuse index
  ├─ reuse compatible KV state
  └─ draft next token
```

---

## 6. Rejection sampling과 end-to-end TV loss

draft model을 빠르게 만드는 것만으로는 부족하다. candidate가 backbone distribution과 다르면 verify에서 자주 reject되어 speedup이 사라진다.

GLM-5.2는 acceptance를 높이기 위해:

- rejection sampling
- end-to-end TV(total variation) loss

를 추가한다.

목표를 단순화하면:

```text
MTP distribution
      ≈
backbone verification distribution
```

을 더 직접적으로 맞추는 것이다.

TV distance는 두 probability distribution의 차이를 측정하므로, draft distribution이 target model과 가까워질수록 speculative acceptance에 유리하다.

---

## 7. Official ablation

Z.AI가 공개한 coding scenario ablation은 7 MTP step 조건에서 다음 acceptance length를 보고한다.

| Method | Acceptance length |
|---|---:|
| Baseline | 4.56 |
| + IndexShare + KVShare | 5.10 |
| + Rejection Sampling | 5.29 |
| + End-to-end TV loss | 5.47 |

최종적으로 baseline 대비 약 20% 증가다.

주의할 점:

- 이 표는 특정 coding scenario와 실험 설정의 acceptance 결과다.
- production engine의 optimal draft token 수와 동일한 숫자가 아니다.
- 더 많은 draft token이 항상 더 빠르지는 않다.

---

## 8. 왜 draft length를 무한히 늘리면 안 되는가

speculative decode throughput은 대략 다음 trade-off를 갖는다.

```text
benefit:
한 번의 verify로 여러 token commit

cost:
draft generation
+ verification compute
+ rejected suffix waste
+ graph/cache complexity
```

acceptance probability가 낮다면 draft를 길게 할수록 waste가 커진다.

따라서 실전에서는:

- workload별 average acceptance length
- p50/p95 acceptance
- MTP overhead
- verify batch efficiency
- ITL

을 측정해서 draft count를 정해야 한다.

current vLLM recipe는 GLM-5.2 MTP를 5 speculative tokens로 예시하고, 다른 runtime/recipe는 workload에 따라 다른 값을 사용할 수 있다.

---

## 9. GLM-5.2 config의 `num_nextn_predict_layers=1`을 해석하는 법

current HF config:

```text
num_nextn_predict_layers = 1
```

이 값만 보고 “1 token만 speculative prediction한다”고 읽으면 안 된다.

GLM 계열은 **MTP parameters를 step 사이에 반복 재사용**하는 구조를 사용한다. checkpoint에 별도의 MTP layer가 하나 있어도 inference iteration을 여러 번 수행할 수 있다.

즉 구분:

```text
number of stored MTP parameter blocks
≠
number of speculative draft iterations
```

이 차이는 checkpoint size와 runtime option을 읽을 때 매우 중요하다.

---

## 10. Transformers backbone과 production speculative engine을 구분한다

generic `GlmMoeDsaForCausalLM`의 standard forward는 backbone causal LM path를 제공한다.

production MTP speculative decoding은 inference engine이:

- MTP checkpoint weights
- draft iteration
- KV state
- index sharing
- verification
- acceptance logic

을 orchestration해야 한다.

따라서 HF에서 normal `generate()`가 동작한다고 vLLM/SGLang의 MTP path까지 검증된 것은 아니다.

---

## 11. vLLM에서 확인할 것

current vLLM recipe는 GLM-5.2에 MTP speculative decoding을 제공한다.

예시 개념:

```text
--speculative-config.method mtp
--speculative-config.num_speculative_tokens 5
```

실전 검증에서는:

1. base generation quality와 MTP generation quality가 동일 범위인가?
2. reasoning/tool parser와 MTP를 동시에 켰을 때 문제없는가?
3. CUDA graph capture가 MTP step shape를 포함하는가?
4. acceptance length가 coding/agent prompt에서 어느 정도인가?
5. throughput gain이 short output에서도 존재하는가?
6. long context에서 index/KV reuse가 실제로 적용되는가?

를 확인해야 한다.

speculative decoding은 **output correctness regression을 먼저 막고 throughput을 측정**해야 한다.

---

## 12. P/D disaggregation과 MTP

P/D 환경에서는 MTP는 decode engine에 특히 중요하다.

```text
Prefill Engine
  long context ingest
  DSA/indexing/cache build
      │ KV transfer
      ▼
Decode Engine
  token generation
  + MTP drafting
  + verification
```

여기서 decode 최적화의 주요 변수는:

- decode batch width
- MoE expert utilization
- MTP acceptance
- KV locality
- speculative verify kernel

이다.

P/D로 prefill interference를 없애도 MTP acceptance가 낮거나 expert dispatch가 비효율적이면 decode throughput은 기대만큼 오르지 않는다.

---

## 13. Kimi 계열과 비교

둘 다 MTP/speculative path를 적극적으로 사용하지만 세부 구조는 모델별로 다르다.

GLM-5.2에서 특히 중요한 점은 **DSA index 자체가 draft overhead가 되기 때문에 IndexShare/KVShare를 MTP 내부까지 확장했다는 것**이다.

즉 GLM-5.2의 speculative design은 backbone attention architecture와 독립적인 부가기능이 아니다.

```text
DSA architecture
      ↓
MTP에도 indexer 필요
      ↓
MTP IndexShare/KVShare
```

으로 연결된 co-design이다.

---

## 14. 한 문장 정리

> **GLM-5.2의 MTP는 ‘한 번에 여러 token을 예측한다’에서 끝나지 않는다. DSA 때문에 생기는 draft-step index/KV 비용까지 공유하고, draft distribution을 backbone에 더 가깝게 학습해 실제 acceptance를 높이는 speculative-decoding subsystem이다.**

## Sources

- GLM-5 Technical Report: https://arxiv.org/abs/2602.15763
- GLM-5.2 official blog: https://z.ai/blog/glm-5.2
- GLM-5.2 config: https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
- vLLM GLM-5.2 recipe: https://recipes.vllm.ai/zai-org/GLM-5.2
