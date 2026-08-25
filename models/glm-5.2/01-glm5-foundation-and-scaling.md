# 01. GLM-5 Foundation & Scaling

작성일: 2026-08-25  
핵심 원문: [GLM-5 Technical Report](https://arxiv.org/abs/2602.15763)

## 1. 왜 GLM-5부터 시작해야 하는가

GLM-5.2는 architecture를 처음부터 갈아엎은 모델이라기보다 **GLM-5의 large-scale MoE + MLA + DSA foundation 위에서 long-context indexing과 agentic training을 확장한 모델**이다.

따라서 GLM-5.2를 제대로 이해하려면 먼저 GLM-5가 어떤 system trade-off를 택했는지 고정해야 한다.

GLM-5 technical report의 대표 scale:

| 항목 | 값 |
|---|---:|
| Total parameters | ~744B |
| Activated parameters | ~40B |
| Pretraining tokens | 28.5T |
| Hidden size | 6144 |
| Dense layers | 3 |
| MoE layers | 75 |
| Routed experts | 256 |
| Experts/token | 8 |
| Shared experts | 1 |
| MoE FFN intermediate | 2048 |
| Attention heads | 64 |
| Q LoRA rank | 2048 |
| KV LoRA rank | 512 |
| Indexer heads | 32 |
| Indexer head dim | 128 |

current GLM-5.2 config도 이 backbone geometry를 거의 그대로 유지한다.

---

## 2. 744B인데 왜 active가 40B인가

Dense 744B model이라면 token 하나를 처리할 때 거의 모든 layer parameter를 통과한다.

Sparse MoE에서는 FFN을 여러 expert로 복제하고 router가 일부만 선택한다.

GLM 계열의 simplified layer:

```text
hidden h
   │
   ├──────── RMSNorm → Attention ─────────┐
   │                                      │
   └──────────────────────────────────────+→ h'
                                          │
                                          ▼
                                      RMSNorm
                                          │
                       ┌──────────────────┴─────────────────┐
                       │                                    │
                shared expert                       routed expert bank
                  always on                         256 중 top-8
                       │                                    │
                       └──────────────────┬─────────────────┘
                                          ▼
                                       residual
```

전체 parameter capacity는 256개의 expert bank에 분산되지만 한 token은 8개 routed expert만 실행한다.

단순 routed activation ratio만 보면:

```math
\frac{8}{256}=3.125\%
```

하지만 active parameter에는 attention, embedding, shared expert, router 등도 포함되므로 전체 active ratio와 동일하지 않다.

### 이 방식의 장점

- total capacity를 크게 늘릴 수 있음
- token당 FLOPs가 total parameter와 같은 비율로 증가하지 않음
- expert specialization 가능

### 시스템 비용

- expert weight가 여러 GPU에 분산됨
- token이 선택된 expert GPU로 이동해야 함
- dispatch/combine all-to-all 발생
- load imbalance가 생기면 tail latency가 늘어남

즉 MoE는 compute를 공짜로 줄이는 기법이 아니라 **dense compute를 routing + communication 문제로 바꾸는 기법**이다.

---

## 3. 왜 expert 수를 늘리면서 layer 수를 제한했는가

GLM-5 report는 model capacity를 늘리면서 layer count를 80-class로 제한한 이유에 expert-parallel communication cost를 연결한다.

MoE layer 하나마다 대략:

```text
local hidden
 → router
 → dispatch(all-to-all)
 → remote/local expert GEMM
 → combine(all-to-all)
 → residual
```

가 반복된다.

layer depth를 무작정 늘리면 token당 EP collective 횟수도 증가한다.

따라서 frontier MoE 설계에서는 다음 세 변수가 동시에 움직인다.

```text
number of layers
× experts per layer
× active experts/token
× placement/parallelism
```

GLM-5는 width를 expert bank로 크게 늘리면서 depth/communication budget을 같이 본다.

이 관점은 실제 vLLM deployment에서도 중요하다. `TP만 크게 하면 된다`가 아니라:

- tensor parallel
- expert parallel
- data/expert parallel
- node boundary
- all-to-all topology

를 같이 최적화해야 한다.

---

## 4. 첫 3개 layer가 Dense인 이유를 어떻게 볼 것인가

current config:

```text
first_k_dense_replace = 3
mlp_layer_types =
  layer 0: dense
  layer 1: dense
  layer 2: dense
  layer 3..77: sparse
```

초기 layer는 raw token representation을 안정적으로 공통 feature space로 올리는 역할이 강하다.

초기부터 모든 token을 expert routing하면:

- 아직 representation이 덜 분화된 상태에서 routing decision이 불안정할 수 있고
- shared low-level feature를 expert마다 중복 학습할 수 있으며
- routing overhead를 일찍부터 지불한다.

따라서 early dense → later sparse는 대형 MoE에서 자주 등장하는 실용적 패턴이다.

다만 위 이유는 일반적인 architecture 해석이며, GLM-5 report가 각 early dense layer의 개별 역할을 명시적으로 증명한 것은 아니다. 문서에서는 **config fact와 interpretation을 구분**한다.

---

## 5. MLA 도입의 scaling 의미

MoE가 parameter width를 키우는 동안 long context는 별도의 memory wall을 만든다.

MHA에서 KV cache는 대략 다음 변수에 비례한다.

```math
KV\ memory \propto L \times H_{kv} \times D_{head} \times 2(K,V) \times bytes
```

context `L`이 수십만~100만 token으로 올라가면 weight memory뿐 아니라 KV cache가 serving capacity를 결정한다.

GLM-5는 DeepSeek-style MLA를 사용해 K/V 정보를 low-rank latent로 압축한다.

current config에서:

```text
KV latent rank = 512
RoPE key       = 64
```

이 compressed state가 long-context cache의 핵심 representation이다.

단, implementation backend에 따라 cache representation이 달라질 수 있다. 예를 들어 generic Transformers sparse-attention path는 현재 expanded K/V를 cache하는 코드 경로가 존재하며 코드에 TODO도 남아 있다. production engine의 MLA-native cache kernel과 vanilla HF implementation을 동일시하면 안 된다.

---

## 6. Attention head를 64개로 구성한 compute 관점

GLM-5 report는 attention head geometry도 serving roofline을 고려해 조정했다고 설명한다.

GLM-5 config:

```text
num_attention_heads = 64
qk_nope_head_dim     = 192
qk_rope_head_dim     = 64
qk total             = 256
v_head_dim           = 256
```

head를 지나치게 많이 두면 decode에서 작은 GEMM/attention kernel 수가 늘고 memory traffic과 launch overhead가 커질 수 있다.

GLM-5는 head dimension을 넓히고 head count를 줄이는 방향을 사용하면서 training/prefill capacity와 decode efficiency의 균형을 잡는다.

여기서 중요한 점:

> **모델 architecture 숫자는 quality만의 선택이 아니라 실제 serving hardware의 arithmetic intensity와 memory behavior까지 반영할 수 있다.**

---

## 7. 28.5T-token training과 context curriculum

GLM-5 report의 training progression은 대략:

```text
large-scale pretraining
      │
      └─ 27T-class corpus
            │
            ▼
mid-training / continued training
      ├─ context expansion 4K → 200K-class
      ├─ long-context data
      └─ DSA transition
            │
            ▼
post-training
      Reasoning RL
         → Agentic RL
         → General RL
         → cross-stage on-policy distillation
```

GLM-5.2에서는 이를 1M context와 더 긴 coding-agent trajectory까지 확장한다.

따라서 1M context를 config의 `max_position_embeddings=1048576` 하나로 이해하면 안 된다.

필요한 것은:

- positional encoding range
- long-context training data
- sparse attention adaptation
- indexer training
- long-horizon task data
- inference engine memory management

전체다.

---

## 8. DSA를 기존 dense attention 모델에 어떻게 넣었는가

GLM-5는 DSA를 scratch architecture로만 학습한 것이 아니라 continued-pretraining 방식으로 도입하는 절차를 설명한다.

개념적으로 두 단계다.

### Stage 1 — indexer warm-up / distillation

먼저 dense attention의 중요도 분포를 lightweight indexer가 따라가게 만든다.

```text
Dense attention distribution
          │ teacher signal
          ▼
lightweight indexer score
```

### Stage 2 — sparse adaptation

이후 실제 top-k sparse attention을 켜고 추가 token으로 adaptation한다.

technical report에서는 DSA transition을 위해 별도의 warm-up과 약 20B-token sparse adaptation을 기술한다.

이 접근이 중요한 이유는 sparse attention에서 가장 어려운 문제가 `attention kernel`이 아니라 **무엇을 버려도 되는지 학습하는 것**이기 때문이다.

---

## 9. GLM-5 foundation에서 GLM-5.2로 무엇이 바뀌는가

GLM-5의 기본 문제:

```text
MLA → KV memory 감소
DSA → main attention compute 감소
MoE → parameter capacity / active FLOPs 분리
MTP → decode acceleration
```

GLM-5.2의 추가 문제:

```text
1M context에서는
DSA indexer 자체도 너무 비싸다
        │
        ▼
IndexShare

긴 speculative loop에서도
MTP indexer/KV 비용이 반복된다
        │
        ▼
MTP IndexShare + KVShare

긴 agent trajectory는
architecture만으로 성공하지 않는다
        │
        ▼
1M-context agent training + long-horizon RL
```

따라서 GLM-5.2를 한 문장으로 요약하면:

> **GLM-5의 sparse MoE/MLA/DSA foundation을 1M-token long-horizon agent workload까지 밀어붙이면서, sparse attention의 남은 index cost와 speculative-decoding cost를 다시 제거한 버전이다.**

---

## 10. Sources

- GLM-5 Technical Report: https://arxiv.org/abs/2602.15763
- GLM-5.2 model config: https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
- GLM-5.2 official blog: https://z.ai/blog/glm-5.2
