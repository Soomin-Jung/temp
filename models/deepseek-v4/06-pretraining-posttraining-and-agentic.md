# 06. Pretraining, Post-training & Agentic Capability

작성일: 2026-08-25

## 1. V4의 성능을 architecture 하나로 설명하면 안 된다

DeepSeek-V4의 성능은 다음 네 층이 함께 만든 결과다.

```text
architecture
+ pretraining data/optimizer
+ specialist RL
+ multi-teacher on-policy distillation
```

특히 최신 Flash-0731 / Pro-0813의 agent benchmark 향상은 backbone shape 변경보다 post-training 변화의 영향이 크다.

---

## 2. Pretraining corpus

V4 technical report는 32T+ 규모의 diverse/high-quality corpus를 사용한다고 설명한다.

강화된 데이터 축:

- mathematics
- programming
- multilingual
- long documents
- scientific papers / technical reports
- agentic data

특히 long-context architecture가 있어도 training data가 짧으면 1M context 능력은 생기지 않는다.

그래서 V4는 architecture와 data curriculum을 같이 본다.

---

## 3. Agentic data는 mid-training에서도 들어간다

V4는 coding capability를 강화하기 위해 pretraining/mid-training 단계부터 agentic data를 포함한다.

이 의미는 agent capability가 post-training tool use prompt만으로 생긴 것이 아니라는 점이다.

```text
pretraining foundation
  ↓
agentic/coding trajectories exposure
  ↓
specialist SFT/RL
  ↓
OPD consolidation
```

으로 계층적으로 만들어진다.

---

## 4. Post-training은 two-stage paradigm

V4의 핵심 post-training 구조:

```text
Stage 1: specialist cultivation
Stage 2: unified consolidation via OPD
```

### Stage 1

도메인별 expert model을 따로 훈련한다.

예:

- math
- coding
- agent
- instruction following

각 specialist는:

```text
SFT
→ domain-specific GRPO RL
```

을 거친다.

---

## 5. Mixed RL을 최종 병합 단계에서 제거

중요한 표현:

> V4가 RL을 없앤 것이 아니다.

specialist 단계에서는 여전히 RL/GRPO를 쓴다.

바뀐 것은 **final unified model merging stage**다.

V3.2-style mixed RL 대신 multi-teacher On-Policy Distillation(OPD)을 사용한다.

```text
Math specialist ───────┐
Code specialist ───────┤
Agent specialist ──────┤
Instruction specialist ┤
...                    ├─→ unified student
other teachers ────────┘
```

---

## 6. On-Policy Distillation

student가 자기 policy로 trajectory를 생성하고, 그 state에서 teacher distribution과 비교한다.

개념 objective:

```math
L = \sum_i w_i D_{KL}(\pi_\theta || \pi_{E_i})
```

여기서 중요한 것은 reverse-KL을 **student trajectory**에서 계산한다는 점이다.

즉 offline teacher trace만 따라가는 distillation과 달리 student가 실제 inference에서 방문할 state distribution에 더 가까운 supervision을 받는다.

---

## 7. Full-vocabulary distillation

V4 OPD는 sampled token 하나의 reward만 보는 방식이 아니라 teacher의 full vocabulary logit distribution을 사용한다.

장점:

- richer dark knowledge
- smoother gradient
- teacher behavior fidelity

단점:

- vocab >100K에서 logit memory가 매우 큼
- teacher가 여러 개면 storage/forward cost 폭증

그래서 별도의 system design이 필요하다.

---

## 8. Teacher scheduling system

Technical report의 핵심 engineering:

- teacher weights는 centralized distributed storage에 offload
- forward 시 on-demand load
- ZeRO-like sharding
- full logits를 저장하지 않고 **last-layer hidden state만 cache**
- training 시 prediction head로 logits 재생성
- samples를 teacher index 기준으로 order해 prediction head residency 최소화
- background async loading/offloading
- KL 계산은 TileLang kernel로 가속

즉 OPD는 단순 loss function이 아니라 distributed systems problem이다.

---

## 9. 왜 hidden state만 cache하는가

vocab size가 약 129K다.

teacher가 여러 개이고 trajectory가 길면 full-vocabulary logits를 모두 저장하는 비용은 막대하다.

그래서:

```text
teacher forward
   ↓
last hidden state cache
   ↓
필요할 때 teacher LM head reload
   ↓
full logits reconstruct
```

한다.

prediction head GEMM을 다시 하는 작은 compute overhead와 huge logit storage를 trade한다.

---

## 10. Million-token RL / OPD infrastructure

1M context에서 RL rollout은 일반 SFT보다 훨씬 까다롭다.

- rollout sequence length variability
- enormous KV cache
- sandbox/tool latency
- reward computation latency
- worker preemption
- long-running agent trajectory failure

V4 report는:

- preemptible/fault-tolerant rollout service
- million-token RL scaling
- sandbox infrastructure for agentic AI

를 별도 infrastructure component로 다룬다.

이건 agent model을 이해할 때 매우 중요하다.

> **agentic capability는 model weight뿐 아니라 학습 시 실제 tool/sandbox trajectory를 안정적으로 대량 생성할 수 있는 infrastructure에서 만들어진다.**

---

## 11. FP4는 post-training rollout에도 사용

V4는 teacher/reference inference와 rollout에 native FP4 weights를 사용해:

- memory traffic 감소
- sampling latency 감소

를 노린다.

training backward는 FP8 mixed precision framework를 재사용하면서 FP4 simulation/dequantization path를 사용한다.

즉 quantization은 serving-only optimization이 아니라 training infrastructure에도 들어간다.

---

## 12. Quick Instruction

V4 report에는 일부 auxiliary task를 별도 small model로 분리하지 않고 special instruction token으로 본체에 태워 실행하는 `Quick Instruction` 아이디어가 등장한다.

핵심:

- 이미 계산된 KV cache 재사용
- 추가 prefill 회피
- query generation / authority/domain 판단 같은 auxiliary task를 병렬화
- 별도 작은 helper model maintenance 감소

Agent system 관점에서 매우 흥미로운 부분이다.

전통 구조:

```text
main model context
→ helper model 다시 prefill
→ auxiliary output
```

Quick Instruction:

```text
same KV state
→ special task token
→ auxiliary generation
```

즉 model-level multiplexing으로 orchestration overhead를 줄인다.

---

## 13. Flash-0731 성능 향상을 어떻게 읽어야 하는가

공식 changelog는 Flash-0731이 Preview와 architecture/size는 같고 post-training을 다시 했다고 명시한다.

그런데 agent benchmark는 크게 상승했다.

예:

```text
Terminal Bench 2.1
61.8 → 82.7
```

이건 매우 중요한 사례다.

> **동일 backbone capacity에서도 agentic post-training / harness / reasoning policy가 실제 agent performance를 크게 바꿀 수 있다.**

GLM-5.2의 EDA agent 사례와도 연결된다.

---

## 14. Pro-0813도 같은 해석이 필요하다

Pro-0813은 Preview 구조 기반 공식 release다.

즉 성능 향상을:

```text
more layers / experts
```

로 설명하면 안 된다.

더 정확한 해석:

```text
same Pro backbone family
+ new target weights
+ improved agentic post-training
+ DSpark production decoding
+ updated reasoning/serving behavior
```

이다.

---

## 15. Architecture vs Capability를 분리하는 표

| 층 | V4의 예 |
|---|---|
| Architecture | SWA/CSA/HCA, mHC, MoE |
| Optimization | Muon, FP4 QAT |
| Foundation data | 32T+ corpus, long docs, agentic data |
| Specialist training | SFT + GRPO |
| Consolidation | multi-teacher OPD |
| Agent infra | sandbox, fault-tolerant rollout |
| Generation | MTP / DSpark |
| Harness | Responses API / coding-agent integration |

모델 benchmark 변화 원인을 볼 때 어느 층이 바뀌었는지 먼저 분해한다.

---

## 16. 출처

- DeepSeek-V4 Technical Report §4–5: https://arxiv.org/abs/2606.19348
- DeepSeek API updates: https://api-docs.deepseek.com/zh-cn/updates/
- Flash-0731: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- Pro-0813: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813
