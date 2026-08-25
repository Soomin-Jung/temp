# 09. Flash vs Pro — Runtime Profile

작성일: 2026-08-25

## 1. 같은 architecture family, 다른 infrastructure class

DeepSeek-V4-Flash와 Pro는 동일한 핵심 architecture를 공유한다.

- mHC
- SWA + CSA + HCA
- DeepSeekMoE
- Hash-MoE bootstrap
- MTP/DSpark lineage
- FP4 expert / FP8 mixed precision

하지만 scale이 크게 달라 실제 serving topology는 거의 다른 급으로 취급해야 한다.

---

## 2. 구조 비교

| 항목 | Flash-0731 | Pro-0813 |
|---|---:|---:|
| Target total params | 284B | 1.6T |
| Active params/token | 13B | 49B |
| Layers | 43 | 61 |
| Hidden size | 4096 | 7168 |
| Q heads | 64 | 128 |
| Head dim | 512 | 512 |
| KV heads | 1 | 1 |
| Q LoRA rank | 1024 | 1536 |
| Routed experts | 256 | 384 |
| Active experts | 6 | 6 |
| Shared experts | 1 | 1 |
| Expert intermediate | 2048 | 3072 |
| CSA top-k | 512 | 1024 |
| Output groups | 8 | 16 |
| mHC copies | 4 | 4 |
| Context | 1M | 1M |
| DSpark Markov rank | 256 | 512 |

---

## 3. Pro가 단순 5.6× 큰 모델은 아니다

총 parameter 비율:

```text
1.6T / 284B ≈ 5.6×
```

active parameter 비율:

```text
49B / 13B ≈ 3.8×
```

그러나 runtime cost ratio는 둘 중 어느 숫자와도 정확히 같지 않다.

Pro는 추가로:

- layer 42% 증가
- hidden width 75% 증가
- attention heads 2×
- CSA top-k 2×
- expert intermediate 1.5×

이므로 attention/cache/retrieval cost까지 함께 커진다.

---

## 4. Weight residency

Sparse MoE는 전체 weight를 매 token 읽지는 않지만 **모델 전체 weight는 GPU/host/storage 어딘가에 resident해야 한다.**

Flash는 native low precision에서 single-node multi-GPU deployment가 현실적인 범위다.

Pro는 약 1.6T target weight 때문에:

- GPU aggregate HBM
- EP sharding
- weight loading time
- startup memory peak

가 훨씬 큰 문제다.

따라서 Pro에서는 expert weight filter / sharded load / fast shared storage가 operationally 중요하다.

---

## 5. Attention runtime

### Flash CSA

```text
64 Q heads
index top-k 512
```

### Pro CSA

```text
128 Q heads
index top-k 1024
```

Pro는 단순 width 증가 외에도 sparse retrieval set 자체가 두 배다.

따라서 long-context decode에서 attention-side bandwidth/compute difference가 active MoE parameter ratio보다 커질 수 있는 구간이 생긴다.

---

## 6. HCA는 Pro에서도 중요한 safety valve

Pro의 1M history를 CSA만으로 처리하면 retrieval candidate/cost가 커진다.

HCA의 128x coarse memory가 interleave되기 때문에 모든 global-memory layer가 top-1024 retrieval을 수행하지 않는다.

즉 Pro scale에서 HCA의 역할은 더 중요하다.

---

## 7. MoE communication

Flash:

```text
256 experts
```

Pro:

```text
384 experts
```

둘 다 token당 6 experts를 쓰므로 activation sparsity는 유지된다.

하지만 Pro는:

- expert tensor가 더 큼
- model shard 수가 많아지기 쉬움
- expert placement domain이 넓어짐

때문에 cross-GPU/cross-node token dispatch가 더 큰 bottleneck이 될 가능성이 높다.

---

## 8. Single-node vs multi-node

### Flash

현재 vLLM recipes는 여러 8-GPU class NVIDIA/AMD system에서 single-node profile을 제시한다.

예:

- H200 ×8
- B200 ×8
- B300 ×8
- MI325X / MI355X ×8

### Pro

aggregate weight memory와 dense component size 때문에 high-memory 8-GPU system 또는 multi-node topology가 더 현실적이다.

구체적인 가능 여부는 checkpoint precision / backend / HBM capacity를 기준으로 계산해야 한다.

---

## 9. DSpark benefit도 다를 수 있다

Pro target verification은 Flash보다 비싸다.

따라서 동일 acceptance를 가정하면 successful speculative token 한 개가 절약하는 target compute의 가치가 더 크다.

반대로 Pro DSpark drafter도 더 크고 Markov rank가 512이므로 draft cost가 올라간다.

결론:

```text
Flash optimal K ≠ Pro optimal K
```

일 가능성이 높다.

checkpoint의 `block_size=5`와 runtime speculative token budget을 그대로 동일하게 고정하기보다 각각 profile해야 한다.

---

## 10. Workload 선택

### Flash가 유리한 경향

- high-QPS agent
- interactive coding
- latency-sensitive tools
- moderate reasoning budget
- cost-efficient long context

### Pro가 필요한 경향

- hard reasoning
- knowledge-heavy task
- difficult long-horizon agents
- high-value low-QPS workload
- quality priority

하지만 최신 Flash-0731의 agentic post-training이 크게 좋아졌기 때문에 `단순 task = Flash, agent = Pro`로 나누는 것도 부정확하다.

실제 benchmark/workload로 routing해야 한다.

---

## 11. Reasoning effort까지 포함한 routing

동일 checkpoint도 reasoning effort에 따라 compute가 달라진다.

예를 들어 platform routing을 설계할 때:

```text
Flash low
Flash high
Flash max
Pro low
Pro high
Pro max
```

는 사실상 서로 다른 service-time distribution을 가진 workload class다.

따라서 model name만으로 autoscaling/resource allocation을 결정하지 말고 reasoning effort까지 telemetry label로 남기는 것이 좋다.

---

## 12. 성능 비교 시 normalization

Flash vs Pro benchmark에서 반드시 맞출 것:

- reasoning effort
- temperature/top-p
- max output tokens
- tool harness
- context length
- DSpark on/off
- concurrency
- TP/EP topology
- GPU type

특히 agent benchmark는 harness 영향이 크다.

최신 checkpoint quality 향상과 serving engine throughput 향상을 같은 benchmark 표에서 섞지 않는다.

---

## 13. 배포 판단 matrix

| 질문 | Flash | Pro |
|---|---|---|
| Weight residency가 쉬운가 | 상대적으로 쉬움 | 매우 어려움 |
| Single-node 가능성 | 높음 | hardware에 강하게 의존 |
| EP 필요성 | 높음 | 매우 높음 |
| Long-context attention cost | 낮은 편 | 더 큼 |
| Agent quality | 매우 강함 | 더 강한 frontier target |
| Cost/QPS | 유리 | 불리 |
| Hard reasoning headroom | 제한적 | 높음 |

---

## 14. 현재 checkpoint 선택 원칙

새 배포라면 특별한 reproduction 목적이 없는 한:

```text
Flash → Flash-0731
Pro   → Pro-0813
```

을 기본 checkpoint로 본다.

Preview를 유지할 이유:

- architecture paper reproduction
- MTP baseline comparison
- DSpark before/after A/B
- historical performance regression

정도다.

---

## 15. 출처

- DeepSeek V4 official: https://deepseek.com/en/news/v4-preview/
- Flash-0731: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- Pro-0813: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813
- vLLM Flash recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- vLLM Pro recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro
