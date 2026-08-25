# 05. MTP, DSpark & Speculative Decoding

작성일: 2026-08-25

## 1. 먼저 세 층을 분리한다

DeepSeek-V4의 speculative decoding을 이해할 때 가장 많이 섞이는 개념은 다음 세 가지다.

```text
1) Base-model MTP objective
2) checkpoint에 동봉된 draft module
3) runtime verification scheduler
```

이 셋은 같지 않다.

---

## 2. V4 backbone은 MTP lineage를 유지한다

V4 technical report는 V3의 Multi-Token Prediction(MTP) 설정을 그대로 유지한다고 설명한다.

기본 의미:

```text
main hidden state
   ↓
auxiliary next-n prediction module
   ↓
future token prediction objective
```

MTP의 1차 목적은 training 중 미래 token prediction signal을 추가하는 것이다.

추론 시에는 이 auxiliary module을 speculative drafter로 재사용할 수 있다.

V4 Preview 계열에서는 이 전통 MTP path가 직접적인 serving drafter였다.

---

## 3. DSpark는 MTP-1의 단순 rename이 아니다

DSpark는 별도의 production speculative-decoding framework다.

논문의 핵심 목표:

- 병렬 draft generation의 throughput
- autoregressive dependency modeling의 draft quality
- request/load에 따라 verification length를 동적으로 결정

을 동시에 가져가는 것이다.

논문이 제안하는 큰 구조:

```text
Target hidden features
       ↓
Parallel draft backbone
       ↓
Sequential correction module
(Markov or RNN)
       ↓
Draft block
       ↓
Confidence estimation
       ↓
Adaptive verification length
       ↓
Target verification
```

---

## 4. 최신 checkpoint와 DSpark

### Flash-0731

config:

```text
dspark_block_size       = 5
dspark_target_layer_ids = [40,41,42]
dspark_markov_rank      = 256
```

### Pro-0813

```text
dspark_block_size       = 5
dspark_target_layer_ids = [58,59,60]
dspark_markov_rank      = 512
```

즉 drafter는 target의 마지막 3개 layer feature를 사용한다.

Pro는 larger rank Markov correction을 사용한다.

---

## 5. Parallel backbone이 필요한 이유

일반 autoregressive drafter는:

```text
t1 → t2 → t3 → t4 → t5
```

처럼 draft token도 순차적으로 생성한다.

draft model이 작아도 5번의 sequential step이 필요하다.

반면 완전 병렬 drafter는:

```text
[t1,t2,t3,t4,t5] simultaneously
```

를 노릴 수 있지만 뒤쪽 token이 앞쪽 draft 결과를 보지 못해 suffix quality가 급격히 나빠질 수 있다.

DSpark는:

```text
parallel backbone
+
cheap sequential correction
```

으로 이 trade-off를 완화한다.

---

## 6. Markov head

production checkpoint의 sequential correction은 Markov head를 사용한다.

개념적으로 각 draft position의 base logit에 직전 sampled draft token 기반 low-rank bias를 더한다.

```math
logit_k = U_k + M(x_{k-1})
```

low-rank factorization:

```text
previous token embedding/index
      ↓
rank-r projection
      ↓
vocabulary correction
```

Flash rank 256, Pro rank 512다.

이 구조의 장점:

- full AR drafter보다 싸다
- purely independent parallel heads보다 intra-block dependency가 있다
- matrix rank를 작게 유지해 sequential overhead를 줄인다

---

## 7. RNN variant와 Markov variant

DSpark paper는 longer prefix state를 담는 recurrent variant도 분석한다.

하지만 production checkpoint는 Markov-style correction을 기본으로 둔다.

이유는 결국 Pareto 문제다.

```text
more sequential modeling
→ higher acceptance
→ but more draft latency
```

Speculative decoding은 draft accuracy 자체가 목적이 아니라 **target 포함 end-to-end throughput/latency**가 목적이다.

따라서 조금 더 정확한 drafter가 항상 더 빠른 것은 아니다.

---

## 8. Confidence head

DSpark의 또 다른 핵심은 draft마다 `몇 token까지 검증할 가치가 있는지` 추정하는 것이다.

뒤쪽 token으로 갈수록 앞 token 하나가 reject되면 suffix 전체의 검증이 낭비된다.

그래서 prefix survival probability를 추정한다.

개념:

```text
P(t1 accepted)
P(t1,t2 accepted)
P(t1,t2,t3 accepted)
...
```

그리고 target verify cost와 accepted-token benefit를 비교해 적절한 verification length를 고른다.

---

## 9. Confidence-scheduled verification

고정 K speculative decoding:

```text
항상 K개 draft
→ 항상 K개 verify
```

DSpark full framework:

```text
request-specific confidence
+
engine throughput/load profile
       ↓
verify length 결정
```

따라서 동일 checkpoint라도 server load와 hardware profile에 따라 optimal K가 달라질 수 있다.

이게 `confidence-scheduled`의 핵심이다.

---

## 10. 논문 결과의 의미

DSpark paper는 production DeepSeek-V4 serving에서 기존 MTP-1 baseline 대비 matched throughput 조건에서 per-user generation speed를 약 60~85% 높였다고 보고한다.

이 숫자는 특정 production environment 결과이므로 그대로 우리 환경에 대입하면 안 된다.

하지만 구조적 의미는 강하다.

> **speculative decoding의 최적화 대상은 acceptance rate가 아니라 latency-throughput Pareto frontier다.**

---

## 11. Checkpoint namespace 함정

최신 checkpoint에서 DSpark tensor가 `mtp.*` namespace 아래 들어갈 수 있다.

예:

```text
mtp.2.markov_head.*
mtp.2.confidence_head.*
```

하지만 namespace가 MTP라고 해서 일반 MTP algorithm으로 실행하는 것은 틀리다.

확인할 것:

```text
dspark_* config fields
markov_head tensors
confidence_head tensors
target layer IDs
```

이 존재하면 DSpark-aware loader/runtime을 사용해야 한다.

---

## 12. vLLM runtime과 논문 DSpark를 분리한다

vLLM에서 `method=dspark`가 존재한다고 해서 해당 버전이 DSpark paper 전체 기능을 모두 구현했다는 뜻은 아니다.

역사적으로 implementation 단계가 나뉘었다.

```text
checkpoint load
→ fixed-K Markov draft
→ confidence head wiring
→ adaptive verification scheduler
```

따라서 version 검증 시 기능 단위를 확인한다.

- DSpark checkpoint load 여부
- Markov correction 여부
- confidence head 사용 여부
- adaptive verification 여부
- CUDA graph 제약
- PP/LoRA/logprobs compatibility

상세 runtime 버전 분석은 기존 문서 참고:

- [DSpark checkpoint와 vLLM 구현 차이](2026-08-18-dspark-checkpoint-and-vllm-implementation.md)

---

## 13. Serving에서 반드시 측정할 metric

단순 output tok/s만 보면 원인을 모른다.

### Draft side

- draft latency
- draft tokens/request
- confidence distribution

### Verification side

- accepted length
- acceptance rate per position
- rejected suffix length
- target verify latency

### End-to-end

- TTFT
- TPOT
- user-perceived tok/s
- aggregate tok/s
- batch size / concurrency
- GPU utilization

### Workload split

- code
- reasoning
- prose
- tool-call JSON
- long-context agent loop

acceptance는 workload에 따라 크게 달라질 수 있다.

---

## 14. CUDA Graph와의 관계

Speculative decoding은 한 iteration에서 실행 path가 더 복잡하다.

```text
draft
→ sample/correct
→ target verify
→ accept/reject
→ replay/state update
```

adaptive verification까지 들어가면 verify length가 request마다 달라진다.

그래서 engine이 이를 efficient CUDA graph shape로 어떻게 bucketize/capture하는지가 매우 중요하다.

`DSpark 지원` 여부와 `DSpark가 CUDA graph에서 효율적으로 동작`하는지는 별개의 질문이다.

---

## 15. 본질

MTP부터 DSpark까지의 evolution은 이렇게 볼 수 있다.

```text
MTP
"미래 token을 하나 더 예측해보자"
       ↓
Speculative MTP
"그 예측을 draft로 쓰자"
       ↓
DSpark
"여러 미래 token을 병렬로 예측하되,
dependency와 verification cost까지 함께 최적화하자"
```

---

## 16. 출처

- DeepSeek-V4 Technical Report: https://arxiv.org/abs/2606.19348
- DSpark paper: https://arxiv.org/abs/2607.05147
- Flash-0731 model/config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- Pro-0813 model/config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813
- Runtime 상세: [2026-08-18-dspark-checkpoint-and-vllm-implementation.md](2026-08-18-dspark-checkpoint-and-vllm-implementation.md)
