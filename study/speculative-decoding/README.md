# Speculative Decoding Study

Speculative decoding을 **알고리즘 → vLLM runtime → scheduler/MBT → 모델별 구현** 순서로 정리한다.

## 읽는 순서

1. [전통 SD, MTP, DFlash, DSpark의 구조적 차이](00-foundations-and-method-lineage.md)
   - proposer와 target verification의 기본 구조
   - MTP, EAGLE, DFlash, DSpark, model-free 방식의 차이
   - acceptance와 latency/throughput trade-off

2. [vLLM SD Method Taxonomy & Selection Guide](01-vllm-method-taxonomy-and-selection.md)
   - proposer / candidate topology / verifier / runtime adaptation의 4축 taxonomy
   - Draft Model, PARD, TLI, EAGLE, EAGLE-3, Medusa, MLP, MTP, DFlash, DSpark, N-gram, Suffix
   - Dynamic SD와 Adaptive Verification을 proposer와 분리
   - v0.28 actual proposer dispatch와 support maturity
   - workload / concurrency / checkpoint 조건별 selection decision tree

3. [vLLM Speculative Decoding Token Budget Deep Dive](02-vllm-token-budget-and-slot-topology.md)
   - `max_num_batched_tokens`, `max_num_scheduled_tokens`, `max_num_seqs`
   - method별 `max_num_new_slots_for_drafting`
   - v0.26/v0.27 static worst-case reservation
   - v0.28 adaptive dual-budget accounting
   - EAGLE3 / P-EAGLE / DFlash / DSpark / MTP / n-gram / draft model / PARD 비교
   - DeepSeek-V4-Flash-0731 + DSpark K=7 + MBT 4K 실패 계산
   - chunked prefill, CUDA Graph, KV lookahead와의 경계

4. [Scheduler Budget, Speculative Decoding & CUDA Graph](../llm-serving-optimization/01-scheduler-token-budget.md)
   - SD를 더 넓은 inference-serving optimization 문맥에서 해석
   - MBT/K/graph mode joint tuning
   - P/D role별 실험 축과 관측 지표

## 모델별 구현

- [DeepSeek-V4 MTP / DSpark](../../models/deepseek-v4/05-mtp-dspark-and-speculative-decoding.md)
- [DeepSeek-V4 DSpark checkpoint와 vLLM 구현](../../models/deepseek-v4/notes/2026-08-18-dspark-checkpoint-and-vllm-implementation.md)
- [GLM-5.2 MTP & Speculative Decoding](../../models/glm-5.2/05-mtp-and-speculative-decoding.md)
- [Kimi-K3 Post-Training / MTP / Speculative Decoding](../../models/kimi-k3/07-posttraining-agentic-rl-and-speculative.md)

## 핵심 mental model

```text
max_num_batched_tokens
= per-iteration input-slot envelope

max_num_scheduled_tokens
= scheduler가 직접 issue할 수 있는 logical token budget

max_num_new_slots_for_drafting
= scheduler 이후 drafter가 추가할 수 있는 request당 input slots
```

v0.26/v0.27의 parallel SD는 drafting headroom을 `max_num_seqs` 기준으로 정적 선점한다.

```math
T = B - SM
```

v0.28은 scheduler token budget과 physical input budget을 분리하고 실제 scheduled request마다 drafting slots를 차감한다.

```math
\sum_i n_i \le T
```

```math
\sum_i (n_i + S) \le B
```

이 차이는 특히 **low-concurrency + long-context chunked prefill + parallel speculative drafting**에서 TTFT와 실제 usable MBT를 크게 바꾼다.
