# LLM Inference Serving Optimization — Mental Model

업데이트: 2026-09-03 KST

이 모듈의 목적은 특정 모델의 "좋은 옵션값"을 외우는 것이 아니다.

목표는 새로운 GPU, 새로운 vLLM version, 새로운 model architecture를 만났을 때 다음 질문을 스스로 풀 수 있는 **운영 수준의 최적화 눈썰미**를 만드는 것이다.

> 이 workload의 병목은 지금 어디에 있고, 내가 바꾸려는 knob는 어떤 자원·스케줄·kernel shape·state lifecycle을 실제로 바꾸는가?

## 1. 최적화를 보는 여섯 축

```text
Workload
   ↓
Scheduler
   ↓
Model architecture
   ↓
Kernel / CUDA Graph
   ↓
Memory / KV / recurrent state
   ↓
GPU / interconnect topology
```

최적화는 이 여섯 축의 교차점에서 일어난다.

### Workload

- input length distribution
- output length distribution
- concurrency
- arrival burst
- prefix reuse
- tool/agent turn pattern
- latency SLO

### Scheduler

- `max_num_batched_tokens`
- `max_num_seqs`
- chunked prefill
- long prefill threshold
- scheduling policy
- speculative token reservation

### Model

- Dense
- MoE
- MLA
- GQA
- Mamba/SSM
- GDN/KDA hybrid
- multimodal encoder
- MTP/drafter

### Execution

- eager
- torch.compile
- CUDA Graph
- kernel backend
- fused op
- graph capture shape

### State / Memory

- weight
- KV cache
- recurrent state
- CUDA Graph pool
- temporary workspace
- activation
- fragmentation

### Hardware

- HBM capacity
- HBM bandwidth
- Tensor Core throughput
- SM occupancy
- PCIe
- NVLink/NVSwitch
- IB/RDMA
- NUMA

## 2. 가장 먼저 해야 할 분류: compute-bound인가 memory-bound인가

### Decode

일반적인 작은 decode batch는 weight를 읽는 비용에 비해 token당 연산량이 낮아 **memory-bandwidth bound**가 되기 쉽다.

batch/concurrency가 커지면 같은 weight load를 더 많은 token이 공유하면서 arithmetic intensity가 올라가고 compute-bound 방향으로 이동한다.

이 변화 때문에 speculative decoding도 concurrency에 따라 이득이 달라진다.

### Prefill

긴 prompt의 matrix multiply는 token dimension이 커서 GPU compute를 더 잘 채운다.

하지만 long context에서는 attention/state scan, KV write, temporary buffer, chunk scheduling이 함께 들어오므로 단순 "prefill = compute-bound" 하나로 끝내면 안 된다.

## 3. MBT는 단순 batch size가 아니다

`max_num_batched_tokens`는 **한 scheduler iteration의 token budget**이다.

작은 MBT:
- decode interference 감소 가능
- tail latency 보호
- prefill을 더 잘게 자름
- iteration 수 증가
- kernel이 작아져 GPU 효율 저하 가능

큰 MBT:
- prefill chunk가 커짐
- 큰 GEMM/kernel shape
- throughput 향상 가능
- active decode가 prefill에 밀릴 수 있음
- workspace/graph shape 부담 증가

즉 최적 MBT는 상수가 아니다.

```text
optimal MBT
  = f(model, GPU, P/D role, context, concurrency, SD, graph mode)
```

vLLM 0.28이 default MBT를 16K로 올렸어도, P/D decode pool이 16K를 그대로 써야 한다는 뜻은 아니다.

## 4. P/D에서 왜 같은 profile을 쓰면 안 되는가

### Prefill pool

목표:
- long prompt를 높은 GPU 효율로 처리
- TTFT와 aggregate prompt throughput의 균형

선호 방향:
- 상대적으로 큰 MBT
- chunked prefill
- compute-efficient shape
- sufficient parallelism

### Decode pool

목표:
- active sequence의 반복 step을 빠르게 처리
- ITL/TPOT 안정성

선호 방향:
- 작은/중간 scheduler budget 후보
- decode-friendly graph mode
- 높은 KV/state capacity
- SD verification cost 관리

따라서 P/D는 같은 model weight를 쓰더라도 **서로 다른 serving workload**다.

## 5. State-based model은 KV-only 모델과 다르다

Mamba, GDN, KDA 같은 recurrent/state 기반 layer가 섞이면 memory capacity를 KV block만으로 계산하면 틀린다.

```text
per-request state
  = attention KV
  + recurrent state
  + conv/state cache
  + alignment/padding cost
```

또 runtime은 state를 arbitrary token granularity로 항상 처리할 수 있는 게 아니다.

예를 들어 특정 hybrid model path에서 state block size가 scheduler budget보다 커야 하는 assertion이 있을 수 있다.

따라서 MBT를 "낮추면 decode가 무조건 좋아진다"가 아니라 **model state block / kernel minimum shape / cache alignment constraint 아래에서 낮춘다.**

## 6. Speculative Decoding은 decode request당 1 token 모델을 바꾼다

일반 decode는 request당 보통 다음 token 1개를 scheduler step에서 진행한다.

speculative decoding에서는 target verification을 위해 draft slot이 추가된다.

vLLM에서는 runtime version에 따라 speculative drafting headroom accounting이 다르다.

- v0.26/v0.27: `max_num_seqs` 기준 worst-case drafting headroom을 정적으로 선점하는 방식
- v0.28: scheduler logical token budget과 physical input-slot budget을 분리하고 실제 scheduled request 수에 따라 drafting slot을 동적으로 차감하는 방식

따라서 `raw MBT - slots × max_num_seqs`를 모든 version에 그대로 적용하지 않는다. method별 `max_num_new_slots_for_drafting`와 version별 scheduler accounting을 함께 본다.

정확한 source-level 계산은 [vLLM Speculative Decoding Token Budget Deep Dive](../speculative-decoding/2026-09-03-vllm-speculative-decoding-token-budget-deep-dive.md)를 따른다.

따라서 K를 늘릴 때:

- acceptance가 올라가면 step 수는 줄 수 있음
- verification batch는 커짐
- compute pressure는 증가
- effective scheduler token budget은 줄 수 있음
- CUDA Graph uniform decode query length도 `1 + K`가 됨

그래서 `num_speculative_tokens`와 MBT를 따로 튜닝하면 안 된다.

상세:
[Scheduler Budget, Speculative Decoding & CUDA Graph](01-scheduler-budget-spec-decode-cudagraph.md)

## 7. CUDA Graph는 "켜기/끄기"가 아니다

현대 vLLM은 workload shape에 따라 여러 graph mode를 제공한다.

- `NONE`
- `PIECEWISE`
- `FULL`
- `FULL_DECODE_ONLY`
- `FULL_AND_PIECEWISE`

P/D decode pool에서는 `FULL_DECODE_ONLY`가 중요한 후보가 된다.

이유:
- decode는 uniform batch가 많음
- prefill/mixed graph memory를 줄일 수 있음
- capture time과 graph memory를 줄이면서 decode launch overhead를 줄일 수 있음

하지만 model의 attention/state backend가 full graph를 어디까지 지원하는지 먼저 봐야 한다.

## 8. Weight parallelism과 state capacity를 같이 본다

TP를 줄이면:
- model weight duplication이 늘 수 있음
- replica 수는 늘릴 수 있음
- per-engine weight footprint가 커짐
- KV/state free HBM이 줄 수 있음

TP를 늘리면:
- weight shard가 작아짐
- KV/state capacity가 늘 수 있음
- collective cost 증가
- request당 더 많은 GPU를 점유
- replica 수 감소

long-context workload에서는 weight FLOPs만 보고 TP1을 고르는 것이 위험하다.

반대로 decode가 아주 짧고 high-RPS라면 작은 TP replica의 독립성이 더 나을 수 있다.

즉:

```text
parallelism decision
  = weights
  + state capacity
  + collective cost
  + replica count
  + workload concurrency
```

## 9. GPU 세대가 바뀌면 같은 옵션의 의미도 바뀐다

H100/H200/B300 사이에서는 단순 FLOPS만 달라지는 게 아니다.

- HBM capacity
- HBM BW
- Tensor Core ratio
- graph capture size practicality
- FP8/FP4 support
- NVLink generation
- intra-node topology
- kernel availability

가 달라진다.

따라서 model profile을 GPU-independent constant로 보지 않는다.

```text
Model Profile
  × Hardware Profile
  × Runtime Version
  × Workload Profile
  -> validated Serving Profile
```

## 10. 운영 튜닝 순서

### Stage 0 — correctness

- eager
- no speculative
- minimal concurrency
- fixed golden set

### Stage 1 — memory envelope

- weight HBM
- KV/state
- graph pool
- max concurrency
- fragmentation headroom

### Stage 2 — phase baseline

- prefill-only
- decode-heavy
- mixed

### Stage 3 — scheduler

- MBT
- max_num_seqs
- chunked prefill
- long threshold

### Stage 4 — graph/kernel

- eager vs graph
- graph mode
- capture sizes
- backend

### Stage 5 — SD

- K
- acceptance
- effective budget
- ITL/throughput

### Stage 6 — topology

- TP/DP/EP
- P:D ratio
- multi-node

### Stage 7 — soak

- real context distribution
- burst
- cancellation
- restart
- cache hit/miss

한 번에 여러 축을 바꾸지 않는다.

## 11. 튜닝 결과를 기록하는 형태

나쁜 기록:

```text
MBT=2048가 제일 빠름
```

좋은 기록:

```text
model:
runtime:
GPU:
topology:
P/D role:

input p50/p90/p99:
output:
concurrency:
prefix hit:

MBT:
max seqs:
SD K:
graph mode:

TTFT p50/p95/p99:
ITL p50/p95/p99:
tokens/s:
GPU util:
HBM:
acceptance:

bottleneck interpretation:
promotion / reject reason:
```

값보다 **왜 그 값이 이 workload에서 유리했는지**를 남긴다.

## 12. 핵심 사고법

최적화 knob를 볼 때 항상 세 질문을 한다.

1. **이 값이 scheduler/kernel/memory에서 실제 무엇을 바꾸는가?**
2. **그 변화가 현재 병목을 줄이는가, 다른 병목으로 옮기는가?**
3. **workload나 GPU가 바뀌면 결론이 뒤집힐 조건은 무엇인가?**

이 세 질문에 답하지 못하면 아직 "튜닝"이 아니라 "숫자 sweep" 단계다.
