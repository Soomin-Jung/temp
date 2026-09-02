# 08. Production Tuning Runbook — 처음 보는 모델을 운영 수준까지 끌어올리는 순서

## 1. 목표

이 Runbook의 목표는 최고 benchmark 숫자를 찾는 것이 아니다.

목표는 다음 조건을 만족하는 production candidate를 만드는 것이다.

- correctness 유지
- workload tail에서 SLO 안정
- KV/state headroom 확보
- preemption 억제
- Prefill/Decode queue balance
- GPU compute 활용
- fabric headroom
- 반복 가능한 configuration
- regression을 설명할 수 있는 observability

## 2. Stage 0 — Architecture inventory

모델 config와 serving implementation에서 먼저 읽는다.

Attention:

- layer count
- attention type per layer
- Q heads / KV heads
- head dimension
- local/sliding/global structure
- latent KV 여부
- sparse/compression 여부

Recurrent:

- state shape
- state dtype
- conv state
- recurrent layer count
- MTP state 증가 방식

MoE:

- expert count
- top-k
- shared expert
- expert size
- EP support

Speculative:

- MTP/draft head 존재
- runtime activation 방식
- supported K
- graph compatibility

출력은 옵션 목록이 아니라 resource model이어야 한다.

예:

| 항목 | Scaling |
|---|---|
| Weight | model constant / TP shard |
| Full KV | context linear |
| Recurrent state | request fixed |
| MTP state | K dependent |
| EP traffic | active tokens dependent |
| P/D transfer | cache representation dependent |

## 3. Stage 1 — Hardware topology inventory

확인:

- GPU SKU
- HBM capacity
- HBM bandwidth
- GPU count/node
- NVLink generation
- NVSwitch 여부
- PCIe topology
- NIC topology
- scale-out network
- NUMA relation

같은 GPU SKU라도 topology가 다르면 좋은 parallelism이 달라진다.

## 4. Stage 2 — Microbenchmark baseline

framework 전에 hardware ceiling을 잡는다.

Single GPU:

- GEMM representative sizes
- HBM bandwidth
- memory copy

Multi GPU:

- P2P bandwidth
- NCCL all-reduce
- all-gather
- reduce-scatter
- send/recv

Multi node라면:

- RDMA bandwidth
- GDR path
- NCCL inter-node

목적은 framework를 hardware peak와 직접 비교하려는 것이 아니라 같은 topology의 현실적인 empirical ceiling을 확보하는 것이다.

## 5. Stage 3 — Minimal correctness serving

최적화를 모두 끄거나 보수적으로 시작한다.

예:

- no speculative decoding
- graph NONE 또는 가장 안정적인 mode
- conservative MBT
- prefix cache off
- async feature off
- single stable topology

검증:

- output correctness
- streaming
- parser/tool call
- long context
- P/D transfer correctness
- restart/recovery

성능 튜닝 전에 correctness gate를 통과한다.

## 6. Stage 4 — Startup memory accounting

매 configuration에서 startup log를 저장한다.

반드시 기록:

- weights memory
- runtime memory
- graph memory
- KV/state pool
- block size
- number of blocks
- max concurrency estimate
- resolved graph mode
- attention backend

이 정보가 없으면 성능이 오른 대신 capacity가 줄어든 것을 놓치기 쉽다.

## 7. Stage 5 — Workload baseline

실제 production distribution 또는 대표 synthetic distribution을 만든다.

Input:

- p50
- p90
- p95
- p99
- maximum supported

Output:

- short
- medium
- long
- reasoning-heavy 별도

Load:

- fixed concurrency
- open-loop RPS

Workload를 한 숫자로 축약하지 않는다.

## 8. Stage 6 — Prefill main effects

고정 topology에서 먼저 세 축을 본다.

A. MBT

목적: large-batch compute knee point.

B. MTP on/off

목적: Prefill side-effect와 state/capacity 변화.

C. CUDA Graph

목적: NONE vs PIECEWISE의 launch/dispatch 효과.

관측:

- TTFT p50/p90/p95
- Prefill phase latency
- prompt tok/s
- queue
- CPU
- SM/Tensor/DRAM
- KV/state capacity

## 9. Stage 7 — Prefill interactions

순서 예:

1. MBT x MTP
2. MTP x CUDA Graph
3. MBT x CUDA Graph
4. survivor ABC

목표는 모든 조합 완주가 아니라 main effect로 제거되지 않은 후보 간 상호작용을 찾는 것이다.

## 10. Stage 8 — Decode sequence ceiling

Decode에서는 max-num-seqs를 main factorial에 넣기 전에 non-binding 지점을 찾는다.

예:

- 256
- 512
- 1024

short-context high-concurrency workload로 saturation을 본다.

512와 1024가 동일하면 이후 512를 고정한다.

hybrid recurrent model + full Decode graph에서는 recurrent-state block constraint도 함께 본다.

## 11. Stage 9 — Decode main effects

A. MBT

예:

- 512
- 1024
- 2048
- 4096 optional

B. MTP K

- OFF
- K1
- K2
- K3

C. CUDA Graph

- NONE
- FULL_DECODE_ONLY
- FULL_AND_PIECEWISE

관측:

- TPOT
- ITL
- generation tok/s
- acceptance
- mean accepted length
- running seqs
- KV/state footprint
- CPU
- HBM
- NVLink

## 12. Stage 10 — Decode interactions

우선순위:

1. MBT x MTP
2. MTP x CUDA Graph
3. MBT x CUDA Graph

MTP K는 token envelope와 graph shape를 동시에 바꾸므로 graph와 독립적인 knob가 아니다.

## 13. Stage 11 — CUDA Graph capture sizing

main configuration winner를 먼저 고른다.

그 뒤에만:

- auto
- smaller max
- larger max
- custom dense bucket

을 비교한다.

필수 관측:

- runtime batch distribution
- graph coverage
- padding waste
- capture startup
- graph memory
- KV capacity
- TPOT

capture list 자체를 먼저 튜닝하지 않는다.

## 14. Stage 12 — Context capacity surface

winner 후보를 context bucket 전체에서 다시 본다.

질문:

- context 증가 시 TTFT slope?
- TPOT slope?
- KV usage slope?
- maximum concurrency?
- P/D transfer bytes/time?
- MTP state overhead 비율?

short-context winner가 long-context winner인지 확인한다.

## 15. Stage 13 — Parallelism sweep

TP/PP/EP를 처음부터 모든 값으로 돌리지 않는다.

먼저 resource reason을 쓴다.

예:

> TP2를 보는 이유는 long-context KV/state capacity를 늘리기 위해서다.

또는:

> TP1을 보는 이유는 Decode collective latency를 제거하기 위해서다.

각 parallelism candidate는 가설이 있어야 한다.

## 16. Stage 14 — Fabric isolation

TP와 P/D transport를 분리한다.

- idle
- Prefill TP-only
- Decode TP-only
- P/D transfer-focused
- combined production-shape

비교:

- NVLink TX/RX
- PCIe TX/RX
- NCCL duration
- P2P status
- errors
- framework throughput

## 17. Stage 15 — P:D ratio

P와 D가 각각 튜닝된 뒤 GPU ratio를 맞춘다.

관측:

- Prefill queue
- Decode queue
- router dispatch
- TTFT
- TPOT
- active KV

Prefill queue만 증가하면 P capacity를 늘린다.

Decode queue만 증가하면 D capacity를 늘린다.

둘 다 안정이면 ratio가 workload arrival mix와 대체로 맞는 것이다.

## 18. Stage 16 — Open-loop saturation

fixed concurrency winner를 실제 arrival-rate curve로 검증한다.

RPS를 올리면서:

- throughput
- TTFT
- TPOT
- queue
- KV usage
- preemption

을 기록한다.

production capacity는 throughput plateau 직전이 아니라 필요한 headroom을 둔 지점으로 잡는다.

## 19. Stage 17 — Long soak

짧은 benchmark로 안 보이는 것:

- memory leak
- allocator fragmentation
- stale graph/cache
- connector resource leak
- error accumulation
- thermal/power throttling
- routing imbalance

때문에 long soak가 필요하다.

## 20. Stage 18 — Failure and recovery

성능 winner도 recovery가 깨지면 production winner가 아니다.

검증:

- process restart
- pod recycle
- engine re-registration
- P/D transfer recovery
- stale request fencing
- cache invalidation
- traffic drain

## 21. Stage 19 — Final scorecard

최종 후보는 다음 축으로 비교한다.

| 축 | 대표 metric |
|---|---|
| Prefill latency | TTFT p95 |
| Prefill throughput | prompt tok/s |
| Decode latency | TPOT/ITL p95 |
| Decode throughput | generation tok/s |
| Queue | waiting/queue p95 |
| Capacity | max concurrency |
| Memory stability | KV usage/preemption |
| SD | acceptance/commit length |
| Compute | SM/Tensor/DRAM |
| Fabric | NVLink/PCIe/NCCL |
| Reliability | error counters |
| Recovery | failure test |

## 22. Production winner의 정의

다음은 winner가 아니다.

- synthetic throughput 최고
- p50 latency 최고
- GPU Util 최고
- KV utilization 최고
- NVLink GB/s 최고
- MTP acceptance 최고

winner는 실제 workload distribution에서:

- SLO를 지키고
- queue가 안정되고
- preemption이 낮고
- 필요한 headroom이 있고
- failure recovery가 되고
- configuration을 설명할 수 있는

조합이다.

## 23. Regression 대응

새 vLLM/GPU/model revision에서 성능이 바뀌면 전체 튜닝을 처음부터 다시 하지 않는다.

먼저 비교:

1. resolved backend
2. graph mode
3. block geometry
4. KV/state capacity
5. scheduler defaults
6. kernel selection
7. communication path

그 뒤 어느 layer의 cost equation이 바뀌었는지 좁힌다.

## 24. 배포 전 체크리스트

### Model

- architecture 확인
- dtype 확인
- context 확인
- parser/template correctness
- MTP support

### Memory

- startup accounting 저장
- max concurrency 확인
- long-context test
- preemption test

### Runtime

- graph resolved mode
- capture size
- scheduler knobs
- prefix/cache policy

### Fabric

- GPU topology
- NCCL/P2P baseline
- P/D transfer path
- error counters

### Observability

- P/D role label
- TTFT/TPOT
- throughput
- queue
- KV
- GPU
- NVLink

### Reliability

- restart
- recycle
- long soak
- failure injection

## 25. 본질

> Production tuning은 옵션 조합 탐색이 아니라 Architecture Resource Model → Hardware Mapping → Phase-specific Tuning → Interaction Test → Saturation → Failure Recovery의 순서로 위험을 제거하는 과정이다.
