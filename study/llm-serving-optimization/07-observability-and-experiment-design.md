# 07. Observability와 Experiment Design — 증상을 원인으로 번역하는 법

## 1. 대시보드의 목적

대시보드는 예쁜 그래프 모음이 아니다.

운영 수준의 대시보드는 네 질문에 답해야 한다.

- 어느 phase가 느린가?
- 어느 resource가 binding인가?
- 어느 knob가 실제 work를 바꾸는가?
- 병목이 어디로 이동했는가?

metric layer는 Request/Queue, Scheduler, Prefill, Decode, KV/State, CUDA Runtime, GPU Compute, HBM, Fabric, Health로 분리한다.

## 2. Request layer

필수 관측:

- request rate
- running requests
- waiting requests
- queue latency
- request success/error
- input token distribution
- output token distribution

이 계층은 workload shape와 admission 상태를 보여준다.

## 3. Prefill layer

필수 관측:

- TTFT p50/p90/p95
- Prefill latency
- prompt tokens/s
- scheduled Prefill tokens
- partial Prefill count

대표 해석:

| 관측 | 1차 가설 |
|---|---|
| prompt tok/s 낮음 + Tensor Active 낮음 + CPU 높음 | dispatch/backend |
| prompt tok/s 낮음 + Tensor Active 높음 | compute/kernel |
| TTFT 높음 + Prefill latency 정상 + queue 높음 | capacity/admission |
| TTFT tail만 악화 | fairness/long-prompt interference |

## 4. Decode layer

필수 관측:

- TPOT
- ITL
- generation tokens/s
- running Decode sequences
- MTP draft tokens
- MTP accepted tokens
- mean acceptance length

대표 해석:

| 관측 | 1차 가설 |
|---|---|
| TPOT 높음 + SM 낮음 + CPU 높음 | launch/dispatch |
| TPOT 높음 + DRAM 높음 | HBM/KV read |
| TPOT 높음 + NVLink 높음 + SM 낮음 | TP communication |
| MTP K 증가 + acceptance 낮음 | speculative waste |
| output tok/s 정체 + running 증가 | decode saturation |

## 5. KV / State layer

필수:

- KV usage
- num blocks
- preemption count
- maximum concurrency estimate
- context distribution

startup metadata로 추가:

- resolved block size
- state page size
- graph memory
- KV pool bytes
- model weight bytes

## 6. GPU metric을 capacity와 activity로 분리한다

Capacity metric:

- framebuffer used/free
- total HBM
- KV pool
- graph pool

Activity metric:

- GPU Util
- SM Active
- Tensor Active
- DRAM Active
- memory-copy utilization

memory used 90%와 memory bandwidth utilization 90%는 전혀 다른 의미다.

## 7. GPU Util만 보면 안 되는 이유

GPU Util은 일정 sampling window에서 GPU가 busy였다는 넓은 신호다.

구분하지 못하는 것:

- Tensor Core compute
- memory stalls
- copy
- collective
- 짧은 kernel 반복

최소 조합은 GPU Util + SM Active + Tensor Active + DRAM Active다.

## 8. Roofline식으로 metric을 읽는다

Compute-bound 후보:

- SM Active 높음
- Tensor Active 높음
- DRAM은 상대적으로 낮음
- GPU clock 정상

Memory-bound 후보:

- Tensor Active 낮거나 중간
- DRAM Active 높음
- HBM throughput 높음

Launch-bound 후보:

- SM Active 낮음
- GPU Util 들쭉날쭉
- CPU core 높음
- kernel duration 짧음

Fabric-bound 후보:

- SM/Tensor 낮음
- NVLink/NCCL 높음
- rank step skew 존재
- collective duration 높음

## 9. NVLink metric을 세 질문으로 분리한다

Topology 질문 — 쓸 수 있는가?

- P2P status
- link state

Activity 질문 — 실제로 썼는가?

- TX bytes/s
- RX bytes/s
- aggregate bandwidth

Reliability 질문 — 정상적으로 썼는가?

- CRC
- replay
- recovery

세 질문을 한 metric으로 대체하지 않는다.

## 10. NVLink traffic attribution

TP2 P/D 환경에서는 P TP collective, D TP collective, P-to-D KV transfer가 같은 fabric에 섞일 수 있다.

따라서 실험을 분리한다.

1. idle
2. Prefill-only
3. Decode-only
4. transfer-focused
5. combined production-shape

이렇게 얻은 baseline을 시간축으로 겹친다.

## 11. PCIe와 NVLink를 같이 본다

P/D transfer에서 NVLink path를 기대한다면 NVLink TX/RX spike와 낮은 PCIe activity의 조합은 좋은 evidence다.

반대로 NVLink가 거의 없고 PCIe가 급증하면 fallback 또는 다른 data path를 의심한다.

단, DCGM은 device-level telemetry이므로 process attribution은 아니다. controlled isolated run이 필요하다.

## 12. Rate metric의 window

짧은 benchmark와 장기 운영 추세는 query 목적이 다르다.

- throughput 시계열: 짧은 rolling window
- selected-run total: dashboard range 기준 increase
- latency percentile: rolling histogram window
- hardware error: selected-range counter delta

고정 24시간 window를 모든 panel에 쓰면 5분 benchmark에서 의미가 깨진다.

## 13. p50/p90/p95를 같이 보는 이유

p50은 typical request를 보여준다.

p90/p95는 queue interference, long prompt, batch collision, cache pressure를 더 잘 드러낸다.

p50은 좋아지고 p95가 나빠지면 평균 kernel 성능은 개선됐지만 scheduling fairness가 악화됐을 수 있다.

## 14. Benchmark noise floor

옵션 차이가 2%인데 run-to-run variance가 5%면 결론을 내릴 수 없다.

anchor configuration을 최소 여러 번 반복한다.

기록:

- mean
- standard deviation
- p95 variability
- GPU background load
- fabric background load

candidate gain이 noise floor를 넘는지 본다.

## 15. One-factor test의 역할

A only, B only, C only는 최종 winner를 찾기보다 factor의 주효과를 이해하는 단계다.

예:

- A = MBT
- B = MTP
- C = CUDA Graph

A가 prompt throughput을 바꾸고, B가 Decode commit rate를 바꾸고, C가 launch overhead를 바꾼다는 것을 먼저 분리한다.

## 16. Pairwise interaction

MBT x MTP 질문:

> MTP의 이득이 token budget에 따라 달라지는가?

MTP x CUDA Graph 질문:

> K가 커지면서 graph shape와 backend path가 달라지는가?

TP x Context 질문:

> long context에서 capacity gain이 collective tax보다 커지는가?

interaction을 보지 않으면 single-factor winner를 합쳤을 때 성능이 깨질 수 있다.

## 17. Full factorial을 무조건 돌리지 않는다

4 MBT x 4 MTP x 5 Graph x 3 TP x 6 Context를 처음부터 돌리면 비용도 크고 해석도 어렵다.

순서:

1. main effect
2. pairwise interaction
3. survivor set
4. reduced full interaction

## 18. Context bucket은 distribution 기반으로 만든다

임의의 예제 길이만 쓰지 않는다.

실제 workload의 p50, p90, p95, p99, max-supported 근처를 bucket으로 잡는다.

synthetic benchmark도 production shape를 대표해야 한다.

## 19. Concurrency test는 두 종류가 필요하다

Fixed concurrency는 kernel/config 비교에 좋다.

예: C=1, 8, 32, 64.

Open-loop arrival rate는 queueing behavior와 saturation point를 찾는 데 좋다.

RPS를 올리면서 throughput plateau와 waiting queue 급증 지점을 찾는다.

## 20. Saturation point

saturation 이전:

- RPS 증가
- throughput 증가
- latency 비교적 안정

saturation 이후:

- RPS 증가
- throughput 정체
- queue/latency 급증

실제 운영 capacity는 이 knee point와 필요한 headroom 사이에서 결정한다.

## 21. P/D에서는 두 queue를 본다

| Prefill queue | Decode queue | 해석 |
|---|---|---|
| 증가 | 안정 | Prefill capacity 부족 |
| 안정 | 증가 | Decode capacity 부족 |
| 둘 다 증가 | 전체 capacity 또는 burst |
| 둘 다 안정 | balanced 가능 |

router queue/dispatch도 별도로 본다.

## 22. 성능 개선의 반증 조건을 미리 쓴다

가설: FULL_DECODE_ONLY가 TPOT를 줄인다.

반증 조건:

- TPOT 차이 없음
- CPU 차이 없음
- graph memory만 증가

가설: TP2가 long-context cell throughput을 높인다.

반증 조건:

- maximum concurrency는 증가
- NVLink tax 때문에 aggregate output tok/s 감소

가설이 반증되면 그 knob를 winner로 유지하지 않는다.

## 23. Run record

각 run에 최소 다음을 기록한다.

- model / revision
- serving-engine revision
- GPU model
- GPU placement
- driver / CUDA
- TP / PP / DP / EP
- MBT
- max-seqs
- MTP K
- graph mode
- capture sizes
- KV/state dtype
- input bucket
- output bucket
- concurrency / RPS
- TTFT
- TPOT
- prompt tok/s
- generation tok/s
- KV usage
- preemption
- SM/Tensor/DRAM
- NVLink/PCIe
- errors

## 24. 전후 비교에서 반드시 같아야 하는 것

- dataset
- request seed
- sampling parameters
- input/output distribution
- concurrency model
- GPU placement
- warmup
- measurement duration

하나라도 바뀌면 factor isolation이 깨진다.

## 25. 도구별 역할

Dashboard:

> 현재 어디가 느린가?

Profiler:

> 그 phase 안에서 어떤 kernel/op가 느린가?

Microbenchmark:

> hardware/backend ceiling은 얼마인가?

Source code:

> 왜 그 execution path가 선택됐는가?

네 도구를 단계적으로 사용한다.

## 26. 핵심 decision tree

Latency가 증가하면 먼저 Queue 증가 여부를 본다.

Queue가 증가하면 capacity/admission/routing을 본다.

Queue가 안정이면 phase를 나눈다.

Prefill이면 Tensor, DRAM, CPU를 비교한다.

Decode이면 CPU, DRAM, NVLink, MTP acceptance를 비교한다.

KV usage가 높고 preemption이 생기면 memory geometry를 우선한다.

## 27. 본질

> Observability의 목적은 모든 metric을 보는 것이 아니라 한 증상을 compute, memory, scheduler, fabric 중 하나의 가설로 빠르게 축소하고 다음 실험을 결정하는 것이다.
