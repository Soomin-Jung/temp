# LLM Serving Optimization — 운영자가 보는 전체 지도

작성일: 2026-09-03

이 모듈은 특정 모델의 추천 옵션을 외우는 문서가 아니다.

목표는 다음 질문을 스스로 풀 수 있는 운영 판단 프레임을 만드는 것이다.

> 이 설정을 바꾸면 어떤 자원을 더 쓰고, 어떤 자원을 덜 쓰며, 어떤 병목을 다른 계층으로 이동시키는가?

LLM serving 최적화는 대부분 다음 여섯 자원의 교환 관계로 설명할 수 있다.

~~~text
Compute
Memory capacity
Memory bandwidth
CPU / launch overhead
Interconnect bandwidth
Scheduler budget
~~~

좋은 튜닝은 단일 metric 최고값을 만드는 일이 아니다.

~~~text
TTFT만 줄였는데 Decode queue가 쌓임
TPOT만 줄였는데 KV capacity가 반토막남
TP를 늘렸는데 NVLink가 포화됨
MTP를 늘렸는데 accepted token보다 speculative state가 더 비쌈
CUDA Graph를 촘촘히 잡았는데 graph memory가 KV cache를 밀어냄
~~~

이런 경우는 local optimization이지 serving-system optimization이 아니다.

## 읽는 순서

1. [00 — Mental Model](00-mental-model.md)
2. [01 — Scheduler, MBT, max-num-seqs](01-scheduler-token-budget.md)
3. [02 — KV Cache, recurrent state와 capacity](02-cache-state-capacity.md)
4. [03 — CUDA Graph, compilation, capture size](03-cuda-graphs-and-compilation.md)
5. [04 — Prefill과 Decode 최적화](04-prefill-decode-optimization.md)
6. [05 — Parallelism과 Fabric](05-parallelism-and-fabric.md)
7. [06 — 모델 아키텍처별 운영 패턴](06-model-architecture-playbook.md)
8. [07 — 관측과 실험 설계](07-observability-and-experiment-design.md)
9. [08 — 운영 튜닝 Runbook](08-production-tuning-runbook.md)
10. [99 — 용어집과 주요 원문](99-glossary-and-references.md)

## 기존 Study와의 관계

- [Attention architecture](../attention/README.md)
- [Serving engineer 관점의 attention/cache/state](../attention/90-serving-engineer-view.md)
- [GPU architecture](../gpu-architecture/README.md)
- [LLM serving bottleneck diagnosis](../gpu-architecture/07-llm-serving-bottleneck-diagnosis.md)
- [NVLink/NVSwitch](../gpu-architecture/10-pcie-nvlink-nvswitch.md)
- [NCCL observability labs](../gpu-architecture/13-nccl-collectives-observability-labs.md)
- [Speculative decoding](../speculative-decoding/00-foundations-and-method-lineage.md)

여기서는 위 개념을 실제 serving parameter와 병목 판단으로 연결한다.

## 핵심 사고 순서

~~~text
1. workload shape
   input length / output length / concurrency / burst / reuse

2. phase
   Prefill / Decode / KV transfer / queue

3. resource
   SM/Tensor / HBM BW / HBM capacity / CPU / NVLink / scheduler

4. architecture
   Full Attention / MLA / GDN-KDA-Mamba / MoE / SD head

5. parameter
   MBT / max-num-seqs / TP / graph / KV dtype / MTP K
~~~

## 가장 중요한 원칙

### Capacity와 Throughput을 분리한다

~~~text
Capacity
  메모리에 몇 개까지 살아있을 수 있는가

Throughput
  단위 시간에 몇 token/request를 끝낼 수 있는가
~~~

### Budget와 Work를 분리한다

~~~text
actual work =
min(
  runnable work,
  token budget,
  sequence budget,
  KV admission,
  backend constraints
)
~~~

### GPU Util 100%는 결론이 아니다

~~~text
GPU Util
SM Active
Tensor Pipe Active
DRAM Active
HBM bytes
NVLink bytes
~~~

를 함께 본다.

### P/D 분리는 최적화 공간을 늘린다

~~~text
Prefill
  large MBT
  large GEMM
  TTFT 중심
  chunked prefill

Decode
  smaller MBT
  full CUDA Graph
  MTP
  TPOT 중심
~~~

### 좋은 값보다 non-binding 값을 찾는다

~~~text
max-num-seqs 256  -> throughput 90
max-num-seqs 512  -> throughput 100
max-num-seqs 1024 -> throughput 100
~~~

이면 512 이상은 성능을 더 이상 제한하지 않는다.

## 최종 목표

~~~text
TTFT p95 증가
Prompt tok/s 감소
GPU Util 98%
Tensor Active 낮음
DRAM Active 높음

=> GPU가 놀지는 않지만 compute-bound는 아님.
   attention/HBM read path를 먼저 의심.
~~~

~~~text
Decode TPOT 증가
SM Active 낮음
CPU core 포화
small batch
graph off

=> GPU compute보다 launch/dispatch overhead가 큼.
   CUDA Graph 후보.
~~~

~~~text
TP2가 TP1보다 KV capacity는 늘었는데 output tok/s 동일
NVLink traffic 높음

=> capacity 개선과 throughput 개선은 다른 문제.
   TP collective가 compute 이득을 상쇄할 수 있음.
~~~

> Serving optimization의 본질은 빠른 옵션을 찾는 것이 아니라 현재 가장 비싼 자원을 찾고, 그 비용을 다른 여유 자원으로 교환하는 것이다.
