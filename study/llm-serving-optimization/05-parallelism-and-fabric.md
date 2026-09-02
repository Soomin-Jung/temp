# 05. Parallelism과 Fabric — TP, PP, DP, EP, NVLink, PCIe, RDMA

## 1. Parallelism은 compute를 나누는 동시에 communication을 만든다

GPU를 더 붙이는 순간 두 효과가 동시에 생긴다.

~~~text
local work 감소
+
communication 증가
~~~

따라서 scale-up의 핵심 질문은 다음이다.

> 나눈 compute/memory 이득이 새 communication tax보다 큰가?

## 2. Tensor Parallelism

TP는 한 layer의 tensor 연산을 여러 GPU rank로 나눈다.

대표 패턴:

~~~text
column-parallel projection
row-parallel projection
attention head shard
state shard
collective
~~~

장점:

~~~text
weight/rank 감소
KV/state/rank 감소 가능
compute/rank 감소
single request latency 감소 가능
~~~

비용:

~~~text
all-reduce
all-gather
reduce-scatter
매 layer synchronization
~~~

## 3. TP는 왜 Decode에서 더 민감한가

Prefill:

~~~text
large GEMM time
>>
collective launch latency
~~~

이면 communication tax가 상대적으로 작다.

Decode:

~~~text
small GEMM
+
frequent collective
~~~

에서는 collective latency가 전체 step의 큰 비중이 될 수 있다.

따라서 다음 결과가 모두 가능하다.

~~~text
TP2 Prefill
  큰 compute gain

TP2 Decode
  capacity는 개선
  latency gain은 작음

TP4 Decode
  KV는 더 여유
  collective tax가 더 커져 throughput 정체
~~~

## 4. TP와 memory capacity

TP가 잘 shard되면:

~~~text
weights/rank 감소
KV heads/rank 감소
recurrent state/rank 감소
~~~

하지만 다음은 일부 replicate될 수 있다.

~~~text
small metadata
일부 KV head
일부 state group
embedding/logits structure
workspace
~~~

따라서 실제 memory scaling은 항상 model config와 backend를 확인한다.

## 5. Head divisibility를 본다

예:

~~~text
KV heads = 4
TP = 2
-> 2 heads/rank

KV heads = 4
TP = 8
-> exact one-head-per-rank shard 불가
-> replication 또는 특수 layout 가능
~~~

TP 숫자가 parameter count에는 좋아 보여도 KV head geometry에는 나쁠 수 있다.

GDN/KDA state group도 같은 관점으로 본다.

## 6. Pipeline Parallelism

PP는 layer 구간을 GPU stage로 나눈다.

장점:

~~~text
weight memory 분산
한 layer 내부 TP collective를 줄일 수 있음
~~~

비용:

~~~text
stage bubble
activation transfer
microbatch scheduling
single-request latency 증가 가능
~~~

Decode small batch에서는 pipeline bubble이 특히 비싸다.

또 MTP/SD가 PP와 제한적으로 호환되는 framework/version이 있을 수 있다.

## 7. Data Parallelism

DP는 모델 replica를 늘린다.

~~~text
request A -> replica 0
request B -> replica 1
~~~

장점:

- throughput scale-out
- request isolation
- layer 내부 TP communication 증가 없음

단점:

- weights 중복
- replica별 KV pool 분리
- routing/load imbalance

long-context model에서는 replica마다 큰 weight를 반복 로드하는 비용과 KV pool fragmentation 때문에 TP와 trade-off가 있다.

## 8. Expert Parallelism

MoE에서는 expert를 GPU에 분산한다.

~~~text
token routing
-> all-to-all
-> expert GEMM
-> all-to-all
~~~

attention이 빨라져도 EP all-to-all이 bottleneck이면 E2E gain은 작다.

따라서 MoE serving에서는 다음을 같이 본다.

~~~text
expert load balance
tokens/expert
all-to-all duration
fabric bytes
expert GEMM utilization
~~~

## 9. Context Parallelism

긴 sequence의 token 축을 여러 GPU로 나눈다.

Full Attention:

~~~text
remote K/V 필요
ring 또는 all-to-all style communication
~~~

Recurrent Attention:

~~~text
chunk boundary state 전달
prefix scan / recurrence dependency
~~~

Sparse/Compressed Attention:

~~~text
global index
compressed memory shard
top-k result exchange
~~~

attention architecture가 CP communication graph를 직접 바꾼다.

## 10. NVLink와 PCIe를 구분한다

NVLink:

- GPU-to-GPU 고대역폭 interconnect
- Hopper NVLink 4는 GPU당 최대 약 900 GB/s급
- Blackwell NVLink 5는 GPU당 최대 약 1.8 TB/s급

PCIe:

- CPU/GPU 및 일부 P2P path
- 일반적으로 NVLink보다 GPU-GPU bandwidth가 낮다.

TP가 NVLink를 타는지 PCIe fallback을 타는지는 topology와 runtime을 확인해야 한다.

## 11. NVSwitch

8-GPU HGX 계열에서는 NVSwitch가 GPU들을 고대역폭 switching fabric으로 연결한다.

중요:

~~~text
NVLink link 존재
!=
모든 pair가 항상 같은 effective bandwidth
~~~

실제 path는 topology, switch generation, concurrent traffic, routing의 영향을 받는다.

## 12. Theoretical peak와 empirical ceiling

GPU spec의 NVLink peak는 hardware capability다.

실제 framework traffic의 기준은 같은 topology의 microbenchmark가 더 유용하다.

~~~text
same GPU pair
same node
same process/container conditions
~~~

에서:

~~~text
NCCL all_reduce_perf
NCCL send/recv
CUDA P2P bandwidth
~~~

를 측정한다.

그 뒤:

~~~text
observed framework GB/s
/
empirical same-topology GB/s
~~~

를 본다.

## 13. TP traffic과 P/D traffic을 분리한다

P/D + TP2 구조:

~~~text
P0 <-> P1
  Prefill TP collective

D0 <-> D1
  Decode TP collective

P pair -> D pair
  KV/state transfer
~~~

DCGM NVLink traffic 하나만 보면 세 traffic이 섞인다.

그래서 workload를 분리한다.

~~~text
N0 idle
N1 Prefill TP only
N2 Decode TP only
N3 P->D transfer-focused
N4 real combined workload
~~~

## 14. Transfer-focused workload

P/D transport를 보고 싶으면 long input / short output이 좋다.

예:

~~~text
128K / 16 output
170K / 16 output
200K / 32 output
~~~

이렇게 하면:

~~~text
large Prefill
large KV/state materialization
large transfer
tiny Decode
~~~

로 transport signal을 분리하기 쉽다.

## 15. NVLink objective evidence

최소 다음을 같이 본다.

~~~text
P2P status
NVLink TX bytes/s
NVLink RX bytes/s
PCIe TX bytes/s
PCIe RX bytes/s
CRC errors
Replay errors
Recovery errors
~~~

질문:

~~~text
1. NVLink path가 가능한가?
2. 실제 traffic이 발생했는가?
3. PCIe가 대신 지배하지 않는가?
4. 높은 traffic에서 오류가 증가하지 않는가?
~~~

## 16. NVLink traffic이 높다고 항상 좋은가

TP를 늘리면 NVLink traffic이 높아진다.

그 자체는 효율의 증거가 아니다.

~~~text
traffic 증가
but
throughput unchanged
~~~

이면 communication tax만 늘었을 수 있다.

좋은 scaling:

~~~text
TP 증가
compute/rank 감소
capacity 증가
throughput 증가
latency 감소 또는 유지
fabric utilization 합리적
~~~

나쁜 scaling:

~~~text
TP 증가
NVLink 급증
SM Active 감소
collective duration 증가
throughput 정체
~~~

## 17. HBM과 NVLink의 상대 속도를 직관으로 잡는다

H200-class 예:

~~~text
HBM bandwidth ~4.8 TB/s
NVLink per-GPU peak ~0.9 TB/s
~~~

GPU 내부 memory traffic보다 GPU 간 traffic의 peak 자체가 낮다.

분산하면 local work를 줄이는 대신 remote synchronization/data movement를 추가한다.

Blackwell 세대에서도 HBM과 NVLink가 모두 빨라지지만 이 교환 관계는 사라지지 않는다.

## 18. Scale-up과 Scale-out

Scale-up:

~~~text
same node
NVLink/NVSwitch
low latency
high bandwidth
~~~

Scale-out:

~~~text
multiple nodes
NIC
IB/RoCE
RDMA/GDRDMA
~~~

멀티노드 TP/EP/CP는 NIC topology까지 보게 된다.

PP는 activation traffic이 stage boundary에 집중되므로 workload에 따라 scale-out에 더 적합할 수 있다.

## 19. GPUDirect RDMA

host-staged path:

~~~text
GPU HBM
-> host memory
-> NIC
-> network
-> host memory
-> remote GPU
~~~

GDRDMA:

~~~text
GPU HBM
<-> NIC DMA
~~~

CPU/host staging을 줄인다.

하지만 기능 지원과 실제 path 사용은 다르다.

검증:

~~~text
DMA-BUF 또는 peer-memory path
NIC-GPU topology
RDMA counters
NCCL logs
bandwidth benchmark
~~~

## 20. Parallelism 선택 질문

### TP

~~~text
weights 때문에 필요한가?
KV/state capacity 때문에 필요한가?
latency 때문에 필요한가?
head/state가 잘 shard되는가?
NVLink tax는 얼마인가?
~~~

### PP

~~~text
weight memory 때문인가?
bubble을 감당 가능한가?
MTP/graph와 호환되는가?
~~~

### DP

~~~text
weight duplication을 감당 가능한가?
routing balance가 되는가?
long-context KV pool이 replica별로 충분한가?
~~~

### EP

~~~text
expert memory 때문인가?
all-to-all fabric이 충분한가?
token imbalance가 심하지 않은가?
~~~

## 21. Fabric 병목 패턴

~~~text
SM Active 낮음
Tensor Active 낮음
NVLink 높음
NCCL kernel duration 높음
rank skew 존재
~~~

이면 communication-bound 가능성이 높다.

반대로:

~~~text
NVLink 낮음
SM/Tensor 높음
~~~

이면 compute가 더 중요한 병목일 수 있다.

## 22. 주요 원문

- NVIDIA NVLink / NVLink Switch
  - https://www.nvidia.com/en-us/data-center/nvlink/
- NVIDIA H200
  - https://www.nvidia.com/en-us/data-center/h200/
- NVIDIA CUDA Programming Guide
  - https://docs.nvidia.com/cuda/cuda-programming-guide/
- 기존 GPU communication study
  - [../gpu-architecture/09-gpu-communication-mental-model.md](../gpu-architecture/09-gpu-communication-mental-model.md)

> Parallelism의 본질은 GPU 수를 늘리는 것이 아니라 local compute/memory 비용을 fabric 비용으로 교환하는 것이다. 좋은 TP/PP/EP 값은 그 교환비가 현재 workload와 hardware에서 이득이 되는 지점이다.
