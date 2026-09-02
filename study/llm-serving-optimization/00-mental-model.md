# 00. LLM Serving Optimization Mental Model

## 1. 하나의 request를 pipeline으로 본다

~~~text
HTTP / parser
  ↓
request queue
  ↓
scheduler admission
  ↓
Prefill
  ↓
KV/state 생성
  ↓
optional P -> D transfer
  ↓
Decode loop
  ↓
streaming / serialization
~~~

뒤 단계의 병목이 앞 단계 metric에도 보일 수 있다.

예를 들어 Decode가 느려지면 request가 오래 running 상태에 남는다.

~~~text
Decode slow
  ↓
running requests 증가
  ↓
KV active set 증가
  ↓
new admission 감소
  ↓
waiting 증가
  ↓
TTFT 증가
~~~

따라서 TTFT 악화가 반드시 Prefill kernel 문제라는 뜻은 아니다.

## 2. 여섯 개 자원

### 2.1 Compute

대표 관측:

~~~text
SM Active
Tensor Pipe Active
FP8/FP16 pipeline activity
achieved FLOPs
kernel duration
~~~

### 2.2 Memory Capacity

대표 소비자:

~~~text
weights
KV cache
recurrent state
CUDA Graph pool
workspace
activation scratch
NCCL buffers
~~~

capacity 부족은 maximum concurrency 감소, preemption, OOM, graph capture 실패로 나타난다.

### 2.3 Memory Bandwidth

Decode는 weights와 KV/state를 반복해서 읽으므로 HBM bandwidth sensitivity가 높다.

Prefill large GEMM은 weight reuse가 커져 compute-bound로 이동하기 쉽다.

### 2.4 CPU / Launch

small-batch Decode에서는 GPU kernel이 짧아서 Python/C++ dispatch와 kernel launch overhead 비율이 커질 수 있다.

CUDA Graph는 이 비용을 줄이는 대표 기술이다.

### 2.5 Interconnect

~~~text
NVLink
NVSwitch
PCIe
InfiniBand
RoCE
GPUDirect RDMA
~~~

TP/EP/CP/P-D transfer가 모두 이 자원을 사용한다.

### 2.6 Scheduler Budget

~~~text
max-num-batched-tokens
max-num-seqs
partial-prefill limits
KV admission
priority
~~~

GPU가 여유 있어도 scheduler budget이 작으면 일을 공급하지 못한다.

## 3. Prefill과 Decode의 roofline은 다르다

### Prefill

~~~text
large token batch
large GEMM
long-sequence attention
~~~

특징:

- weight reuse가 높다.
- arithmetic intensity가 높아지기 쉽다.
- Tensor Core 효율을 끌어올리기 쉽다.
- MBT가 중요한 knob가 된다.

KPI:

~~~text
TTFT
prompt tok/s
prefill latency
queue latency
~~~

### Decode

~~~text
1 token/request/step
small or skinny GEMM
weight reread
KV/state reread
frequent collectives
frequent launches
~~~

특징:

- memory-bound 가능성이 크다.
- CPU launch overhead 비율이 커질 수 있다.
- CUDA Graph 효과가 커진다.
- speculative decoding 효과가 커질 수 있다.

KPI:

~~~text
TPOT
ITL
generation tok/s
accepted draft length
~~~

## 4. 최적화는 병목 이동이다

~~~text
Before
Decode = launch-bound

CUDA Graph 적용
  ↓

After
Decode = HBM-bound
~~~

이후 CUDA Graph를 더 만져도 효과가 작다.

또 다른 예:

~~~text
TP1
KV capacity 부족

TP2
KV capacity 해결
  ↓
NVLink collective 증가
  ↓
communication-bound
~~~

좋은 운영자는 한 knob의 효과보다 다음 병목이 어디로 이동했는지를 본다.

## 5. Little's Law와 KV pressure

대략:

~~~math
L = \lambda W
~~~

Decode가 느려져 W가 커지면 같은 RPS에서도 running request 수가 늘어난다.

그 결과 KV pressure도 커진다.

즉 TPOT 악화는 latency뿐 아니라 memory capacity 문제를 증폭할 수 있다.

## 6. Batch의 두 얼굴

Batch 증가 장점:

~~~text
weight reuse 증가
GEMM efficiency 증가
occupancy 증가
~~~

단점:

~~~text
queue wait 증가
tail latency 증가
active KV set 증가
spec verification waste 증가
collective payload 증가
~~~

throughput-optimal batch와 latency-optimal batch는 다를 수 있다.

## 7. Workload shape가 먼저다

### Long input / short output

~~~text
170K input
128 output
~~~

Prefill, KV transfer, cache capacity가 중요하다.

### Short input / long output

~~~text
4K input
4K output
~~~

Decode, MTP, CUDA Graph가 중요하다.

### High-concurrency chat

scheduler, continuous batching, HBM bandwidth가 중요하다.

### Long-context agent

prefix reuse, state transfer, routing, cache persistence가 같이 중요해진다.

## 8. Architecture가 cost equation을 바꾼다

~~~text
Full Attention
  KV bytes proportional to context

MLA
  bytes/token 감소
  context scaling은 여전히 존재

GDN/KDA/Mamba
  fixed recurrent state 비중 존재

Hybrid
  fixed state + token-growing KV

MoE
  expert GEMM + all-to-all 추가
~~~

## 9. Hardware generation도 cost equation을 바꾼다

H200-class Hopper는 약 4.8 TB/s HBM bandwidth와 GPU당 900 GB/s NVLink 4를 제공한다.

Blackwell Ultra-class는 최대 8 TB/s HBM과 GPU당 1.8 TB/s NVLink 5 세대로 올라간다.

하드웨어가 바뀌면 좋은 TP/precision/batch도 바뀔 수 있다.

## 10. 운영자가 항상 던질 질문

1. 이 metric은 capacity인가 activity인가?
2. 이 knob은 실제 work를 바꾸나 상한만 바꾸나?
3. 지금 Prefill인가 Decode인가?
4. context 증가 시 무엇이 선형으로 늘어나는가?
5. TP 증가 시 무엇을 shard하고 무엇을 replicate하는가?
6. CUDA Graph가 줄이는 비용은 compute인가 launch인가?
7. SD gain보다 draft/state 비용이 크지 않은가?
8. NVLink traffic은 TP인가 P/D transfer인가?
9. 성능이 좋아진 대신 KV capacity를 잃지 않았는가?
10. local winner가 system winner인가?

> 운영 수준의 눈썰미는 옵션 이름을 많이 아는 데서 생기지 않는다. 증상을 자원 모델로 번역하고, 그 자원을 소비하는 execution path를 좁히는 능력에서 생긴다.
