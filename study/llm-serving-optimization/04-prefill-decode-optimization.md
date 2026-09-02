# 04. Prefill과 Decode를 따로 최적화하는 법

## 1. P/D를 분리하는 순간 최적화 목적이 달라진다

단일 engine에서는 Prefill과 Decode가 동일 scheduler, 동일 CUDA Graph 정책, 동일 MBT, 동일 cache pool 안에서 경쟁한다.

P/D disaggregation은 이 충돌을 줄이고 role-specific tuning을 가능하게 한다.

~~~text
Prefill Engine
  입력 전체를 처리
  KV/state 생성
  TTFT와 prompt throughput 중심

Decode Engine
  KV/state를 이어받음
  output token 반복 생성
  TPOT/ITL와 output throughput 중심
~~~

따라서 P/D 환경에서 가장 먼저 해야 할 일은 같은 옵션을 양쪽에 복사하는 것이 아니라 각 옵션이 어느 phase에 의미가 있는지를 분리하는 것이다.

## 2. Prefill의 비용 구조

~~~text
embedding/projection
large GEMM
attention / recurrent scan
FFN or MoE
KV/state write
TP collective
P->D transfer preparation
~~~

Prefill에서 중요한 자원:

~~~text
Tensor Core compute
HBM write/read
long-sequence attention
CPU scheduling
TP fabric
KV/state output bandwidth
~~~

## 3. Decode의 비용 구조

~~~text
scheduler
-> one-token or speculative token batch
-> layer forward
-> TP collective
-> KV/state update
-> sampling / acceptance
-> next step
~~~

Decode에서 중요한 자원:

~~~text
weight HBM read
KV/state read
small GEMM efficiency
CPU launch
CUDA Graph coverage
TP collective latency
MTP acceptance
~~~

## 4. Prefill MBT의 의미

작은 MBT:

~~~text
+ step latency 짧음
+ fairness 좋음
- GEMM efficiency 낮을 수 있음
- 긴 prompt가 많은 step으로 분할
~~~

큰 MBT:

~~~text
+ prompt tok/s 상승 가능
+ large GEMM 효율 증가
- one-step latency 증가
- queue tail 악화 가능
- workspace/cache pressure 증가 가능
~~~

대표 sweep:

~~~text
8K
16K
32K
64K optional
~~~

## 5. Decode MBT의 의미

Decode에서는 active sequence 수와 MTP K가 실제 token envelope를 정한다.

~~~text
MTP OFF:
tokens ~= active seqs

MTP K:
tokens ~= active seqs * (1 + K)
~~~

따라서 Prefill의 32K를 Decode에 그대로 가져오는 것은 대부분 의미가 없다.

대표 sweep:

~~~text
512
1024
2048
4096 optional
~~~

2K가 충분하다면 4K는 non-binding이다.

## 6. Prefill MTP는 왜 주력 최적화가 아닌가

MTP는 기본적으로 Decode에서 비싼 target call당 확정 token 수를 늘리는 기술이다.

Prefill에서 MTP on/off를 볼 이유는 성능 gain 자체보다 다음 side effect 때문이다.

~~~text
extra state/cache allocation
hybrid recurrent state shape 변화
CUDA Graph compatibility 변화
P/D transfer layout 변화
CPU/GDN path 변화
~~~

즉 Prefill MTP 실험의 질문은 다음에 가깝다.

> MTP를 켜야 하는 전체 serving configuration에서 Prefill이 얼마나 손해를 보는가?

## 7. Decode MTP는 핵심 축이다

MTP K 증가:

~~~text
drafted candidates 증가
-> accepted tokens/cycle 증가 가능
-> target cycles/token 감소 가능
~~~

하지만:

~~~text
draft overhead 증가
verification token 증가
state/KV 증가
graph shape 증가
maximum concurrency 감소
~~~

도 생긴다.

실제 winner metric:

~~~text
generation tok/s
TPOT
ITL
accepted length
draft cost
KV/state capacity
queue stability
~~~

## 8. Prefill CUDA Graph

1차 비교:

~~~text
NONE
PIECEWISE
~~~

확인:

~~~text
TTFT
prompt tok/s
CPU util
SM Active
Tensor Active
graph memory
~~~

PIECEWISE가 이득이 없다면 Prefill은 launch-bound가 아닐 가능성이 높다.

## 9. Decode CUDA Graph

1차:

~~~text
NONE
FULL_DECODE_ONLY
~~~

후속:

~~~text
FULL_AND_PIECEWISE
~~~

Decode-only role에서는 FULL_DECODE_ONLY가 unused Prefill graph를 피할 수 있다는 장점이 있다.

## 10. Prefill TP의 목적

TP 증가:

~~~text
weight/rank 감소
KV/state/rank 감소 가능
compute/rank 감소
~~~

대신:

~~~text
layer별 collective 증가
NVLink dependence 증가
~~~

Prefill은 large GEMM이므로 TP를 늘렸을 때 compute reduction이 communication tax를 이길 가능성이 Decode보다 높을 수 있다.

그러나 long context에서는 attention/HBM path가 지배하면 TP scaling이 선형이지 않다.

## 11. Decode TP의 목적

장점:

~~~text
weights/rank 감소
KV/state/rank 감소
single-rank compute 감소
~~~

비용:

~~~text
매 layer collective
small batch일수록 collective latency 비중 큼
~~~

즉:

~~~text
TP = latency knob
+
capacity knob
+
fabric knob
~~~

이다.

## 12. P/D ratio

~~~text
P queue grows
D queue stable
-> Prefill capacity 부족

P queue stable
D queue grows
-> Decode capacity 부족
~~~

GPU를 더 넣을 때:

~~~text
P scale out
D scale out
P TP 변경
D TP 변경
~~~

중 무엇이 맞는지 queue와 phase latency로 판단한다.

## 13. P/D transfer를 별도 phase로 본다

~~~text
Prefill compute
-> KV/state materialization
-> transport
-> Decode remote load
-> Decode
~~~

구간별로:

~~~text
T_total_to_decode_ready
=
T_prefill
+
T_transfer_prepare
+
T_transport
+
T_remote_load
~~~

를 생각한다.

## 14. Long input / short output

~~~text
170K input
128 output
~~~

우선순위:

~~~text
1. Prefill TTFT
2. KV/state capacity
3. P->D transfer
4. Decode
5. MTP
~~~

## 15. Short input / long output

~~~text
4K input
4K output
~~~

우선순위:

~~~text
1. Decode TPOT
2. CUDA Graph
3. MTP
4. Decode MBT/max-seqs
5. Prefill
~~~

## 16. Long input / long output

~~~text
170K input
2K output
~~~

P/D 전체 balance가 중요하다.

~~~text
Prefill compute
KV transfer
Decode HBM/state
MTP
active KV capacity
~~~

중 하나를 과도하게 최적화하면 다른 곳이 bottleneck이 된다.

## 17. Prefill 실험 설계

고정:

~~~text
TP
PP
max-num-seqs
KV dtype
input distribution
concurrency
GPU placement
~~~

변수:

~~~text
A = MBT
B = MTP
C = CUDA Graph mode
~~~

순서:

~~~text
A
B
C
A x B
A x C
B x C
survivor ABC
~~~

## 18. Decode 실험 설계

먼저 one-time non-binding check:

~~~text
max-num-seqs
256 / 512 / 1024
~~~

그 다음:

~~~text
A = MBT
B = MTP K
C = CUDA Graph mode
~~~

후속:

~~~text
capture size
context-length curve
~~~

## 19. Context dimension은 별도 축이다

8K에서 winner인 configuration이 170K에서 winner라는 보장은 없다.

context 증가:

~~~text
Full-attention KV 증가
Decode HBM read 증가
maximum concurrency 감소
P/D transfer bytes 증가
~~~

따라서 최소:

~~~text
8K
32K
64K
128K
long-tail production bucket
max supported bucket
~~~

에서 재검증한다.

## 20. Cell winner를 고른다

Prefill winner와 Decode winner를 각각 골라도 두 값을 합치면 cell winner가 아닐 수 있다.

~~~text
P best:
very large MBT

D best:
very high MTP K

combined:
P produces KV faster than D consumes
D queue grows
KV active set grows
~~~

최종 기준:

~~~text
P queue stable
D queue stable
TTFT stable
TPOT stable
KV headroom
preemption low
fabric headroom
~~~

## 21. 본질

> P/D 최적화의 본질은 두 엔진을 각각 가장 빠르게 만드는 것이 아니라 Prefill 생산률과 Decode 소비률, 그리고 그 사이 KV transport를 하나의 flow-control 문제로 맞추는 것이다.
