# 02. KV Cache, Recurrent State와 Capacity

## 1. KV cache라는 말을 세분화한다

모든 LLM이 같은 종류의 cache를 갖는 것은 아니다.

~~~text
Token-growing KV
Latent KV
Windowed KV
Compressed KV
Recurrent state
Conv state
Speculative state
~~~

각 구조는 context length에 대한 memory scaling이 다르다.

## 2. Full Attention

한 layer의 logical KV:

~~~math
M_{layer}
=
2 \times T \times H_{kv} \times d_h \times b
~~~

- T: cached tokens
- Hkv: KV heads
- dh: head dimension
- b: bytes per element
- 2: K와 V

전체 L layer:

~~~math
M_{KV}
=
2LTH_{kv}d_hb
~~~

따라서 context 2배는 KV memory 2배다.

## 3. GQA

GQA는 여러 Q head가 더 적은 수의 KV head를 공유한다.

~~~text
Hkv 감소
→ KV bytes/token 감소
→ Decode HBM read 감소
→ long-context capacity 증가
~~~

GQA의 serving 이점은 parameter count보다 cache bytes와 bandwidth에서 더 직접적으로 드러난다.

## 4. MLA

MLA는 token별 cache representation을 latent로 줄인다.

~~~text
per-token bytes 감소
but
context-linear scaling은 남음
~~~

즉 long-context capacity는 좋아지지만 recurrent attention처럼 context-independent state가 되는 것은 아니다.

## 5. Recurrent / Mamba / GDN / KDA

recurrent attention은 history를 token별 KV가 아니라 state로 누적한다.

~~~text
S_t = f(S_t-1, x_t)
~~~

memory scaling을 개념적으로:

~~~text
O(context)
→
O(state size)
~~~

로 바꾼다.

하지만 state 자체가 작다고 보장되지는 않는다.

matrix recurrent state는 head 수와 state dimension에 따라 request당 수십~수백 MiB가 될 수 있다.

## 6. Conv state도 별도다

Mamba/GDN 계열은 recurrent matrix 외에 short convolution state를 가질 수 있다.

~~~text
Request
├─ recurrent matrix state
└─ short-conv state
~~~

Speculative decoding에서는 candidate token들을 처리하기 위해 추가 conv/state slot이 필요할 수 있다.

## 7. Hybrid model

Qwen3.5/3.6처럼 recurrent layer와 full-attention layer가 섞이면:

~~~math
M_{request}
=
M_{recurrent}
+
T \times M_{full-attn-per-token}
~~~

이다.

따라서:

~~~text
short context
→ recurrent state 비중 큼

long context
→ full-attention KV 비중 증가
~~~

이것이 context에 따라 최적 MTP K와 maximum concurrency가 바뀔 수 있는 이유다.

## 8. Hybrid cache page alignment

Serving allocator는 서로 다른 cache type을 같은 physical pool에서 관리하기 위해 page size를 맞출 수 있다.

~~~text
attention page
recurrent state page
~~~

둘의 physical bytes가 다르면 framework는:

~~~text
attention block size 증가
or
state page padding
~~~

을 선택할 수 있다.

따라서 사용자가 block-size=256을 지정했다고 실제 resolved hybrid physical block이 256 token이라고 단정하면 안 된다.

startup에서 반드시 확인한다.

~~~text
resolved attention block size
resolved recurrent block size
page padding
number of blocks
~~~

## 9. 왜 state page와 token block이 헷갈리는가

recurrent state는 한 block이 여러 token 구간의 마지막 state snapshot을 대표할 수 있다.

반면 full-attention KV block은 그 구간의 token KV 자체를 보유한다.

그래서:

~~~text
Mamba/GDN block size
≠
scheduler MBT
≠
CUDA Graph capture size
~~~

이다.

세 숫자가 우연히 비슷해도 의미는 완전히 다르다.

## 10. Fragmentation을 두 종류로 본다

### Internal fragmentation

마지막 KV block이 일부만 사용되는 waste.

~~~text
큰 block
→ metadata 감소 가능
→ tail waste 증가 가능
~~~

### Structural padding

hybrid cache type을 같은 page bytes로 맞추면서 생기는 waste.

~~~text
state page < attention page
→ state page padding
~~~

두 waste의 원인이 다르므로 block-size 실험도 목적을 명확히 해야 한다.

## 11. Speculative decoding과 state memory

K가 커지면:

~~~text
draft/verify 한 cycle에서 commit token 증가 가능
TPOT 개선 가능
~~~

하지만 recurrent model에서는 candidate별 state transition을 처리해야 한다.

~~~text
S_t
→ S_t+1
→ S_t+2
→ S_t+3
~~~

reject 시 accepted prefix 상태로 복귀해야 한다.

그래서 framework는:

~~~text
extra speculative state blocks
state copy
rollback/postprocess metadata
~~~

를 사용할 수 있다.

결과:

~~~text
MTP K 증가
→ latency 개선 가능
→ state footprint 증가
→ maximum concurrency 감소 가능
~~~

이다.

## 12. Context-length capacity curve가 핵심 산출물이다

운영에서는 총 KV GB 하나보다 다음 curve가 중요하다.

~~~text
context length
→ maximum active requests
~~~

예:

| Context | Max concurrency |
|---:|---:|
| 8K | 400 |
| 32K | 180 |
| 64K | 90 |
| 128K | 44 |
| 200K | 27 |

이 curve는 다음의 합성 결과다.

~~~text
architecture
KV dtype
state dtype
TP
block geometry
graph memory
workspace
MTP K
~~~

## 13. TP가 cache에 미치는 영향

Full Attention KV head가 TP rank에 잘 shard되면:

~~~text
TP 증가
→ per-rank KV 감소
~~~

recurrent state도 state/head dimension이 TP-sharded되면 per-rank footprint가 줄 수 있다.

하지만 항상 정확히 1/TP가 아니다.

~~~text
KV head divisibility
state group replication
small tensor replication
alignment/padding
~~~

을 확인한다.

즉 TP는 parameter size만 보고 정하는 값이 아니다.

## 14. KV dtype과 state dtype을 분리한다

BF16 KV:

~~~text
2 bytes/element
~~~

FP8 KV:

~~~text
1 byte/element + scale/metadata
~~~

하지만 hybrid model은:

~~~text
attention KV = FP8
recurrent conv state = BF16/FP16
recurrent SSM state = FP32
~~~

처럼 서로 다른 precision을 동시에 사용할 수 있다.

따라서 KV FP8을 켰다고 전체 request cache가 절반이 되는 것은 아니다.

## 15. CUDA Graph memory도 같은 HBM을 먹는다

GPU HBM budget:

~~~text
total HBM
- weights
- runtime allocations
- CUDA Graph pools
- temporary workspace
- NCCL buffers
- allocator reserve
=
KV/state pool
~~~

그래서 graph capture size를 늘릴 때 TPOT만 보면 안 된다.

~~~text
graph coverage 증가
→ launch overhead 감소 가능

but

graph pool 증가
→ KV/state pool 감소
→ concurrency 감소 가능
~~~

이다.

## 16. P/D에서도 cache는 사라지지 않는다

Prefill engine:

~~~text
prompt compute
KV/state 생성
KV/state write
transfer
~~~

Decode engine:

~~~text
remote KV/state load
state retain
KV/state read
generation
~~~

P/D는 cache cost를 없애지 않는다.

대신 compute phase를 분리하고 새로운 data movement stage를 추가한다.

~~~text
P write
→ P-to-D transport
→ D read
~~~

따라서 transfer time과 bytes도 capacity study에 포함해야 한다.

## 17. Prefix caching은 architecture마다 의미가 다르다

Full Attention:

~~~text
prefix token KV blocks reuse
~~~

Recurrent:

~~~text
prefix boundary state snapshot reuse
~~~

Hybrid:

~~~text
KV blocks
+
recurrent state snapshot
~~~

을 동시에 일치시켜야 한다.

recurrent state snapshot을 너무 촘촘히 저장하면 fixed-state advantage를 잃을 수 있다.

## 18. Preemption의 비용도 architecture마다 다르다

Full Attention:

~~~text
recompute KV
or
offload KV
~~~

Recurrent:

~~~text
recompute state
or
save/restore state
~~~

Hybrid:

~~~text
두 종류 모두 처리
~~~

따라서 동일 preemption count라도 실제 penalty는 모델마다 다르다.

## 19. Capacity 관측 방법

### Startup

~~~text
weights memory
graph memory
available KV/state memory
resolved block size
number of GPU blocks
estimated max concurrency
~~~

### Runtime

~~~text
KV usage
preemption
running
waiting
queue latency
~~~

### Workload

~~~text
input p50/p95/p99
output p50/p95/p99
concurrency distribution
~~~

세 층을 함께 기록한다.

## 20. 좋은 headroom이란

KV utilization이 낮을수록 좋은 것은 아니다.

너무 낮으면 GPU memory를 놀린다.

하지만:

~~~text
sustained 95~100%
+
preemption 증가
+
waiting 증가
~~~

이면 burst headroom이 없다.

운영 winner는 평균 utilization 최고가 아니라 workload tail을 버틸 수 있는 configuration이다.

## 21. 본질

> Cache 최적화는 GB를 최대화하는 일이 아니라 실제 context distribution에서 필요한 active request 집합을 preemption 없이 유지할 수 있는 memory geometry를 만드는 일이다.
