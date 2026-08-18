# 90. Serving Engineer 관점의 Attention: KV Cache, State, Kernel, Prefix Cache와 분산 추론

작성일: 2026-08-18  
상위 문서: [Attention Architecture Study Guide](README.md)

## 1. 이 장의 목표

모델 논문에서 attention은 정확도와 계산 복잡도로 설명되지만, 실제 vLLM 같은 serving engine에서는 다음 객체로 나타난다.

- 몇 byte의 cache/state를 request마다 보유하는가?
- decode 한 step에서 HBM에서 몇 byte를 읽는가?
- prefill에서 어떤 kernel이 Tensor Core를 채우는가?
- prefix hit 시 무엇을 재사용해야 하는가?
- request를 preempt/resume할 때 어떤 state를 보존해야 하는가?
- TP/CP에서 cache/state가 어떻게 shard되고 통신되는가?
- CUDA Graph가 어떤 mutable state를 capture해야 하는가?

이 장의 목표는 attention 이름을 **실제 추론 자원과 SLO**로 번역하는 것이다.

---

# 2. 가장 먼저 Prefill과 Decode를 분리한다

## 2.1 Prefill

**Prefill(프리필, prompt token 전체를 모델에 넣어 hidden state와 cache/state를 만드는 단계)**은 많은 token을 동시에 처리한다.

대표 병목:

- QKV/FFN/MoE large GEMM
- long sequence attention
- context parallel communication
- KV/state write

긴 context에서 TTFT(Time To First Token)를 지배한다.

## 2.2 Decode

**Decode(디코드, 새 output token을 한 step씩 생성하는 단계)**는 request마다 token 수가 작고 model weights/cache를 반복적으로 읽는다.

대표 병목:

- HBM weight read
- KV/state read/write
- small/skinny GEMM
- collective synchronization
- scheduler overhead

ITL(Inter-Token Latency)과 TPOT(Time Per Output Token)을 지배한다.

같은 attention도 prefill과 decode에서 최적 kernel이 다를 수 있다.

---

# 3. Conventional KV Cache 계산

GQA/MHA의 한 layer logical cache:

$$
M_{layer}
=
2\times T\times H_{kv}\times d_h\times b
$$

- $T$: cached tokens
- $H_{kv}$: KV heads
- $d_h$: head dimension
- $b$: bytes/element
- `2`: K와 V

전체 $L$ layer라면:

$$
M_{KV}
=
2LTH_{kv}d_hb
$$

### 예시

- $L=64$
- $H_{kv}=8$
- $d_h=128$
- $T=200000$
- BF16, $b=2$

이면:

$$
M_{KV}
=2\times64\times200000\times8\times128\times2
$$

$$
\approx 52.4\text{ GB}
$$

의 logical KV가 한 sequence에 필요하다.

FP8이면 raw element bytes는 절반 수준이지만 scale metadata, alignment와 implementation overhead는 별도다.

이 계산 하나만으로도 long-context serving에서 GQA/MLA/recurrent attention이 왜 중요한지 알 수 있다.

---

# 4. KV Cache Capacity와 실제 Concurrency는 다르다

KV cache가 `200K request 20개분` 있다고 해서 실제로 20개가 동시에 decode된다는 뜻은 아니다.

실제 concurrency는 다음에 의해 더 낮아질 수 있다.

- model FLOPs
- MoE expert GEMM throughput
- all-to-all communication
- TP collective
- max-num-batched-tokens
- scheduler policy
- prefill/decode interference
- memory bandwidth
- CUDA graph batch bucket
- request length distribution

따라서 KV capacity는 **memory upper bound**이지 throughput capacity가 아니다.

---

# 5. Attention Mechanism별 Request State

## 5.1 MHA/GQA

```text
Request
 └─ KV blocks per layer
```

Token 수에 따라 증가한다.

## 5.2 MLA

```text
Request
 └─ latent KV blocks + position component per layer
```

Token당 bytes는 작지만 token 수에 따라 증가한다.

## 5.3 Sliding Window

```text
Request
 └─ recent W-token KV ring/buffer
```

순수 local layer라면 bounded state가 가능하다.

## 5.4 Recurrent GDN/KDA

```text
Request
 ├─ recurrent matrix state per linear layer
 └─ short-convolution state per linear layer
```

Context length에 무관한 fixed state다.

## 5.5 Hybrid Recurrent + Full/MLA

```text
Request
 ├─ recurrent state
 ├─ conv state
 └─ token-wise global KV/latent blocks
```

Qwen3.6와 Kimi-K3가 이 범주다.

## 5.6 Compressed Attention

```text
Request
 ├─ compressed long-range entries
 └─ recent raw local entries
```

DeepSeek-V4 같은 구조다.

---

# 6. HBM Bandwidth 관점

Decode에서 full attention 한 layer의 현재 query는 cached K/V를 읽는다.

읽기 traffic은 대략:

$$
B_{read}\propto 2TH_{kv}d_hb
$$

이다.

Context $T$가 커질수록 ITL이 증가하는 주요 이유 중 하나다.

### GQA

$H_{kv}$를 줄여 bytes를 줄인다.

### MLA

token당 cached dimension을 줄인다.

### Sparse Attention

실제로 읽는 token count를 $T\rightarrow K$로 줄인다.

### Compressed Attention

history entry count를 $T\rightarrow T/m$으로 줄인다.

### Recurrent Attention

과거 token cache read 자체를 fixed state read로 바꾼다.

각 방법은 HBM traffic을 줄이는 **축이 다르다.**

---

# 7. Arithmetic Intensity

**Arithmetic Intensity(아리스메틱 인텐시티, 메모리에서 읽은 byte당 수행하는 연산량)**는 GPU utilization을 이해하는 핵심이다.

$$
AI=\frac{FLOPs}{Bytes\ transferred}
$$

Prefill large GEMM은 AI가 높아 compute-bound가 되기 쉽다.

Decode small batch는 weights/KV를 많이 읽지만 계산 재사용이 낮아 memory-bound가 되기 쉽다.

Attention compression이 decode에서 큰 의미를 갖는 이유는 FLOPs보다 **bytes/token**을 줄이는 효과가 크기 때문이다.

---

# 8. FlashAttention과 PagedAttention을 혼동하지 않는다

## 8.1 FlashAttention

같은 exact softmax attention을 tile/online-softmax로 계산해 HBM I/O와 intermediate materialization을 줄이는 **compute algorithm/kernel**이다.

## 8.2 PagedAttention

KV cache를 fixed-size physical blocks로 나눠 request의 logical sequence와 physical memory 위치를 분리하는 **serving memory-management algorithm**이다.

```text
FlashAttention
  → attention 계산 방법

PagedAttention
  → KV 저장/주소 관리 방법
```

두 기술은 함께 사용할 수 있다.

---

# 9. Block Size와 Fragmentation

Paged KV cache는 fixed-size block을 사용한다.

Block size가 크면:

- metadata/lookup overhead 감소
- contiguous kernel access에 유리할 수 있음
- 마지막 partially-filled block waste 증가

Block size가 작으면:

- fragmentation 감소
- block table/allocator overhead 증가
- kernel locality가 나빠질 수 있음

Hybrid state 모델은 conventional KV block 외에 recurrent-state block이 별도로 존재할 수 있어 allocator design이 더 복잡해진다.

---

# 10. Prefix Caching: Full Attention

동일 prefix를 가진 두 request:

```text
Request A: [system prompt][document][question A]
Request B: [system prompt][document][question B]
```

Full attention에서는 `[system prompt][document]`의 KV block을 hash 기반으로 공유할 수 있다.

필요 조건:

- token sequence 동일
- model/config 동일
- position state 동일
- cache dtype/adapter 등 호환

Prefix caching은 prefill compute와 KV duplication을 줄인다.

---

# 11. Prefix Caching: Recurrent Attention

Recurrent state에서는 prefix 끝의 state가 필요하다.

$$
Prefix_{0:N}ightarrow S_N
$$

그리고 ShortConv가 있다면:

$$
Prefix_{0:N}ightarrow(S_N,C_N)
$$

가 필요하다.

### 문제

Full attention은 prefix 중간의 KV blocks가 모두 그 자체로 재사용 가능하지만 recurrent model은 arbitrary prefix boundary에서 state snapshot이 없으면 해당 지점까지 recurrence를 다시 계산해야 한다.

### 가능한 설계

1. KV/prefix block boundary마다 recurrent state snapshot 저장
2. 큰 checkpoint interval만 저장하고 중간 구간 recompute
3. host/offload tier에 state snapshot 보관
4. prefix cache의 logical block과 recurrent state block을 별도 관리

### Trade-off

Snapshot을 너무 자주 저장하면:

$$
O(\text{number of prefix blocks}\times\text{large recurrent state})
$$

가 되어 fixed-state memory advantage를 잃는다.

이것이 KDA/GDN prefix caching이 conventional KV cache보다 어려운 이유다.

---

# 12. Prefix Caching: Compressed Attention

DeepSeek-V4 같은 sequence compressor가 있을 때 prefix boundary가 compressor window와 정확히 일치하지 않을 수 있다.

특히 overlapping compressor라면 새 suffix의 첫 compressed entry가 prefix 끝 token 일부와 함께 계산될 수 있다.

검토 항목:

- compression stride
- compression kernel/window
- block boundary
- local raw branch
- indexer representation
- compressed-entry hash

따라서 raw token hash만으로 compressed state를 재사용할 수 있는지 framework 구현을 확인해야 한다.

---

# 13. Continuous Batching

**Continuous Batching(컨티뉴어스 배칭, running batch에서 완료된 request를 제거하고 waiting request를 즉시 투입하는 scheduler 방식)**은 높은 GPU utilization의 핵심이다.

Request마다 variable state가 있으므로 batch slot mapping이 필요하다.

### Conventional model

```text
slot → block_table → KV pages
```

### Recurrent hybrid model

```text
slot
 ├─ recurrent_state_ptr
 ├─ conv_state_ptr
 └─ global_KV_block_table
```

Request swap 시 모든 state pointer의 ownership이 정확해야 한다.

---

# 14. Preemption

GPU memory pressure 또는 scheduling policy로 request를 잠시 빼는 **Preemption(프리엠션, 실행 중단 후 재개)**에서 state를 어떻게 처리할지 결정해야 한다.

## 14.1 Recompute

KV/state를 버리고 나중에 prompt/prefix부터 다시 계산한다.

- GPU memory 절약
- resume TTFT/compute 증가

## 14.2 Swap/Offload

CPU memory/NVMe/remote tier로 state를 이동한다.

- recompute 감소
- PCIe/network bandwidth 필요

Recurrent matrix state는 token 수와 무관하지만 layer당 state 자체가 클 수 있어 offload frequency가 높으면 PCIe traffic이 커질 수 있다.

---

# 15. Speculative Decoding과 State Fork

Speculative decoding에서 target sequence가:

```text
committed prefix
   + candidate1
   + candidate2
   + candidate3
```

까지 임시로 진행됐는데 candidate2에서 reject됐다고 하자.

### KV model

candidate2 이후 KV block/slot을 버리면 된다.

### Recurrent model

State도 candidate token마다 순차적으로 변한다.

$$
S_N\rightarrow S_{N+1}\rightarrow S_{N+2}\rightarrow S_{N+3}
$$

candidate1만 accept됐다면 $S_{N+1}$로 돌아가야 한다.

방법:

- step별 state snapshot
- branch copy
- checkpoint + replay
- multi-token verification-specific recurrent kernel

따라서 speculative decoding의 메모리 이득/손실을 recurrent model에서 별도로 평가해야 한다.

---

# 16. CUDA Graph

**CUDA Graph(쿠다 그래프, 반복 GPU launch sequence를 capture/replay하여 CPU launch overhead를 줄이는 기능)**는 decode latency에 중요하다.

Graph capture가 쉬우려면:

- tensor shape
- address/layout
- control flow
- workspace

가 안정적이어야 한다.

Hybrid model에서는:

- KV block table은 request마다 다름
- recurrent state pointer도 slot마다 다름
- sparse top-k index는 runtime마다 다름
- compressed cache length가 달라짐

같은 dynamic 요소가 존재한다.

Serving framework는 indirection table이나 piecewise graph를 이용해 이를 처리한다.

따라서 모델 load 시 `cudagraph capture succeeded`만 보는 것보다 실제 attention backend가 graph replay path를 타는지 확인해야 한다.

---

# 17. Tensor Parallelism

**Tensor Parallelism(텐서 패럴렐리즘; TP)**은 한 layer tensor 연산을 여러 GPU에 분할한다.

### Full Attention

일반적으로 Q/K/V heads 또는 projection dimensions를 rank에 shard한다.

GQA에서 KV head 수가 TP size보다 작거나 나눠떨어지지 않으면 KV head를 일부 replicate할 수 있다.

### MLA

- latent KV rank
- Q/K/V up/down projection
- absorbed matrices
- RoPE component

중 어느 axis를 shard할지 고려한다.

### GDN/KDA

Recurrent state를 head/key/value dimensions 중 어느 axis로 shard하는지 kernel이 결정한다.

TP size가 state/head shape와 맞지 않으면:

- state replication
- extra all-reduce/all-gather
- padding

이 발생할 수 있다.

> TP 숫자를 model parameter size만 보고 선택하지 말고 attention head/state divisibility도 본다.

---

# 18. Context Parallelism

**Context Parallelism(컨텍스트 패럴렐리즘; CP)**은 긴 sequence token 축을 여러 GPU로 나눈다.

### Full Attention

query shard가 remote K/V를 필요로 하므로 all-to-all/ring attention 같은 communication이 필요하다.

### Linear/Recurrent Attention

Chunk별 local computation 후 boundary recurrent state를 다음 context shard로 전달할 수 있다.

이론적으로 raw KV 전체 교환보다 state communication이 작을 수 있지만 recurrence dependency와 prefix-scan algorithm이 필요하다.

### Sparse Attention

Global indexer가 어느 shard의 token이 top-k인지 결정해야 한다.

### Compressed Attention

compressed memory를 CP 단위로 shard할 수 있지만 compressor boundary와 global dense/sparse search를 조율해야 한다.

Attention mechanism은 CP communication graph를 직접 바꾼다.

---

# 19. Data Parallelism과 Expert Parallelism

Attention은 주로 TP/CP와 직접 연결되지만 MoE 모델에서는 **Expert Parallelism(엑스퍼트 패럴렐리즘; EP)**과도 간접적으로 영향을 주고받는다.

한 iteration에서:

```text
Attention
  ↓
MoE all-to-all / expert GEMM
  ↓
Attention
```

이 반복된다.

Attention을 줄였더라도 MoE expert path가 bottleneck이면 전체 throughput은 크게 증가하지 않을 수 있다.

예를 들어 recurrent attention으로 KV가 충분해도 MoE all-to-all이나 expert GEMM saturation 때문에 running sequence 수가 늘지 않을 수 있다.

따라서 attention optimization 효과는 전체 layer profile에서 측정해야 한다.

---

# 20. KV Cache Quantization

KV dtype을 BF16에서 FP8로 줄이면 raw cache byte는 대략 절반이 된다.

$$
b:2\rightarrow1
$$

하지만 실제 비용에는:

- scale tensor
- dequantization/requantization
- kernel support
- accuracy impact
- mixed precision RoPE component

가 있다.

DeepSeek-V4처럼 architecture 자체가 compression과 low precision을 함께 설계한 모델에서는 단순 `FP8 KV on/off`와 cache representation이 다를 수 있다.

---

# 21. Cache Offloading과 Attention 종류

CPU/remote tier로 KV/state를 offload할 때 transfer unit이 다르다.

### GQA/MLA

token block을 offload/load한다.

### Recurrent Attention

sequence state snapshot을 offload할 수 있다.

State는 fixed-size지만 한 request의 모든 history를 대표하므로 miss 시 일부 token block만 가져오는 방식과 semantics가 다르다.

### Sparse/Compressed

- selected historical blocks
- compressed blocks
- local raw blocks

등 tiering granularity를 설계해야 한다.

Cache-aware router도 `prefix KV block locality`뿐 아니라 future recurrent/ compressed state semantics를 고려해야 할 수 있다.

---

# 22. TTFT와 ITL Trade-off

긴 prompt에서 `max-num-batched-tokens` 같은 scheduler budget을 작게 하면 chunked prefill iteration 수가 늘어 TTFT가 증가할 수 있다.

반대로 decode token을 더 자주 scheduling하면 ITL이 줄어들 수 있다.

Attention 구조는 이 trade-off를 바꾼다.

- Full attention: prefill chunk가 커질수록 attention compute 증가
- Linear attention: chunkwise recurrence kernel shape에 따라 optimum chunk 존재
- Sparse/Compressed: index/compression overhead가 chunk마다 발생 가능

따라서 scheduler knob의 최적값을 모델 family가 달라도 그대로 복사하면 안 된다.

---

# 23. 주요 Metrics

Attention architecture를 배포할 때 최소한 다음을 함께 본다.

## Latency

- TTFT p50/p95/p99
- ITL p50/p95/p99
- TPOT
- end-to-end latency

## Scheduler

- running requests
- waiting requests
- preemptions
- batch tokens
- batch sequences

## Cache/State

- KV usage
- prefix hit rate
- local compute / cache-hit path ratio
- recurrent state allocation
- offload hit/miss

## GPU

- SM utilization
- Tensor Core utilization 가능 지표
- HBM bandwidth/utilization
- GPU memory
- NVLink/NVSwitch traffic
- IB/RDMA traffic

## MoE

- expert load imbalance
- EP all-to-all
- expert GEMM utilization

단일 `GPU util 95%`만으로 bottleneck을 판단하지 않는다.

---

# 24. 실전 모델 비교 프레임

새 모델 A/B를 동일 GPU에서 비교할 때:

1. 같은 context distribution
2. 같은 output length distribution
3. 같은 concurrency sweep
4. 같은 cache dtype
5. 같은 scheduler budget
6. kernel/backend 로그 확보
7. warm-up/CUDA graph 완료 후 측정

을 맞춘다.

그 뒤 다음을 분리한다.

```text
TTFT 차이
 ├─ attention prefill
 ├─ MoE/FFN
 ├─ scheduler chunking
 └─ CP/TP communication

ITL 차이
 ├─ weight bandwidth
 ├─ KV/state bandwidth
 ├─ recurrent kernel
 ├─ global attention interval
 └─ collective
```

---

# 25. Attention별 Deployment Checklist

## Conventional GQA

- [ ] query/KV head divisibility with TP
- [ ] KV dtype
- [ ] FlashAttention backend
- [ ] block size
- [ ] prefix caching
- [ ] max model length

## MLA

- [ ] latent-cache backend 사용 여부
- [ ] matrix absorption 경로
- [ ] FlashMLA 또는 optimized MLA kernel
- [ ] RoPE cache component dtype
- [ ] TP sharding

## GDN/KDA

- [ ] recurrent state dtype/shape
- [ ] ShortConv state
- [ ] chunkwise prefill kernel
- [ ] fused recurrent decode kernel
- [ ] state-aware prefix caching
- [ ] preemption/resume correctness
- [ ] SD rollback correctness

## Sparse Attention

- [ ] indexer backend
- [ ] top-k config
- [ ] paged sparse gather
- [ ] sparse attention kernel
- [ ] indexer precision

## Compressed Attention

- [ ] compressor kernel
- [ ] compressed block allocator
- [ ] local raw branch
- [ ] compression boundary/prefix cache
- [ ] layer별 compression ratio

---

# 26. 로그에서 확인해야 하는 것

Model config에 기능이 있다고 실제 optimized path를 탄다는 보장은 없다.

로그/프로파일에서 확인한다.

- selected attention backend
- FlashAttention/FlashMLA/FlashKDA/FlashQLA backend
- fallback to eager/Triton/reference implementation 여부
- cudagraph mode
- kernel compile result
- FP8 KV support
- prefix cache support
- hybrid cache manager mode
- TP/CP backend

특히 최신 모델은 `지원됨`과 `고성능 optimized path가 완성됨` 사이에 큰 차이가 있을 수 있다.

---

# 27. Architecture를 SLO로 번역하기

| Architecture 변화 | 기대되는 시스템 효과 | 반드시 확인할 부작용 |
|---|---|---|
| MHA→GQA | KV byte/decode read 감소 | quality/head sharing, TP replication |
| GQA→MLA | token당 cache 대폭 감소 | specialized kernel, latent layout |
| Global→SWA | bounded local cache/compute | long-range exact retrieval |
| Global→Sparse | read/compute token 수 감소 | index/top-k/gather overhead |
| Raw→Compressed | cache entry 수 감소 | compression information loss |
| Full→Recurrent | history-length-independent state | finite state capacity, state management |
| Pure→Hybrid | efficiency/quality 절충 | cache manager와 kernel 복잡도 증가 |

---

# 28. 최종 Mental Model

Serving engineer 입장에서 attention은 다음 네 질문으로 귀결된다.

### 28.1 무엇을 저장하는가?

```text
raw KV?
latent KV?
compressed entries?
fixed recurrent state?
```

### 28.2 새 token마다 무엇을 읽는가?

```text
all history?
recent window?
top-k?
compressed history?
one fixed state?
```

### 28.3 Prefix/branch를 어떻게 복제하는가?

```text
KV blocks?
state snapshots?
compressed boundaries?
```

### 28.4 GPU에서는 어떤 kernel/communication으로 실행되는가?

```text
FlashAttention?
FlashMLA?
FlashKDA/FlashQLA?
sparse gather?
TP/CP collectives?
```

이 네 가지를 알면 논문의 `efficient attention`이라는 표현을 실제 GPU memory, TTFT, ITL, throughput으로 번역할 수 있다.

---

# 29. 핵심 정리

1. KV cache capacity는 concurrency의 memory upper bound일 뿐 compute throughput을 보장하지 않는다.
2. Decode에서는 KV/state HBM traffic이 ITL에 직접 영향을 준다.
3. GQA, MLA, sparse, compressed, recurrent attention은 서로 다른 axis의 memory traffic을 줄인다.
4. FlashAttention은 model attention을 바꾸지 않고 exact attention의 GPU I/O를 줄인다.
5. PagedAttention은 attention 수식을 바꾸지 않고 KV cache allocation/fragmentation을 해결한다.
6. Recurrent attention은 prefix caching, preemption, speculative rollback에서 state snapshot 문제가 생긴다.
7. Hybrid model은 KV cache manager와 recurrent state manager를 동시에 요구한다.
8. TP/CP scaling은 attention state shape와 communication graph에 따라 크게 달라진다.
9. Attention 최적화가 커도 MoE/collective가 병목이면 end-to-end throughput gain이 작을 수 있다.
10. 실제 배포에서는 `architecture support`와 `optimized kernel path`를 반드시 구분해 검증해야 한다.

---

# 30. 참고 자료

- FlashAttention  
  https://arxiv.org/abs/2205.14135
- PagedAttention / vLLM  
  https://arxiv.org/abs/2309.06180
- DeepSeek FlashMLA  
  https://github.com/deepseek-ai/FlashMLA
- MoonshotAI FlashKDA  
  https://github.com/MoonshotAI/FlashKDA
- Qwen FlashQLA  
  https://github.com/QwenLM/FlashQLA
- vLLM documentation  
  https://docs.vllm.ai/
