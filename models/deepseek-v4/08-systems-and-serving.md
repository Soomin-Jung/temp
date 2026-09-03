# 08. DeepSeek-V4 Systems & Serving

작성일: 2026-08-25

## 1. V4는 model architecture만 보면 실제 serving cost를 설명할 수 없다

DeepSeek-V4 inference는 최소 다섯 개의 resource domain이 겹친다.

```text
Dense / attention compute
Compressed-memory compute
KV/index cache bandwidth
MoE expert compute + all-to-all
Speculative draft + verification
```

따라서 단일 `tokens/s` 숫자보다 각 sub-path를 분리해서 profile해야 한다.

---

## 2. V4의 cache는 하나가 아니다

일반 Transformer의 KV cache 이미지는:

```text
K cache + V cache per layer
```

정도로 단순하다.

V4는 layer type에 따라 state가 다르다.

### SWA-only layer

```text
local raw K=V window cache
```

### CSA layer

```text
local SWA cache
+ compressed c4 attention cache
+ sparse indexer cache
```

### HCA layer

```text
local SWA cache
+ c128 heavily-compressed cache
```

따라서 prefix caching / KV transfer / P-D disaggregation에서 `KV`라는 단일 tensor 묶음으로 생각하면 구현 세부를 놓친다.

---

## 3. 1M context가 가능한 이유

V4의 context efficiency stack:

```text
1. MQA-like shared K=V
2. recent 128-token exact SWA
3. c4 compressed memory
4. c4 memory에 DSA top-k
5. c128 coarse memory
6. FP8 attention cache
7. FP4 indexer cache
```

vLLM의 architecture analysis는 1M context의 BF16-equivalent cache가 V3.2-style stack보다 크게 작아진다고 설명한다.

중요한 것은:

> 1M context support는 scheduler option이 아니라 cache representation architecture의 결과다.

---

## 4. Prefill과 Decode의 병목이 다르다

### Long prefill

주요 비용:

- compressor generation
- indexer scoring
- attention over long memory
- mHC hidden traffic
- MoE token dispatch

긴 prompt에서는 sequence-parallel/context-parallel 계열 최적화가 중요하다.

### Decode

주요 비용:

- per-token attention cache reads
- sparse index lookup
- MoE expert weight bandwidth / EP communication
- DSpark draft/verify
- CUDA launch / graph overhead

따라서 P/D disaggregation 관점에서도 V4는 prefill/decode resource balance가 크게 다를 수 있다.

---

## 5. Tensor Parallelism

V4는 dense attention projection이 작지 않다.

Pro 기준:

```text
hidden = 7168
128 query heads × 512 head dim
```

TP는:

- Q projection
- grouped output projection
- LM head
- shared expert path

등을 shard하는 데 중요하다.

하지만 MoE expert bank까지 TP만으로 나누면 expert execution/communication 효율이 나빠질 수 있다.

그래서 vLLM recipe는 MoE checkpoint에 EP 사용을 적극 권장한다.

---

## 6. Expert Parallelism

EP는 routed expert bank를 GPU들에 분산한다.

Token flow:

```text
router
→ token dispatch
→ destination expert GPU
→ expert compute
→ result return/reduce
```

병목:

- all-to-all latency
- token imbalance
- small expert GEMM efficiency
- inter-node network

특히 Pro는 384 expert이고 checkpoint도 매우 크므로 EP는 memory placement 자체에도 중요하다.

---

## 7. TP + EP를 함께 봐야 하는 이유

V4에는 두 종류의 큰 matrix가 섞여 있다.

### Dense replicated/shared components

- attention projections
- mHC parameters
- embedding / LM head
- shared expert

### Sparse components

- routed experts

TP만 줄이면 sparse expert placement가 비효율적이고, EP만 늘리면 dense weights replication이 커질 수 있다.

따라서 최적 topology는:

```text
TP × EP × DP
```

joint design 문제다.

---

## 8. FP4 expert backend

V4의 routed experts는 native FP4 checkpoint다.

서빙 엔진 입장에서 확인할 것:

- checkpoint FP4 format
- GPU architecture의 FP4 native support
- kernel이 dequant-before-GEMM인지 native low-precision path인지
- scale format
- shared expert precision
- router precision

`FP4 checkpoint`라는 말만으로 실제 tensor-core path를 알 수 없다.

Blackwell 계열에서는 NVFP4 variant도 존재하며 native DeepSeek FP4/FP8 checkpoint와 kernel selection이 다를 수 있다.

---

## 9. DeepGEMM / FlashInfer / FlashMLA를 구분한다

V4의 NVIDIA runtime에는 여러 backend가 동시에 개입할 수 있다.

### DeepGEMM 계열

주로 expert/dense low-precision GEMM 경로.

### FlashInfer sparse path

V4 sparse attention / cache backend에 사용될 수 있다.

### FlashMLA naming

vLLM이 V4 attention을 기존 sparse-MLA infrastructure 위에서 구현하기 때문에 file/class 명에 MLA가 등장한다.

따라서 장애가 발생하면:

```text
MoE GEMM failure?
attention kernel failure?
indexer kernel failure?
cache layout failure?
```

를 먼저 나눈다.

기존 H100 SM90 DeepGEMM IMA 사례는:

- [2026-08-18-vllm-0.27-deepgemm-sm90-cuda-ima.md](notes/2026-08-18-vllm-0.27-deepgemm-sm90-cuda-ima.md)

참고.

---

## 10. CUDA Graph

V4 current vLLM source는 graph capture와 dynamic sparse path 사이를 명시적으로 조정한다.

attention projection/input preparation은 capture에 남기고, 일부 sparse indexer/attention path는 controlled eager break를 사용한다.

따라서 운영에서 볼 metric:

- graph capture 성공 여부
- graph replay hit ratio
- piecewise/eager break 빈도
- batch-size bucket
- decode shape
- speculative verify shape

이다.

`--enforce-eager`가 안정적이라고 해서 production 최적 설정은 아니다. 먼저 kernel/graph bug를 분리하기 위한 diagnostic mode로 본다.

---

## 11. Hybrid KV Cache Manager

V4처럼 layer마다 cache type/size가 다른 model은 uniform KV allocation 가정이 비효율적이다.

hybrid KV cache manager는:

- SWA
- compressed cache
- sparse indexer state

처럼 서로 다른 cache spec을 인식해 page allocation을 조정해야 한다.

V4 recipe에서 hybrid KV cache manager를 유지하는 이유다.

---

## 12. Prefix caching

V4 prefix state는 단순 raw KV가 아니다.

prefix reuse에 필요한 state가 layer별로 다르다.

```text
SWA state
compressed entries
indexer cache/state
position/compression boundary metadata
```

그래서 prefix hash hit만 맞는다고 끝이 아니다.

implementation은 compression boundary와 cache layout까지 재현해야 한다.

---

## 13. P/D Disaggregation에서의 핵심 질문

V4를 P/D로 나누면 prefill side가 만들어 decode side로 넘겨야 할 state를 정확히 정의해야 한다.

질문:

1. local SWA cache를 어떻게 전달하는가?
2. CSA compressed cache는 physical 어떤 layout인가?
3. indexer cache도 transfer 대상인가, decode에서 rebuild 가능한가?
4. HCA compressed entries는 어떤 boundary까지 완성됐는가?
5. FP8/FP4 cache format을 connector가 그대로 운반하는가?
6. page/block metadata를 어떻게 동기화하는가?

즉 V4에서 KV transfer는 `K,V tensor copy`보다 넓은 문제다.

---

## 14. vLLM P/D recipe의 의미

현재 vLLM recipe는 DeepSeek-V4에서 Mooncake/NIXL 기반 P/D 예제를 제공한다.

하지만 production 적용 시 반드시 별도 검증할 것:

- connector가 V4 hybrid cache spec을 정확히 지원하는지
- sparse/indexer state transfer 구현 버전
- network transport
- decode cache registration
- prefix/block ID consistency

recipe 존재와 특정 vLLM/connector 조합의 production stability는 별개다.

---

## 15. DSpark와 serving load

Speculative decoding benefit은 concurrency와 workload에 따라 달라진다.

낮은 concurrency:

```text
latency reduction benefit ↑
```

높은 concurrency:

```text
target GPU가 이미 batch로 잘 차면
extra draft compute가 throughput을 해칠 수도 있음
```

그래서 vLLM recipe도 spec decoding을 latency/small-batch feature로 분류한다.

DSpark adaptive verification은 이 trade-off를 request/load-aware하게 최적화하려는 방향이다.

---

## 16. Reasoning mode와 capacity planning

최신 V4 checkpoint는 reasoning effort를 제공한다.

agent/reasoning workload에서는 output length 자체가 크게 달라질 수 있다.

따라서 capacity planning 시:

```text
model = same
reasoning effort = different
```

이어도:

- output token count
- KV residency time
- request service time
- DSpark acceptance behavior
- scheduler occupancy

가 달라진다.

`model TPS` 하나로 SLO를 설계하면 안 된다.

---

## 17. Flash와 Pro의 serving 성격

### Flash

- 284B / 13B active
- 43 layers
- 256 experts
- smaller retrieval top-k
- single-node 8-GPU class deployment이 현실적

### Pro

- 1.6T / 49B active
- 61 layers
- 384 experts
- larger hidden/attention/expert widths
- memory footprint 때문에 multi-GPU/possibly multi-node topology가 훨씬 중요

동일 V4 architecture family지만 infrastructure class는 크게 다르다.

---

## 18. Current vLLM recipe 기준

2026-08-25 조회 기준 vLLM recipes에는:

- `DeepSeek-V4-Flash-0731`
- `DeepSeek-V4-Pro-0813`

을 위한 NVIDIA serving profile이 존재한다.

대표 설정 축:

```text
--kv-cache-dtype fp8
--enable-expert-parallel
--tensor-parallel-size ...
--tokenizer-mode deepseek_v4
--reasoning-parser deepseek_v4
--speculative-config method=dspark
```

다만 recipe의 minimum version과 실제 production baseline은 반드시 kernel/GPU별 회귀 테스트로 결정한다.

---

## 19. Profiling checklist

### Attention

- SWA time
- compressor time
- indexer time
- sparse/global attention time
- output projection

### MoE

- router
- dispatch
- expert GEMM
- shared expert
- all-to-all

### mHC

- pre/post mix kernel
- hidden memory bandwidth

### DSpark

- draft time
- accepted length
- verify time

### Cache

- bytes/token
- cache hit
- compression entry count
- index cache size

---

## 20. 출처

- V4 Technical Report: https://arxiv.org/abs/2606.19348
- vLLM V4 architecture blog: https://vllm-project.github.io/2026/04/24/deepseek-v4.html
- vLLM Flash recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- vLLM Pro recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro
- vLLM V4 source: https://github.com/vllm-project/vllm/tree/main/vllm/models/deepseek_v4
