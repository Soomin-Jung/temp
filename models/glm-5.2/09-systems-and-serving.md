# 09. Systems & Serving — NVIDIA / vLLM / SGLang

작성일: 2026-08-25  
목적: GLM-5.2 architecture가 실제 inference system에서 어떤 resource/parallelism/kernel requirement를 만드는지 정리한다.

## 1. 먼저 model-level constraint를 다시 본다

GLM-5.2는 일반적인 dense decoder가 아니다.

```text
~743–744B total / ~39–40B active MoE
+ MLA
+ DSA top-k=2048
+ IndexShare
+ 1M native context
+ MTP speculative decoding
```

따라서 serving 병목도 여러 종류가 겹친다.

```text
weight capacity     → giant MoE
communication       → expert parallel
KV capacity         → 1M context
attention compute   → DSA
index compute       → IndexShare
serial decode       → MTP
CPU/runtime         → long-context scheduling/cache metadata
```

한 optimization만 켜고 `GLM-5.2 최적화 완료`라고 할 수 없다.

---

## 2. Current vLLM ecosystem baseline

current vLLM recipe는 GLM-5.2를 다음처럼 설명한다.

- ~743B total / ~39B active
- context 1,048,576
- vLLM 0.23.0+ recipe baseline
- FP8: practical NVIDIA default
- 8×H200/H20: single-node FP8 serving
- 8×B200: FP8 KV cache와 함께 full 1M-context target
- Blackwell: NVFP4 variant 지원
- MTP speculative decoding
- GLM reasoning/tool parser

이 숫자는 **recipe의 verified/reference configuration**이고, 현재 플랫폼의 vLLM 0.26/0.27 계열에서 그대로 copy하는 것이 아니라 regression을 다시 해야 한다.

engine version이 올라가면 특히:

- DSA kernel
- DeepGEMM/MoE kernel
- MTP
- CUDA graph
- distributed backend

이 바뀔 수 있다.

---

## 3. Weight memory부터 계산한다

단순 raw parameter memory approximation:

```text
744B × BF16(2 bytes) ≈ 1.49 TB
744B × FP8 (1 byte) ≈ 744 GB
744B × 4-bit         ≈ 372 GB
```

실제 runtime은:

- quantization metadata/scales
- embeddings/non-quantized weights
- CUDA/NCCL memory
- activation workspace
- expert buffers
- KV cache
- graph capture pools

이 추가된다.

current vLLM recipe가 FP8 variant에 raw 744GB보다 큰 practical memory requirement를 제시하는 이유다.

따라서 `8×H200 = 1128GB니까 1M도 충분하겠지`라고 계산하면 안 된다. **weight가 fit하는 것과 usable KV budget이 충분한 것은 별개다.**

---

## 4. 1M context에서 bottleneck이 이동한다

Z.AI 공식 설명:

GLM-5.2가 200K-class에서 1M으로 올라가면 primary inference bottleneck이 점점:

- KV-cache capacity
- long-context kernel overhead
- CPU-side cache management
- scheduler/runtime overhead
- cache transfer pipeline

으로 이동한다.

DSA/IndexShare가 attention FLOPs를 줄였기 때문에 오히려 다른 overhead가 더 잘 보인다.

이것은 중요한 시스템 법칙이다.

> **한 병목을 architecture에서 제거하면 그 다음 병목이 end-to-end latency를 지배한다.**

---

## 5. KV cache는 MLA만 보고 계산하면 안 된다

model architecture 수준에서는 MLA의 compressed KV rank가 512라 KV memory가 크게 줄어든다.

하지만 production memory accounting에는 다음이 들어간다.

```text
MLA latent / KV representation
+ RoPE key state
+ DSA indexer cache
+ sparse-attention metadata
+ block table
+ prefix-cache metadata
+ speculative/MTP state
```

그리고 engine backend가 실제로 compressed latent를 cache하는지 expanded K/V를 cache하는지도 확인해야 한다.

따라서 GLM-5.2를 올릴 때 가장 먼저 engine에서 확인할 것은:

```text
“이 backend의 GLM-MoE-DSA KV cache physical layout이 무엇인가?”
```

이다.

---

## 6. DSA kernel support는 load support와 다르다

세 단계로 나눠야 한다.

### Level 1 — Model loads

`GlmMoeDsaForCausalLM` checkpoint가 로드되고 output이 나온다.

### Level 2 — Correct sparse attention

top-k indices를 사용해 결과가 reference와 맞는다.

### Level 3 — Efficient sparse kernel

selected position만 실제로 gather/read하는 specialized sparse MLA/SFA kernel을 사용한다.

Level 1/2만 되고 Level 3가 없으면 1M에서 성능이 무너질 수 있다.

profiler에서 확인:

- sparse kernel name
- indexer GEMM
- top-k kernel
- gather/cache access
- dense fallback 여부

---

## 7. IndexShare runtime validation

current config는 layer별 `full/shared` pattern을 갖는다.

runtime에서 검증할 것:

```text
Full layer
  index projection + score + topk

Shared layer
  no indexer compute
  previous topk reuse
```

profiler상 기대:

- shared layer에서 indexer QK kernel 없음
- top-k op 없음
- sparse attention kernel은 그대로 존재

즉 attention layer 수가 줄어드는 것이 아니라 **index selection subgraph만 줄어든다.**

---

## 8. MoE parallelism: TP만으로 끝나지 않는다

GLM-5.2는 256 expert bank 때문에 EP가 중요하다.

### TP-centric deployment

장점:

- 단순한 model sharding
- attention/dense projection과 일관된 parallel layout

단점:

- expert weight도 TP 방식으로 쪼개면 MoE-specific locality 이점을 덜 활용할 수 있음

### EP/TEP/DEP 계열

expert를 rank별로 배치하고 token을 dispatch한다.

장점:

- expert GEMM locality
- expert bank scale-out

비용:

- all-to-all
- imbalance
- network topology dependence

실전에서는 request shape에 따라 TP/EP 조합을 검증해야 한다.

---

## 9. H200과 Blackwell을 다르게 봐야 한다

### H200

- large HBM 141GB/GPU
- FP8 weight가 현실적
- NVLink/NVSwitch intra-node가 중요
- full 1M은 concurrency/KV headroom까지 고려하면 매우 공격적인 target

vLLM recipe는 8×H200 FP8 single-node serving을 제시하지만 full 1M target은 B200 쪽을 명시한다.

### B200/B300

- 더 큰 HBM
- NVFP4 expert quantization option
- 1M KV headroom에 유리
- Blackwell-specific kernels 활용 가능

GLM-5.2-NVFP4 계열은 모든 tensor를 4-bit로 만드는 것이 아니라 MoE expert linears 중심으로 quantization하고 attention/shared/early dense path는 더 높은 precision을 유지하는 구성도 존재한다.

따라서 `NVFP4 = 모델 전체 4bit`로 단순화하지 않는다.

---

## 10. P/D disaggregation이 잘 맞는 이유

GLM-5.2 prefill과 decode의 병목이 매우 다르다.

### Prefill

```text
long input
→ DSA index construction
→ sparse attention
→ large token-batch MoE
→ KV/index cache build
```

### Decode

```text
1-step tokens
→ sparse cache read
→ latency-sensitive MoE
→ MTP draft/verify
```

따라서 P/D를 분리하면 각 phase를 다른 parallelism/replica ratio로 최적화할 수 있다.

```text
Prefill pool
  high memory bandwidth
  long-context optimized
      │
      │ KV / cache transfer
      ▼
Decode pool
  low ITL
  MTP optimized
  concurrency optimized
```

Z.AI GLM-5 RL infra에서도 P/D disaggregation을 multi-turn rollout interference 감소에 사용했다고 설명한다.

---

## 11. KV transfer와 long context

P/D에서는 GLM-5.2 cache가 크므로 KV transfer가 새로운 병목이 된다.

확인할 것:

- transfer representation이 compressed MLA state인가?
- indexer cache도 transfer해야 하는가, decode에서 reconstruct 가능한가?
- prefix cache ownership은 어디인가?
- request migration 시 sparse metadata를 어떻게 넘기는가?
- IB/RDMA path를 타는가?

NVIDIA Dynamo의 GLM-5.2 long-context recipe는 B200/H200에서 aggregated/disaggregated 구성을 제공하고, KV-aware routing·cache offload·high-speed transfer를 함께 다룬다.

P/D architecture를 볼 때 모델 KV만이 아니라 **DSA index state까지 transfer contract에 포함되는지**를 확인해야 한다.

---

## 12. Context Parallelism과 DSA

1M context는 한 GPU/rank의 cache capacity를 넘어 sequence dimension을 shard하는 Context Parallelism(CP)을 유도한다.

하지만 DSA indexer는 global top-k를 골라야 한다.

sequence가 rank별로 나뉘면 각 rank가 local top-k만 골라서는 dense reference와 다른 selection이 될 수 있다.

vLLM Ascend의 GLM-5.2 DCP design은 이를 명시적으로 해결한다.

```text
Indexer cache → replicate across DCP ranks
SFA KV cache → shard across DCP ranks
```

이 설계는 NVIDIA에서도 동일 구현을 쓴다는 뜻이 아니라, **DSA + CP가 갖는 본질적 distributed constraint를 보여주는 reference**로 볼 수 있다.

---

## 13. MTP와 CUDA Graph

MTP가 켜지면 decode graph는:

```text
base decode
+ draft iteration(s)
+ verification
+ acceptance/commit
```

으로 복잡해진다.

CUDA graph에서 확인할 것:

- decode batch size별 capture
- speculative token count 고정/가변
- top-k sparse index shape
- expert routing shape
- graph break
- first-request JIT/warmup

MTP를 켰는데 latency가 나빠진다면 acceptance만 볼 것이 아니라 **graph capture/fallback 여부**까지 봐야 한다.

---

## 14. Agent workload용 benchmark shape

chat benchmark만으로 GLM-5.2를 평가하면 모델 target과 맞지 않는다.

최소 traffic class를 분리한다.

### Short chat

```text
ISL 2K–8K / OSL 200–2K
```

### Repo coding

```text
ISL 32K–128K / OSL 4K–32K
```

### Long agent

```text
ISL 128K–500K+
long multi-turn prefix reuse
OSL cumulative tens of K
```

### Maximum-context stress

```text
500K–1M
low concurrency
cache-capacity / indexer / CPU stress
```

NVIDIA Dynamo의 GLM-5.2 recipe도 long-context agentic traffic을 별도 workload로 다룬다.

---

## 15. Metrics matrix

### Latency

- TTFT
- ITL
- E2E latency
- time/tool-step

### Throughput

- input tok/s/GPU
- output tok/s/GPU
- requests/s
- accepted speculative tokens/s

### Memory

- weight HBM
- KV HBM
- indexer cache
- graph pool
- CPU/offload cache

### Sparse attention

- indexer time
- top-k time
- sparse attention time
- selected block/token utilization

### MoE

- expert tokens/rank
- all-to-all time
- max/mean expert load
- shared expert overlap

### Cache

- prefix hit rate
- cache transfer bandwidth
- eviction/preemption count
- P/D transfer latency

---

## 16. Validation order

GLM-5.2를 플랫폼에 넣는다면 다음 순서가 안전하다.

```text
1. FP8 single-node correctness
2. reasoning/tool parser correctness
3. DSA sparse kernel 확인
4. IndexShare profiler 확인
5. MoE TP/EP topology 비교
6. MTP OFF/ON quality parity
7. MTP acceptance/throughput
8. 32K → 128K → 256K context scaling
9. prefix caching
10. multi-node / native multiprocess
11. P/D + KV transfer
12. 500K/1M stress
13. real agent trace replay
```

처음부터 1M + MTP + P/D + multi-node를 모두 켜면 failure domain을 분리하기 어렵다.

---

## 17. 기존 temp deployment 기록과 연결

과거 검토에서는 GLM-5.2-FP8을 Network A H200 multi-node/IB에서 고려했다.

현재 관점에서는 다시 검토할 때:

- Ray 장기 의존보다 vLLM native multiprocess/backend 우선
- 최신 vLLM GLM-MoE-DSA path 확인
- H200에서는 realistic context/concurrency target을 먼저 결정
- DSA/IndexShare/MTP가 모두 최적화되는 image/version 고정
- multi-node EP/TP topology와 IB traffic profiler 확인
- 필요하면 P/D를 별도 phase로 확장

하는 것이 맞다.

---

## 18. 한 문장 정리

> **GLM-5.2 serving은 ‘744B 모델을 GPU에 올리는 문제’가 아니다. MoE communication, 1M KV capacity, DSA indexer, IndexShare, MTP, CPU scheduling이 서로 병목을 넘겨받는 구조이므로 architecture-aware profiling이 필수다.**

## Sources

- GLM-5.2 official blog: https://z.ai/blog/glm-5.2
- vLLM GLM-5.2 recipe: https://recipes.vllm.ai/zai-org/GLM-5.2
- NVIDIA Dynamo recipes: https://docs.nvidia.com/dynamo/dev/recipes/browse
- vLLM Ascend GLM-5.2 reference: https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/GLM5.2.html
- GLM-5 Technical Report: https://arxiv.org/abs/2602.15763
