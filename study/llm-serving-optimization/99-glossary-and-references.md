# 99. Glossary, Source Map, and Reading Order

## 1. 핵심 용어

### Arithmetic Intensity

메모리에서 읽은 byte당 수행하는 연산량.

높을수록 compute-bound로 이동하기 쉽고, 낮을수록 memory-bandwidth 영향이 커지기 쉽다.

### Capacity

메모리에 보유할 수 있는 request/state/KV의 양.

Throughput과 다르다.

### Throughput

단위 시간에 완료하거나 생성하는 request/token 수.

Capacity가 크다고 throughput이 자동으로 큰 것은 아니다.

### Binding Constraint

현재 성능을 실제로 제한하는 상한.

예: MBT를 1K에서 2K로 올렸을 때 throughput이 증가하면 1K가 binding일 수 있다.

### Non-binding Constraint

더 키워도 성능이 거의 변하지 않는 상한.

예: max-num-seqs 512와 1024가 같은 throughput이면 512 이상은 non-binding일 수 있다.

### Prefill

입력 prompt 전체를 처리해 KV/state를 만드는 phase.

대표 KPI는 TTFT와 prompt tokens/s.

### Decode

출력 token을 autoregressive하게 생성하는 phase.

대표 KPI는 TPOT, ITL, generation tokens/s.

### TTFT

Time To First Token.

queue, Prefill compute, KV transfer, remote-load가 모두 영향을 줄 수 있다.

### TPOT

Time Per Output Token.

Decode compute, HBM, graph, TP collective, SD가 영향을 준다.

### ITL

Inter-Token Latency.

streaming token 사이 시간.

### MBT

max-num-batched-tokens.

한 scheduler/model iteration의 token budget 상한.

### max-num-seqs

한 iteration에서 처리하는 sequence 수의 상한.

scheduler뿐 아니라 metadata, graph, recurrent-state path에도 영향을 줄 수 있다.

### KV Cache

Full Attention류에서 과거 token의 Key/Value representation을 보유하는 cache.

### Recurrent State

Mamba/GDN/KDA류에서 과거 sequence를 fixed-size state로 누적한 표현.

### Hybrid Cache

token-growing KV와 recurrent state를 동시에 보유하는 구조.

### CUDA Graph

반복 kernel launch sequence를 capture/replay하여 CPU submission/launch overhead를 줄이는 CUDA 기능.

### PIECEWISE

graph-compatible partition만 CUDA Graph로 실행하고 unsupported op는 graph 밖에서 실행하는 vLLM mode.

### FULL_DECODE_ONLY

uniform Decode는 full graph, Prefill/mixed는 graph 없이 실행하는 vLLM mode.

P/D Decode instance에 특히 자연스러운 후보.

### MTP

Multi-Token Prediction.

training objective 또는 runtime speculative drafter로 사용될 수 있다.

checkpoint capability와 runtime activation을 구분한다.

### TP

Tensor Parallelism.

layer tensor를 여러 GPU rank에 shard.

compute/memory를 줄이는 대신 collective traffic을 만든다.

### PP

Pipeline Parallelism.

layer 구간을 stage로 나눔.

weight capacity를 늘릴 수 있지만 bubble과 stage transfer가 생긴다.

### DP

Data Parallelism.

model replica를 여러 개 두고 request를 분산.

throughput scale-out에 좋지만 weights/KV pool이 replica별로 중복된다.

### EP

Expert Parallelism.

MoE expert를 GPU에 분산.

all-to-all communication과 load balance가 핵심이다.

### CP

Context Parallelism.

sequence token 축을 GPU에 분산.

attention architecture마다 communication graph가 크게 다르다.

### NVLink

NVIDIA GPU-to-GPU high-bandwidth interconnect.

기능 지원과 실제 traffic은 구분해서 검증한다.

### NVSwitch

여러 NVLink endpoint를 switching fabric으로 연결하는 장치/기술.

### GDRDMA

GPUDirect RDMA.

NIC가 GPU memory와 직접 DMA하는 path를 사용해 host staging을 줄이는 기술.

---

## 2. Source-of-truth 우선순위

serving optimization에서 사실을 확인할 때 우선순위:

1. pinned serving-engine source code
2. pinned model config/checkpoint
3. official serving-engine docs
4. GPU vendor programming guide / product docs
5. model technical report
6. benchmark observation
7. issue discussion
8. secondary blog

performance recommendation은 source만으로 확정하지 않는다.

source는 path를 설명하고, benchmark가 winner를 결정한다.

---

## 3. vLLM / CUDA 핵심 원문

### vLLM CUDA Graph design

https://docs.vllm.ai/en/latest/design/cuda_graphs/

확인할 것:

- NONE
- PIECEWISE
- FULL
- FULL_DECODE_ONLY
- FULL_AND_PIECEWISE
- backend capability
- runtime downgrade
- dispatcher

### vLLM configuration API

https://docs.vllm.ai/en/latest/api/vllm/config/

확인할 것:

- CompilationConfig
- SchedulerConfig
- CacheConfig
- ParallelConfig

### NVIDIA CUDA Programming Guide

https://docs.nvidia.com/cuda/cuda-programming-guide/

CUDA Graph의 host submission overhead, graph capture/replay, memory/execution model을 읽는다.

---

## 4. NVIDIA Hardware / Fabric 원문

### NVLink / NVLink Switch

https://www.nvidia.com/en-us/data-center/nvlink/

세대별 GPU당 NVLink bandwidth와 switch domain 차이를 본다.

### H200

https://www.nvidia.com/en-us/data-center/h200/

H200 SXM 기준 HBM bandwidth와 NVLink capability를 본다.

### Blackwell Ultra architecture

https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/

HBM, NVLink 5, NV-HBI, Tensor Core와 attention acceleration 세대 차이를 읽는다.

중요:

> hardware peak는 framework throughput의 직접 기대값이 아니다. 동일 topology microbenchmark로 empirical ceiling을 별도로 만든다.

---

## 5. Attention / Model Architecture 기존 Study

### Foundations

[../attention/00-foundations.md](../attention/00-foundations.md)

### Full Attention / GQA

[../attention/01-full-attention.md](../attention/01-full-attention.md)

### MLA

[../attention/02-mla-low-rank-attention.md](../attention/02-mla-low-rank-attention.md)

### Linear / Recurrent Attention

[../attention/03-linear-recurrent-attention.md](../attention/03-linear-recurrent-attention.md)

### Hybrid Attention

[../attention/04-local-sparse-compressed-hybrid.md](../attention/04-local-sparse-compressed-hybrid.md)

### Serving Engineer View

[../attention/90-serving-engineer-view.md](../attention/90-serving-engineer-view.md)

---

## 6. Qwen

기존 Study:

[../attention/11-qwen.md](../attention/11-qwen.md)

운영 시 특히 볼 것:

- layer_types
- GDN state shape
- full-attention layer ratio
- KV heads/head dim
- MTP
- state dtype
- graph backend support

Qwen hybrid 계열을 conventional GQA 모델의 KV 공식만으로 해석하지 않는다.

---

## 7. DeepSeek

기존 Study:

[../attention/12-deepseek.md](../attention/12-deepseek.md)

주요 구조:

- MLA
- MoE
- MTP
- sparse/compressed attention 계열

운영 시:

- latent KV
- expert parallelism
- all-to-all
- sparse indexer
- compressed cache
- speculative verification

을 각각 분리한다.

---

## 8. Kimi

기존 Study:

[../attention/10-kimi-k3.md](../attention/10-kimi-k3.md)

Kimi-K3 공식 공개 기술 보고서:

https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf

운영 시:

- recurrent/linear state
- global attention branch
- MoE
- TP state shard
- EP fabric
- prefix-state handling

을 함께 본다.

---

## 9. Speculative Decoding

기존 Study:

[../speculative-decoding/2026-08-18-speculative-decoding-mtp-dspark.md](../speculative-decoding/2026-08-18-speculative-decoding-mtp-dspark.md)

핵심 운영식:

- draft cost
- target verification cost
- expected committed tokens per cycle
- state/KV overhead
- graph shape
- load-dependent opportunity cost

acceptance rate 하나로 winner를 고르지 않는다.

---

## 10. GPU / Communication 기존 Study

### GPU Architecture

[../gpu-architecture/README.md](../gpu-architecture/README.md)

### Memory / Data Movement

[../gpu-architecture/03-memory-and-data-movement.md](../gpu-architecture/03-memory-and-data-movement.md)

### Metrics

[../gpu-architecture/06-metrics-and-observability.md](../gpu-architecture/06-metrics-and-observability.md)

### LLM Serving Bottleneck Diagnosis

[../gpu-architecture/07-llm-serving-bottleneck-diagnosis.md](../gpu-architecture/07-llm-serving-bottleneck-diagnosis.md)

### GPU Communication Mental Model

[../gpu-architecture/09-gpu-communication-mental-model.md](../gpu-architecture/09-gpu-communication-mental-model.md)

### PCIe / NVLink / NVSwitch

[../gpu-architecture/10-pcie-nvlink-nvswitch.md](../gpu-architecture/10-pcie-nvlink-nvswitch.md)

### RDMA / GDRDMA

[../gpu-architecture/11-rdma-and-gpudirect-rdma.md](../gpu-architecture/11-rdma-and-gpudirect-rdma.md)

### NCCL Labs

[../gpu-architecture/13-nccl-collectives-observability-labs.md](../gpu-architecture/13-nccl-collectives-observability-labs.md)

---

## 11. 추천 반복 학습 루프

처음에는 이 순서로 읽는다.

1. 00 Mental Model
2. 01 Scheduler
3. 02 Cache
4. 03 CUDA Graph
5. 04 P/D
6. 05 Fabric
7. 06 Architecture
8. 07 Observability
9. 08 Runbook

그 다음 실제 장애나 benchmark 결과를 가지고 역순으로 읽는다.

1. 07에서 증상을 분류
2. 06에서 architecture cost를 찾음
3. 05/03/02/01에서 resource path를 좁힘
4. 08에서 재실험

이 반복이 운영 눈썰미를 만든다.

---

## 12. 마지막 문장

> 좋은 serving engineer는 옵션을 많이 외우는 사람이 아니다. workload를 보고 어떤 자원 항이 커질지 예측하고, metric으로 그 가설을 반증하며, architecture와 hardware가 바뀌어도 같은 사고법으로 다시 최적점을 찾는 사람이다.
