# 04. NVIDIA GPU 세대와 주요 제품 비교

작성일: 2026-08-18  
선행 문서: [메모리와 데이터 이동](03-memory-and-data-movement.md)  
다음 문서: [정밀도와 Tensor Core](05-numerical-precision-and-tensor-cores.md)

## 1. 비교 전에 고정할 기준

세대 비교는 적어도 다음 여섯 축으로 해야 한다.

1. Compute Capability와 instruction set
2. SM과 Tensor Core의 구조
3. 지원 numerical format과 accumulation
4. memory capacity, type, bandwidth, cache
5. PCIe, NVLink, NVSwitch, C2C 같은 I/O
6. MIG, confidential computing, media/graphics engine 같은 product feature

제품표의 FLOPS 하나만 비교하면 다음 차이를 놓친다.

- 같은 architecture 안의 form factor별 clock/power 차이
- sparse와 dense 수치의 2배 차이
- FP4, FP8, BF16처럼 precision 자체가 다른 수치
- HBM capacity와 bandwidth
- scale-up communication 능력
- 실제 kernel/software support

## 2. 현대 NVIDIA architecture의 흐름

| Architecture | 대표 compute capability | 대표 제품 | 핵심 변화 | 주된 의미 |
|---|---:|---|---|---|
| Volta | 7.0 | V100 | 1세대 Tensor Core, Independent Thread Scheduling, HBM2, NVLink 2 | 범용 GPU에 deep-learning matrix engine이 본격 통합 |
| Turing | 7.5 | T4 | 2세대 Tensor Core, INT8/INT4, RT Core, FP/INT 동시 실행 개선 | inference와 visual AI의 저전력 가속 |
| Ampere | 8.0 / 8.6 | A100, A30, A10 | 3세대 Tensor Core, TF32/BF16/FP64 Tensor, 2:4 sparsity, async copy, MIG, NVLink 3 | AI training·HPC·multi-tenant data center 통합 |
| Ada Lovelace | 8.9 | L4, L40, L40S | 4세대 Tensor Core와 FP8, 3세대 RT Core, AV1 media engine | graphics·media·AI를 함께 처리하는 data-center visual compute |
| Hopper | 9.0 | H100, H200, GH200 | 4세대 Tensor Core, FP8 Transformer Engine, TMA, block cluster, distributed shared memory, NVLink 4 | Transformer와 대규모 scale-up 최적화 |
| Blackwell | 10.0 | B200, GB200 | 5세대 Tensor Core, 2세대 Transformer Engine, FP6/FP4, dual-reticle NV-HBI, NVLink 5 | 저정밀 AI와 rack-scale model parallelism 확대 |
| Blackwell Ultra | 10.3 | B300, GB300 | 더 큰 HBM, NVFP4/attention 처리 강화, 공개 구조상 TMEM 확대 | reasoning·test-time scaling·긴 context inference 강화 |
| Vera Rubin | 공개 제품 정보 기준, 변경 가능 | Rubin GPU, Vera Rubin NVL72 | HBM4, NVLink 6, decode/front-end 개선 | 차세대 rack-scale AI; 현재 수치는 preliminary |

### 2.1 Compute Capability가 architecture 이름과 완전히 1:1은 아니다

Ampere는 A100의 8.0과 GA10x 계열의 8.6이 있다. Blackwell도 B200/GB200의 10.0, B300/GB300의 10.3, RTX PRO/GeForce의 12.0, GB10의 12.1처럼 product family에 따라 다르다.

따라서 software build target은 architecture 이름이 아니라 실제 GPU의 compute capability까지 확인해야 한다.

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## 3. 세대별 구조 변화

### 3.1 Volta: Tensor Core와 독립 thread scheduling

Volta GV100은 SM에 8개의 1세대 Tensor Core를 넣어 FP16 input과 FP32 accumulation의 matrix multiply를 가속했다. 또한 Independent Thread Scheduling을 도입해 lane별 execution state를 더 유연하게 관리했다.

운영·개발 관점의 변화:

- deep learning의 핵심 GEMM이 일반 FP32 CUDA pipeline과 다른 hardware path를 사용
- warp-synchronous programming에 명시적 synchronization 필요
- HBM2와 NVLink 2를 사용한 high-end data center platform 확립

### 3.2 Turing: inference precision과 graphics AI

Turing은 Tensor Core에 INT8과 INT4를 추가하고 RT Core를 도입했다. T4는 낮은 power envelope에서 inference, video, virtual graphics를 함께 처리하도록 설계됐다.

중요한 변화:

- quantized inference의 hardware acceleration 확대
- FP32와 INT32 pipeline의 동시 처리 개선
- unified L1/shared memory 구조 개선

### 3.3 Ampere: data center 범용화

A100 GA100의 3세대 Tensor Core는 TF32, BF16, FP64 Tensor 연산과 2:4 structured sparsity를 지원한다.

주요 기능:

- TF32: FP32 range를 유지하면서 Tensor Core 학습 경로 제공
- BF16: FP16보다 넓은 exponent range
- FP64 Tensor Core: HPC matrix 연산 강화
- 2:4 sparsity: 조건을 만족하는 matrix의 effective throughput 향상
- asynchronous global-to-shared copy와 barrier
- L2 persistence control
- 1세대 MIG: compute, cache, memory partition을 hardware 수준에서 분리
- NVLink 3: A100 SXM에서 GPU당 600 GB/s bidirectional 표기

### 3.4 Ada Lovelace: L40S는 다른 목적의 GPU다

L40S는 Ada AD102 계열이다. 4세대 Tensor Core와 FP8, 3세대 RT Core, NVENC/NVDEC를 갖고 48 GB GDDR6를 사용한다.

L40S의 성격:

- text/multimodal AI inference와 training 가능
- rendering, simulation, video encode/decode에 강함
- PCIe Gen4 x16, 864 GB/s GDDR6
- L40S에는 NVLink와 MIG가 없고, vGPU는 time-sliced 방식이 중심

L40S의 높은 nominal FP8 FLOPS만 보고 H100과 같은 대형 LLM scale-up 성능을 기대하면 안 된다. HBM bandwidth/capacity와 NVLink topology가 다르다.

### 3.5 Hopper: H100과 H200

Hopper GH100의 핵심은 Transformer 연산과 asynchronous data movement다.

- 4세대 Tensor Core
- E4M3/E5M2 FP8와 Transformer Engine
- TMA: multidimensional tensor를 global↔shared memory로 비동기 이동
- thread block cluster와 distributed shared memory
- DPX instructions
- 2세대 MIG와 confidential computing
- NVLink 4: SXM에서 GPU당 900 GB/s bidirectional 표기

H100 SXM과 H200 SXM의 관계:

| 항목 | H100 SXM | H200 SXM | 해석 |
|---|---:|---:|---|
| Architecture / CC | Hopper / 9.0 | Hopper / 9.0 | 같은 instruction feature 계열 |
| Memory | 80 GB HBM3 | 141 GB HBM3e | H200가 model·KV capacity에 유리 |
| Memory bandwidth | 3.35 TB/s | 4.8 TB/s | memory-bound workload에서 H200 이점 |
| NVLink | 4세대, 900 GB/s | 4세대, 900 GB/s | 동일 세대 scale-up fabric |

H200의 핵심 이점은 `H100보다 새로운 Tensor Core 명령`이 아니라 더 큰 HBM과 약 1.4배의 memory bandwidth다. 실제 compute specification은 SKU clock에 따라 일부 차이가 있지만 구조적 차이의 중심은 memory subsystem이다.

### 3.6 Blackwell: dual-reticle과 저정밀 AI

Blackwell compute GPU는 두 reticle-sized die를 NV-HBI로 연결해 CUDA에서 하나의 accelerator처럼 동작하게 한다. 공개 자료 기준 NV-HBI bandwidth는 10 TB/s다.

주요 기능:

- 5세대 Tensor Core
- 2세대 Transformer Engine
- FP8, FP6, NVFP4와 micro-tensor scaling
- B200 최대 180 GB HBM3e, 8 TB/s
- GB200의 공개 L2 capacity 126 MB
- NVLink 5: GPU당 1.8 TB/s bidirectional
- hardware decompression, enhanced RAS/secure AI 기능
- Cluster Launch Control: 미실행 block을 cancel하고 work stealing에 활용

Blackwell에서 중요한 software 경계는 `sm_100` native code와 PTX compatibility다. Hopper 전용 custom extension은 재빌드나 최신 wheel이 필요할 수 있다.

### 3.7 Blackwell Ultra: B300/GB300

Blackwell Ultra는 Blackwell의 programming model을 유지하면서 reasoning과 attention workload를 강화한다.

- compute capability 10.3
- GPU당 최대 288 GB HBM3e, 최대 8 TB/s 공개 사양
- GPU당 1.8 TB/s NVLink 5
- Blackwell 대비 1.5배 dense FP4 FLOPS, 2배 attention performance라는 system-level 공개 비교
- 공개 SM 구조에 256 KB Tensor Memory(TMEM) 설명

`B300`과 `GB300`은 같지 않다. B300은 x86 HGX/DGX에도 쓰이는 GPU SKU이고, GB300은 Grace CPU와 결합한 Grace Blackwell Ultra platform을 가리킨다.

### 3.8 Vera Rubin: preliminary 영역

2026-08-18 공개 자료는 Rubin GPU에 대해 다음을 제시한다.

- GPU당 최대 288 GB HBM4
- GPU당 최대 22 TB/s memory bandwidth
- 6세대 NVLink, GPU당 3.6 TB/s
- decode와 front-end efficiency 개선

이 값은 preliminary이며 출시 SKU와 system configuration에 따라 바뀔 수 있다. 운영 설계의 확정 baseline이 아니라 forward-looking capacity planning에만 사용한다.

## 4. 대표 데이터센터 GPU 사양

아래는 비교를 위한 대표 SKU/form factor다. 같은 제품의 PCIe, NVL, SXM 변형은 수치가 다를 수 있다.

| 대표 제품 | Architecture / CC | Device memory | Memory BW | Scale-up interconnect | 구조적 포지션 |
|---|---|---:|---:|---|---|
| A100 80GB SXM | Ampere / 8.0 | 80 GB HBM2e | 2.039 TB/s | NVLink 3, 600 GB/s | TF32/BF16, MIG, mature AI/HPC |
| L40S PCIe | Ada / 8.9 | 48 GB GDDR6 ECC | 864 GB/s | PCIe Gen4 x16, NVLink 없음 | AI + graphics + media |
| H100 SXM | Hopper / 9.0 | 80 GB HBM3 | 3.35 TB/s | NVLink 4, 900 GB/s | Transformer compute와 scale-up |
| H200 SXM | Hopper / 9.0 | 141 GB HBM3e | 4.8 TB/s | NVLink 4, 900 GB/s | 큰 model·KV와 memory-bound inference |
| B200 SXM | Blackwell / 10.0 | 최대 180 GB HBM3e | 최대 8 TB/s | NVLink 5, 1.8 TB/s | FP4/FP6와 rack-scale AI |
| B300 SXM | Blackwell Ultra / 10.3 | 최대 288 GB HBM3e | 최대 8 TB/s | NVLink 5, 1.8 TB/s | reasoning·attention·capacity 강화 |
| Rubin GPU | Rubin / preliminary | 최대 288 GB HBM4 | 최대 22 TB/s | NVLink 6, 3.6 TB/s | 차세대 rack-scale platform |

모든 NVLink 수치는 NVIDIA의 GPU당 aggregate bidirectional 표기를 따른다. sparse performance 수치는 의도적으로 표에서 제외했다. 서로 다른 precision과 sparsity 조건의 PFLOPS를 한 열에 놓으면 오히려 판단을 흐리기 때문이다.

## 5. Feature matrix

| 기능 | A100 | L40S | H100/H200 | B200 | B300 |
|---|---|---|---|---|---|
| Tensor Core generation | 3세대 | 4세대 | 4세대 | 5세대 | 5세대 강화 |
| BF16 / TF32 | 지원 | 지원 | 지원 | 지원 | 지원 |
| FP8 Tensor path | 미지원 | 지원 | 지원 + Transformer Engine | 지원 + TE 2세대 | 지원 + TE 2세대 |
| FP6 / NVFP4 | 미지원 | 미지원 | 미지원 | 지원 | 지원·강화 |
| Structured sparsity | 2:4 | 지원 | 지원 | 지원 | 지원 |
| TMA / block cluster | 미지원 | 미지원 | 지원 | 지원 | 지원 |
| MIG | 지원 | 미지원 | 지원 | 지원 | 지원 SKU/profile 확인 |
| NVLink 대표 scale-up | 3세대 | 없음 | 4세대 | 5세대 | 5세대 |
| RT/media engine | compute SKU 중심 | 강함 | RT 없음, decode/JPEG SKU 기능 | compute SKU 중심 | compute SKU 중심 |

`지원`은 hardware feature가 존재한다는 뜻이다. framework와 kernel이 실제 사용하는지는 별도 확인한다.

## 6. LLM 관점에서 제품을 읽는 법

### 6.1 모델이 올라가는가

```math
M_{required}
=M_{weights}+M_{KV}+M_{workspace}+M_{runtime}
```

첫 판단은 HBM/GDDR capacity다. MoE의 active parameter가 작아도 전체 expert weight를 보유해야 하므로 storage 요구량은 total parameter와 sharding에 좌우된다.

### 6.2 decode가 빠른가

작은 batch decode는 weight/KV 이동 대비 연산 재사용이 낮아 memory bandwidth의 영향을 크게 받는다. 이때 H200가 H100보다 유리한 이유가 Tensor Core 세대가 아니라 HBM3e bandwidth일 수 있다.

### 6.3 prefill이 빠른가

긴 prompt와 충분한 batch의 큰 GEMM은 Tensor compute와 attention kernel 효율이 중요하다. FP8/FP4 최적 kernel과 shape가 사용되면 세대 차이가 크게 나타난다.

### 6.4 여러 GPU로 잘 확장되는가

TP/EP에서 GPU당 NVLink bandwidth, NVSwitch topology, collective latency가 중요하다. L40S 여러 장을 PCIe로 묶는 것과 H200 HGX NVSwitch domain은 GPU 수가 같아도 전혀 다른 system이다.

### 6.5 운영 격리가 필요한가

MIG 지원 여부와 profile, memory/compute isolation, application compatibility를 본다. time-slicing은 MIG의 memory·fault isolation과 같지 않다.

## 7. 세대가 바뀔 때 software 검증 목록

- CUDA Toolkit과 driver minimum/compatibility
- PyTorch wheel의 CUDA와 architecture support
- custom extension의 `sm_XX` cubin과 PTX 포함 여부
- Triton, FlashAttention, DeepGEMM, CUTLASS kernel support
- NCCL과 NVLink/NVSwitch generation support
- DCGM/DCGM Exporter field support
- container base image의 driver capability
- quantization format과 Transformer Engine library 지원
- CUDA Graph capture, compiler graph, fused operator 호환성

새 GPU의 peak specification이 아무리 높아도 fallback kernel이나 PTX JIT만 사용하면 기대 성능이 나오지 않을 수 있다.

## 8. 자가 점검

1. H100에서 H200으로 바뀔 때 가장 큰 구조적 변화는 무엇인가?
2. L40S의 FP8 수치가 높아도 H200 대체재로 단순 계산할 수 없는 이유는 무엇인가?
3. B300용 native extension target은 B200의 `sm_100`과 같다고 가정해도 되는가?
4. A100과 H100의 NVLink 600/900 GB/s는 단방향 수치인가?
5. Rubin 수치를 현재 구매·운영 baseline으로 확정하면 안 되는 이유는 무엇인가?

## 9. 주요 원문

- NVIDIA, [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)
- NVIDIA, [Volta Tuning Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/)
- NVIDIA, [Turing Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/)
- NVIDIA, [Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- NVIDIA, [A100 GPU Specifications](https://www.nvidia.com/en-us/data-center/a100/)
- NVIDIA, [L40S GPU Specifications](https://www.nvidia.com/en-us/data-center/l40s/)
- NVIDIA, [Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- NVIDIA, [H100 GPU Specifications](https://www.nvidia.com/en-us/data-center/h100/)
- NVIDIA, [H200 GPU Specifications](https://www.nvidia.com/en-us/data-center/h200/)
- NVIDIA, [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- NVIDIA, [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- NVIDIA, [Inside the Vera Rubin Platform](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)

> 본질: GPU 세대의 진짜 차이는 peak FLOPS 한 줄이 아니라, **어떤 정밀도로 계산하고 데이터를 얼마나 가까이·빠르게 공급하며 여러 GPU를 어떤 비용으로 하나의 작업에 묶는가**에 있다.

