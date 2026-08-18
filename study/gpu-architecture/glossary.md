# GPU Architecture 용어집

작성일: 2026-08-18  
분류: `study/gpu-architecture`  
표기: 영문 원어 · 통용 발음 · 이 자료에서의 의미

## A. 제품과 구조

| 용어 | 읽기 | 의미 |
|---|---|---|
| Architecture | 아키텍처 | 한 세대의 명령, 실행, 메모리, 인터커넥트 설계 원칙. Hopper, Blackwell 등이 이에 해당한다. |
| Die | 다이 | wafer에서 잘라낸 하나의 반도체 조각. 제품 하나가 다이 하나라는 보장은 없다. |
| Chip | 칩 | 문맥에 따라 die 또는 패키지 전체를 가리킨다. 모호하므로 범위를 확인한다. |
| Package | 패키지 | die, interposer/substrate, HBM 등을 전기·기계적으로 결합한 단위. |
| SKU | 에스케이유 | 활성 SM, 메모리 용량, clock, power, form factor가 정해진 판매 제품 구성. |
| Form factor | 폼 팩터 | SXM, PCIe card, NVL처럼 물리·전기 연결 방식을 정한 제품 형태. |
| System/Platform | 시스템/플랫폼 | GPU뿐 아니라 CPU, NIC, NVSwitch, 전력·냉각, software stack까지 포함한 실행 환경. |
| GPC | 지피시 | Graphics Processing Cluster. 여러 TPC/SM과 raster 관련 블록을 묶는 상위 단위. |
| TPC | 티피시 | Texture Processing Cluster. NVIDIA GPU 계층에서 보통 여러 SM을 묶는 단위. |
| SM | 에스엠 | Streaming Multiprocessor. warp를 스케줄하고 CUDA/Tensor/load-store/SFU 명령을 실행하는 핵심 단위. |
| CUDA Core | 쿠다 코어 | 주로 scalar FP/INT 명령을 처리하는 실행 lane을 가리키는 마케팅·구조 용어. CPU core와 동일한 독립 코어가 아니다. |
| Tensor Core | 텐서 코어 | 작은 matrix multiply-accumulate를 높은 처리량으로 실행하는 전용 연산기. |
| RT Core | 알티 코어 | ray traversal/intersection을 가속하는 graphics 전용 블록. |
| Copy Engine | 카피 엔진 | 연산 kernel과 독립적으로 DMA copy를 진행할 수 있는 엔진. |
| Compute Capability | 컴퓨트 캐퍼빌리티 | CUDA 기능과 binary 호환성을 나타내는 `major.minor` 버전. 제품 서열 점수가 아니다. |

## B. CUDA 실행

| 용어 | 읽기 | 의미 |
|---|---|---|
| Host | 호스트 | CUDA 관점의 CPU와 그 memory 영역. |
| Device | 디바이스 | CUDA kernel을 실행하는 GPU와 device memory. |
| Kernel | 커널 | GPU에서 많은 thread가 병렬 실행하는 함수. 운영체제 kernel과 다른 뜻이다. |
| Grid | 그리드 | 한 kernel launch가 만든 모든 thread block의 집합. |
| Thread block | 스레드 블록 | 하나의 SM에 배치되어 shared memory와 block synchronization을 공유하는 thread 그룹. |
| Thread | 스레드 | CUDA 프로그램의 논리적 실행 인스턴스. 하드웨어에서는 warp 단위로 발행된다. |
| Warp | 워프 | NVIDIA GPU가 함께 스케줄하는 32개 thread 묶음. |
| SIMT | 심트 | Single Instruction, Multiple Threads. thread별 상태를 유지하면서 warp 단위로 명령을 실행하는 모델. |
| Divergence | 다이버전스 | 같은 warp의 thread가 서로 다른 control-flow 경로를 선택해 실행 효율이 낮아지는 현상. |
| Warp scheduler | 워프 스케줄러 | ready warp에서 다음 명령을 선택해 실행 파이프에 발행하는 SM 구성요소. |
| Eligible warp | 엘리저블 워프 | operand와 dependency가 준비되어 즉시 명령을 발행할 수 있는 warp. |
| Resident warp | 레지던트 워프 | SM 자원을 할당받아 현재 상주하지만 실행 준비 상태일 필요는 없는 warp. |
| Occupancy | 오큐펀시 | SM이 지원하는 최대 active warp 대비 실제 active warp의 비율. 성능 그 자체는 아니다. |
| Latency hiding | 레이턴시 하이딩 | 한 warp가 기다리는 동안 다른 ready warp를 실행해 지연을 감추는 방식. |
| Stream | 스트림 | CUDA 작업의 순서를 정의하는 queue. 서로 다른 stream은 조건이 맞으면 overlap할 수 있다. |
| CUDA Graph | 쿠다 그래프 | 반복되는 작업 의존관계를 캡처해 launch overhead를 줄이는 실행 방식. |
| Thread Block Cluster | 스레드 블록 클러스터 | 여러 block이 가까운 SM에 함께 배치되고 cluster-level 기능을 사용하는 Hopper 이후 실행 계층. |
| DSM | 디에스엠 | Distributed Shared Memory. cluster 내 block들이 서로의 shared memory에 접근하는 기능. |
| TMA | 티엠에이 | Tensor Memory Accelerator. Hopper에서 다차원 tensor 이동과 address 계산을 비동기화하는 장치. |
| PTX | 피티엑스 | NVIDIA의 virtual ISA. driver가 target GPU용 machine code로 JIT할 수 있다. |
| cubin | 큐빈 | 특정 GPU target용으로 생성된 CUDA binary code object. |

## C. 메모리와 데이터 이동

| 용어 | 읽기 | 의미 |
|---|---|---|
| Register | 레지스터 | thread별 operand를 저장하는 가장 가까운 storage. 부족하면 occupancy 저하나 spill이 생긴다. |
| Local memory | 로컬 메모리 | thread-private 주소 공간이지만 물리적으로는 device memory에 놓일 수 있다. register와 다르다. |
| Shared memory | 셰어드 메모리 | 같은 block의 thread가 공유하는 software-managed on-chip memory. |
| L1/L2 cache | 엘원/엘투 캐시 | 반복되는 memory access를 가까운 계층에서 처리하는 hardware-managed cache. L2는 GPU 전체가 공유한다. |
| HBM | 에이치비엠 | High Bandwidth Memory. GPU package 가까이에 적층해 매우 넓은 interface를 제공하는 memory. |
| GDDR | 지디디알 | graphics double data rate memory. board 위 개별 memory package를 GPU와 연결한다. |
| VRAM | 브이램 | GPU가 사용하는 video/device memory의 통칭. HBM 또는 GDDR일 수 있다. |
| Pinned memory | 핀드 메모리 | page fault로 이동되지 않도록 고정된 host memory. 안정적인 DMA와 async copy에 유리하다. |
| Unified Memory | 유니파이드 메모리 | 하나의 가상 주소 공간과 page migration으로 host/device memory 관리를 단순화하는 CUDA 기능. |
| DMA | 디엠에이 | CPU core가 byte를 직접 복사하지 않고 device engine이 memory 전송을 수행하는 방식. |
| PCIe | 피시아이 익스프레스 | CPU, GPU, NIC, storage 등을 연결하는 범용 serial I/O fabric. generation과 lane 수가 대역폭을 좌우한다. |
| BDF | 비디에프 | PCI domain:bus:device.function 주소. 장치의 topology 위치를 식별한다. |
| BAR1 | 바 원 | GPU framebuffer 일부를 CPU 또는 peer device 주소 공간에 노출하는 PCIe memory window. |
| P2P | 피투피 | 두 device가 host staging 없이 서로의 memory에 직접 접근하거나 복사하는 경로. |
| NVLink | 엔브이링크 | NVIDIA GPU/CPU 사이의 고대역폭 point-to-point interconnect. 세대별 속도와 기능이 다르다. |
| NVSwitch | 엔비스위치 | 여러 NVLink endpoint를 연결해 다수 GPU 간 fabric을 만드는 switch ASIC. |
| NVLink-C2C | 엔브이링크 시투시 | NVIDIA CPU와 GPU 또는 chiplet 사이 coherent die/package 연결에 쓰이는 interconnect. |
| GPUDirect RDMA | 지피유다이렉트 알디엠에이 | NIC 같은 peer device가 GPU memory를 CPU bounce buffer 없이 DMA하는 기술. |
| GPUDirect Storage | 지피유다이렉트 스토리지 | storage와 GPU memory 사이 data path에서 불필요한 CPU copy를 줄이는 기술. |
| RoCE | 로시 | RDMA over Converged Ethernet. Ethernet 위에 RDMA transport를 제공한다. |
| InfiniBand | 인피니밴드 | RDMA를 기본으로 제공하는 HPC/AI cluster interconnect 기술. |
| NUMA | 누마 | CPU socket별로 memory 접근 latency/bandwidth가 달라지는 non-uniform memory architecture. |

## D. 정밀도와 연산

| 용어 | 읽기 | 의미 |
|---|---|---|
| FLOP | 플롭 | floating-point operation 한 번. FMA를 2 FLOP으로 세는 관례가 흔하다. |
| FLOPS | 플롭스 | 초당 floating-point operation 수. 어떤 정밀도와 sparsity 조건인지 반드시 붙여야 한다. |
| FP32 | 에프피 서티투 | 32-bit IEEE 계열 floating point. 범용 계산과 일부 accumulation에 사용한다. |
| TF32 | 티에프 서티투 | FP32 범위와 축소된 mantissa를 사용해 Tensor Core 처리량을 높이는 NVIDIA 형식/실행 모드. |
| FP16 | 에프피 식스틴 | 16-bit floating point. 높은 처리량과 낮은 memory 사용량을 제공하지만 exponent 범위가 좁다. |
| BF16 | 비에프 식스틴 | FP32와 같은 exponent bit 수를 가진 16-bit floating point. AI 학습·추론에 널리 사용한다. |
| FP8 | 에프피 에이트 | E4M3, E5M2 같은 8-bit floating format. scaling과 정확도 관리가 중요하다. |
| NVFP4 | 엔브이 에프피 포 | Blackwell 계열에서 AI용으로 지원하는 NVIDIA 4-bit floating format. 별도 scale과 software path가 필요하다. |
| INT8/INT4 | 인트 에이트/포 | 8-bit/4-bit integer quantization 형식. scale, zero point, kernel 지원이 성능·정확도를 좌우한다. |
| Accumulator | 어큐뮬레이터 | multiply 결과를 합산하는 더 높은 정밀도의 내부 값 또는 register. |
| Quantization | 퀀타이제이션 | weight/activation/KV를 더 적은 bit 형식으로 근사 표현하는 과정. |
| Dequantization | 디퀀타이제이션 | 양자화 값을 연산 가능한 더 높은 정밀도로 복원하거나 scale을 적용하는 과정. |
| Structured sparsity | 스트럭처드 스파시티 | 정해진 작은 그룹 패턴에서 일부 값이 0이라는 제약을 이용해 연산을 건너뛰는 방식. |
| Transformer Engine | 트랜스포머 엔진 | tensor 통계와 scaling을 관리해 FP8/저정밀 Transformer 연산을 지원하는 hardware·software 조합. |
| Arithmetic intensity | 어리스메틱 인텐시티 | memory에서 이동한 byte당 수행한 연산 수. Roofline에서 compute/memory bound를 가르는 축. |
| Roofline model | 루프라인 모델 | peak compute와 memory bandwidth로 attainable performance 상한을 해석하는 모델. |

## E. 관측과 운영

| 용어 | 읽기 | 의미 |
|---|---|---|
| NVML | 엔브이엠엘 | NVIDIA Management Library. utilization, memory, power, temperature 등 관리 지표 API. |
| `nvidia-smi` | 엔비디아 에스엠아이 | NVML 기반의 명령행 관리·관측 도구. |
| DCGM | 디시지엠 | Data Center GPU Manager. fleet telemetry, health, diagnostics, profiling field를 제공한다. |
| GPU Utilization | 지피유 유틸라이제이션 | 표본 구간 중 하나 이상의 kernel이 실행 중이었던 시간 비율. 연산기 peak 달성률이 아니다. |
| Memory Utilization | 메모리 유틸라이제이션 | NVML에서 global memory read/write가 일어난 표본 시간 비율. memory capacity 사용률이 아니다. |
| FB memory usage | 에프비 메모리 사용량 | framebuffer/device memory의 total, used, free 용량. bandwidth activity와 다르다. |
| SM Active | 에스엠 액티브 | DCGM에서 SM별로 하나 이상의 warp가 active였던 시간 비율의 평균. |
| SM Occupancy | 에스엠 오큐펀시 | 관측 중 active warp 수가 이론상 최대와 비교해 어느 정도였는지 나타내는 지표. |
| Tensor Active | 텐서 액티브 | Tensor pipe가 active였던 cycle 비율 계열 지표. Tensor peak FLOPS와 동일하지 않다. |
| DRAM Active | 디램 액티브 | device memory interface가 read/write traffic으로 active였던 cycle 비율. |
| Achieved bandwidth | 어치브드 밴드위스 | workload가 실제 달성한 byte/s. 이론 대역폭과 구분한다. |
| Latency | 레이턴시 | 한 작업이나 요청이 완료될 때까지 걸린 시간. |
| Throughput | 스루풋 | 단위 시간당 처리한 request, token, byte 또는 operation 수. |
| Tail latency | 테일 레이턴시 | p95/p99처럼 분포 꼬리의 지연. 평균으로 숨겨지는 최악 사용자 경험을 나타낸다. |
| Throttling | 스로틀링 | power, temperature, reliability 등의 이유로 clock이 제한되는 상태. |
| XID | 엑스아이디 | NVIDIA driver가 kernel log에 남기는 GPU error event code. 코드와 주변 로그를 함께 해석한다. |
| ECC | 이씨씨 | memory bit error를 검출·정정하는 기능과 그 error count. corrected/uncorrected를 구분한다. |
| Row remapping | 로 리매핑 | 결함이 의심되는 DRAM row를 예비 row로 교체하는 memory reliability 기능. |
| CUPTI | 컵티 | CUDA Profiling Tools Interface. profiler와 telemetry가 CUDA activity/counter를 수집하는 API. |
| Nsight Systems | 엔사이트 시스템즈 | CPU, CUDA, communication을 전체 timeline에서 분석하는 profiler. |
| Nsight Compute | 엔사이트 컴퓨트 | 선택한 CUDA kernel의 instruction, warp stall, cache, memory, pipe를 분석하는 profiler. |
| Counter multiplexing | 카운터 멀티플렉싱 | 동시에 셀 수 없는 hardware counter를 시간 분할 또는 여러 pass로 수집하는 방식. |

## F. LLM Serving

| 용어 | 읽기 | 의미 |
|---|---|---|
| Prefill | 프리필 | input token 전체를 병렬 처리해 첫 KV cache와 첫 token 조건을 만드는 단계. |
| Decode | 디코드 | 이미 생성된 KV cache를 읽으며 보통 한 sequence당 다음 token 하나씩 생성하는 반복 단계. |
| KV cache | 케이브이 캐시 | attention의 과거 key/value tensor를 저장해 이전 token 연산을 재사용하는 device memory 영역. |
| Continuous batching | 컨티뉴어스 배칭 | request가 끝날 때마다 새 request를 batch에 넣어 decode 자원을 지속 활용하는 scheduling 방식. |
| PagedAttention | 페이지드 어텐션 | KV block을 page처럼 관리해 allocation fragmentation과 공유를 다루는 vLLM 방식. |
| Prefix caching | 프리픽스 캐싱 | 동일한 prompt prefix의 KV를 재사용해 prefill 연산을 줄이는 기능. |
| TP | 티피 | Tensor Parallelism. 한 tensor 연산을 여러 GPU rank로 나누고 collective로 부분 결과를 결합한다. |
| PP | 피피 | Pipeline Parallelism. model layer 구간을 여러 stage로 나누어 실행한다. |
| DP | 디피 | Data Parallelism. model replica가 서로 다른 request/batch를 처리한다. |
| EP | 이피 | Expert Parallelism. MoE expert를 여러 rank에 분산하고 token을 all-to-all로 전달한다. |
| NCCL | 니클 | NVIDIA Collective Communications Library. all-reduce, all-gather, reduce-scatter, all-to-all 등을 제공한다. |
| TTFT | 티티에프티 | Time To First Token. 요청 도착부터 첫 output token까지의 시간. |
| ITL | 아이티엘 | Inter-Token Latency. 연속 output token 사이의 시간. |
| TPOT | 티팟 | Time Per Output Token. 출력 token당 걸린 시간으로 ITL과 유사하지만 집계 정의를 확인해야 한다. |
| E2E latency | 엔드투엔드 레이턴시 | 요청 도착부터 전체 응답 완료까지의 시간. |
| Speculative decoding | 스페큘러티브 디코딩 | draft가 여러 후보 token을 제안하고 target model이 검증해 target 호출당 완료 token 수를 늘리는 방식. |

## G. 자주 혼동하는 짝

| A | B | 핵심 차이 |
|---|---|---|
| GPU memory used | Memory Utilization | 전자는 할당된 용량, 후자는 read/write가 있었던 시간 비율이다. |
| GPU Utilization | SM/Tensor throughput | 전자는 kernel 실행 시간, 후자는 실행 자원의 실제 활동·처리량이다. |
| HBM bandwidth | PCIe bandwidth | device memory 내부 경로와 host/device I/O 경로다. |
| Architecture | SKU | 세대 설계와 그 설계를 잘라 만든 판매 제품이다. |
| CUDA Core | CPU core | SIMD/SIMT execution lane과 독립적인 범용 latency-optimized core다. |
| Occupancy | Utilization | 상주 가능한 warp 채움 정도와 시간상 실행 활동 정도다. |
| FP8 support | FP8 speedup | 기능 존재와 workload가 최적 kernel로 실제 이득을 얻는 것은 별개다. |
| NVLink present | NCCL uses NVLink | 물리 기능 존재와 collective runtime의 실제 경로 선택은 별개다. |
| TTFT | Prefill duration | TTFT에는 queue, scheduling, transfer 등 prefill 외 시간이 포함된다. |
| ITL | kernel duration | ITL에는 scheduling, collective, queue, CPU overhead도 포함될 수 있다. |

## H. 공식 기준 문서

- NVIDIA, [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- NVIDIA, [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- NVIDIA, [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/)
- NVIDIA, [DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/)
- NVIDIA, [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

> 본질: 용어를 정확히 구분하면 관측값이 물리 구조, 실행 상태, 서비스 결과 중 어디에 속하는지가 보인다.

