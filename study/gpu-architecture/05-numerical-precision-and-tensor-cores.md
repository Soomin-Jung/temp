# 05. Numerical Precision과 Tensor Core

작성일: 2026-08-18  
선행 문서: [NVIDIA 세대와 주요 GPU 비교](04-nvidia-architecture-generations.md)  
다음 문서: [GPU 메트릭과 관측 도구](06-metrics-and-observability.md)

## 1. 정밀도를 줄이면 무엇이 줄어드는가

숫자 하나의 bit 수를 줄이면 다음 세 비용을 줄일 가능성이 있다.

1. Memory capacity: 같은 HBM에 더 많은 weight/KV를 저장
2. Memory traffic: 같은 tensor를 더 적은 byte로 이동
3. Compute cost: 지원 Tensor Core가 더 많은 low-precision operation을 처리

그러나 네 번째 비용이 생긴다.

4. Accuracy와 scaling/quantization 관리 비용

따라서 `FP8은 BF16보다 2배 작다`와 `모델이 2배 빨라진다`는 같은 명제가 아니다.

## 2. Floating-point의 기본 구조

일반적인 binary floating-point는 sign, exponent, fraction/mantissa로 구성된다.

```math
x=(-1)^s \times 2^{e-bias} \times (1.f)
```

- sign bit: 양수/음수
- exponent: 표현 가능한 값의 범위
- fraction/mantissa: 유효 숫자의 세밀함

| Format | Sign | Exponent | Fraction | 저장 크기 | 핵심 성격 |
|---|---:|---:|---:|---:|---|
| FP32 | 1 | 8 | 23 | 4 bytes | 넓은 range와 높은 precision |
| TF32 | 1 | 8 | 10 | input/storage는 보통 FP32 interface | FP32 range, 줄인 multiply precision, FP32 accumulate |
| BF16 | 1 | 8 | 7 | 2 bytes | FP32와 같은 exponent 폭, 낮은 fraction precision |
| FP16 | 1 | 5 | 10 | 2 bytes | BF16보다 좁은 range, 더 많은 fraction bits |
| FP8 E4M3 | 1 | 4 | 3 | 1 byte | 비교적 precision 우선, range가 좁음 |
| FP8 E5M2 | 1 | 5 | 2 | 1 byte | range 우선, precision이 더 낮음 |
| NVFP4 | 1 | format-specific | format-specific | raw value 4 bits + scale metadata | block scaling을 전제로 한 Blackwell AI format |
| INT8 | 정수 | - | - | 1 byte | scale/zero-point 등 quantization 필요 |
| INT4 | 정수 | - | - | raw value 4 bits + metadata | 높은 압축, 오차·kernel 제약 증가 |

TF32는 model weight를 19-bit 파일로 저장하는 일반 dtype이라고 이해하면 안 된다. FP32 input interface에서 Tensor Core가 multiply에 TF32 precision을 사용하고 FP32로 accumulate하는 math mode에 가깝다.

FP4/INT4의 이론적 0.5 byte는 packed raw value만 센 값이다. 실제 checkpoint와 runtime memory에는 scale, metadata, padding, group alignment가 추가된다.

## 3. BF16과 FP16

둘 다 16 bits지만 trade-off가 다르다.

- BF16: exponent 8 bits라 큰 값과 작은 값의 range가 넓어 training stability에 유리
- FP16: fraction 10 bits라 같은 범위 안의 세밀함은 더 좋지만 overflow/underflow에 취약

training에서는 FP32 master weight 또는 accumulator, loss scaling, mixed precision recipe가 함께 사용될 수 있다. inference에서는 model quality와 kernel support가 허용하면 BF16/FP16 weight와 activation을 사용할 수 있다.

## 4. FP8: 저장 형식만이 아니다

Hopper의 Transformer Engine은 layer별 tensor 통계를 사용해 FP8과 16-bit 경로, scaling을 관리한다.

### 4.1 E4M3와 E5M2

- E4M3: fraction이 하나 더 많아 precision에 유리, range는 더 좁음
- E5M2: exponent가 하나 더 많아 range에 유리, precision은 더 낮음

training에서 forward/backward tensor의 성격에 따라 format과 scale을 선택할 수 있다. inference checkpoint의 FP8 weight format과 training-time Transformer Engine 동작은 구분해야 한다.

### 4.2 L40S의 FP8과 H100의 Transformer Engine

L40S의 4세대 Tensor Core도 FP8 Tensor 연산을 지원한다. 그러나 H100의 Hopper architecture에서 강조되는 Transformer Engine, TMA, block cluster, HBM/NVLink system 기능까지 L40S가 가진다는 뜻은 아니다.

`FP8 지원`은 한 줄짜리 동일 기능 표가 아니라 다음을 분해해 확인해야 한다.

- hardware instruction 지원
- input format과 accumulator
- per-tensor 또는 block scaling 방식
- library/framework kernel 지원
- checkpoint serialization format
- activation/KV까지 FP8인지 weight-only인지

## 5. Blackwell의 FP6와 NVFP4

Blackwell 2세대 Transformer Engine은 micro-tensor scaling을 사용해 작은 block 단위로 scale을 적용하고 FP4 accuracy를 개선한다.

NVFP4의 장점:

- BF16 대비 weight/activation traffic을 크게 줄일 잠재력
- 같은 HBM에 더 큰 model 또는 KV/workspace 수용
- 5세대 Tensor Core의 높은 low-precision throughput

제약:

- scale metadata와 conversion overhead
- operator별 FP4 지원 차이
- outlier와 layer sensitivity
- checkpoint calibration 또는 quantization-aware recipe
- hardware와 CUDA/library version 종속성

FP4를 `소수점 4자리`로 이해하면 안 된다. 총 4 bits의 매우 제한된 code space와 scale을 결합하는 low-precision representation이다.

## 6. Tensor Core가 하는 일

Tensor Core는 matrix multiply-accumulate를 tile 단위로 처리한다.

```math
D=A\times B+C
```

일반적인 mixed-precision pattern은 다음과 같다.

- $A$, $B$: FP16/BF16/FP8/FP4 같은 낮은 precision
- multiply: hardware가 지원하는 low-precision Tensor path
- accumulate: FP16 또는 FP32 등 더 높은 precision
- output: 필요 dtype으로 cast/scale

Tensor Core operation은 세대별 PTX instruction과 tile shape가 다르다. WMMA, MMA, WGMMA, compiler/library abstraction이 이를 감싼다.

### 6.1 왜 matrix shape가 중요한가

Tensor Core는 고정된 tile 형태의 hardware operation을 효율적으로 수행한다. matrix dimension이 tile/alignment에 맞지 않으면 다음이 발생할 수 있다.

- padding
- edge tile의 낮은 lane utilization
- 일반 CUDA core fallback
- extra transpose/layout conversion
- 작은 GEMM에서 launch와 setup overhead가 지배

따라서 theoretical PFLOPS는 충분히 큰, 잘 정렬된 GEMM에서 접근 가능한 상한이다.

## 7. Structured Sparsity

Ampere 이후 Tensor Core는 대표적으로 2:4 structured sparsity를 지원한다. 연속된 네 weight 중 두 개가 zero인 규칙을 만족하도록 pruning하고 metadata와 함께 압축한다.

```text
dense:   [w0, w1, w2, w3]
2:4:     [w0,  0, w2,  0] + position metadata
```

hardware는 non-zero operand만 처리해 effective throughput을 높일 수 있다.

주의:

- 일반적인 unstructured zero가 많다는 사실만으로 sparse Tensor Core path가 되지 않는다.
- model을 2:4 pattern으로 만들고 accuracy를 회복하는 training/fine-tuning이 필요할 수 있다.
- datasheet의 `dense | sparse` 두 수치를 구분해야 한다.
- sparse peak 2배가 end-to-end 2배를 뜻하지 않는다.

## 8. Dense·sparse·precision peak를 섞지 않는 법

다음 표기라면 항상 footnote를 먼저 읽는다.

```text
FP8 Tensor: 2,000 TFLOPS | 4,000 TFLOPS*
* structured sparsity 사용
```

비교 원칙:

1. 같은 precision끼리 비교
2. dense끼리 비교
3. 같은 form factor와 power envelope 확인
4. accumulator 조건 확인
5. 실제 model이 해당 dtype/sparsity를 사용하는지 확인

FP4 sparse peak와 BF16 dense peak를 나란히 놓고 `몇 배 빠르다`고 계산하면 운영 성능 예측으로 사용할 수 없다.

## 9. LLM memory 계산

### 9.1 Weight memory

단순화하면 다음과 같다.

```math
M_{weights}\approx N_{parameters}\times bytes_{per\ weight}
```

예를 들어 70B parameter의 raw weight만 보면:

| Format | 이론 raw weight 크기 |
|---|---:|
| FP32 | 약 280 GB |
| BF16/FP16 | 약 140 GB |
| FP8/INT8 | 약 70 GB + metadata 가능 |
| 4-bit | 약 35 GB + scale/metadata/padding |

실제 runtime은 embedding tie, quantization metadata, allocator fragmentation, CUDA graph pool, communication buffer, workspace를 추가로 요구한다.

MoE에서는 `active parameters`가 token당 compute를 설명하지만 전체 checkpoint storage는 total parameters에 가깝다. expert parallel은 expert weight를 GPU에 나눠 저장할 수 있지만 통신 비용이 생긴다.

### 9.2 KV cache memory

표준 MHA/GQA 계열의 근사식은 다음과 같다.

```math
M_{KV}\approx
B\times S\times L\times 2\times H_{KV}\times D\times bytes
```

- $B$: sequence 수
- $S$: 저장 token 수
- $L$: layer 수
- 2: Key와 Value
- $H_{KV}$: KV head 수
- $D$: head dimension

GQA는 KV head 수를 줄여 cache를 절감한다. MLA와 latent attention은 cache representation이 다르므로 이 식을 그대로 쓰지 않는다. TP에서 KV가 어떻게 shard/replicate되는지도 framework 구현을 확인한다.

### 9.3 Capacity와 bandwidth의 동시 효과

KV dtype을 BF16에서 FP8로 줄이면:

- token capacity 증가
- attention에서 읽는 byte 감소 가능
- quantize/dequantize와 accuracy 비용 발생
- 지원 kernel이 없으면 conversion overhead가 이득을 깎음

## 10. 왜 quantization이 항상 빨라지지 않는가

| 원인 | 설명 |
|---|---|
| Kernel fallback | hardware는 지원하지만 framework가 최적 kernel을 사용하지 않음 |
| Dequant overhead | 매 layer scale 적용과 conversion 비용이 큼 |
| Small shape | Tensor Core tile을 충분히 채우지 못함 |
| Memory already hidden | 병목이 통신·launch·queue라 weight byte 감소가 영향 없음 |
| Extra metadata | scale, zero-point, packing/unpacking traffic |
| Mixed operators | GEMM 외 attention, normalization, sampling은 다른 dtype/path 사용 |
| Accuracy constraint | 일부 layer를 higher precision으로 남겨 fusion/flow가 복잡해짐 |

속도 평가는 model load 성공이나 peak FLOPS가 아니라 같은 workload의 TTFT, ITL, token/s, power/token으로 검증한다.

## 11. 메트릭으로 Tensor path 확인하기

운영 수준:

- `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`
- `DCGM_FI_PROF_SM_ACTIVE`
- `DCGM_FI_PROF_DRAM_ACTIVE`
- achieved application token/s

Kernel 수준:

- Nsight Compute Tensor pipe utilization
- instruction mix와 MMA instruction
- achieved occupancy
- L1/L2/DRAM throughput
- kernel duration과 launch count

Tensor Active가 낮다는 사실만으로 문제는 아니다. sampling, reshape, memory copy, attention reduction처럼 Tensor Core를 쓰지 않는 phase도 있다. operator와 timeline에 맞춰 해석한다.

## 12. 자가 점검

1. BF16과 FP16은 같은 2 bytes인데 무엇이 다른가?
2. TF32 weight 파일의 크기를 19 bits/parameter로 계산해도 되는가?
3. MoE active parameter가 30B면 30B weight만 HBM에 있으면 되는가?
4. datasheet의 sparse peak를 dense workload 예측에 바로 써도 되는가?
5. KV cache FP8이 capacity는 늘렸지만 latency를 악화시킬 수 있는 경우는 무엇인가?

## 13. 주요 원문

- NVIDIA, [Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- NVIDIA, [Structured Sparsity on Ampere](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)
- NVIDIA, [Hopper Architecture In-Depth — FP8 and Transformer Engine](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- NVIDIA, [Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- NVIDIA, [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- NVIDIA, [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/)

> 본질: 낮은 정밀도의 가치는 숫자를 작게 만드는 데 있지 않고, **허용 가능한 오차 안에서 계산·저장·이동해야 할 정보량을 함께 줄이는 데** 있다.

