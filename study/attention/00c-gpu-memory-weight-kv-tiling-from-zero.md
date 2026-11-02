# 00c. GPU 메모리에서 실제 연산까지: Weight Tile, GEMM, KV Cache를 처음부터

> 작성: 2026-11-03  
> 독자: “SM Cache가 작으면 어떻게 거대한 Weight와 10GiB 이상의 KV를 계산하나?”에서 출발하는 LLM 서빙 엔지니어  
> 선행: [행렬곱 원소별 계산과 FMA(00a)](00a-matrix-gemm-to-qkv-from-zero.md)  
> 후속: [TP, KV Shard, All-Reduce 비용(00b)](00b-tp-and-llm-inference-cost-from-first-principles.md), [1회차 발표 플로우](presentation-session-01-hardware-attention-kv-cache.md)  
> 범위: Hopper 계열 GPU의 **설명용 실행 모델**, Qwen3.8-27B의 FFN/Full Attention 구성. 특정 vLLM 버전의 커널 트레이스와는 구분한다.

## 먼저 가장 중요한 답

1. **모델 가중치(Weight)는 많은 학습된 Tensor의 집합**이다. 그중 Q/K/V/FFN/Output Projection Weight는 보통 2차원 행렬로 표현한다. RMSNorm의 Scale처럼 벡터인 가중치도 있고, 모델/커널의 실제 저장 형식은 Packing·Quantization·Transposition 등에 따라 달라진다.
2. **Weight Tile은 별도로 학습된 작고 독립적인 행렬이 아니다.** 큰 가중치 행렬에서 GPU Kernel이 연산을 위해 **일시적으로 선택한 부분 영역**이다. Tile은 계산 단위이고 Weight Shard(TP)는 분산 **소유/배치 단위**이므로 다르다.
3. 모델 Weight와 각 요청의 KV Cache는 통상 **GPU Global Memory(HBM)**에 상주하고, 작은 연산 단위를 처리할 때 L2/L1/Shared Memory/Registers를 활용한다. Cache Hit이면 HBM을 다시 읽지 않을 수도 있다.
4. **10.498GiB KV를 한 SM에 모두 넣지 않는다.** 그것은 16개 Full Attention 층·4개 KV Head·172K Token의 **전체 논리적 저장량**이다. 개별 Kernel은 맡은 Layer/Head/Token 구간의 K/V Tile을 단계적으로 접근한다.
5. **작게 나눠 읽는 것은 10GiB를 없애는 일이 아니다.** 전체 과거를 조회하는 Full Decode라면 최종적으로 긴 KV History를 읽는 총량이 크다. 이것이 용량 제약과 대역폭 제약을 동시에 이해해야 하는 이유다.

## 1. GPU 메모리의 ‘단위’는 무엇인가?

### 1.1 네 층위: 저장소와 작업대를 구분한다

```text
H200/H100 같은 Hopper GPU 한 장 (개념도)
┌────────────────────────────────────────────────────┐
│ HBM / Global Memory — GPU에 붙은 대용량 DRAM        │
│  • Weight 행렬들, KV Cache, Activation, Workspace   │
│  • 커널 실행이 끝나도 할당이 유지되는 큰 데이터      │
│                                                    │
│                  ⇅  Memory traffic                 │
│ ┌────────────────────────────────────────────────┐ │
│ │ L2 Cache — GPU 칩 전체 SM에서 공유하는 Cache   │ │
│ └────────────────────────────────────────────────┘ │
│             ⇅                         ⇅            │
│ ┌───────────────────────┐ ┌──────────────────────┐ │
│ │ SM 0                  │ │ SM 1 ... SM n       │ │
│ │ L1 Cache/Shared Mem   │ │ L1/Shared           │ │
│ │ Register File         │ │ Registers           │ │
│ │ CUDA / Tensor Cores   │ │ CUDA / Tensor Cores  │ │
│ └───────────────────────┘ └──────────────────────┘ │
└────────────────────────────────────────────────────┘

중요: 이 도식은 계층 관계를 보여준다.
모든 Load가 ‘HBM → L2 → L1 → Shared → Register’를 반드시
순서대로 통과한다는 뜻은 아니다. 캐시 우회/비동기 전송/
Direct Global Load/Compiler 최적화 등 실제 데이터 경로는 다양하다.
```

| 메모리 | 물리 범위 | 무엇을 보관/관리하는가? | 자주 하는 오해 |
|---|---|---|---|
| HBM (Global) | GPU 카드/패키지의 대용량 DRAM | 지속 할당된 Weight, KV, Activation | “한 번 연산하면 가중치가 SM에 영구적으로 남는다” |
| L2 Cache | GPU 전반, 여러 SM이 공유 | 최근 사용한 Global Memory의 Cache Line 등 | “모델을 전부 L2로 복사해야 한다” |
| L1 Cache / Shared Memory | SM 가까이, 한 SM의 제한된 온칩 자원 | 데이터 자동 Cache(L1), Kernel이 명시 배치한 Tile(Shared) | “L1과 Shared는 언제나 같은 동작” |
| Registers | SM의 Thread별 자원 | Accumulator, 인덱스, 스칼라 중간값 등 | “행렬 전체를 Register에 저장한다” |

CUDA Memory Space를 가장 먼저 구분하자. **Register와 Shared는 SM 근처에 존재**하지만 사용 범위가 다르며, Shared Memory는 Thread Block 내 협업과 데이터 재사용을 위해 제어한다. L1 Cache는 주로 HW Cache이고 GPU 전반에 별도의 L2가 있다.

### 1.2 실제 크기는 어느 정도 차이인가?

- **H200 SXM HBM3e**: 약 **141 GB** (벤더 표기, GB; 실제 사용 가능 할당 용량과 정확히 같다고 단정하지 않음).
- **H100 SXM Hopper 아키텍처의 공개 예시**: GPU 전체 L2 **50 MB**, SM 하나의 **최대 Configurable Shared Memory 약 228 KB**, SM 하나의 Register File **256 KB**. H200 역시 Hopper 계열이지만 이 H100 수치가 H200 특정 SKU/커널의 실제 사용량이라는 뜻은 아니다.
- 한 Kernel의 Shared Memory 예약량은 SM 전체 최대치보다 작을 수 있고, **하나의 SM에는 여러 Thread Block이 동시 상주할 수 있어** 각 Block의 작업 공간도 제한된다.

감각적으로 생각하자: **HBM은 큰 창고, L2는 여러 작업대가 공유하는 중간 보관함, Shared/Register는 한 작업자가 당장 쓰는 작은 작업대**다. 이 비유는 물리 계층 이해를 돕기 위한 것이며 캐시 동작을 완전히 기술하지 않는다.

## 2. “모델 Weight는 작은 행렬 집단인가?”를 정확히 고치자

### 2.1 거대한 모델은 다양한 Weight Tensor로 이루어진다

```text
Model
 ├── Embedding Matrix                     [Vocab, Hidden]
 ├── Layer 0
 │    ├── (Full Attn이면) WQ, WK, WV, WO  (Matrix)
 │    ├── (GDN이면) 여러 Projection       (Matrix)
 │    ├── FFN Wgate                        (Matrix)
 │    ├── FFN Wup                          (Matrix)
 │    ├── FFN Wdown                        (Matrix)
 │    └── Norm Scale                      (Vector)
 ├── Layer 1: 그 Layer의 다른 Weight Tensor들
 ...
 ├── Layer 63: 그 Layer의 다른 Weight Tensor들
 └── LM Head / Output Weights

모델의 ‘27B’는 대략 전체 학습 파라미터 수의 명칭이지
‘27B × 27B 크기 행렬 하나’라는 뜻이 아니다.
```

이 모델은 여러 Layer에서 서로 다른 학습된 가중치를 사용한다. **같은 FFN 구조를 64번 재사용하더라도 Weight 값은 층마다 다르다.** 요청마다 Weight를 복제하는 대신 하나의 모델 인스턴스에서 여러 요청이 같은 Weight를 공유할 수 있다.

### 2.2 진짜 큰 가중치 행렬 하나

Qwen3.8-27B의 FFN Up Projection 중 하나를 예로 들어보자.

```math
W_{\mathrm{up}}:[5120,17408]
```

BF16 원소는 2 Byte이고, 가중치의 원소 수는:

```math
5120\times17408=89{,}128{,}960\ \mathrm{elements}
```

따라서 저장량은:

```math
89{,}128{,}960\times2=178{,}257{,}920\ \mathrm{Byte}
=\boxed{170\ \mathrm{MiB}}
```

**이 한 행렬이 170 MiB**다. ‘모델은 작은 행렬들만 모아둔 것’이라고 이해하면 오해가 생긴다. 큰 행렬과 작은 Weight Tensor가 함께 존재하며, 전체가 매우 크다.

### 2.3 Weight Tile은 저장된 Weight와 무엇이 다른가?

```text
한 FFN Up Weight (HBM에 논리적으로 존재)
       17408개의 출력 Feature 열 →
  ┌──────────────────────────────────────────────┐
  │ ┌─────────┐                                  │
  │ │ Tile    │                                  │
  │ │128×128  │ ← 방금 계산에 필요해서 선택한 부분│
  │ └─────────┘                                  │
  │                                              │
  │                  Wup [5120,17408]             │
  │                                              │
  └──────────────────────────────────────────────┘
     ↑
   입력 Feature 행 5120개

Wup 전체 = Weight Tensor / 연산 도중 선택한 조각 = Weight Tile.
큰 가중치 Tensor를 미리 128×128 조각으로
물리 파일에 저장해야만 Tile을 쓸 수 있다는 뜻은 아니다.
```

**128×128 BF16 Tile은 128×128×2 = 32,768 Byte = 32 KiB.** 이 값은 Tile에 들어 있는 원소의 *논리적 크기*다. 실제 Memory Load는 Thread Layout, Alignment, Coalescing, Cache Line 등으로 구현별로 달라진다.

그리고 **Tensor Core가 직접 [128,128]을 한 번에 단일 명령으로 계산하는 것도 아니다.** Kernel이 큰 Tile을 더 작은 하드웨어 명령(MMA Instruction)의 조각으로 나누어 처리할 수 있다.

## 3. 실제 GEMM 하나가 어떻게 타일로 계산되는가?

### 3.1 한 번에 64개 Token의 FFN Up Projection

이번 배치의 Input이 64 Token이라고 가정한다. (엔진의 실제 Chunk/Batched Token 수와는 다를 수 있다.)

```math
X:[64,5120],\quad W:[5120,17408],\quad Y=XW:[64,17408]
```

Y 전체의 한 출력 블록을 **64 Token × 128 Feature**로 계산하자.

```text
                 W [5120 × 17408]
                 출력 열 0..127 부분
                ┌──────────────┐
 K축 0..127     │ W(0) 128×128 │
 K축 128..255   │ W(1) 128×128 │
 ...            │ ...          │
 K축 4992..5119 │ W(39)        │
                └──────────────┘
                        ↑ K축 따라 40번

X [64 × 5120]                       Y [64 × 17408]
┌─────────────────────┐              ┌───────────────────┐
│ X(0) [64×128]      │ ───────────→ │ Y tile [64×128]  │
│ X(1) [64×128]      │   accumulate │  계속 더해 완성   │
│ ...                 │              └───────────────────┘
│ X(39)               │
└─────────────────────┘

그 뒤 출력 열 128..255, 256..383, ... 구간을 계산한다.
실제 고성능 GEMM 커널은 Tile을 동시에 병렬 실행하고
X/Weight tile을 재사용하도록 배치한다.
```

원리는 다음과 같다.

```math
Y_{0:64,\;0:128}=\sum_{j=0}^{39}
X_{0:64,\;128j:128(j+1)}
W_{128j:128(j+1),\;0:128}
```

행렬 Shape: 각 단계의 [64,128] × [128,128] = [64,128]. **모두 같은 출력 Tile의 부분 결과이므로 40개를 더해야 한다.**

한 단계에서 필요한 **원소 크기**는:

| 항목 | Shape | 데이터 유형 가정 | 개념적 Bytes |
|---|---|---|---:|
| X Tile | [64,128] | BF16 | 16 KiB |
| Weight Tile | [128,128] | BF16 | 32 KiB |
| 결과 Accumulator | [64,128] | FP32 | 32 KiB |
| 합계 | 해당 없음 | 해당 없음 | **80 KiB** |

**80 KiB는 세 데이터 크기의 합일 뿐, 반드시 80 KiB를 Shared Memory에 한번에 적재한다는 의미가 아니다.** 일반적으로 Accumulator의 상당 부분은 Register로 보관되며, Shared/Registers/Cache 역할과 동시 상주 Block 수에 따라 실제 사용량이 달라진다. 이 크기 비교는 “SM당 228 KB 정도의 Shared Memory로 170 MiB Weight를 다 처리할 수 있나?”라는 질문에 대한 출발점이다.

출력 Tile 하나의 근사 FLOP 수:

```math
2\times64\times128\times5120
=83{,}886{,}080\ \mathrm{FLOP}\ (\approx83.89\ MFLOP)
```

- K=5120을 128씩 나눴으므로 **40회**의 부분 계산이 필요하다.
- N=17408의 출력을 128개 열씩 나눴으므로 **136개 출력 열 Tile**이 존재한다.
- 다른 Thread Block / SM이 서로 다른 출력 Tile을 **동시에** 처리할 수 있다. 40회는 개념적 K 축 누산 횟수이지, 모든 GPU가 40번 ‘전역 동기화’한다는 뜻이 아니다.
- Tile을 통한 실행의 목적은 **작은 작업 공간에서 큰 행렬곱 완성 + 재사용 + 병렬성**이다. Tile이 크면 항상 빠르거나 작으면 항상 느린 것도 아니다.

### 3.2 실제 실행 시 누가 데이터를 가져오나?

“GPU가 레이어 1에서 Weight 행렬 하나를 통째로 SM에 로드한 다음 연산”하는 것이 아니다.

1. 보통 모델 로딩 시 Weight Tensor는 HBM에 적재된다. 일부 Offload/Quantization/TP 경로는 예외다.
2. Layer의 GEMM Kernel이 실행되면 SM들에 **작업할 Output Tile**들이 분배된다.
3. 각 작업은 필요한 X/Weight의 부분 데이터를 전역 메모리 경로에서 읽고, L2/L1/Shared/Register를 활용한다. Kernel 구현에 따라 비동기 복사/Prefetch 등이 가능하다.
4. 부분 곱을 Tensor Core MMA 등의 명령으로 누산한다.
5. 완성된 Output Tile을 필요한 곳에 기록하거나 후속 연산과 Fusion한다.
6. 후속 Layer는 이전 Layer가 만든 Activation을 입력으로, **해당 Layer 고유 Weight**를 사용한다.

**단순한 “한 레이어 → 다음 레이어”는 수학적 의존성이고, 실제 GPU에서는 여러 SM이 한 Layer의 서로 다른 Tile을 동시에 계산하는 병렬 실행**이다. Weight를 한 번 SM에 두면 다음 Token까지 영구 유지한다는 보장은 없다. L2 등에 일부 데이터가 남으면 Cache Hit로 재사용될 수 있다.

### 3.3 Prefill의 Weight 재사용 vs Decode의 작은 행렬

```text
같은 Weight W:[5120,17408]

Prefill (예: N=8192)           Decode (예: B=1)
X:[8192,5120]                 X:[1,5120]
      × W                         × W
큰 행렬곱 / 여러 Token         매우 얇은 행렬곱·GEMV 유사
Weight Tile을 여러 Token       같은 Weight Tile을 한 Token
행에 걸쳐 재사용할 여지        계산에만 써 재사용률이 낮음

실제 Decode B>1 또는 여러 요청 묶음에서는 GEMM Shape/효율이 달라짐
```

이 때문에 Prefill은 연산 밀도(Arithmetic Intensity)가 커져 Compute 자원을 활용하기 쉽고, 작은 Batch Decode는 HBM에서 Weight를 공급하는 비용의 비중이 커지기 쉽다. 단, **항상 Compute/Memory Bound라고 단정하지 않는다.** Kernel Shape, GDN/Attention Backend, Context Length와 Batching이 영향을 준다.

---

## 4. “10GiB KV Cache를 SM이 어떻게 읽나?”: 저장 범위부터 나누자

### 4.1 10.498 GiB가 한 GPU/한 Layer/한 Head의 크기는 아니다

Qwen3.8-27B의 **Full Attention 16층만** 세어보자. 4개의 KV Head, head_dim=256, BF16, Context 172000 Token이다.

| 범위 | 계산 | Raw Storage |
|---|---|---:|
| Token 1개, K만, Head 1개 | 256×2B | **512 Byte** |
| Token 1개, K와 V, Head 1개 | 2×256×2B | **1 KiB** |
| Token 1개, K와 V, Layer 1개(4 Head) | 4×1KiB | **4 KiB** |
| Token 1개, K와 V, Full 16층 합 | 16×4KiB | **64 KiB** |
| 172K Token, Head 1개, Layer 1개 | 172000×1KiB | **167.969 MiB** |
| 172K Token, 16층·4 Head 합 | 172000×64KiB | **10.498 GiB** |

**해석: 10.498 GiB는 Full 16개 층의 모든 Head·모든 토큰의 합계다.** 이 데이터 전체를 단일 Attention Kernel이 한 번에 SM으로 가져오는 것이 아니다. 또한 GDN recurrent/convolution state는 **포함하지 않는 수치**다.

### 4.2 현재 한 KV Head의 64 Token만 가져온다고 가정

한 Head만 보면 64개 Token의 K와 V는:

```math
K_{\mathrm{tile}}:[64,256]\times2\ \mathrm{B}=32\ \mathrm{KiB}
```

```math
V_{\mathrm{tile}}:[64,256]\times2\ \mathrm{B}=32\ \mathrm{KiB}
```

합쳐 **64 KiB**다.

```text
GPU HBM: 한 Layer, KV Head 하나의 172K 기록
[KV token 1..64] [65..128] [129..192] ... [171969..172000]
       │ 64 Token짜리 논리적 블록            ▲ 마지막 32개
       ↓
  SM이 처리하는 K/V Tile (가정)
     K [64,256] = 32 KiB
     V [64,256] = 32 KiB
       │
       ├─ 현재 Query와 K의 Dot Product / Score
       ├─ Softmax 누산 통계 갱신
       └─ Value의 가중합 Partial Output 갱신
       ↓
     다음 K/V Tile 처리 ... 이어서 최종 출력 완성

172000/64 = 2687.5
→ 2687개 완전한 64-Token 타일 + 나머지 32-Token 타일 하나
(논리 예제: 실제 Paged Blocks나 Tensor Layout과는 무관)
```

**중요:** Tile 64개는 설명을 위해 고른 수치다. 실제 FlashAttention/Decode Kernel의 Tile Shape, Attention Head를 어떻게 묶는지, GPU SM 분배, SRAM 할당은 Backend마다 다르며 Head별 K/V를 Tensor Core에 공급하는 방식도 다양하다.

### 4.3 “전체 Softmax”인데 왜 작게 나눌 수 있나?

Full Attention에서는 **현재 Query와 모든 과거 Key의 점수를 상대적으로 정규화**해야 하므로, 각 Tile의 Softmax를 별개로 계산한 뒤 그냥 평균 내면 틀린 결과가 된다.

FlashAttention 계열의 **Online Softmax**는 처리한 Tile들의 최대 점수 m, 지수합 l, Value의 누적 가중합(Accumulator)을 갱신하며, 이전 Tile의 누적값을 새 최대 점수 기준으로 재조정한다.

```math
m_{\mathrm{new}}=\max(m_{\mathrm{old}},\max(S_{\mathrm{tile}}))
```

```math
l_{\mathrm{new}}=e^{m_{\mathrm{old}}-m_{\mathrm{new}}}l_{\mathrm{old}}+
\sum_j e^{s_j-m_{\mathrm{new}}}
```

```math
a_{\mathrm{new}}=e^{m_{\mathrm{old}}-m_{\mathrm{new}}}a_{\mathrm{old}}+
\sum_j e^{s_j-m_{\mathrm{new}}}v_j,\qquad o=\frac{a_{\mathrm{final}}}{l_{\mathrm{final}}}
```

위 식은 **하나의 Query 행에 대한 개념적 유도**이고, 실제 구현은 Query Tile을 함께 처리하거나 커널별 Reduction을 사용한다. 이론상 전체 Score 행렬을 HBM에 보관하지 않고 Softmax Attention을 계산할 수 있으며, 부동소수점 산술 순서에 따른 작은 수치 차이는 발생할 수 있다.

### 4.4 작은 Tile로 처리하면 HBM 대역폭 문제가 사라질까?

**아니야.** 현재 Query가 전체 과거를 조회해야 한다면 전체 Context의 과거 K/V에 접근해야 한다. Tile로 나누는 것은 **온칩 작업 공간 사용과 중간 Score 저장 비용을 줄이는 실행 방법**이지, 논리적으로 조회해야 하는 K/V 자체를 자동으로 없애는 방법이 아니다.

- **Capacity**: 전체 Cache가 HBM에 들어가나?
- **Bandwidth**: 필요한 KV를 HBM에서 얼마나 빨리 공급할 수 있나?
- **On-chip Resource**: 한 번 처리할 Tile을 어떤 크기로 잡고 재사용하나?
- **Compute**: Dot Product / MMA / Softmax를 얼마나 빠르게 실행하나?

GQA라면 여러 Q Head가 같은 KV Head를 공유하므로 **K/V 타일 재사용**이 가능하다. 다만 실효 재사용 정도는 실제 Kernel의 Schedule과 L2 Cache Hit 등에 달려 있다.

### 4.5 TP=4일 때에는 Head별로 GPU가 나뉠 수 있다

```text
Qwen GQA: 4 KV Head × Full Attention 16층

TP=1:
GPU0 → Head0,1,2,3의 모든 T Token
     → Raw KV 10.498 GiB (Full 16층 기준)

TP=4 (대표적인 Head Sharding 가정):
GPU0 → KV Head0의 모든 T Token → 약 2.625 GiB
GPU1 → KV Head1의 모든 T Token → 약 2.625 GiB
GPU2 → KV Head2의 모든 T Token → 약 2.625 GiB
GPU3 → KV Head3의 모든 T Token → 약 2.625 GiB

GPU0은 Token 1..T/4만 저장하는 것이 아님.
각 GPU가 자기 Head에 대해 전체 Context를 보관함.
```

TP=8의 단순한 KV Head 분할은 Head 수 4보다 Rank 수가 많아서 불가능하므로, vLLM의 일반적 Head Parallel 구현은 **KV Head를 복제**할 수 있다. KV Cache가 TP에 무조건 정확하게 반비례한다고 가정하면 잘못이다. 자세한 계산은 [00b](00b-tp-and-llm-inference-cost-from-first-principles.md)의 7~9절 참조.

---

## 5. Tiling, Cache, PagedAttention, TP를 한 번에 구분하기

| 이름 | 분할하는 대상 | 목적 | 모델 계산 결과가 바뀌는가? |
|---|---|---|---|
| GEMM Tiling | 입력/가중치 행렬의 **계산 블록** | 온칩 메모리에 맞춰 Compute와 Reuse | 수학적으로 동일, 반올림 차이 가능 |
| Attention Tiling/FlashAttention | Q/K/V의 **연산 블록** | Attention Score의 큰 HBM Materialization 회피 | 수학적으로 동일, 오차 가능 |
| L2/L1 Cache | 최근 접근한 Memory Cache Line | 반복 Load의 HBM 접근 회피 | 결과 동일 |
| PagedAttention / KV Block | KV의 **물리 할당·논리↔물리 주소 관리** | Fragmentation, 성장 및 공유 관리 | Attention 수식 자체와 별개 |
| Tensor Parallel | Weight의 Feature/Head와 부분 계산 결과 | 여러 GPU에 연산·Memory 분산 | Collective 필요, 부동소수점 순서 변화 가능 |
| Context Parallel | Token/Sequence 구간과 관련 State | 긴 Context를 GPU에 분산 | 추가 정보 교환 필요 |

**Tiling은 하나의 GPU 내부에서도 실행하며, TP와는 다른 차원이다.** TP4 GPU 각각도 자신의 Weight Shard를 다시 Tile로 나누어 계산한다.

---

## 6. 발표에서 핵심이 되는 ‘한 장의 그림’

**제안 삽입 위치:** 1회차 슬라이드 7의 Tensor Core 소개 직후, 또는 슬라이드 8 Transformer Block 설명 직전. 시간이 부족하면 슬라이드 7 오른쪽의 확대 그림으로 결합한다.

```text
[모델/요청 데이터 저장]
                 HBM
    Weight Wup:[5120,17408]  (BF16 170 MiB)
    KV Cache: 16 Full layers, T=172K (BF16 10.498 GiB)
                 │
                 │ On-demand Loads / Reuse
                 ▼
                L2
                 │
       ┌─────────┴──────────┐
       │ SM: Shared/Register│
       │                     │
       │ GEMM Weight Tile    │
       │   [128,128],32 KiB  │
       │ Input Tile          │
       │   [64,128],16 KiB   │
       │       ↓             │
       │ Tensor Core MMA     │
       │   Accumulate Output │
       └─────────────────────┘
                 │
                 ▼
       Output Tile → HBM / 다음 Kernel

[Decode Attention에서는]
HBM KV History → K/V Tile 64 Token씩(64 KiB/head)
               → QK^T + Online Softmax + PV 누적
               → 다음 Tile을 반복해 결과 완성

‘전체 저장량’과 ‘한 번의 작업 공간’은 크기와 역할이 다르다.
```

**발표자가 말할 문구:** “Weight 행렬은 매우 크지만 Tensor Core가 그것을 통째로 들고 계산하는 건 아닙니다. SM은 HBM에서 작은 Tile을 받아 수천만 번의 곱셈-덧셈을 누적해 결과를 완성합니다. KV도 마찬가지로 전체 10GiB를 SM에 적재하지 않습니다. 다만 나눠 가져와도 전체 과거를 훑는 데 필요한 총 Byte는 없어지지 않기 때문에 Memory Bandwidth가 중요한 겁니다.”

---

## 7. 교재 스스로 검증할 계산 문제

1. BF16 [5120,17408] Weight 행렬 하나는 원소 몇 개, 몇 MiB인가?
2. [128,128] BF16 Weight Tile 하나는 몇 KiB인가?
3. [64,5120] × [5120,17408]의 결과 Shape는?
4. Output Tile [64,128]은 K=5120을 128씩 나누면 누산 40회, 출력 열 N=17408을 128씩 나누면 136개 Column Tile이 되는 이유는?
5. Head dimension 256, BF16일 때 한 Head의 Token 하나에 K와 V가 각각 몇 Byte인가?
6. 172K·Full 16층·4KV Head의 KV가 약 10.498 GiB여도 **단일 SM이 그 전체를 담지 않아도 되는 이유**는?
7. 타일 크기를 2배로 키우면 항상 빠른가? (아니오 — Shared/Register 압박, Occupancy 및 Reuse·Kernel 최적화 균형 필요.)
8. FlashAttention, PagedAttention, TP는 각각 무엇을 다르게 분할하는가?

**정답:** 89,128,960 원소 / 170 MiB; 32 KiB; [64,17408]; K 누산 40회 및 N 출력 조각 136개; K 512 Byte/V 512 Byte; 한 번에 작업할 일부만 온칩에 보유하며 Tile별 누적; 조건마다 다름; Compute Tile·KV Allocation Block·GPU별 Feature/Head Shard.

---

## 8. 출처와 검증 수준

- CUDA의 Global/Shared/Registers 메모리 스코프 및 On-chip vs Off-chip 정의: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html
- Hopper H100의 L2 50MB, SM당 최대 Shared 228KB, Register File 256KB, SM 구조: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- NVIDIA FlashAttention Tile의 HBM→온칩 SRAM, QK/Softmax/PV 및 Online Softmax 설명: https://developer.nvidia.com/blog/co-designing-ai-model-attention-for-fast-interactive-long-context-inference
- FlashAttention의 Tile 크기·Occupancy·Online Softmax 트레이드오프: https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/
- Qwen3.8-27B의 config와 실제 Layer별 Projection/FFN/KV의 정의: [PR #27 종합 마스터 노트](LLM-serving-hardware-attention-master-notes.md) 12~15장.
- 수치 170 MiB, 32/16/32 KiB, 167.969 MiB 및 10.498 GiB, Tile 40·136회 등은 **위 Shape/Byte 가정으로 계산한 산술값**. 특정 vLLM 커널의 실제 Load Transaction/Cache Hit를 측정한 값이 아니다.
