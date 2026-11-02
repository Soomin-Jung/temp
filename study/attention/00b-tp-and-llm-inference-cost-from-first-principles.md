# 00b. LLM 물리적 비용을 처음부터 계산하기 — GEMM, FFN, Attention, KV Cache와 Tensor Parallelism

> 대상: LLM 추론 시스템 배포 엔지니어, 행렬곱 및 분산 추론의 기초부터 학습할 독자  
> 위치: [00a 행렬·GEMM·QKV 입문](00a-matrix-gemm-to-qkv-from-zero.md) 다음, [종합 마스터 노트](LLM-serving-hardware-attention-master-notes.md)의 13~16장 앞  
> 목적: **TP=1에서 실제로 무엇을 계산하는지 → TP=N에서 어느 축을 나누는지 → KV와 통신을 몇 Byte로 세는지 → GPU 사양에서 어떤 한계를 만나는지** 이해한다.  
> 모델 예시: PR #27 원고의 Qwen3.8-27B 텍스트 추론, Full/Gated Attention 16층과 GDN 48층.  
> 범위 제한: GDN의 recurrent 수식·상태 샤딩/통신 최적화는 후속 심화 과정. 여기서는 **Full Attention과 모든 층의 Dense FFN**을 확실히 이해한다.

> **GPU가 실제로 작은 데이터 조각을 어떻게 처리하는지가 아직 낯설다면:** [00c: HBM·L2·SM·Weight Tile·KV Tiling 실물 크기 계산](00c-gpu-memory-weight-kv-tiling-from-zero.md)을 먼저 읽는다. **Weight Tile은 TP Weight Shard가 아니며**, 한 GPU 안에서도 GEMM은 Tile 단위로 나눠 처리한다.

**핵심 판정:** TP는 단순히 “모델 파라미터를 GPU 수로 나눈다”가 아니다. **레이어 안에서 출력 Feature 또는 입력 Feature, Attention Head를 나누고, 필요한 곳에서 결과를 다시 결합하는 실행 전략**이다. Weight, KV, 활성화, FLOP, 통신량은 **각각 다른 분할 규칙**을 갖는다.

## 0. 이 문서의 지도

```text
[1] 숫자·행렬·FMA → GEMM의 원소 계산 → FLOPs
          ↓
[2] Tensor Core가 처리하는 행렬 블록
          ↓
[3] TP=1: Q/K/V Projection + Attention + FFN
          ↓
[4] KV Cache: 재계산을 피하지만 Token별 저장·조회 비용 발생
          ↓
[5] Prefill / Decode: 어떤 Shape와 비용을 만들까?
          ↓
[6] TP=N: FFN Feature 분할 / Attention Head 분할
          ↓
[7] KV Cache의 Head Sharding / Replication
          ↓
[8] All-Reduce의 부분 결과 합산 / rank별 bytes
          ↓
[9] FLOP·Byte·Latency를 H200 Compute/HBM/Link에 연결
```

반드시 분리할 개념 네 가지:

| 개념 | 질문 | 단위 |
|---|---|---|
| FLOP | 연산을 얼마나 많이 하는가? | FLOP |
| Throughput | 1초에 얼마나 계산할 수 있는가? | FLOP/s |
| Capacity | 동시에 얼마나 저장할 수 있는가? | GB, GiB |
| Bandwidth | 1초에 얼마나 이동할 수 있는가? | GB/s |

GB는 보통 10^9 Byte, GiB는 2^30 Byte다. **BF16 1원소 = 2 Byte**를 기본으로 계산하되, 실제 엔진은 FP8/양자화/Block Allocation/Workspace와 별도 상태를 가질 수 있다.

---

## 1. 원점: 행렬의 한 칸은 어떻게 계산되는가?

### 1.1 숫자 하나를 곱하는 데서 시작

스칼라 두 개 3과 4를 곱하면 12다. 벡터 두 개의 내적은 **같은 위치 원소끼리 곱한 결과를 전부 더한다**.

```math
\begin{bmatrix}1&2&3\end{bmatrix}\begin{bmatrix}7\\9\\11\end{bmatrix}
=1\times7+2\times9+3\times11=58
```

이 58이 **출력 행렬의 딱 한 칸**이다.

### 1.2 행렬 두 개의 실제 숫자 계산

```math
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}\;(2\times3),\qquad
B=\begin{bmatrix}7&8\\9&10\\11&12\end{bmatrix}\;(3\times2)
```

조건: **왼쪽 행렬의 열 수(3) = 오른쪽 행렬의 행 수(3)**. 출력 크기는 **왼쪽 행 수 × 오른쪽 열 수 = 2×2**.

각 칸은 다음과 같다.

```math
C_{0,0}=1\cdot7+2\cdot9+3\cdot11=58,\quad
C_{0,1}=1\cdot8+2\cdot10+3\cdot12=64
```

```math
C_{1,0}=4\cdot7+5\cdot9+6\cdot11=139,\quad
C_{1,1}=4\cdot8+5\cdot10+6\cdot12=154
```

```math
C=AB=\begin{bmatrix}58&64\\139&154\end{bmatrix}\;(2\times2)
```

인덱스를 일반화하면:

```math
C_{i,j}=\sum_{k=0}^{K-1} A_{i,k}B_{k,j},\qquad
A:[M,K], B:[K,N], C:[M,N]
```

**행렬곱과 원소별 곱(Hadamard Product)은 다르다.** 위 식은 각 출력 칸이 전체 K축을 훑어 곱하고 더한다. 원소별 곱은 동일 위치만 곱하고 합산하지 않는다.

### 1.3 FMA는 왜 2 FLOP인가?

**Fused Multiply-Add(FMA)**는 수학적으로 c ← a×b+c를 수행하는 명령 형태다. 곱셈 1회 + 덧셈 1회로 **관례상 2 FLOP**으로 센다. FMA가 하나의 명령이라는 사실과 FLOP이 하나라는 주장은 다르다. FMA는 보통 곱셈과 덧셈을 결합해 중간 반올림을 줄인다.

위 2×3 및 3×2의 예에서는:
- 실제 산술식 기준: 출력 4칸 × (곱셈 3회 + 합산 2회) = **20 scalar 연산**.
- GEMM의 표준 FLOP 근사: 출력 4칸 × K(3) × 2 = **24 FLOP**.
- 두 값의 차이: 각 출력 칸의 최초 누산을 0에서 시작하는 FMA 관례 및 경계 효과. K가 크면 상대 차이가 작아진다.

따라서 문서에서 사용하는 GEMM의 대표적인 계산량은:

```math
\boxed{F_{\mathrm{GEMM}}\approx 2MNK\;\mathrm{FLOP}}
```

**M은 처리할 Token 행 수, K는 입력 Feature 차원, N은 출력 Feature 차원**으로 해석할 수 있다. 주의: 수식의 N은 Output Feature이고, 뒤 절에서 Token 수를 T라고 표기하여 혼동을 막는다.

### 1.4 GEMM vs GEMV, Tensor Core

- **GEMM**은 일반적인 Matrix×Matrix 곱. Prefill처럼 여러 Token을 행으로 묶으면 큰 행렬 곱이 된다.
- **GEMV**는 Matrix×Vector 곱. Decode의 단일 Token, 작은 Batch에서는 원리상 GEMV/skinny GEMM에 가까운 형태가 나타난다.
- **CUDA Core**는 다양한 일반 산술을 위한 실행 유닛이며, **Tensor Core**는 타일 단위의 Matrix Multiply-Accumulate(MMA)를 높은 처리량으로 수행하도록 특화된 연산 자원이다.
- **Tensor Core 사용이 보장되는 것은 아니다.** Shape·Dtype·정렬·커널 구현·Tensor Core 지원에 따라 실제 선택이 달라진다.
- Softmax, SiLU, RMSNorm, Residual, RoPE, 메모리 이동은 전체가 GEMM이라는 뜻이 아니다.

---

## 2. TP=1에서 LLM Layer는 무엇을 계산할까?

### 2.1 전체 경로: Attention/Token Mixing과 FFN은 다른 역할

```text
Input Hidden States X: [T, d]
    │
    ├─ RMSNorm
    ├─ Q/K/V Projection  →  Attention(Q,K,V)  →  Output Projection
    │                        (또는 GDN Token Mixing)
    ├─ Residual
    ├─ RMSNorm
    ├─ FFN: Gate Projection / Up Projection → SiLU× → Down Projection
    └─ Residual → Next Layer
```

위 그림은 설명용 블록이며 실제 Norm/Output Gate/Fusion 등의 세부는 모델 구현에 따른다.

- Attention(Full): 과거 **서로 다른 Token 위치**에서 유효한 정보를 가져와 섞는다.
- FFN(Dense SwiGLU): **각 Token 위치별 Feature 차원**을 학습된 가중치로 변환한다. 다른 Token의 FFN 출력을 직접 조회하지 않지만, 입력 X에는 이전 Attention이 집계한 정보가 들어 있다.
- 두 블록 모두 많은 Projection/GEMM을 수행한다.
- Qwen3.8-27B에서는 **16개 Gated Full Attention 층 + 48개 GDN 층**, **64개 층 모두에 Dense FFN**이 있다.

### 2.2 Q/K/V Projection은 모두 GEMM이다

하나의 Full Attention 층 기준:

```math
X:[T,5120],\;W_Q:[5120,6144],\;W_K:[5120,1024],\;W_V:[5120,1024]
```

```math
Q=XW_Q:[T,6144],\quad K=XW_K:[T,1024],\quad V=XW_V:[T,1024]
```

- Q head 24 × head dimension 256 = 6144.
- KV head 4 × head dimension 256 = 1024.
- **5120/24를 head dimension으로 계산하면 틀린다.** 이 모델의 head_dim은 config에 256으로 명시된다.
- 실제 Qwen Gated Full Attention의 Q Projection은 **Attention 출력 Gate**를 함께 만들어 단순 Q 부분만의 폭보다 넓다. FFN/Attention 비용 회계에서는 이를 다시 포함한다.

그다음 Head별로 분리한 Query와 대응 KV로 점수를 계산한다.

### 2.3 Self-Attention도 GEMM 두 번을 포함한다

한 Head에서 Prompt 길이를 T, Head 차원을 256으로 두자.

```math
Q_h:[T,256],\quad K_h:[T,256],\quad V_h:[T,256]
```

```math
S_h=\frac{Q_hK_h^\top}{\sqrt{256}}+M:\ [T,T]
```

여기서 S의 (i,j) 원소는 **i번째 Query와 j번째 Key의 내적**이다. Causal Mask M은 현재 위치가 미래 토큰을 읽지 못하게 막는다. Mask 이후 각 Query 행의 Softmax로 확률 가중치를 만든다.

```math
P_h=\operatorname{softmax}(S_h):[T,T],\qquad
O_h=P_hV_h:[T,256]
```

따라서 **Q/K/V Projection → QK^T → Softmax → PV → Output Projection**에서 Projection과 QK^T와 PV가 행렬 곱이며 Softmax는 행별 비선형 정규화다.

참고: 위 [T,T]는 **논리적 Score Shape**이지, FlashAttention 등의 구현이 반드시 HBM에 T×T 점수 행렬을 저장한다는 뜻이 아니다.

### 2.4 MHA / MQA / GQA의 수식은 동일하다

위 Scaled Dot-Product 식은 **MQA의 정의가 아니다.** MHA/MQA/GQA 모두 같은 Head별 점수·가중합 구조를 사용하고, Head의 K/V 공유 방식이 다르다.

| 방식 | Q Head 수 | KV Head 수 | 특징 |
|---|---:|---:|---|
| MHA (비교 가정) | 24 | 24 | Q마다 전용 KV |
| MQA (비교 가정) | 24 | 1 | 모든 Q가 1개 KV 공유 |
| GQA (Qwen 실제) | 24 | 4 | 6개 Q Head가 1개 KV 공유 |

```math
O_i=\operatorname{softmax}\!\left(
\frac{Q_iK_{g(i)}^\top}{\sqrt{d_h}}+M\right)V_{g(i)}
```

g(i)는 Query Head i가 매핑되는 KV Head 번호다. **KV Head 감소 = KV Projection 폭·KV 저장과 읽기 감소**이지 **Q Head 또는 Attention Score의 수가 같은 비율로 감소**한다는 뜻은 아니다.

---

## 3. FFN을 반드시 별도의 거대 GEMM으로 세어야 한다

Qwen의 Dense SwiGLU FFN, d=5120, m=17408을 간소화해 쓰자.

```math
G=\operatorname{SiLU}(XW_g),\quad U=XW_u,\quad
Y=(G\odot U)W_d
```

| 구간 | 입력·가중치 | 출력 | FLOP 근사 |
|---|---|---|---|
| Gate | [T,5120] × [5120,17408] | [T,17408] | 2Tdm |
| Up | [T,5120] × [5120,17408] | [T,17408] | 2Tdm |
| 원소별 SiLU × | 두 개 [T,17408] | [T,17408] | 별도, GEMM 아님 |
| Down | [T,17408] × [17408,5120] | [T,5120] | 2Tdm |

```math
\boxed{F_{\mathrm{FFN,layer}}=6Tdm
=534{,}773{,}760\,T\ \mathrm{FLOP}}
```

64층의 주요 FFN 계산량은 **token당 약 34.2255 GFLOP**이며, 170K prompt라면:

```math
F_{\mathrm{FFN,64layers}}=64\cdot6\cdot170000\cdot5120\cdot17408
\approx 5.818\ \mathrm{PFLOP}
```

이는 **170K Prefill 전체 작업량**이지 초당 처리 능력(PFLOP/s)이 아니다.

전체 연산은 FFN뿐 아니라 QKV/Output Projection, GDN QKV와 상태 연산, Full Attention의 QK/AV, Norm, Gate, LM Head 등으로 구성된다.

---

## 4. KV Cache의 물리적 이유 — 무엇을 저장하고 무엇을 다시 읽는가?

### 4.1 Causal Decoding에서 과거 K/V는 재사용할 수 있다

과거 토큰의 각 레이어 K/V는 미래 토큰이 추가되어도 **동일한 causal prefix 기준으로 계산된 과거 표현**을 재사용할 수 있다. 따라서 매 Decode Step마다 Prompt 전체를 다시 Projection·Attention·FFN 처리하지 않고, **새 Token의 Q/K/V만 계산해 K/V를 추가**한다.

그러나 새 Q는 **과거 저장된 K 전체를 비교**하고 V를 집계하므로 과거 Context 조회 비용은 남는다.

### 4.2 KV 원소 수와 Byte 계산

**Full Attention Layer 1개, Token 1개**에 대해:

```math
M_{\mathrm{KV/token/layer}}=2H_{kv}d_hb
```

2는 K와 V, Hkv는 KV Head 수, dh는 Head Dimension, b는 원소당 Byte다.

Qwen의 GQA 4KV, dh=256, BF16=2 Byte이면:

```math
2\times4\times256\times2=4096\ \mathrm{Byte}=4\ \mathrm{KiB}
```

16개 Full Attention Layer 전체:

```math
16\times4096=65536\ \mathrm{Byte}=64\ \mathrm{KiB/token}
```

172,000 Token을 하나의 요청이 점유한다면:

```math
172000\times65536/2^{30}\approx10.498\ \mathrm{GiB/request}
```

이 수치는 **요청 1건의 Full Attention KV Raw Payload**다. GDN recurrent/conv state와 Block rounding, Workspace, Weight, 실행상 중복 등이 빠져 있다.

---

## 5. Prefill과 Decode를 GEMM·Attention·Byte로 구분하자

| 지점 | Prefill | Decode (batch가 작을 때) |
|---|---|---|
| X의 행 수 | chunk 안의 다수 Token | 각 활성 Sequence당 1 Token |
| QKV/FFN Projection | 큰 GEMM → weight tile 재사용 유리 | skinny GEMM/GEMV → weight 이동 비중 증가 |
| Full Attention 본체 | Prompt 내 다수 Q와 과거 K; causal O(T²) | 새 Q가 길이 T의 과거 K/V 조회; step당 O(T) |
| 상태 | Prompt 전체 위치의 KV 생성 | KV append + 과거 KV read |
| 대표 병목 | 연산, HBM, kernel, scheduler 모두 가능 | 작은 batch에서 Weight/KV HBM 또는 통신 지연 가능 |

**기호 유의:** 전체 Context Length T와 한 번의 GPU 호출에서 처리하는 Token 행 수 N은 다르다. Chunked Prefill의 N은 chunk 크기이며, Decode에서 N은 활성 Sequence 수에 가깝다. Decode 시 T는 여전히 수만~수십만일 수 있다.

170K Causal Prefill을 주요 계산만 세면 (종합 마스터 노트 13.7~13.8절과 같은 조건):

| 주요 산술 범위 | 총 FLOP 근사 |
|---|---:|
| 모든 64층 FFN | 약 5.818 PFLOP |
| 16 Full + 48 GDN의 Projection 및 Gate | 약 2.461 PFLOP |
| 16개 Full Attention의 QK/AV | 약 5.682 PFLOP |
| **위 부분 합** | **약 13.961 PFLOP** |

GDN State Core·Conv, LM Head, Softmax, RMSNorm 등은 **계산 범위 밖**이다. 이 표의 FLOP 비중을 실행 시간 비중으로 해석하면 안 된다.

**핵심:** FFN은 T에 비례해 증가하지만 **64개 층 모두**에서 크고, Full Attention QK/AV는 인덱스별 허용 Token 쌍 때문에 O(T²)로 증가한다. “Prefill의 Compute는 Attention만”이라는 주장은 틀리다.

---

## 6. TP=1 → TP=N: 왜 Feature를 분할한다고 하는가?

### 6.1 TP와 PP와 CP는 자르는 축이 다르다

| 방법 | 주로 자르는 축 | 핵심 |
|---|---|---|
| TP (Tensor Parallel) | **동일 Layer 내부의 Weight Feature/Head** | 모든 TP Rank가 같은 Layer의 다른 부분 계산 |
| PP (Pipeline Parallel) | **Layer 구간** | 서로 다른 Rank가 다른 Layer 담당 |
| CP (Context Parallel) | **Sequence/Context 위치** | 서로 다른 Rank가 서로 다른 Token 구간의 State·계산 담당 |
| DP/Replica | **요청 또는 Batch 그룹** | 모델 사본 여러 개로 처리량 확장 |

따라서 “TP는 Feature를 쪼갠다”는 설명은 **대표적인 Column/Row Parallel GEMM에 대해 정확**하다. Attention에서는 Feature 축을 Head 축으로 그룹화해 분할하는 것이 일반적이다. Layer 자체를 떼어 다른 GPU에 순서대로 주는 것은 PP다.

### 6.2 Column Parallel: 결과의 서로 다른 열을 각 GPU가 담당

일단 숫자 행렬로 해보자.

```math
X=\begin{bmatrix}2&3\end{bmatrix},\qquad
W=\begin{bmatrix}1&2&3&4\\5&6&7&8\end{bmatrix}
```

TP=1의 전체 결과:

```math
Y=XW=\begin{bmatrix}17&22&27&32\end{bmatrix}
```

TP=2에서는 W의 **열**을 나눈다.

```math
W_0=\begin{bmatrix}1&2\\5&6\end{bmatrix},\quad
W_1=\begin{bmatrix}3&4\\7&8\end{bmatrix}
```

```math
Y_0=XW_0=[17,22],\quad Y_1=XW_1=[27,32]
```

완성된 원래 Y는 **열 방향 연결** 결과다. 하지만 바로 다음 연산이 각 조각을 받도록 설계되었다면 **이 단계의 All-Gather가 필요하지 않다**.

### 6.3 Row Parallel: 각각의 부분 합을 더해야 원래 출력

이번엔 다음 가중치를 사용하자.

```math
W_d=\begin{bmatrix}1&0\\0&1\\1&1\\2&1\end{bmatrix}
```

GPU0은 Y0=[17,22]와 위 행렬 첫 두 행, GPU1은 Y1=[27,32]와 마지막 두 행을 사용한다.

```math
Z_0=[17,22]\begin{bmatrix}1&0\\0&1\end{bmatrix}=[17,22]
```

```math
Z_1=[27,32]\begin{bmatrix}1&1\\2&1\end{bmatrix}=[91,59]
```

```math
\boxed{Z=Z_0+Z_1=[108,81]}
```

여기는 열을 잇는 것이 아니라 **같은 위치 원소를 합산해야** 한다. 다음 Layer가 완전한 Z를 모든 Rank에서 필요로 하는 일반적인 TP 배치라면 **All-Reduce(sum)**를 사용한다.

### 6.4 SwiGLU FFN의 TP=4 Shape

```text
공통 Input X                     [T,5120]
       ├── GPU0 Gate/Up columns    [5120,4352] / [5120,4352]
       ├── GPU1 Gate/Up columns    [5120,4352] / [5120,4352]
       ├── GPU2 Gate/Up columns    [5120,4352] / [5120,4352]
       └── GPU3 Gate/Up columns    [5120,4352] / [5120,4352]

각 GPU:  SiLU(G_r) ⊙ U_r         [T,4352]
각 GPU:  (SiLU(G_r)⊙U_r) Wdown_r
               Wdown_r           [4352,5120]
               partial output     [T,5120]
                               ↓
               All-Reduce SUM    [T,5120] on all TP ranks
```

- Gate/Up는 **Column Parallel**.
- Down은 **Row Parallel**.
- 중간 Activation은 Rank당 약 1/4 폭.
- 각 Rank는 **최종 결과 [T,5120]에 대한 부분 합**을 갖고, 하나의 Collective로 합산한다.
- 순수 선형 FLOP을 균등 분할한다면 Rank당 대략 **FFN FLOPs/4**. 하지만 norm·activation·launch·통신 및 복제 상태는 그렇지 않다.
- All-Reduce가 한 번인지, Reduce-Scatter/All-Gather로 나뉘는지는 Sequence Parallel 등 구현에 따라 달라진다.

---

## 7. TP에서 Attention은 QKV까지 나눈다

### 7.1 Full Attention에서 Head 기준 분할

Qwen의 Q Head는 24개, KV Head는 4개다. TP=4일 때 Rank당:

| 항목 | TP=1 전체 | TP=4 Rank당 |
|---|---:|---:|
| Q Head | 24 | 6 |
| K Head | 4 | 1 |
| V Head | 4 | 1 |
| Q 출력 폭 | 6144 | 1536 |
| K 출력 폭 | 1024 | 256 |
| V 출력 폭 | 1024 | 256 |
| Local Attention Output 폭 | 6144 | 1536 |

입력 X는 일반적인 TP 경로에서는 Rank마다 **필요한 전체 Hidden Feature [T,5120]**를 갖는 상태에서 Column Parallel QKV를 실행할 수 있다.

```math
W_{Q,r}:[5120,1536],\;W_{K,r}:[5120,256],\;W_{V,r}:[5120,256]
```

각 Rank에서 해당 Q Head 6개를 자신이 가진 KV Head 1개에 대응시켜 계산한다. 따라서 Head 병렬 경로에서는 **Attention 내부 QK^T/Softmax/PV 자체에 TP Rank 사이의 All-Reduce가 기본적으로 필요하지 않다.**

Attention 결과의 마지막 Output Projection은 **Row Parallel**로 분할한다.

```math
O_r:[T,1536],\quad W_{O,r}:[1536,5120],\quad
Y=\sum_{r=0}^{3}O_rW_{O,r}
```

이 부분 결과 Y의 합산에 All-Reduce가 들어갈 수 있다.

**Qwen 특이사항:** Gated Full Attention의 Q Output Gate와 Output Projection 등 추가 Projection/Elementwise를 포함한다. 위는 일반 Head Sharding 원리 설명용이며, vLLM 특정 버전의 실제 Kernel Dispatch·Fusion·GDN State Sharding을 그대로 보장하지 않는다.

### 7.2 이 모델의 TP 크기와 Head 개수의 제약

| TP | Q Head/Rank | KV Head/Rank | KV 특징 |
|---:|---:|---:|---|
| 1 | 24 | 4 | 전체 |
| 2 | 12 | 2 | Head 분할 |
| 4 | 6 | 1 | Head 분할 |
| 8 | 3 | 1 (예) | **KV Head가 Rank별 복제** |

TP8에서는 Q=24가 8로 나뉘지만 KV=4는 8로 나눌 수 없다. vLLM의 QKVParallelLinear는 **KV Head가 Q Head보다 적을 때 필요하면 KV Head를 복제**하는 경로를 가진다. 여기서는 **각 원래 KV Head를 두 Rank가 가진다**는 단순화한 사례를 사용했다.

주의: TP8의 KV Head 복제는 **TP Group 전체에 모든 KV가 전부 복제**된다는 뜻이 아니다. 예에서는 원래 각 Head가 두 Rank에만 중복 존재한다.

실제 모델 전체의 지원 TP Size는 GDN Head/State 배치, 사용 Backend, dtype 및 구현 제약까지 확인해야 한다.

---

## 8. KV Cache는 TP에서 어떻게 나뉘는가?

### 8.1 각 GPU가 갖는 것은 Context 일부가 아니라 담당 Head의 모든 Context

```text
TP=4, Qwen GQA, 한 요청의 T개 Token

GPU0 HBM : K0[1..T] + V0[1..T]  (Head 0)
GPU1 HBM : K1[1..T] + V1[1..T]  (Head 1)
GPU2 HBM : K2[1..T] + V2[1..T]  (Head 2)
GPU3 HBM : K3[1..T] + V3[1..T]  (Head 3)

각 GPU는 동일 요청의 전체 길이 T를 보지만,
Head 축의 서로 다른 K/V만 저장한다.
```

GPU0이 Token 1..T/4, GPU1이 다음 구간을 맡는 것은 일반 TP의 기본 동작이 아니다. 이는 CP 등 Sequence Sharding 계열의 문제다.

### 8.2 Logical Raw KV와 Physical Aggregate KV의 차이

한 요청 T=172000, BF16, Full Attention 16개 층만 세면:

| TP | GPU당 KV KiB/token | GPU당 KV GiB/request | 전체 GPU KV GiB/request |
|---:|---:|---:|---:|
| 1 | 64 | 10.498 | 10.498 |
| 2 | 32 | 5.249 | 10.498 |
| 4 | 16 | 2.625 | 10.498 |
| 8 (KV 2중 복제 사례) | 16 | 2.625 | **20.996** |

복제 유무는 **동일한 모델의 계산 결과와는 별개로 실제 HBM 점유량을 바꾼다.** KV payload 이외에 Paged Block Allocation, fragmentation, Prefix Cache, GDN State, CUDA Graph/Activation/Workspace 등이 별도로 들어간다.

참고: TP를 키워서 전체 Weight가 반으로 줄어드는 이상적인 상황에서도 **KV Head 복제 때문에 KV가 같은 비율로 줄지 않을 수 있다**. 반대로 Prefix Shared Blocks가 있으면 요청별 KV 논리값 합과 실제 물리 할당량이 다를 수 있다.

---

## 9. All-Reduce는 언제, 얼마나 통신하는가?

### 9.1 All-Reduce와 All-Gather를 구분

- All-Reduce(sum): 여러 GPU가 **같은 shape의 부분 합**을 내고, **모든 GPU가 합계**를 받음.
- All-Gather: 서로 다른 데이터 조각을 **연결**해 모든 GPU가 전체를 받음.
- Reduce-Scatter: 부분 합을 수행하되 합산 결과의 **각 조각만 각 Rank가 보유**.
- 단일 Layer 출력 결합은 배치에 따라 All-Reduce 하나일 수도, Reduce-Scatter/All-Gather 조합일 수도 있다.
- Row Parallel 뒤 All-Reduce, Column Parallel 직후 통신 생략은 흔한 Megatron-style TP 설계다. 임의의 모든 모델·버전에서 “Layer마다 정확히 2회”로 외우지 않는다.

### 9.2 통신 페이로드는 Weight 전체가 아니다

TP=4, BF16, Row Parallel 최종 Hidden Tensor Shape가 [N,5120]이면 Collective 입력의 크기는:

```math
S=N\times5120\times2\;\mathrm{Byte}
```

예를 들어:
- Decode 활성 요청 N=8: **S=81,920 Byte=80 KiB**.
- Prefill Chunk N=8192: **S=83,886,080 Byte=80 MiB**.

전송하는 대상은 이 시점에서 합산할 **부분 출력 Activation**이다. Weight Shard 전체나 KV Cache 전체를 매 Layer All-Reduce하는 것이 아니다.

### 9.3 Ring All-Reduce의 통신량을 직접 유도

P개 Rank, 각 Rank의 입력 Tensor 크기 S Byte. 이상적인 Ring 알고리즘은 Reduce-Scatter P−1 단계 + All-Gather P−1 단계를 수행하며, 단계마다 약 S/P Byte를 한 방향으로 주고받는다.

```math
\boxed{V_{\mathrm{send/rank}}=2\frac{P-1}{P}S}
```

수신량도 같다면 **각 Rank의 송·수신 합계는 2Vsend**다. 이것을 단순히 “Network 총 전송량”이라고 부르면 방향과 Rank의 합산 기준이 모호해진다.

TP=4에서는 2×(3/4)=1.5배이므로:

| N | Collective 입력 S | Rank당 송신 | Rank당 수신 | Rank당 송+수 |
|---:|---:|---:|---:|---:|
| 8 | 80 KiB | **120 KiB** | **120 KiB** | 240 KiB |
| 8192 | 80 MiB | **120 MiB** | **120 MiB** | 240 MiB |

동일 메시지의 Collective가 L회면 총 전송량 근사는 Rank당 송신 L×Vsend. **실제 횟수 L은 구현과 Layer 구성에 의해 결정**되며, Qwen GDN까지 포함한 고정 수치로 단정하지 않는다.

### 9.4 시간 모델: 작은 메시지는 latency, 큰 메시지는 bandwidth

이상적인 Ring에 대해:

```math
t_{\mathrm{ring}}\approx2(P-1)\alpha+
\frac{2(P-1)}{P}\frac{S}{B_{\mathrm{eff}}}
```

- α: 한 통신 단계의 시작/동기화에 해당하는 latency (단순 모델).
- B_eff: **실제 Collective 데이터 경로에서 해당 방향으로 달성한 대역폭**.
- 이는 계산 결과를 예측하는 간단한 모형. NCCL은 Topology와 메시지 크기에 따라 Ring 이외의 경로와 여러 채널을 사용할 수 있다.

Decode의 [8,5120] 메시지는 크기가 작아도 64개 층에서 반복되는 동기화가 중요할 수 있다. 반면 Prefill 큰 Chunk는 실제 전송 Byte/BW 부담이 커질 수 있다. TP가 증가하면 **GPU당 GEMM 작업이 작아져 효율이 떨어질 수 있음**도 고려해야 한다.

---

## 10. H200 스펙으로 계산의 경계를 해석하기

| Hardware Constraint | H200 SXM 공식 사양·범위 | 배포 해석 |
|---|---|---|
| Compute-bound | FP32 67 TFLOP/s, BF16 Tensor Core **약 990 TFLOP/s Dense** / 1,979 TFLOP/s Sparsity | GEMM 등 해당 Precision·Kernel의 이론적 연산 상한 |
| Memory-capacity-limited | 141 GB HBM3e / GPU | Weight Shard + KV Shard + GDN State + Buffer가 수용되는가 |
| Memory-bandwidth-bound | 4.8 TB/s HBM / GPU | Decode Weight/KV 조회 및 Kernel 데이터 공급 |
| Communication-limited | NVLink 900 GB/s (GPU당 양방향 합산 사양), PCIe Gen5 128 GB/s 양방향 합산 | All-Reduce/All-Gather의 실제 링크 경로와 실효 대역폭 |
| 노드 간 통신 | **GPU 스펙표만으로 알 수 없음** | NIC 수·IB/RoCE Link Speed·Topology·NCCL 설정 별도 확인 |

**중요한 정정:** NVIDIA H200 표의 BF16 1,979 TFLOP/s는 **With Sparsity** 조건이다. 일반적인 Dense BF16의 이론 성능 기준은 약 990 TFLOP/s다. 또한 NVLink 900 GB/s는 한 방향 900 GB/s라는 뜻이 아니라 GPU의 총 양방향 사양이다. NIC 400 Gb/s는 400 GB/s가 아니라 명목 **50 GB/s**다.

### 10.1 시간을 계산할 때 세 개의 서로 다른 하한

```math
t_{\mathrm{compute}}\gtrsim\frac{F}{P_{\mathrm{eff}}},\quad
t_{\mathrm{HBM}}\gtrsim\frac{D_{\mathrm{HBM}}}{B_{\mathrm{HBM,eff}}},\quad
t_{\mathrm{comm}}\gtrsim\text{latency}+\frac{D_{\mathrm{link}}}{B_{\mathrm{link,eff}}}
```

- 한 Kernel의 연산과 데이터 이동이 겹칠 수 있으면 단순화된 **Roofline 하한은 두 항의 최댓값**에 가깝다.
- 여러 Kernel 사이 순서 의존성과 Collective는 전체 Critical Path를 만든다.
- Peak FLOP/s로 실제 tok/s를 확정할 수는 없다.
- GPU Util 100%가 Tensor Core Peak 활용률 100%라는 뜻도 아니다.

### 10.2 Arithmetic Intensity를 GEMM에서 유도

```math
I=\frac{\mathrm{FLOP}}{\mathrm{HBM\ transferred\ Byte}}
```

기본 행렬 곱 X:[N,d], W:[d,m], Y:[N,m], 모두 b Byte/element이고 각 원소가 HBM에서 이상적으로 **단 한 번만** 읽히거나 써진다고 하면:

```math
F=2Ndm,\quad D_{\min}=b(Nd+dm+Nm),\quad
I_{\mathrm{ideal}}=\frac{2Ndm}{b(Nd+dm+Nm)}
```

- N=1일 때 큰 Weight를 읽어 한 Token에만 쓰면 낮은 FLOP/Byte.
- N이 크면 같은 Weight Tile을 여러 Token에서 재사용할 가능성이 커져 FLOP/Byte 증가.
- 실제 Kernel IO는 tiling/cache/fusion/quantization에 따라 달라진다.

### 10.3 Capacity만 맞으면 끝인가?

아니다. 예를 들어 27B 파라미터 × BF16 2 Byte ≈ **54 GB**는 Weight Raw 개략값이지만, 실제 Checkpoint 구조와 비텍스트 컴포넌트, 추가 Tensor, dtype, Padding을 확인해야 한다. 여기에 요청별 KV와 Recurrent State 및 Workspace가 추가된다. TP로 Weight Shard가 작아도 **통신이 느리면 TTFT/ITL 목표를 못 맞출 수 있다**.

---

## 11. 분산 구조의 기본 검증 절차

다음 절차는 실제 운영 환경에서 확인할 수 있는 **검증 계획**이다. 여기서 계산한 것은 이론치이며 실제 Benchmark를 수행하지 않았다.

1. Checkpoint revision·config·dtype·실행 엔진/Backend·TP Size를 고정한다.
2. Rank별 Q/K/V/FFN Weight Shape와 KV Head 수, KV 복제 여부를 **실제 로딩된 Tensor/Log**로 확인한다.
3. 각 Rank의 Cache Block 개수, Head 수, dtype, 전체 VRAM 및 CUDA Graph/Workspace를 기록한다.
4. 고정 Prompt/Output/동시성에서 TP1, TP2, TP4 등을 비교한다. GPU 총 개수가 다르면 자원량 차이임을 함께 기록한다.
5. Nsight/NCCL Trace로 Row Parallel 뒤 Collective의 실제 횟수, 메시지 Byte, latency와 겹침 여부를 측정한다.
6. TTFT, ITL/TPOT, Request Throughput 및 Queueing을 함께 기록한다.
7. TP8 또는 Hybrid GDN은 **구현 지원/State 분할·복제**를 먼저 검증한다. 본문의 단순 Head Shard 예만으로 지원을 선언하지 않는다.

### 계산 연습

- [ ] A:[3,4], B:[4,5]이면 출력 Shape와 GEMM FLOPs 근사는?
- [ ] FFN d=5120, m=17408, N=4에서 Gate/Up/Down 각각의 결과 Shape는?
- [ ] TP2 Column Parallel 예제에서 Y0와 Y1은 합산인가, 연결인가?
- [ ] TP2 Row Parallel 예제에서 부분 합 2개를 왜 같은 위치끼리 더하는가?
- [ ] Q Head24, KV Head4, TP4이면 Rank당 Head는 몇 개인가?
- [ ] TP8에서 KV Head 복제가 발생하면 전체 KV Payload가 TP4보다 커질 수 있는가?
- [ ] [8,5120] BF16 출력의 Ring TP4 All-Reduce에서 Rank당 송신 KiB는?
- [ ] Attention QK/AV의 O(T²)와 FFN의 O(T)를 구분해 170K의 비용 구성을 설명할 수 있는가?
- [ ] H200의 BF16 1,979 TFLOPS를 일반 Dense 모델의 실효 계산 성능으로 그대로 적용하면 왜 잘못인가?

**정답 핵심:** [3,5], 약 120 FLOP; 각 Gate/Up [4,17408], Down [4,5120]; Column은 연결, Row는 합; 6Q/1KV; TP8 예제에서는 중복 발생; 120 KiB/rank; Head와 State가 다르게 분할되므로 단일 “1/TP” 규칙은 없다.

---

## 12. 발표에 적용하는 순서와 후속 학습

이 내용의 **발표용 1회차**는 [슬라이드·발표자 흐름](presentation-session-01-hardware-attention-kv-cache.md)에서 별도로 관리한다. 이번 발표는 “**연산량·데이터 이동량이 왜 나오는가, TP가 그 둘을 어떻게 교환하는가**”까지 완결한다.

**후속 주제:** Gated DeltaNet의 recurrence / state update / Conv State / Chunked Scan, P/D Disaggregation, FlashAttention/PagedAttention/CUDA Graph 및 CP 상세는 **이번 회차의 필수 범위에서 제외**한다. 도입의 Qwen 전체 64층 구조는 소개하되, 계산의 일관성을 위해 **Full Attention + 모든 Dense FFN + 명시된 일부 GDN Projection**만 정량화하고 생략 범위를 표기한다.

## 참고·근거

1. PR #27 종합 노트: [LLM-serving-hardware-attention-master-notes.md](LLM-serving-hardware-attention-master-notes.md), 2·4~7·12~15장 (Qwen Config 및 170K 부분 FLOP 회계).
2. 초심자용 수학·Attention 원고: [00a-matrix-gemm-to-qkv-from-zero.md](00a-matrix-gemm-to-qkv-from-zero.md).
3. vLLM 공식 Linear API: https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/linear/ (ColumnParallelLinear, RowParallelLinear, QKVParallelLinear; Row Parallel의 reduce_results 및 KV Head replication 가능성).
4. NVIDIA Megatron/NeMo TP 개요: https://docs.nvidia.com/nemo-framework/user-guide/25.04/nemotoolkit/features/parallelisms.html .
5. NVIDIA Megatron-LM Tensor Model Parallelism의 대표 원리: https://arxiv.org/abs/1909.08053 .
6. NVIDIA H200 제품·표의 Sparsity 단서: https://www.nvidia.com/ko-kr/data-center/h200/ .
7. Lenovo H200 Precision별 Dense/Sparsity 분리 표: https://lenovopress.lenovo.com/lp1944-nvidia-h200-141gb-gpu .
8. NVIDIA Context Parallelism: https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/context_parallel.md .

원문 자료, 공개 모델 설정, 이 문서의 산술적 유도, 실제 서빙 엔진 구현을 **같은 근거 수준으로 혼동하지 말 것**.