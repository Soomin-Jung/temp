# 발표 1회차 마스터 플로우 — Hardware / Attention / KV Cache

**발표 제목:** LLM Inference에서 발생하는 물리적 비용 이해하기  
**과정:** Hardware / Attention / KV Cache (1/3)  
**시간 목표:** 설명 약 74분, Q&A 약 15분 (총 89분), 축약 시 약 50분+10분  
**대상:** GPU Serving을 운영하지만 GEMM·Attention·TP 행렬 분할을 수학적 원리까지 배우려는 팀원  
**핵심 교재:** [행렬 입문 00a](00a-matrix-gemm-to-qkv-from-zero.md) → [분산·비용 유도 00b](00b-tp-and-llm-inference-cost-from-first-principles.md) → [전체 참고 마스터](LLM-serving-hardware-attention-master-notes.md)

> **발표의 최종 목표:** 처음의 “Qwen3.8 27B를 배포해 주세요”라는 요청에 대해, Weight·FFN/Attention FLOP·KV·통신 메시지의 크기를 계산한 다음 H200의 FLOP/s·HBM GB·HBM GB/s·GPU Link GB/s와 연결할 수 있게 한다.

## 핵심 설계 결정

- **TP=1 설명을 먼저 완결**하고 TP=2/4/8은 뒷부분에 배치한다. 처음부터 분산을 말하면 행렬 하나의 의미가 흐려진다.
- **GEMM → Tensor Core → QKV 및 FFN → Attention → KV Cache → Prefill/Decode → TP/Collective → Hardware Bound** 순서.
- Full Attention과 **Dense FFN을 동등하게 취급**한다. FFN이 모든 64층에서 수행된다는 점은 반드시 명시한다.
- **Qwen 아키텍처는 처음에 잠깐 소개하고 끝에 같은 그림과 숫자로 돌아온다.** Qwen Hybrid의 GDN을 깊게 가르치는 회차는 아니다.
- 한 장에는 장문의 정의를 넣지 않는다. **수식 1~2개, 그림 하나, 강조 문구 하나**를 원칙으로 한다.
- 각 장의 오른쪽에는 선택적으로 **“그림에서 무엇이 나뉘는가/합쳐지는가” 화살표**를 그린다.
- Peak FLOPS와 FLOPs, Link Bandwidth와 실제 Collective Throughput, Raw KV와 실제 Allocated HBM을 구분한다.

---

## A. Introduction + 하드웨어 지도 (슬라이드 1~4, 약 10분)

### 슬라이드 1 (2분) — 고객사의 한마디

**화면:** 상단 “고객사: Qwen3 27B 배포해주세요. 모델 좋다더라..”. 아래에 **고객 요구 → TTFT/ITL/Concurrency/Context 질문** Funnel과 **GPU·HBM·Network의 유한 자원** 두 영역.

**발표자가 풀 말:** 배포는 모델 실행 명령이 아니라 우선순위와 제약을 맞추는 엔지니어링. 목표 성능과 Workload에 따라 적합한 배포 구성이 달라진다.

**전환:** 그래서 모델을 먼저 눈에 익히되, 계산은 밑바닥부터 진행한다.

### 슬라이드 2 (2분) — Qwen3.8-27B 아키텍처 앵커

**화면:** 지금 제작한 아키텍처 도식 + 최소 Config. 64 Layers / 48 GDN + 16 Full Attention / hidden_size 5120 / 24 Q / 4 KV / head_dim 256 / intermediate_size 17408 / 262144 max position.

**발표자가 풀 말:** “이 수치들은 오늘 계산할 때 다시 등장한다. 현재는 세부를 설명하지 않는다.” 64층 모두 FFN이 있음을 간단히 표시한다.

**전환:** 그 숫자들이 어떤 하드웨어 자원을 점유하는지 보자.

### 슬라이드 3 (3분) — GPU 서버를 이루는 자원

**화면:** 이미 수집한 HGX 8-GPU 서버 그림과 GPU SM/L2/HBM 그림 + 네 물리적 제약. Compute / HBM Capacity / HBM Bandwidth / Communication Bandwidth.

**발표자가 풀 말:** GPU의 연산량과 저장 용량은 다른 지표이며, GPU 내부 HBM과 GPU↔GPU NVLink/PCIe, 노드↔노드 IB를 구분한다. HBM Capacity는 엄밀히 Bandwidth-type bound와 다른 **수용 가능성 제약**이다.

**전환:** 제품 스펙에서 각 제약의 수치가 무엇인지 찾아본다.

### 슬라이드 4 (3분) — H200 사양 표: 네 자원에 형광펜

**화면:** H200 SXM 1 GPU 기준 FP32 67 TFLOPS, BF16 Dense 약 990 TFLOPS (Sparsity 1,979), 141GB HBM3e, 4.8 TB/s HBM, NVLink 900 GB/s *양방향 합산*을 강조.

**발표자가 풀 말:** 이론 성능은 실측값이 아니다. 모델, dtype, topology, Workload의 실제 효율을 모르면 초당 Token 수를 확정할 수 없다. 노드 간 IB 대역폭은 GPU 카드 스펙표에 없음.

**전환:** FLOP이 무엇인지 알아야 Compute 수치가 의미를 갖는다.

---

## B. GEMM 기초에서 실제 LLM 연산으로 (슬라이드 5~10, 약 26분)

### 슬라이드 5 (4분) — 벡터 내적과 행렬 한 칸의 계산

**화면:** A:[2,3] × B:[3,2] → C:[2,2]에서 **C(0,0)=1×7+2×9+3×11=58**을 행/열 색깔로 연결. 다른 결과 64,139,154도 순서대로 등장.

**발표자가 풀 말:** 왼쪽의 **행**과 오른쪽의 **열**을 고른 다음 같은 위치 원소를 곱해 더하면 출력의 한 칸. “M×K와 K×N을 곱하면 M×N”의 이유를 직접 계산으로 설명한다.

**전환:** 원소가 많아지면 연산 횟수를 어떤 공식으로 세는가?

### 슬라이드 6 (5분) — FMA → FLOP → GEMM 연산량

**화면:** c=a×b+c (FMA), 관례상 2 FLOP. “각 출력 칸 K회 누산 × 출력 칸 MN개”라는 화살표와 **F≈2MNK**. 2×3·3×2 예제의 20 scalar op vs 24 FLOP 근사 차이를 작은 주석으로.

**발표자가 풀 말:** 연산 명령 수와 FLOP은 같지 않다. FLOP과 FLOP/s의 단위도 다르다. 연산량과 시간의 구분을 강조한다.

**전환:** 이런 행렬 블록 연산을 높은 처리량으로 실행하는 물리 자원이 Tensor Core다.

### 슬라이드 7 (3분) — CUDA Core와 Tensor Core

**화면:** CUDA Core 일반 산술, Tensor Core 타일 MMA **D=AB+C** 비교. SM 내부 위치는 공식 GPU 그림에서 하이라이트. “특화”이지 “모든 연산을 수행”이 아님.

**발표자가 풀 말:** GEMM Shape, dtype, kernel의 선택에 따라 Tensor Core 이용 여부·효율이 달라진다. Softmax·Norm·Activation은 모두 GEMM이라고 부르지 않는다.

**전환:** LLM에서 이 GEMM은 어디서 반복될까?

### 슬라이드 8 (4분) — Transformer Layer의 두 큰 연산: Mixing과 FFN

**화면:** Hidden State → Norm → QKV/Attention(또는 GDN) → Residual → Norm → Gate/Up/Down FFN → Residual. **Attention과 FFN 박스를 동등한 크기**로. “Token 간 정보 교환 vs 각 Token Feature 변환”.

**발표자가 풀 말:** FFN은 Attention의 부속이 아니라 거대한 행렬 연산의 독립 축. Qwen에서는 64층 모두 FFN 실행. Attention도 Projection을 여러 개 수행한다.

**전환:** 우선 FFN에서 GEMM이 정확히 몇 번 나오는지 계산해 보자.

### 슬라이드 9 (5분) — FFN: Gate/Up/Down의 3 GEMM

**화면:** [T,5120]×[5120,17408] 두 갈래(Gate, Up) → SiLU× → [T,17408]×[17408,5120]. 핵심 수식 **F_FFN≈6Tdm**.

**발표자가 풀 말:** Input Token 수 T가 늘면 FFN의 행 수도 늘어 **연산량이 T에 선형 비례**한다. SwiGLU의 Gate/Up/Down 사이 원소별 연산은 별도. “Attention만 Compute를 먹는다”는 오해 제거.

**전환:** 이제 Attention의 Projection도 정확히 같은 행렬곱이다.

### 슬라이드 10 (5분) — QKV Projection에서 Attention 본체까지

**화면:** XWQ, XWK, XWV → QKᵀ → Scale/Mask/Softmax → PV → WO. Qwen GQA 24 Q / 4 KV 숫자는 오른쪽 표. 핵심 식은 **S=QKᵀ/√d+M**, **O=Softmax(S)V**.

**발표자가 풀 말:** 행렬곱은 QKV 생성, Score 곱, Value 가중합, Output에 있다. Self-Attention은 **같은 Input Representation에서 Q/K/V를 만든다**는 뜻. Softmax 식 자체는 MQA가 아니라 모든 MHA/MQA/GQA의 공통 원리.

**전환:** 현재 Query가 과거 Key와 비교한다면, 과거 Key를 어디에 두어야 할까?

---

## C. KV Cache, Prefill/Decode와 병목 (슬라이드 11~14, 약 19분)

### 슬라이드 11 (5분) — MHA / MQA / GQA: 누가 KV를 공유하나?

**화면:** Q24/KV24, Q24/KV1, Q24/KV4를 색으로 매핑한 **세 구조**. 동일한 Attention 식. MHA/MQA는 실제 Qwen 아키텍처가 아니라 비교용 가정이라는 표기.

**발표자가 풀 말:** Query는 그대로 많지만 KV Head를 공유할수록 Token당 저장/읽기 비용이 감소한다. GQA가 QK/AV의 Query Head 연산을 정확히 6분의 1로 줄인다는 뜻은 아니다.

**전환:** 그렇다면 과거 K/V를 매 토큰마다 다시 계산하는 대신 저장할 수 있다.

### 슬라이드 12 (5분) — KV Cache는 왜 생기는가?

**화면:** Without Cache → 과거 Prefix 재계산, With Cache → 지난 K/V 보관 + 새 K/V 추가 + 현재 Q가 과거 Cache 조회. HBM 화살표와 “Recompute ↓, Memory ↑”.

**발표자가 풀 말:** Causal 모델의 과거 Token 표현은 미래 Token이 추가되어도 같은 Prefix에 대한 값으로 재사용 가능하다. Cache는 완성 문장/정답 저장소가 아니라 **Layer별 중간 K/V 상태**다. Cache가 있어도 각 새 Query의 긴 History Read는 남는다.

**전환:** Token 하나를 얼마나 저장하는지 이제 계산할 수 있다.

### 슬라이드 13 (4분) — KV Cache Byte를 한 칸씩 유도

**화면:** **2(K,V)×L×Hkv×head_dim×dtype_B**. Qwen 16×4×256×2×2 = **64 KiB/token** → 172K Token ≈ **10.498 GiB/request**. 물리적 Allocator 값과 구분 주석.

**발표자가 풀 말:** Weight는 요청 간 공유하지만 Cache는 요청 이력마다 존재한다. Context와 동시성이 커질수록 HBM Capacity가 압박된다. GDN 48층의 상태는 **이 식에 포함하지 않으며** 나중에 따로 다루겠다.

**전환:** 이 저장 비용과 계산량이 Prefill과 Decode에서 다르게 나타난다.

### 슬라이드 14 (5분) — Prefill/Decode를 다시 보는 눈

**화면:** Prefill X:[T,d] 거대한 GEMM 및 Attention causal token pair, Decode X:[B,d] skinny GEMM + Cache read T. **FFN O(T), Full Attention Prefill O(T²), Decode 새 Token Attention O(T)**.

**발표자가 풀 말:** Prefill은 전체 Prompt의 많은 위치를 계산한다. 첫 생성 Token은 마지막 Prompt 위치의 Logits에서 선택할 수 있다. 이후 Decode는 생성한 Token을 다음 입력으로 사용한다. “Prefill은 항상 Compute-bound / Decode는 무조건 Memory-bound”라고 단정하지 않고 Batch, Length, Kernel, Link에 따른 경향을 설명한다.

**전환:** 지금까진 GPU 한 장의 모델이다. GPU 수를 늘리면 계산 비용은 어떻게 이동할까?

---

## D. TP=N에서 연산·상태·통신을 세기 (슬라이드 15~18, 약 19분)

### 슬라이드 15 (4분) — TP와 PP가 자르는 축

**화면:** 같은 64층을 여러 GPU에 분배하는 PP 그림과, 같은 Layer의 Feature/Head를 분할하는 TP 그림. 기본값 **TP=1 → TP=4**.

**발표자가 풀 말:** PP는 Layer 구간, TP는 **한 Layer 안의 Weight Feature/Attention Head 축**을 분할한다. CP처럼 Token 시간축을 나누는 것은 일반적인 TP와 별개다. 전체 모델이 무조건 1/TP Byte로 정확히 나뉘는 것은 아니다.

**전환:** FFN의 세 행렬로 분할 결과의 원리를 보여준다.

### 슬라이드 16 (5분) — FFN TP: Column/Row Parallel과 All-Reduce

**화면:** 숫자 예제 **X=[2,3]**, Column 두 조각의 출력 **[17,22] | [27,32]**, Row 부분 출력 **[17,22] + [91,59] = [108,81]**. 이어 Qwen FFN 17408을 TP4 각 4352 Feature로 나눈 그림.

**발표자가 풀 말:** Column Parallel은 **서로 다른 출력 Feature**이므로 즉시 합산하지 않는다. Row Parallel의 **부분 합을 더해야 완전한 Output**이므로 All-Reduce가 필요하다. Rank당 FFN 주요 계산은 균등 분할에서 약 1/TP로 감소할 수 있지만 통신/Launch/작은 GEMM 효율은 남는다.

**전환:** Attention도 QKV와 Output Projection의 Head·Feature 축을 나눈다.

### 슬라이드 17 (5분) — Attention TP와 KV Head Sharding / Replication

**화면:** Q24/KV4 → TP1(24/4), TP2(12/2), TP4(6/1), TP8(3/1, KV 복제). GPU0~3이 **서로 다른 KV Head의 1..T Context**를 갖는 그림. Rank별 KV 10.498 / 5.249 / 2.625 / 2.625 GiB, TP8 전체는 중복으로 20.996 GiB.

**발표자가 풀 말:** QKV 가중치도 Head 기준으로 나뉜다. 각 Rank가 자기 Head의 Attention을 수행하므로 **QKV Projection 직후 반드시 All-Reduce하는 것은 아니다**. Output Projection의 부분 출력 합산에 Collective가 생긴다. TP8의 Head 복제는 Backend 가정이며 GDN State까지 같은 방식으로 나뉜다는 뜻이 아니다.

**전환:** 남은 비용은 실제로 얼마를 보내는가이다.

### 슬라이드 18 (5분) — Collective Byte 계산 → H200 Bound → 처음 질문 회수

**화면:** **S=N×5120×2 Byte**, TP4 Ring **Rank send=2×(3/4)×S**. Decode N8 → S=80 KiB, Rank send 120 KiB, Recv 120 KiB. Prefill Chunk N8192 → S=80 MiB, Rank send 120 MiB. 오른쪽 H200 990 TFLOP/s Dense, HBM 141GB/4.8TB/s, NVLink 900GB/s bi-dir, NIC 별도. 초기 Qwen 모델 Config 작은 재등장.

**발표자가 풀 말:** All-Reduce는 모델 Weight나 전체 KV Cache를 매번 복사하는 것이 아니라 **합산할 Hidden Activation**을 교환한다. 실제 Kernel과 Collective 성능은 Peak보다 낮을 수 있으므로 TTFT/ITL와 함께 Benchmark로 확인한다. 처음 고객 질문에 “모델 Weight가 들어가니 가능” 대신 **FLOP + Raw Cache + Collective + 실제 측정**으로 답한다.

**마무리:** “같은 GPU라도 연산, 저장, 읽기, 교환 중 어느 비용이 지배하는지는 Workload와 배포 구성에 따라 다르다.”

---

## 발표 시간 조절

**기본:** 슬라이드 1~18 = 약 **74분**, Q&A 15분 = **89분**. 충분한 중간 질의가 예상되면 5~10분 줄인다.

**60분 축약:** 슬라이드 5·6의 직접 숫자 연습을 3분 줄이고, 8·9를 압축(4분), 11의 MHA/MQA 비교 단축(2분), 15의 PP를 1장 안에서만 짚기(2분), 17의 TP8 복제는 후속 문서로(3분), 18의 H200 표를 회수로만(5분) 운영해 발표 50분 전후 + Q&A 10분으로 조정한다. 축약 중에도 **KV Byte / FFN FLOP / Ring Message Byte** 3개 산술식은 남긴다.

## 시각 자료 작업 원칙

- 수식은 **행렬 원소 색 연결 → Shape → FLOP → HBM/통신 Byte** 순서로 확대한다.
- **단위와 Scope를 매번 기입:** “Token 하나 / Layer 하나 / GPU Rank 하나 / 요청 하나 / 전체 Rank”.
- 각 그래프에 **이론 산술량인지, 대역폭 이론 Peak인지, 실제 측정치인지**를 표기한다.
- 원본 Qwen 도식에 존재하는 Gated DeltaNet 부분은 숨기지 않는다. 다만 **상태 수식은 이번 회차에 깊게 다루지 않는다**는 의도적 범위 제한을 밝힌다.
- 기존 완성된 1~4번 슬라이드는 최대한 유지하고, 5번 이후를 이 원고대로 구성한다.
- 다음 회차 후보: GDN의 Recurrent State, Full Attention과의 State 비교, P/D 분리, Flash/PagedAttention, CUDA Graph 및 통신 최적화.

## 이번 회차를 듣고 직접 답해야 하는 질문

1. A:[2,3]×B:[3,2]의 결과가 왜 [2,2]이며 C(0,0)는 어떻게 구하는가?
2. GEMM FLOPs를 2MNK로 근사하는 이유와 FMA 한 명령의 계산 의미는?
3. LLM의 FFN Gate/Up/Down은 Attention 밖에서 어떤 GEMM을 수행하는가?
4. Qwen3.8-27B의 Q Head 24와 KV Head 4는 다른 Weight Shape를 어떻게 만드는가?
5. 16 Full Attention Layer가 172K BF16 Token의 Cache로 10.498 GiB를 요구하는 이유는?
6. Prefill의 큰 FFN GEMM과 Full Attention O(T²)를 왜 동시에 세어야 하는가?
7. TP4에서 FFN의 17,408 Feature와 Q/KV Head는 각각 어떻게 나뉘는가?
8. TP8에서 KV Head 복제가 생기는 대표적인 조건은?
9. Row Parallel 후 All-Reduce를 하는 수학적 이유와 80 KiB Input의 Ring Rank당 송신량은?
10. 최종 FLOP/s·GB·GB/s·Link GB/s를 실제 TTFT·ITL과 왜 일대일 대응시킬 수 없는가?
