# LLM 추론의 하드웨어 비용과 Attention·KV 메모리

## 기술교류회 학습용 마스터 노트

**기준일: 2026-11-02 · 대상: LLM 배포·서빙 엔지니어**

> 고객이 “Qwen 27B를 배포해 달라”고 요청하면, 배포 엔지니어는 먼저 속도·동시성·입력 길이·출력 길이·지연 목표를 물어야 한다. GPU 자원은 유한하고, 하나의 구성으로 낮은 지연·높은 처리량·긴 문맥·낮은 비용을 모두 극대화할 수는 없다. 이 교재는 그 판단에 필요한 연산, 메모리, 통신의 근거를 만든다.

## 문서 사용법과 증거 구분

이 자료는 슬라이드용 요약이 아니라 논문·수식·모델 설정·서빙 판단을 연결한 학습 원고다. 본문 인용표시 `[R#]`는 끝의 참고자료로 연결된다. **[공식]**은 공급자 공식 보고서·모델 카드·설정, **[논문]**은 원 논문, **[구현]**은 공식 프레임워크 문서·코드, **[유도]**는 제시한 가정에서 식으로 계산한 값, **[설명 예시]**는 개념을 보이기 위한 가정이다. 논리적 tensor 크기와 실제 allocator의 물리 할당량을 구분한다. 제품·모델 구조는 자료가 확인된 시점과 checkpoint 범위를 함께 적는다. 공개되지 않은 폐쇄 모델의 내부 구조는 추측하지 않는다.

함께 보는 자료: 저장소의 `study/attention/00-foundations.md`부터 `90-serving-engineer-view.md`까지는 수학과 attention 계보별 심화 노트이고, `models/kimi-k3`, `models/deepseek-v4`, `models/glm-5.2`는 모델별 조사 노트다. 그 자료들은 탐색 지도와 맥락을 제공하며, 이 문서에서 쓰는 release 이름·구성 수치·공식 claim은 현재 접근 가능한 원 논문·공식 model card/config와 다시 대조했다. 문서의 한국어 본문 글꼴은 Word에서 **맑은 고딕(Malgun Gothic)**으로 표시되도록 지정한다.

자료 업데이트 기준은 2026-11-02다. 웹에 보이는 최신 이름만으로 구조를 추론하지 않고, 가능한 경우 공식 config / model card를 우선한다. 특히 “Qwen3-27B”는 널리 알려진 Qwen3 dense 모델명과 혼동될 수 있다. 이 문서의 48 Gated DeltaNet + 16 Gated Full Attention 예시는 공식 공개 자료에서 확인된 **Qwen3.5-27B** 구성으로 다룬다. **Qwen3.8-27B라는 이름은 이 기준일에 공식 자료로 확인하지 못했다.** [R12]

## 목차

1. 발표의 문제 설정과 한 페이지 요약  
2. 선수 개념: FLOP, FLOPS, 대역폭, 용량, latency  
3. CPU·GPU·GEMM·projection  
4. GPU 내부 구조와 데이터 이동  
5. 모델 실행 흐름: 레이어, dense, MoE, hybrid  
6. Prefill과 autoregressive decode  
7. Attention 공통식과 KV cache 용량 유도  
8. MHA·MQA·GQA·MLA 비교  
9. SWA·CSA·HCA: 긴 문맥에서 저장과 선택  
10. Recurrent state와 convolution state  
11. 하이브리드 모델과 공개 모델 사례  
12. CUDA Graph의 역할과 한계  
13. 메모리·통신 용량 계산 사례  
14. 운영 의사결정 흐름과 점검표  
15. Attention·Serving 최적화 계보  
16. 참고: P/D disaggregation  
17. 발표자 자기 점검  
18. 주석이 포함된 참고문헌  
19. 용어집

---

# 1. 발표의 문제 설정과 한 페이지 요약

## 1.1 출발 질문

> 고객: “Qwen 27B 배포해주세요. 좋다고 하네요.”  
> 엔지니어: “속도가 우선인가요? 동시 처리량인가요? prompt가 얼마나 길고, 답변은 몇 토큰인가요? 목표 TTFT·TPOT와 GPU 자원은요?”

이 되물음은 요구를 귀찮게 하는 절차가 아니라 **하드웨어 배치를 정하는 입력값을 수집하는 일**이다. 같은 가중치라도 짧은 prompt·낮은 동시성에서는 작은 단일 GPU가 경제적일 수 있고, 장문 prompt·긴 출력·다수 동시성에서는 KV cache 용량과 HBM 대역폭이 먼저 한계가 된다. 모델이 한 장의 GPU에 안 들어가면 tensor parallelism(TP)이 필요할 수 있지만, GPU를 나누면 collective 통신 비용이 생긴다. GPU끼리 NVLink/NVSwitch가 있는지, 노드 사이에 고속 fabric이 있는지에 따라 최적 구성이 달라진다.

## 1.2 기억할 판단 프레임

| 질문 | 지배적인 자원 | 주로 볼 지표 | 배포상 결과 |
|---|---|---|---|
| 한 요청의 prompt를 얼마나 빨리 읽나? | 연산량, attention IO, batch | TTFT, prefill tokens/s, GPU util | 큰 GEMM/attention 효율과 prompt 길이 |
| 다음 출력 토큰을 얼마나 빨리 내나? | weight·KV HBM 읽기, decode batch, latency | TPOT/ITL, decode tokens/s | KV head 구조, KV dtype, HBM bandwidth |
| 동시 요청을 얼마나 수용하나? | VRAM 용량, KV cache pool, fragmentation | active sequences, KV blocks, queue | cache 크기와 allocator가 직접 제한 |
| TP를 늘리면 빨라지나? | shard별 계산 절감 대 all-reduce 비용 | step time, collective time | 연결 topology와 workload가 관건 |
| 장문 문맥이 가능하나? | KV/cache 상태의 길이 증가, prefill 비용 | cache bytes/request, prefill latency | GQA/MLA/SWA/압축/상태형 구조 |

핵심 메시지: **FLOPS는 계산 상한, HBM bandwidth는 데이터 공급 속도, VRAM은 동시 상주량, GPU 간 link는 분할 실행의 통신 속도, latency는 왕복·동기화 시간**이다. 이 지표는 서로 바꿔 쓸 수 없다. Prefill과 decode는 같은 모델을 실행해도 서로 다른 성능 구간을 본다. KV cache와 recurrent state는 매 요청이 점유하는 상태이므로, 모델 weight만 계산해서는 serving 동시성을 예측할 수 없다. [R1–R4]

---

# 2. 선수 개념: FLOP, FLOPS, 대역폭, 용량, latency

## 2.1 단위부터 구분하기

| 용어 | 뜻 | 단위·예 | 혼동하면 생기는 오류 |
|---|---|---|---|
| FLOP | floating-point operation 한 번. FMA를 곱셈+덧셈 2 FLOP으로 세는 관례가 흔함 | 연산 횟수 | `FLOP`을 초당 성능으로 잘못 읽음 |
| FLOPS | 초당 부동소수점 연산 횟수 | TFLOP/s, PFLOP/s | peak FLOPS를 실제 모델 처리량으로 간주 |
| 메모리 용량 | 동시에 저장 가능한 데이터 크기 | GB, GiB | 대역폭과 동일시 |
| 대역폭 | 시간당 이동 가능한 데이터 양 | GB/s, TB/s | 용량 크면 빠르다고 단정 |
| latency | 한 연산/전송을 시작하고 결과를 얻는 데 드는 시간 | ns, μs, ms | 대역폭만으로 짧은 메시지 시간을 예측 |
| 통신량 | GPU 사이에 옮기는 byte 합 | MB, GB | 단방향/양방향 수치를 혼합 |

1 GB는 십진 기준 10⁹ byte, 1 GiB는 2³⁰ byte다. 제품 자료의 GB와 시스템의 GiB를 섞으면 표시 용량이 약 7% 차이 난다. “900 GB/s NVLink” 같은 수치도 GPU-to-GPU 경로, 방향성, 집계 방식, SKU 및 시스템 구성 기준을 확인해야 한다. [R5]

## 2.2 Roofline 사고법

실제 성능은 연산 상한과 메모리 공급 상한 중 작은 쪽에 제한된다.

`성능 ≤ min(peak FLOP/s, 유효 대역폭[byte/s] × 산술 강도[FLOP/byte])`

산술 강도는 `수행 FLOP / HBM에서 이동한 byte`다. 계산을 한 번 더 쪼개 여러 번 HBM을 읽으면 산술 강도가 낮아져 memory-bound가 된다. 큰 GEMM은 입력 타일을 재사용해 산술 강도가 높은 편이라 compute-bound에 가까워질 수 있다. 단일 token decode의 weight 읽기와 과거 KV 읽기는 재사용이 제한되거나 batch 규모가 작아 대역폭·launch latency에 묶이기 쉽다. 이 설명은 전형적인 경향이지, 모델과 kernel 전체를 단정하는 법칙은 아니다. [R2]

### 예시

같은 1 TB 데이터라 해도 1 TB VRAM은 “얼마나 담나”를 말하고, 4 TB/s HBM은 이상적으로 “몇 초에 순차 이동하나”를 말한다. 대역폭만으로 실제 시간을 계산할 때는 접근 패턴, 캐시 적중, 동시 요청, kernel 효율, page/block layout, 동기화가 빠져 있으므로 계산은 하한에 불과하다.

---

# 3. CPU·GPU·GEMM·projection

## 3.1 CPU와 GPU의 역할

CPU는 복잡한 분기·작은 작업·낮은 지연 제어 흐름에 강하다. API 처리, tokenization, scheduler, 요청 큐, sampling 제어, CUDA kernel launch, 네트워크·스토리지 관리가 전형적인 CPU 측 작업이다. GPU는 대량의 독립 연산을 병렬로 수행하도록 설계되어 큰 행렬곱, attention, elementwise tensor 작업에 강하다. 실제 서빙은 CPU와 GPU 중 하나만 쓰는 구조가 아니라, CPU가 실행을 조정하고 GPU가 무거운 수치 연산을 처리하는 pipeline이다. [R3]

## 3.2 GEMM이란?

GEMM은 General Matrix-Matrix Multiplication, 즉 보통 `C = αAB + βC` 형태의 행렬곱 연산이다. LLM에서 dense layer의 핵심 계산은 입력 activation 행렬에 학습된 weight 행렬을 곱하는 것이다. batch 또는 sequence 길이가 커지면 여러 token의 입력을 행렬의 행으로 묶으므로 GPU에 큰 GEMM으로 전달할 수 있다.

가령 hidden size `d=5120`, batch 내 처리 위치 `N`인 입력 `X ∈ R^(N×d)`와 weight `W ∈ R^(d×d_out)`이면 `Y=XW`이고 대략 `2N d d_out` FLOP(FMA를 2개로 계산)다. 실제 계산은 dtype, bias, quantization, activation fusion 및 kernel에 따라 조금 다르다.

## 3.3 Projection은 가중치 행렬 곱이다

Projection은 벡터/hidden representation을 학습된 선형변환으로 다른 feature 공간에 사상하는 것이다. 새로운 종류의 연산을 뜻하는 단어라기보다 `Y=XW`에 붙는 역할 이름이다.

- **Q/K/V projection:** 한 token hidden state `x`에서 `q=xW_Q`, `k=xW_K`, `v=xW_V`를 만든다. 구현은 세 GEMM을 합쳐 QKV fused projection으로 할 수 있다.
- **Output projection:** attention head별 결과를 concat한 뒤 `y=concat(heads)W_O`로 residual stream 폭으로 돌려놓는다.
- **MLP projection:** gated MLP는 보통 up/gate projection 두 개를 만들고, 비선형·게이트를 적용한 뒤 down projection으로 hidden 폭에 되돌린다. 예: `u=xW_u`, `g=σ(xW_g)`, `y=(u⊙g)W_d`.

각 projection은 weight를 읽고 곱하는 연산이다. projection의 차원과 위치 수가 FLOPs를 좌우하고, decode의 batch가 작을 때는 큰 이론 FLOPs가 낮은 실제 이용률로 이어질 수 있다. [R1, R3]

---

# 4. GPU 내부 구조와 데이터 이동

## 4.1 계층 구조

CUDA 프로그래밍 모델의 논리적 실행 계층은 grid → thread block → thread다. 한 block은 한 SM(Streaming Multiprocessor)에 배치되어 그 SM의 실행 자원을 사용한다. NVIDIA 세대별 SM의 내부 스케줄러·실행 파이프·캐시 구성·Tensor Core 세대는 달라지므로, “SM 하나가 정확히 이 구성”이라는 설명은 제품 세대와 compute capability를 지정해야 한다. SM은 **GPU 안에서 block을 실행·스케줄하는 multiprocessor 단위**로 이해하면 충분하다. [R3, R4]

| 구성요소 | 직관과 기능 | LLM serving 영향 |
|---|---|---|
| thread | 하나의 논리 실행 주체 | elementwise·index 작업 등을 분할 |
| warp | NVIDIA GPU에서 보통 32개 thread의 실행 묶음 | SIMD식 실행 및 divergence 비용 |
| register | thread가 쓰는 가장 가까운 소규모 상태 | 레지스터 과다 사용은 occupancy를 낮출 수 있음 |
| shared memory | block 내 thread가 공유하는 온-chip 메모리 | 타일 재사용·협업. 용량 제한·bank conflict 고려 |
| L1 / texture cache | SM 근처 cache 자원; 세대에 따라 shared memory와 일부 자원 공유 | 재사용 가능 data의 일부를 빠르게 접근 |
| L2 | GPU 전체가 공유하는 on-chip cache 계층 | HBM 왕복 감소에 도움, 큰 KV 전체가 상주하는 공간은 아님 |
| HBM | GPU 외부 패키지의 고대역폭 device memory | weight, KV cache, activation, workspace의 주요 저장소 |

## 4.2 CUDA Core와 Tensor Core

CUDA Core는 일반적인 scalar/vector ALU 연산과 각종 SIMT 명령을 처리하는 실행 lane을 일컫는 NVIDIA 제품 용어다. Tensor Core는 작은 행렬 조각에 대한 행렬 곱-누산을 고처리량으로 수행하는 전용 연산 유닛이다. GEMM 커널은 tile을 준비하고 Tensor Core MMA 연산을 호출하며, 주소 계산·제어·activation·정규화 등은 CUDA Core/기타 고정 기능 유닛과 메모리 파이프라인이 함께 담당할 수 있다. 정확한 실행 경로는 컴파일러·커널·GPU 세대·dtype에 따라 달라진다. [R4]

Tensor Core는 모든 GPU 연산을 대신하지 않는다. 지원 dtype·shape·정렬·sparsity 조건을 만족하는 행렬 연산에 특히 효과적이다. 연산 크기가 작거나 memory-bound, 제어 흐름·불규칙 gather가 주된 작업이면 Tensor Core peak 처리량에 도달하지 않을 수 있다. 그래서 `GPU-Util`이 95%라고 해서 Tensor Core가 정격 FLOPS로 계속 일하고 있다는 뜻은 아니다. GPU util은 대개 일정 샘플 구간 동안 GPU가 busy였는지 보여주는 지표다. kernel 종류, SM occupancy, Tensor Core utilization, HBM read/write, stall reason, batch 크기 등과 함께 해석해야 한다.

## 4.3 GPU 안과 GPU 사이의 데이터 경로

**GPU 내부:** thread가 HBM의 global memory에서 data를 읽는다. 필요한 tile은 L2/SM 근처 cache를 거치고, kernel은 일부 tile을 register나 shared memory에 가져와 계산 후 결과를 global memory에 쓴다. 구체적인 cache 경유는 cache policy와 kernel 접근에 따라 달라진다. 모든 attention KV가 매 순간 SRAM에 올라와 있는 것은 아니다. 큰 KV 전체는 보통 HBM에 있고, kernel tile이 작업에 필요한 부분을 순차/부분적으로 가져온다. [R3, R7]

**GPU 간:** 같은 node의 GPU는 PCIe 또는 NVLink 경로로 통신하고, HGX 시스템은 NVSwitch를 통해 다수 GPU의 NVLink 경로를 연결할 수 있다. 노드 간 GPU 통신은 NIC와 Ethernet/InfiniBand 등의 fabric을 통한다. GPUDirect RDMA는 지원되는 환경에서 NIC가 GPU memory와 직접 data를 주고받게 해 CPU staging을 줄이는 기술이지, 네트워크 latency나 congestion을 없애는 것은 아니다. [R5, R6]

| 연결 | 범위 | 해결하는 문제 | 주의점 |
|---|---|---|---|
| PCIe | CPU↔GPU 및 GPU↔GPU (플랫폼 의존) | 범용 I/O·장치 연결 | PCIe 세대/레인/스위치 topology에 따라 차이 큼 |
| NVLink | 주로 node 내 GPU↔GPU 고속 링크 | TP/collective, peer memory access | aggregate·direction 수치를 제품 정의대로 읽을 것 |
| NVSwitch | node 내 다수 NVLink 간 switching fabric | GPU pair 간 통신 경로의 확장 | 구성에 따라 대역폭과 oversubscription 다름 |
| InfiniBand / Ethernet fabric | node↔node 통신 | 분산 TP·EP·collective | NIC 속도와 GPU-NIC locality, RDMA, congestion이 실제값 좌우 |

## 4.4 공개 하드웨어 수치 읽기 예시 (제품·form factor 기준)

아래는 공식 NVIDIA 자료에서 확인한 예시다. H100/H200의 SXM HGX와 PCIe/NVL SKU는 메모리·NVLink 구성이 다를 수 있다. bandwidth 표의 node aggregate는 전체 8-GPU system 합이며 GPU당 수치와 혼용하면 안 된다. 특히 NVLink 900 GB/s는 GPU-to-GPU bandwidth의 제품 정의를 따른 값으로, 링크마다 900 GB/s라는 뜻이 아니다. [R5, R38]

| 플랫폼 예시 | GPU 메모리 | GPU HBM 대역폭 | GPU 간 link / 노드 통신 | 해석 |
|---|---:|---:|---|---|
| H100 SXM | 80 GB HBM3 | 약 3.35 TB/s (datasheet SKU 기준) | NVLink 4, 최대 900 GB/s GPU-to-GPU; HGX 8-GPU | compute, memory, link 숫자는 특정 SXM/HGX 구성을 기준으로 확인 |
| H200 SXM | 141 GB HBM3e | 약 4.8 TB/s | NVLink 4, 최대 900 GB/s GPU-to-GPU; HGX 8-GPU | 긴 context cache 용량·대역폭 증가가 H100 대비 중요한 차이 |
| B200 SXM | 180 GB HBM3e | 최대 약 8 TB/s급 공식 플랫폼 표 | NVLink 5 / NVSwitch; HGX node aggregate는 공식 table 별도 확인 | 차세대 link·HBM은 SKU/system 자료를 함께 봐야 함 |

H100 datasheet, H200 datasheet, HGX reference architecture는 발표자료의 정확한 장비와 일치하는지 확인한다. “H200이니까 NVLink 900 GB/s”는 모든 H200 형태에서 자동으로 성립하지 않는다. PCIe 카드, NVL bridge pair, SXM HGX는 서로 다른 topology다. 제품 수치는 구매·벤치마크 대상의 exact SKU/form factor가 확정되면 해당 datasheet를 다시 대조한다. [R5, R38]

## 4.4 TP collective가 만드는 비용

Tensor parallelism은 각 layer의 행렬 차원을 GPU들로 나누어 계산한다. 일부 projection은 output/column 차원으로 나누고, 뒤에서 합치는 reduce가 필요하다. 다음 projection은 input/row 차원으로 나뉠 수 있고, partial output 합산을 위해 all-reduce 또는 reduce-scatter/all-gather 계열 통신이 생긴다. 구현은 모델·parallel layout에 따라 다르다. NCCL은 collective primitive를 제공한다. [R6]

단순 ring all-reduce의 전송량 근사로 rank 수 `P`, tensor payload `S byte`일 때 rank당 send+receive 합을 약 `2(P−1)/P × S`로 보는 모델이 흔하다. 실제 시간은 `message latency × 단계 수 + bytes / effective bandwidth + synchronization`에 가깝고, topology·overlap·알고리즘에 따라 달라진다. GPU 수를 늘리면 각 GPU 계산량은 줄지만, collective 횟수·통신량·동기화 지연은 남는다. 작은 decode GEMM에서는 계산 절감보다 매 token collective가 더 비싸질 수 있어 scaling이 sublinear 또는 퇴행할 수 있다. [R6]

### 흔한 오해 / 정정

> “GPU가 두 배면 2배 빠르다.” → 모델 병렬 분할, collective, GPU 간 링크, scheduler, 작은 batch, memory locality 때문에 선형 scaling이 보장되지 않는다. 모델이 한 GPU에 들어가고 latency가 중요하면 TP를 키우는 것이 오히려 손해일 수 있다.

---

# 5. 모델 실행 흐름: 레이어, dense, MoE, hybrid

## 5.1 token 하나가 모든 layer를 통과하는가?

일반적인 decoder-only autoregressive 모델에서 한 sequence의 새 token은 layer 1부터 마지막 layer까지 순차적으로 hidden state를 전달한다. 각 layer의 계산이 끝나야 다음 layer가 시작하고, 최종 hidden state에서 vocabulary logits를 계산해 다음 token을 고른다. **MoE여도 token은 layer stack을 진행**하지만 각 MoE layer에서는 전체 expert 대신 router가 선택한 일부 expert만 실행한다. Hybrid attention이어도 token은 모든 layer를 통과하되, layer마다 token mixing mechanism이 다를 수 있다. [R1]

## 5.2 Dense·MoE·Hybrid 구분

- **Dense:** 각 token이 각 layer의 attention과 dense FFN을 계산한다. 총 parameter가 대체로 token당 활성 parameter와 일치한다.
- **MoE:** 각 token hidden state를 router에 넣어 expert subset(top-k)을 선택한다. 전체 model weight는 커도 token당 활성 FFN 연산은 줄일 수 있다. 대신 expert weight 총량은 VRAM/분산 메모리에 계속 필요하고, expert parallelism을 쓰면 GPU 간 token dispatch/all-to-all이 발생할 수 있다.
- **Hybrid attention:** layer별로 full attention, sliding window, MLA/압축 attention, linear/recurrent attention 등을 조합한다. token은 layer stack은 통과하되, 어떤 layer는 과거 token의 KV를 읽고, 다른 layer는 recurrent state를 update할 수 있다.

따라서 “토큰 하나가 나오기 위해 모든 레이어를 통과한다”는 말은 기본적으로 맞지만, 모든 layer가 같은 attention을 쓰거나 모든 MoE expert를 실행한다는 의미는 아니다. 모델에 auxiliary/refinement/draft head가 붙으면 generation 경로에 추가 branch가 있을 수 있으므로 해당 architecture를 따로 확인해야 한다.

## 5.3 요청 상태의 종류와 수명

| 데이터 | 의미 | 수명 | 보통 저장 위치 / 용량 경향 |
|---|---|---|---|
| weight | 학습된 파라미터 | model process 동안 유지 | GPU HBM 또는 CPU/offload; 모델 크기·dtype에 비례 |
| activation | 현재 prefill/decode 계산 중 layer 사이의 중간 tensor | layer/step 처리 뒤 재사용 여부에 따라 짧음 | HBM 및 kernel 내부 register/shared; batch·sequence·workspace에 따라 peak |
| KV cache | full/local attention의 과거 K와 V | sequence 종료·eviction까지 | GPU HBM 기본; token 수와 attention layer 수에 비례 |
| recurrent state | 과거 sequence 정보를 압축해 update한 상태 | sequence 종료까지 요청별 유지 | GPU HBM; 보통 context 길이에 선형 증가하지 않고 state dimension에 의해 결정 |
| convolution state | causal conv가 다음 입력에 필요로 하는 최근 입력 history | sequence 종료까지 | GPU HBM; 보통 kernel size−1 길이의 최근 값 중심 |
| workspace | GEMM/attention/collective 등의 임시 buffer·scratch | 해당 연산 또는 실행 graph 기간 | HBM, library/engine에 따라 예약·재사용 |
| metadata | block table, scale, sequence length, state index 등 | cache/state와 유사 | CPU와 GPU 양쪽; 작지만 allocator·정렬에 영향 |

이 수명 구분은 용량 계획의 토대다. Weight를 GPU 수로 나누어도 KV·activation·workspace·CUDA Graph pool이 남는다. 요청이 늘면 weight는 그대로지만 요청별 KV/state는 증가한다.

---

# 6. Prefill과 autoregressive decode

## 6.1 Prefill: prompt를 context로 바꾸기

요청 prompt가 token id `x₁…xₙ`으로 들어오면 모델은 전체 prompt positions를 여러 layer에 처리한다. Full attention layer에서 각 position의 Q/K/V projection을 계산하고 causal mask로 미래 위치를 가린 attention을 실행한다. Prefill이 끝나면 모든 prompt 위치의 K/V가 이후 decode에 쓰일 cache에 존재한다. Hybrid model에서는 full-attention layer별 K/V 외에도 linear/recurrent layer의 sequence scan 결과 및 마지막 state가 만들어진다. “모든 위치의 hidden state를 모두 저장해 둔다”는 뜻은 아니다. 중간 activation은 보통 다음 layer에 전달된 뒤 해제·재사용할 수 있고, 이후 generation에 필요한 KV/recurrent state가 보존된다. [R1, R7]

## 6.2 Decode: 한 token씩 다음 state를 갱신

현재 마지막 token hidden state가 layer를 순차 통과한다. Full-attention layer는 새 Q/K/V를 만들고, 새 K/V를 해당 sequence cache에 추가한 뒤 현재 Q로 과거 K/V를 조회한다. `softmax(QKᵀ/√d + mask)V` 결과와 residual/MLP 계산을 거쳐 다음 layer로 이동한다. Hybrid recurrent layer는 새 입력으로 state를 update한다. 마지막 layer의 logits에서 sampling/greedy/beam 등의 정책으로 다음 token을 선택하고, 이를 다음 decode step 입력으로 쓴다. [R1, R8]

## 6.3 TTFT와 TPOT/ITL

- **TTFT (time to first token):** 요청 접수부터 첫 출력 token이 사용자에게 도달하기까지의 지연. tokenization, queueing, prefill, 첫 decode, network/streaming이 반영될 수 있다.
- **TPOT (time per output token):** 생성된 출력 token들의 평균 추가 지연/처리 간격을 보는 지표. 제품별 계측 정의를 확인해야 한다.
- **ITL (inter-token latency):** 연속 출력 token 사이의 지연. p50/p95/p99와 batch 영향을 같이 본다.

TTFT의 핵심은 긴 prompt prefill과 queueing인 경우가 많고, TPOT/ITL은 decode의 HBM·KV 접근·batch·scheduler·collective가 중요해진다. 단, prefill과 decode는 엔진에서 겹쳐 실행될 수 있으므로 특정 부하에서 실측해야 한다.

| 항목 | Prefill 경향 | Decode 경향 |
|---|---|---|
| token 작업량 | prompt token을 여러 개 병렬 처리 | sequence당 한 step에서 보통 1 token 처리 |
| GEMM | 큰 `N` dimension을 묶기 쉬움 | batch가 작으면 skinny GEMM |
| attention | 긴 입력 전체에 대한 연산 | 새로운 query가 과거 cache를 읽음 |
| 주된 압박 | compute, attention IO, activation peak, TTFT | HBM BW, KV read, weight read, launch/collective latency |
| 처리량 조정 | batch·chunked prefill | continuous batching·active sequence 합산 |

## 6.4 Batching 정책은 서로 다른 문제를 다룬다

- **Batching:** 여러 sequence를 한 GPU step에서 함께 계산해 weight reuse와 GPU occupancy를 높인다. batch가 커지면 처리량이 오를 수 있지만 개별 latency·KV 사용량도 증가한다.
- **Continuous batching:** 끝난 sequence 자리에 새 요청을 넣어 static batch의 빈 slot을 줄인다. 길이가 다른 요청이 섞이는 서버에서 GPU 이용률과 처리량에 유리하다. [R9]
- **Chunked prefill:** 긴 prompt를 chunk로 나눠 decode와 스케줄링을 공정하게 섞거나 prefill activation peak/큰 단일 작업을 제어한다. 전체 FLOPs를 마법처럼 없애지는 않으며 chunk 단위·scheduler 때문에 효율 trade-off가 있다. [R10]

운영 입력으로는 prompt 길이, output 길이, 동시성, 요청 도착률, batch 정책을 분리해서 기록한다. prompt 길이는 prefill 시간과 KV의 시작 크기를 좌우하고, output 길이는 decode step 수와 cache 증가 기간을 늘린다. 동시성은 요청별 cache를 더한다. 평균만 보면 긴 꼬리 요청이 가려질 수 있으므로 p50/p95/max를 따로 본다.

---

# 7. Attention 공통식과 KV cache 용량 유도

## 7.1 Scaled dot-product attention

query `q`는 “무엇을 찾을지”, key `k`는 “어떤 내용을 어느 정도로 매칭할지”, value `v`는 “매칭된 위치에서 가져올 정보”의 역할을 한다. 한 query는 모든 유효 key와 dot product를 계산하고, scale 및 causal/window mask를 적용한 뒤 softmax 가중치를 value에 곱한다.

`Attention(Q,K,V) = softmax((QKᵀ)/√dₕ + mask) V`

causal decoder에서 위치 `t`는 미래 위치 `>t`를 보지 않는다. full attention은 이전 모든 위치를 대상으로 한다. 이 식의 **수학적 대상**과 FlashAttention 등 **그 식을 더 적은 HBM IO로 실행하는 kernel**은 다른 계층이다. [R1, R7]

## 7.2 왜 K와 V를 저장하나?

현재 token `t`의 다음 계산은 각 attention layer에서 이전 positions의 K/V를 참조한다. cache가 없으면 매 decode step마다 과거 token representation을 layer 전체에서 다시 계산해야 한다. 과거 token들이 고정되어도 각 레이어에서 projection/attention을 반복하게 되어 낭비가 크다. KV cache는 이전 위치의 K/V를 재사용하고, 새 token의 K/V만 추가한다.

길이 `T`까지 생성한다고 하자. 모든 시점에 prefix를 처음부터 재계산하는 단순한 접근은 layer당 token-prefix 처리량 합이 `1+2+…+T=O(T²)`가 된다. cache는 과거 K/V를 재사용하여 각 새 token에서 과거 cache attention을 계산하고 새 K/V를 추가한다. Attention score 조회 자체는 full attention에서 출력 길이에 따라 계속 커지고, 모든 과거 K/V를 매 step 읽는 비용은 사라지지 않는다. 즉 cache는 반복 projection/hidden-state 계산을 피하지만 full context attention을 상수 비용으로 만들지는 않는다. [R1, R7]

## 7.3 Full-attention KV bytes 유도

기호를 고정한다.

| 기호 | 정의 | 단위 |
|---|---|---|
| `B` | 동시에 cache를 가진 batch/sequence 수 | sequence |
| `T` | sequence마다 cache에 든 token 수 | token/sequence |
| `L` | KV를 저장하는 full-attention layer 수 | layer |
| `H_kv` | layer당 KV head 수 | head/layer |
| `d_h` | 각 head의 채널 차원 | element/head |
| `b` | KV 원소 하나의 저장 크기 | byte/element |
| `2` | Key와 Value 두 tensor | K,V |

sequence 하나, layer 하나, token 위치 하나에 저장되는 원소 수는 `2 × H_kv × d_h`다. 여기에 dtype byte `b`, token `T`, layer `L`, 동시 sequence `B`를 곱한다.

`M_KV = B × T × L × 2 × H_kv × d_h × b` byte

여기서 2는 K와 V다. K와 V의 dtype 또는 layout이 다르면 더 일반적으로 `B×T×L×H_kv×d_h×(b_K+b_V)`가 된다. 위 기본식은 모든 cache position과 layer에 동일한 head 수·head dimension·byte 수를 가정한 raw tensor 이론값이다. [R8]

### 단위 검산

`sequence × token/sequence × layer × K,V × head/layer × element/head × byte/element = byte`

### GQA 예시 계산

설명용 모델을 `L=32`, `H_q=32`, `H_kv=8`, `d_h=128`, BF16 (`b=2`)라 둔다. sequence 하나당 token 하나의 full-attention KV는 `32×2×8×128×2 = 131,072 byte = 128 KiB`다. 8,192 token이면 `1 GiB`다. MHA (`H_kv=32`)였다면 같은 나머지 조건에서 네 배 `4 GiB`다. 이는 raw KV tensor 계산이고, allocator/block metadata와 workspace는 별도다.

## 7.4 TP shard는 GPU 개수로 자동 나누면 안 된다

TP에서 attention heads가 rank별로 나뉘면 local cache는 `H_kv,local` 또는 local latent dimension을 기준으로 계산해야 한다. 하지만 KV head 수가 TP degree보다 적거나 head 배치 제약이 있으면 KV head가 rank에 복제될 수 있다. 구현에 따라 GQA KV head를 나누거나 replicated K/V를 둘 수 있다. layer 수·pipeline parallel·context parallel·cache layout도 바뀐다. 따라서 `전체 KV / GPU 수`는 가정일 뿐이다. 각 rank의 model config, parallel group, cache spec, 실제 block allocation을 확인해야 한다. vLLM처럼 attention type마다 cache group/shape가 다른 최신 serving 구현은 단순 수식보다 세부화가 필요하다. [R11]

---

# 8. MHA·MQA·GQA·MLA 비교

각 방법은 query가 attention으로 정보를 읽는 방식의 변형이며, 저장/연산 절감은 다르게 나타난다. 표기상 query head 수를 `H_q`, KV head 수를 `H_kv`, layer 수를 `L`, token 수를 `T`, head dimension을 `d_h`, 저장 dtype byte를 `b`로 둔다. 모델별 projection dimension은 각 config를 따른다.

## 8.1 MHA: Multi-Head Attention

### 구조와 계산

MHA는 여러 query head가 서로 다른 projection을 사용하고, 보통 각 Q head에 대응하는 독립 K/V head가 있다. 표준 equal-head 사례는 `H_q=H_kv=H`다. head `i`는 `softmax(q_iK_iᵀ/√d_h)V_i`를 계산하고, 모든 head를 concat해 output projection을 거친다.

### Cache 공식

token·layer당 `2×H×d_h×b` byte. 전체 `B×T×L×2×H×d_h×b` byte다. KV 표현력이 head마다 독립인 만큼 단순 GQA/MQA보다 cache bytes가 많다. KV head가 query head와 같다는 것은 일반적 정의에 가까운 설명이지만, multi-head 변형 전체가 반드시 이 구성이라고 볼 필요는 없다.

### 비용과 trade-off

Projection 및 attention head 수가 크고 cache bandwidth/용량이 커진다. 충분한 GPU 병렬성이 있고 모델이 이 구조로 학습되었다면 표현력의 유연성이 있지만, MHA를 항상 최고 정확도로 단정할 수는 없다. 모델 규모·데이터·학습·distillation·추론 커널이 결과를 바꾼다.

## 8.2 MQA: Multi-Query Attention

### 구조와 계산

여러 query head가 하나의 KV head를 공유한다: `H_kv=1`, `H_q>1`. 각 query head는 동일 K/V를 이용해 별도 query로 attention을 계산한다. Query 수가 줄어드는 것은 아니며, K/V projection 및 저장/읽기 측 비용이 줄어든다. [R13]

### Cache 공식 및 실무 영향

token·layer당 `2×1×d_h×b`, 전체 `B×T×L×2×d_h×b`. 같은 `H_q`인 MHA 대비 raw cache는 대략 `1/H_q`다. decode에서 cache read byte가 작아져 메모리 병목을 완화할 수 있지만, 모든 head가 같은 K/V를 공유하면 학습 시 표현 선택지가 줄 수 있다. MQA 논문은 품질을 보존하는 학습을 제시했지만, 어느 모델에서도 품질 저하가 없다는 보장은 아니다. 구현이 K/V를 query head 수만큼 복제해서 계산하면 예상 이점을 잃을 수 있다.

## 8.3 GQA: Grouped-Query Attention

### 구조와 group 관계

`H_q` query head를 같은 크기 그룹으로 나누고 각 그룹의 query들이 한 KV head를 공유한다. `G = H_q/H_kv`를 group size라 두면 `H_q = G×H_kv`, `G`는 보통 정수다. MHA는 `G=1`, MQA는 `H_kv=1`, 즉 `G=H_q`인 극단 사례다. [R14]

### Cache·연산

token·layer당 `2×H_kv×d_h×b`; 전체 `B×T×L×2×H_kv×d_h×b`. 동일한 query head 수에서 MHA 대비 KV 크기 비율은 `H_kv/H_q=1/G`다. Query별 score/value 계산은 계속 있으므로 attention FLOPs가 그 비율만큼 줄어드는 것은 아니다. Memory traffic/저장량 감소가 핵심이다.

### 품질과 속도

GQA는 MHA와 MQA 사이에서 cache bandwidth 절감과 KV 표현력 사이의 선택지를 준다. GQA 원 논문은 MQA로 uptraining하는 절차를 포함해 결과를 평가했다. 결과를 구조 자체의 보편적 accuracy ranking으로 확대하면 안 된다. 추론 속도는 cache traffic이 지배적이면 좋아질 가능성이 있지만, kernel이 KV 공유를 효율적으로 broadcast하지 못하거나 compute-bound이면 차이는 작을 수 있다. TP rank 배치에 따라 GQA KV replication이 생길 수 있어 rank별 cache를 확인한다. [R14]

### 숫자 예

Qwen3.5-27B의 공개 config 요약은 full-attention이 `H_q=24`, `H_kv=4`, `d_h=256`이다. 그러면 GQA group size는 6이고, BF16 기준 raw KV bytes/token/full layer는 `2×4×256×2=4096 byte=4 KiB`다. Full attention layer 16개라면 전체 `64 KiB/token`; prompt 170K와 output 2K, 즉 172K cache tokens에 `10.49 GiB/request`다. 이 예의 config와 산술은 11장의 모델 사례에서 다시 확인한다. [R12]

## 8.4 MLA: Multi-Head Latent Attention

### 단순 head 수 감소와 다른 점

MLA는 GQA처럼 KV head 수를 줄이는 데서 멈추지 않고, token별 key/value 표현을 latent vector로 압축해 저장한다. DeepSeek-V2 보고서는 low-rank joint KV compression과 decoupled RoPE를 설명한다. 개념적으로 `c_t^{KV}=W_D^{KV}h_t`라는 낮은 차원의 latent를 저장하고, 각 attention head가 필요한 K/V 표현을 projection으로 복원하거나 algebra적으로 projection을 흡수(absorb)할 수 있게 계산을 재배열한다. Position encoding/RoPE가 key에 들어가는 경로는 latent 압축과 충돌하지 않도록 decoupled position component를 두는 설계가 핵심이다. [R15]

### Cache 식

모델 구현이 token마다 압축 latent 차원 `d_c`와 별도 positional-key 차원 `d_r`를 저장한다면 단순화한 bytes/token/layer는 `(d_c+d_r)×b`다. 전체 cache는 `B×T×L×(d_c+d_r)×b`로 나타낼 수 있다. 이는 설계 설명용 일반형이다. 실제 MLA 구현은 latent, RoPE key, layer sharing, cache dtype, auxiliary/index state 등을 따로 계산해야 한다.

### 연산·속도·품질

cache read/write는 크게 줄일 수 있지만, projection·복원·attention score 연산은 남는다. Absorption은 decode 시 projection과 head 연산 순서를 바꾸어 KV를 head별로 복원하지 않고 latent에 흡수된 weight로 계산하는 algebraic optimization이다. 그 효용은 kernel 구현, quantization, batch 및 GPU memory/compute roofline에 달렸다. MLA의 압축 latent가 단순한 “K/V tensor를 작게 저장”하는 방법으로 끝나지 않는 이유는 positional path와 per-head attention의 정확한 수학적 등가성, 로딩 가능한 weight shape, cache quantization 경로까지 설계에 포함되기 때문이다. 구조가 충분히 학습되면 MHA 대비 품질이 같거나 더 좋은 특정 결과도 가능하지만, 모든 모델에 적용되는 명제는 아니다. [R15]

### 구현 주의와 모델 예

DeepSeek-V2가 MLA의 대표 공개 사례이며, 이후 DeepSeek V3 계열과 공개 GLM-5 계열 문서에 MLA 기반 구조가 보고된다. 정확한 release/config마다 attention이 추가로 sparse/local 방식과 결합될 수 있다. [R15, R24]

## 8.5 비교 매트릭스

| 축 | MHA | MQA | GQA | MLA |
|---|---|---|---|---|
| 토큰·layer 저장량 | `2H_qd_hb` | `2d_hb` | `2H_kvd_hb` | latent 중심 `(d_c+d_r)b` 등 구현별 |
| Full model raw KV | `B T L 2H_qd_hb` | `B T L 2d_hb` | `B T L 2H_kvd_hb` | cache에 실제 저장하는 latent/pos/부가 상태의 합 |
| 절감 지점 | 기준 | KV head 하나 공유 | 그룹별 KV 공유 | joint latent 압축 + algebraic reordering |
| Attention 계산 | query head별 QKᵀ, V 혼합 | head 수만큼 query 계산 | head별 query 계산 | 복원 또는 absorb 경로로 동일 기능 계산 |
| Decode 영향 | KV traffic 클 수 있음 | cache traffic 최저 경향 | 중간 trade-off | 낮은 cache bytes, 구현/compute 비용 추가 가능 |
| 품질 경향 | head별 KV 자유도 큼 | 공유 제약 큼 | 중간 선택 | 학습된 저랭크 표현력에 의존 |
| 구현 주의 | memory footprint | KV broadcast·head sharing | TP에서 KV shard/replica | latent·RoPE·absorb·quantization 호환 |
| 예시 | 초기 Transformer 계열 다수 | PaLM 계열 논문·일부 모델 | Qwen/Llama 등 config별 확인 | DeepSeek-V2/V3, 일부 후속 계열 |

**정확도 주의:** “MHA가 항상 최고, MQA가 항상 최저”라는 단정은 피한다. 정확도는 학습 방법, uptraining, 데이터, parameter 수, architecture와 평가에 따라 달라진다. 논문 표본의 benchmark 결과는 그 실험의 checkpoint와 학습 조건에 대한 증거다.

---

# 9. SWA·CSA·HCA: 긴 문맥에서 저장과 선택

## 9.1 SWA: Sliding Window Attention

### 정의와 cache 식

query position `t`가 최근 window `W` positions만 참조하는 causal attention이다. 유효 KV 길이를 `T_eff=min(T,W)`라 하면 local layer `L_s`의 raw cache는 `B×T_eff×L_s×2×H_kv×d_h×b`다. 전체 문맥 `T`가 계속 늘어도 cache가 최대 `W` 주변에서 제한된다. Full-attention layer `L_f`와 섞이면 `M=B×2H_kvd_hb×(L_fT + L_s min(T,W))` (동일한 head dims/dtype 가정)이다. [R16]

### 장점·주의

긴 context에서 local layer의 KV 용량과 attention 계산을 제한한다. 그러나 오래된 token을 버리면 멀리 떨어진 정보에 직접 접근할 수 없고, 검색·인용·장기 회상 품질이 떨어질 수 있다. sink token 유지, global layer, retrieval, summary memory 등 보완책은 별도 정책이다. 구현 시 sliding cache ring/block mapping, sequence 위치, window 경계와 prefix reuse 정책을 확인한다. SWA의 cache가 고정 길이라고 해도 prefill 계산 전체가 항상 window 길이로 감소하는 것은 아니며 kernel·layer 구조에 따라 다르다.

## 9.2 DeepSeek-V4의 CSA: Compressed Sparse Attention

### 용어와 흐름

DeepSeek V4 기술보고서와 공개 구현에서 **Compressed Sparse Attention (CSA)**라는 명칭을 사용한다. 공개 구현 설명에서는 먼저 raw 최근 window의 sliding K/V 경로를 두고, 별도 compressor가 만든 sequence-level compressed entries에 대해 indexer가 query별 top-k를 선택하는 구조로 설명한다. 즉 오래된 원본 token마다 전체 raw K/V를 유지해 무작위로 복구한다고 단순화하면 안 된다. 오래된 문맥의 주 attention 경로는 압축 entry이고, 원본 raw token K/V는 local window 범위에서 유지된다. “모든 과거 원본 token을 매 query마다 읽는” full attention과 다르다. 정확한 압축 stride, indexer, top-k 및 cache 구조는 checkpoint/report/구현 버전을 함께 표기해야 한다. [R17, R18]

개념식은 다음과 같이 쓸 수 있다. 원 입력 `h₁…h_T`에서 압축 단위 `m`마다 `c_j = Compress(h_{(j−1)m+1:jm})`를 만들고, query `q_t`가 index score `s_{t,j}=Index(q_t,c_j)`에서 후보 집합 `I_t=TopK_j(s_{t,j})`를 고른다. 본 attention은 `I_t`에 해당하는 compressed key/value와 최근 raw local-window K/V를 함께 대상으로 softmax weighted sum을 계산한다. 이 식은 CSA의 정보 흐름을 단순화한 개념식이며 DeepSeek 내부 projection의 exact formula를 대체하지 않는다.

DeepSeek-V4 공개 Transformers config 문서에 나온 V4-Pro 기본값 사례는 CSA compression rate `m=4`, index top-k `k=1024`, HCA compression rate `m′=128`이다. 이 값은 해당 문서의 Pro default에 대한 것이며 모든 V4 checkpoint나 Flash 변형에 공통인 상수가 아니다. 저장·계산 절감은 압축률, 선택 K, local tail 크기, indexer 비용, batch/query 수, 데이터 재사용에 따라 달라진다. Top-K는 무료가 아니다. indexer projection/score, 후보 metadata, gather의 비연속 access가 추가 비용이다. [R17, R18]

## 9.3 HCA: Heavily Compressed Attention

DeepSeek-V4 자료의 용어는 **Heavily Compressed Attention (HCA)**다. 공개 Transformers 문서는 HCA가 최근 raw sliding-window K/V와 함께 compressor가 만든 compressed entries를 attention 대상으로 두며, 이 구현에서 HCA layer는 available compressed entries를 볼 수 있고 CSA처럼 per-query top-k indexer를 쓰지 않는 cache path를 설명한다. 즉 local/raw branch는 recent window이고 global-long-range branch는 heavily compressed representation이다. V4 config에서 HCA 압축률, CSA의 압축률/top-k는 모델별 설정값으로 읽어야 한다. “단순 hierarchical attention”이라고 번역해 별도의 tree algorithm으로 단정할 근거는 없다. HCA는 더 강한 압축 representation을 사용해 CSA의 query-wise selection과 다른 저장·계산 trade-off를 만든다. [R17, R18]

## 9.4 Full vs SWA vs CSA vs HCA

| 방식 | 저장 단위 | Query가 보는 대상 | cache 증가 | 검색/선택 | 주요 비용 변수 | 손실·품질 경향 |
|---|---|---|---|---|---|---|
| Full attention | token별 K,V | 모든 과거 token | `O(T)`/layer | 없음, 전체 조회 | `T`, heads, dimension | reference 정보 보존, 계산·메모리 증가 |
| SWA | 최근 token K,V | 최근 `W`개 | `O(min(T,W))` | window | `W`, local layer 수 | 오래된 원문 직접 접근 제한 |
| CSA | raw local window + compressed entries + indexer metadata | query별 선택 compressed entries + 최근 raw K/V | 압축률·top-k·tail에 의존 | compressed entries top-k | compression `m`, top-k `k`, tail, index cost | 선택 실패·압축으로 세부 정보 손실 가능 |
| HCA | raw local window + heavily compressed entries | 최근 raw K/V + available compressed entries | 높은 압축에 따른 축소 | 공개 구현 설명상 CSA식 query top-k indexer 없음 | compression ratio, layer schedule | 표현 압축 trade-off; 보고서 성능 범위에 한정 |

V4 공개 자료에 제시된 CSA/HCA compression rates나 KV 절감 비율은 해당 보고서의 모델/checkpoint 및 cache 정의에 속한다. 이를 CSA/HCA 일반식의 보장 성능처럼 인용하면 안 된다. 최신 Transformers 구현 설명은 config와 cache를 이해하는 보조자료지만, 논문이 공식 원문인 경우 원 보고서의 표기와 차이가 있으면 원 보고서를 우선하고 구현 차이를 병기한다. [R17, R18]

---

# 10. Recurrent state와 convolution state

## 10.1 Full KV 대신 상태를 이어가는 이유

Recurrent/linear attention 계열은 각 token의 과거 K/V를 전부 보관하기보다, 과거에서 누적한 고정 크기 또는 제한 크기 상태를 새 token마다 update한다. 따라서 많은 구조에서 memory per sequence가 context length에 선형으로 늘지 않는다. 대신 상태가 압축한 history로부터 정보를 읽으므로 full attention과 표현·retrieval 동작이 같지 않다.

일반적인 state recurrence를 `S_t = F(S_{t−1}, x_t)` 및 `y_t=G(S_t,x_t)`로 표시할 수 있다. `S_t`는 특정 아키텍처의 모든 state tensor를 축약한 표기다. 구현에선 레이어·head·gate별로 여러 텐서가 있다.

## 10.2 Convolution state

causal convolution kernel size `k`가 다음 입력의 출력을 계산할 때 최근 `k−1`개 입력이 필요하다면 그 tail buffer를 보관한다. 새 입력 `x_t`가 들어오면 `[x_{t−k+1},…,x_{t−1},x_t]`를 만들고 convolution을 수행한 뒤 가장 오래된 값을 빼고 새 값을 유지한다. 이 상태는 convolution의 finite receptive history로, recurrent attention의 memory matrix와 다른 역할이다.

## 10.3 Mamba selective state와 DeltaNet 계열 matrix state

- **Mamba/SSM:** 연속 state-space model의 선택적 파라미터화와 scan으로 긴 sequence를 처리한다. 입력에 따라 state transition 및 write/read 성격이 변하도록 selective mechanism을 둔다. 기본적인 직관은 sequence 전체 KV 대신 state를 반복 update하는 것이다. 실제 state 크기·scan 방식은 구현/config에 의존한다. [R19]
- **Mamba-2 / SSM duality:** state-space recurrence와 attention의 구조적 연결을 보이고, hardware-efficient parallel algorithm을 제안한다. 이는 recurrence state의 고정 메모리와 attention 스타일 계산 사이의 관계를 이해하는 근거다. [R20]
- **DeltaNet:** associative memory 관점의 linear attention으로, 새 key/value가 들어올 때 현재 예측과 target의 차이를 반영하는 delta-rule update를 사용한다. 단순히 state에 `k⊗v`를 더하는 것이 아니라 이전 기억을 보정하는 형태다. [R21]
- **Gated DeltaNet / KDA:** gate로 state의 유지·감쇠를 조절하고 delta update를 결합해 forgetting과 memory update를 제어한다. Kimi Linear는 KDA와 MLA를 섞은 hybrid model로 제시되었다. layer별 head 수와 state tensor shape를 raw cache처럼 한 줄로 계산하면 안 된다. [R22, R23]

개념적인 gated delta update 예시: `S_t = α_t S_{t−1} + β_t (v_t − S_{t−1}k_t) k_tᵀ`와 같이 쓸 수 있다. `α_t`, `β_t`의 생성 방식과 수식은 논문마다 다르므로 이 식은 update/forgetting 직관을 위한 축약형이다. 정확한 Gated DeltaNet/KDA 수식은 해당 원 논문 정의를 따른다.

## 10.4 상태 메모리 계산 관점

full KV의 기본은 `B×T×L×…`로 token 수에 비례한다. state 기반 구조는 대략 `B×Σ_l state_bytes(layer_l)`로, context `T` 대신 레이어와 state dimension·dtype에 의해 정해지는 고정 또는 제한 상태 크기를 가진다. 하지만 상태 하나가 1 tensor라는 뜻은 아니다. 예를 들어 recurrent matrix state, convolution tail, per-head norm/gate metadata가 함께 있을 수 있다. 동시 요청 B가 커지면 state 메모리도 보통 B에 비례하므로 “linear attention은 상태 메모리가 0”이라는 말은 틀리다.

Prefill에서는 전체 prompt에 대해 parallel scan/chunk scan을 수행해 recurrent state를 구축한다. Decode에서는 토큰 하나를 입력으로 state가 갱신된다. 어느 단계가 빠른지는 커널 구현, sequence 길이, batch, state dimension에 달려 있다. [R19–R23]

---

# 11. 하이브리드 모델과 공개 모델 사례

## 11.1 Qwen3.5-27B 구성으로 계산하기

공식 Qwen 공개 모델 카드와 Transformers model doc에서 확인한 예시는 **Qwen3.5-27B**다. 모델 명칭을 Qwen3-27B 또는 Qwen3.8-27B로 바꾸어 부르면 구성 확인을 잘못할 수 있다. 공개 요약에는 64 decoder layer, `16 × (3 × Gated DeltaNet → FFN → 1 × Gated Attention → FFN)` pattern, Gated DeltaNet 48 V heads/16 QK heads/head dim 128, Gated Attention Q24/KV4/head dim 256이 기재되어 있다. Qwen3.6-27B가 공개된 경우에도 공개 문서는 architecture/model_type이 같다고 설명하지만, 이 산술 예시는 model card가 명시한 Qwen3.5-27B config를 가리킨다. [R12]

### token 하나의 layer별 흐름

1. token hidden state는 64 layer를 순차 통과한다.
2. 16개 pattern group 각각에서 Gated DeltaNet 3개와 Gated Full/Gated Attention 1개가 나온다. 따라서 `16×3=48` Gated DeltaNet layer와 `16×1=16` full/Gated Attention layer다.
3. 각 full-attention layer에서 현재 token의 K/V가 생성되어 해당 layer cache에 추가되고, query는 보존된 과거 K/V를 읽는다.
4. 각 Gated DeltaNet layer는 현재 token으로 그 layer의 recurrent state를 update한다. 이 layer가 full attention과 같은 방식으로 모든 context K/V를 저장한다고 보지 않는다.
5. convolution state가 구성에 존재하면 causal conv tail도 별도 update된다. recurrent matrix state와 convolution tail은 역할도 크기도 다른 state다.
6. state는 sequence/request별로 유지되고 해당 sequence가 끝나면 회수된다. prefix sharing, beam branching, speculative decoding 등은 실제 ownership 정책을 바꿀 수 있다.

### Raw full-KV 산술

Qwen3.5-27B full-attention config 값 `L_f=16`, `H_kv=4`, `d_h=256`를 사용한다. BF16/FP16 `b=2`면 `bytes/token = 16×2×4×256×2 = 65,536 byte = 64 KiB`. FP8 1 byte면 32 KiB/token이다. 170K prompt + 2K generated tokens를 `T=172,000`으로 계산하면 BF16 raw global/full KV는 `172,000×65,536 = 11,272,192,000 byte = 10.50 GiB/request`; FP8이면 약 5.25 GiB/request다. `170K`를 170,000으로 둔 산술 예다. tokenizer가 의미하는 실제 token count는 요청마다 정확히 센다.

| 동시 요청 | BF16 full-KV raw | FP8 full-KV raw |
|---:|---:|---:|
| 1 | 10.50 GiB | 5.25 GiB |
| 4 | 42.01 GiB | 21.00 GiB |
| 8 | 84.02 GiB | 42.01 GiB |
| 16 | 168.03 GiB | 84.02 GiB |

표는 full-attention KV에 대한 계산이다. Gated DeltaNet recurrent/convolution state, block metadata, scale, alignment, allocator fragmentation, activation peak, workspace, graph pool, model weights를 포함하지 않는다. 실제 vLLM/Transformers/기타 engine의 hybrid cache layout과 sequence scheduling을 읽어야 최종 VRAM budget이 된다. [R11, R12]

### Prefill과 decode에서 하이브리드 state

170K prompt prefill 동안 16개 full-attention layer는 각 position의 K/V를 cache에 채우고, 48개 recurrent layer는 전체 sequence의 입력을 scan/chunk 형태로 처리해 요청의 최종 recurrent state를 얻는다. Decode token 하나는 각 full layer에 새 K/V를 추가하고 긴 cache를 참조하며, 각 recurrent layer는 현재 token을 state에 적용한다. 따라서 hybrid stack은 “KV 저장이 없다”가 아니라 **일부 layer는 context 길이에 따른 KV를, 다른 layer는 sequence별 recurrent/convolution state를 저장**하는 구조다.

## 11.2 Kimi K3

MoonshotAI 공식 Kimi K3 README는 모델을 2.8T total / 104B active MoE, 93 layers, 1 dense layer, 69 Kimi Delta Attention (KDA) + 24 Gated MLA layer, hidden dimension 7168, 96 attention heads, 1M context로 설명한다. 공식 attention 명칭은 **KDA와 Gated MLA**다. `KLA`로 부르거나 69 KDA + 24 Gated MLA를 근거 없이 다른 명칭으로 바꾸지 않는다. [R25]

공개 README의 layer/head summary만으로 Gated MLA의 정확한 per-token cache tensor byte를 산출할 수 없으므로 수치를 추정하지 않는다. KDA layer의 recurrent/convolution states는 full KV와 분리해 구현 config에서 계산해야 한다. 공식 public checkpoint와 deployment engine의 support, quantization path 및 cache layout도 일치하는지 배포 시 확인해야 한다.

## 11.3 GLM-5.3와 GLM-5.3-Flash 구분

기준일 현재 공개 자료는 GLM-5.3과 GLM-5.3-Flash를 별도 checkpoint로 다룬다. GLM-5.3은 GLM-5.2 계열의 동일 base architecture에 post-training을 적용했다는 공식 발표/공개 기술 설명이 확인된다. **GLM-5.3-Flash는 동일한 모델이라고 단정할 수 없다.** 공개 config/Model coverage는 Flash에 45-layer hybrid 구조(34 KDA + 11 KPool-indexed DSA), 320B total/18B active, multimodal 구성 등을 보고한다. GLM-5.3의 78-layer MLA/DSA 계열과 Flash의 45-layer KDA/DSA 구성은 구별해야 한다. 공식 Z.ai 자료가 특정 Flash config를 명시하지 않는 숫자는 supporting implementation docs에 근거한 값임을 표시한다. [R26, R27]

## 11.4 DeepSeek-V4-Pro와 V4-Flash, V4.1-Flash

DeepSeek 공식 V4 공개 자료는 Pro와 Flash를 MoE 및 1M context preview family로 설명하고 **SWA + CSA + HCA** 계열의 hybrid attention을 명시한다. CSA는 Compressed Sparse Attention, HCA는 Heavily Compressed Attention으로 부른다. vLLM model recipe와 HF config는 특정 공개 checkpoint의 cache/model serving 구성에 대한 implementation evidence다. [R17, R18, R28]

V4.1-Flash 문서는 CED(Causal Encoder-Decoder), CSA2의 Full/Reindex/Reuse mode, cross-layer main KV/index sharing, hierarchical sparse indexer, FP4 KV cache, 그리고 SWA Bounded Replay를 공개 주장으로 설명한다. 보도문/논문 초록에서 말하는 890 bytes/token 및 V4-Flash 대비 cache 절감률은 **그 V4.1-Flash 보고서의 global/persistent cache 정의와 해당 checkpoint 범위**에만 적용한다. 이는 모든 CSA2 모델의 공식 보장 bytes/token이 아니다. `Bounded Replay`는 최근 window를 replay해 SWA KV state를 재구성한다는 배포 최적화 설명이며, global KV의 SRAM 상주를 뜻하지 않는다. 현재 보고서·HF 모델 카드에 공개된 claim과 실제 deployment engine의 작동 경로는 분리해서 기록한다. [R29]

## 11.5 공개 모델 구조 비교표

| 모델/정확한 버전 | Dense/MoE | 공개 확인된 Attention | layer pattern/config | KV/state 영향 | 근거와 확실성 |
|---|---|---|---|---|---|
| Qwen3.5-27B | dense 27B | 3:1 Gated DeltaNet / Gated Attention | 64 = 48 GDN + 16 full; full Q24/KV4/d256 | full-KV는 16 layer만 계산; recurrent+conv state 추가 | 공식 model card/config [R12] |
| Qwen3.6-27B | dense 27B | 같은 Qwen3.5 architecture family라 공개 Transformers 문서 | config/version별 확인 | Qwen3.5 동일이라 보고된 범위, 배포 config 확인 필요 | 공식 문서 [R12] |
| Kimi K3 | MoE 2.8T/104B active | KDA + Gated MLA; AttnRes도 별도 depth mixing | 93, 1 dense, 69 KDA + 24 Gated MLA | recurrent state와 latent cache 혼합; exact bytes 공개요약만으로 불가 | 공식 README [R25] |
| GLM-5.2 | MoE | MLA + DSA/IndexShare | 공개 model card/config 기준 78 layer family | MLA cache + sparse index/cache reuse | 공식 config/tech report [R26] |
| GLM-5.3 | GLM-5.2 same base, post-trained (공개 claim) | MLA + DSA 계열 | 정확한 release config를 기준으로 함 | GLM-5.2 계열 cache; version check | 공식 발표 + config [R26] |
| GLM-5.3-Flash | MoE 320B/18B active | KDA + KPool-indexed DSA | 45 layer: 34 KDA + 11 DSA (지원 구현 문서) | recurrent state + sparse MLA KV/index cache | Z.ai card 및 NVIDIA NeMo/vLLM recipe [R27] |
| DeepSeek-V4-Pro preview / checkpoint-specific | MoE | SWA + CSA + HCA | exact schedule은 report/config에서 checkpoint별 확인 | local window + compressed/indexed branches | 공식 report, config [R17, R18] |
| DeepSeek-V4-Flash-0731 | MoE 284B/13B active | SWA + CSA + HCA | 0731 release; base dimensions 확인은 config 참조 | 보고서 설정 기준 long-context KV reduction | official release/model card [R17, R28] |
| DeepSeek-V4.1-Flash | MoE | CED + CSA2 + SWA Bounded Replay | 40-layer encoder/decoder (paper/model card scope) | 890 B/token global cache claim; bounded replay storage 별도 | official model card/paper abstract [R29] |
| Llama 4 Scout/Maverick | MoE family | GQA + local/global attention details vary by release | official config must distinguish model variants | local attention cache bounded in local layers; GQA in global | Meta model card/config [R30] |
| Gemma 3 | dense/MoE varies by size | alternating local sliding-window and global attention | sizes/config specify exact schedule | local layers cap cache; global layers grow with T | Google official model card [R31] |
| Mistral Large 2 / Mixtral releases | dense or MoE by model | public configs commonly GQA; do not generalize whole brand | checkpoint config required | KV heads/layers from exact config | official config/model card [R32] |
| GPT-4/4o/5 family | proprietary | attention/KV design not publicly confirmed in sufficient detail | public architecture details unavailable | public cache bytes cannot be derived | 공개 자료로 확인 불가 [R33] |
| Claude 3.x/4.x family | proprietary | internal attention mechanism not publicly confirmed | no public layer/head config | public cache bytes cannot be derived | 공개 자료로 확인 불가 [R34] |
| Gemini 2.x/3.x family | proprietary | public technical reports disclose selected methods, not all serving config | version-by-version disclosure | no uniform cache formula available | 공개 자료로 확인 불가 [R35] |

**운영 대조 원칙:** model card, downloaded config, runtime framework's parsed config, and live engine initialization logs must agree. Product name similarity is not evidence of identical architecture. Checkpoint release suffixes, vision vs text-only, instruct vs base, and quantized variants can differ.

---

# 12. CUDA Graph의 역할과 한계

## 12.1 정확한 정의

CUDA Graph는 반복되는 GPU 작업과 dependency를 graph로 capture/instantiate한 뒤 replay하여 CPU 측 launch overhead를 줄이는 실행 기능이다. graph가 attention·GEMM·sampling을 새로 계산하거나 FLOPs를 줄이지 않는다. capture는 실행 경로를 기록하고, instantiate는 replay 가능한 graph 실행 구조를 준비하고, replay는 그래프에 등록된 node 작업들을 재실행한다. [R36]

## 12.2 Shape·주소가 안정적이어야 하는 이유

graph node는 kernel과 memory address, dependency가 캡처 당시의 실행 경로에 묶인다. 동적 batch/token shape, 매 step 달라지는 buffer 주소, 조건부 Python control flow는 그대로 재사용하기 어렵다. 엔진은 shape bucket, padded batch slot, 고정 input buffer, graph pool 등의 방법으로 실행 조건을 일정하게 만든다. 너무 많은 graph variant를 캡처하면 graph memory pool과 capture warmup 비용이 증가한다.

## 12.3 Prefill과 decode

- **Decode:** 매 step 한 token 처리가 반복되므로 고정/padded batch와 안정된 buffer를 이용한 graph replay가 launch overhead를 줄이기 쉽다.
- **Prefill:** prompt 길이는 가변적이므로 길이/batch bucket, piecewise graph, chunked prefill 등의 실행 방식이 가능하다. Bucket을 많이 두면 graph pool 및 warmup 부담이 커지고, padding은 추가 계산을 수행할 수 있다.
- **Graph는 workload를 사라지게 하지 않는다.** memory-bandwidth-bound KV read, collective, 긴 GEMM의 FLOPs를 줄이는 기능이 아니며, capture가 지원하지 않는 operation은 graph 경로에서 fallback될 수 있다.

vLLM 최신 문서는 `FULL`, `FULL_DECODE_ONLY`, `PIECEWISE` 등 capture mode와 version-specific behavior를 설명한다. 2026-11-02 기준 문서 페이지는 계속 변할 수 있으므로, 실제 배포 버전의 vLLM 문서를 확인한다. [R36]

---

# 13. 메모리·통신 용량 계산 사례

## 13.1 기준 예: Qwen3.5-27B, 170K + 2K

앞서 확인한 full-attention 16 layers, 4 KV heads, head dimension 256을 기준으로 한다. BF16 cache raw는 `64 KiB/token`; 172K token에서 `10.50 GiB`/request다. FP8 cache 저장을 실제로 지원하고 scale/metadata 등 별도 비용을 빼면 논리 payload는 반이 되어 약 `5.25 GiB`/request다. Full attention layer 수가 context length 전체 cache를 만드는 대상이다. Gated DeltaNet recurrent state와 convolution state는 별도 추가값이며, 정확한 runtime cache layout 없이 이를 0으로 처리하지 않는다. [R12]

### 운영 VRAM budget 식

`VRAM_total ≈ W_resident + KV_full + S_recurrent + A_peak + workspace + graph_pool + allocator_reserve + metadata + fragmentation`

이 항목을 하나의 “모델 크기” 수치로 뭉치지 않는다. weight memory는 quantization에 따라 달라지고, KV/state는 동시 요청·길이, activation/workspace는 batch·prefill chunk·kernel에 따라 바뀐다. CUDA Graph pool 및 allocator reserve는 프레임워크가 사전에 확보할 수 있다.

## 13.2 용량과 처리량은 별도 계산

위 표의 concurrency 8은 raw cache가 84.02 GiB BF16이라는 뜻이다. “8 concurrent면 tokens/s가 8배”라는 뜻이 아니다. throughput은 GPU compute/HBM bandwidth/KV read 양/batch/scheduler/collective/network와 요청 길이의 함수다. 실제 serving에서는 cache pool에서 block이 얼마나 배정되었는지, prefix sharing이 있는지, scheduler가 prefill을 chunk하는지 측정한다.

## 13.3 KV transfer link 이상 하한

예를 들어 한 요청의 BF16 full KV 10.50 GiB를 GPU/worker 사이로 완전히 옮긴다고 가정하자. 900 GB/s 링크를 10진 GB/s payload로 그대로 곱한 절대 이상 하한은 `약 11.27 GB / 900 GB/s ≈ 12.5 ms`다. 400 Gb/s network는 byte-rate 50 GB/s이므로 같은 11.27 GB를 옮기는 하한은 `≈225 ms`다. 이 값은 통신 방향/aggregate 수치를 개별 경로에 그대로 적용한 **설명용 계산**이고 실제 endpoint bandwidth가 아니다. Protocol header, synchronization, contention, NIC-GPU topology, DMA setup, transfer overlap, cache layout/scale/metadata, queueing으로 실제 시간은 더 크다. NVLink aggregate가 GPU pair의 단방향 payload bandwidth를 뜻한다고 오해하면 안 된다. [R5, R6]

## 13.4 TP에서 추가로 따질 항목

- weight shard: weight 용량은 local rank별로 줄 수 있으나 replicated embedding/norm/router 등은 모델 구현에 달렸다.
- KV shard/replica: local KV bytes는 head mapping 및 cache implementation에 따른다. GQA head가 rank보다 적으면 replication 가능성이 있다.
- collectives: layer 사이 activation/reduction traffic은 token 수·hidden dimension·TP degree·backend topology에 따라 발생한다.
- latency: 매 token collective가 있다면 decode latency에 비중이 클 수 있다. prefill에서는 계산과 통신이 겹칠 여지가 다르다.
- 멀티노드: NIC throughput만 보지 말고 GPU-NIC affinity, rail count, RDMA mode, switch oversubscription, NCCL algorithm을 함께 확인한다.

## 13.5 Prompt/output/concurrency 사례 감각

1) prompt가 170K로 길면 첫 응답 이전의 prefill compute와 cache fill이 크다.  
2) output이 2K면 생성 내내 cache가 늘고 매 step마다 과거 정보 조회가 반복된다.  
3) 동시성이 16이면 동일한 모델 weight는 공유해도 각 요청별 KV/state가 필요하다.  
4) max-model-len을 크게 열어둔 값이 곧 모든 요청의 실제 할당량은 아니지만, 엔진의 cache block pool은 GPU 잔여 메모리를 기준으로 동시 수용 가능 token을 계산한다.  
5) 실무 capacity planning은 예상 길이분포와 활성 요청 수의 p95를 기준으로 하고, 장애/replica headroom 및 prefill peak를 남긴다.

---

# 14. 운영 의사결정 흐름과 점검표

## 14.1 요청에서 구성을 고르는 순서

1. 사용자 SLO를 묻는다: TTFT, TPOT/ITL, p95/p99, 동시성, 요청 도착률.
2. prompt/output 분포와 최대 token 길이, prefix reuse율, cancellation을 확인한다.
3. 정확한 model checkpoint/config, dtype/quantization, attention layer pattern을 고정한다.
4. weight가 한 GPU에 들어가는지, KV/state를 남긴 뒤 동시성 요구를 만족하는지 계산한다.
5. capacity 부족이면 먼저 weight quantization, KV dtype, context cap, batching, cache block sizing, model variant를 평가한다.
6. 모델이 나뉘어야 하면 TP/PP/EP를 선택하고 NVLink·PCIe·IB topology 및 collective 비용을 측정한다.
7. prefill·decode 혼합 부하로 TTFT/ITL, HBM BW, cache allocation, OOM/eviction을 검증한다.

## 14.2 하드웨어 지표 점검표

| 관찰 | 가능한 원인 | 후속 계측 |
|---|---|---|
| GPU util 높고 tokens/s 낮음 | memory stalls, 작은 kernel 반복, collective, 낮은 occupancy | SM/Tensor utilization, HBM BW, NCCL 시간, kernel profile |
| TTFT만 높음 | 긴 prefill, queueing, tokenization | prefill tokens/s, scheduler queue, chunk sizes |
| ITL이 context 길이에 따라 느려짐 | full KV read, bandwidth saturation | bytes/token, HBM BW, KV dtype, batch |
| context 짧은 요청도 동시성 낮음 | weight/activation/workspace/graph pool 여유 부족 | memory breakdown, captured graph sizes |
| TP 확장 시 성능 악화 | collective overhead/link bottleneck | NCCL trace, topology, rank mapping |
| GPU 메모리 여유인데 OOM | allocator reserve, transient activation, fragmentation | peak memory, block utilization, graph pool |

## 14.3 올바른 보고 문장

“GPU 사용률이 90%이므로 compute-bound”라고 쓰지 말고, “고정된 prompt/output 분포에서 SM active 90%, Tensor Core pipe 45%, HBM read 88% peak, NCCL 12% wall time였으며, batch 증가 시 ITL은 이만큼 악화됐다”처럼 **측정 관찰과 원인 가설을 분리**한다.

---

# 15. Attention·Serving 최적화 계보

| 시기/아이디어 | 해결하려 한 문제 | 제안/계층 | 남은 trade-off |
|---|---|---|---|
| 2017 Transformer attention [R1] | recurrence 없이 병렬 sequence modeling | Q/K/V scaled dot-product, multi-head | full attention 계산·KV가 길이에 따라 증가 |
| 2019 MQA [R13] | decode에서 K/V bandwidth 감소 | 여러 Q head가 KV head 하나 공유; 구조/학습 변화 | 공유에 따른 표현 제약·uptraining 필요 |
| 2023 GQA [R14] | MQA와 MHA 사이의 실용적 선택 | query groups별 KV head 공유 | 적절한 group 수와 구현이 중요 |
| 2022 FlashAttention [R7] | HBM IO가 attention 병목 | exact attention을 tiling/online softmax로 계산하는 IO-aware kernel | 모델 cache 자체를 줄이지 않음; shape/backend 의존 |
| 2023 PagedAttention [R9] | KV allocation fragmentation, dynamic request memory | KV를 block 단위로 매핑해 allocation/공유 효율화 | attention 수학을 바꾸지 않으며 cache 총 raw bytes를 줄이는 건 아님 |
| 2022–24 MLA [R15] | KV cache bandwidth·용량 | 저랭크 latent 저장, positional path decoupling, absorption | 구조/커널 복잡성 및 추가 projection |
| local/sparse/selected attention [R16, R24] | context 길이에 따른 모든 token 접근 비용 | SWA, sparse indexer, retrieval | 멀리 있는 정보 접근 제한/검색 비용 |
| 2023 Mamba/SSM [R19] | 긴 sequence에서 attention의 quadratic cost | input-selective recurrent state + hardware-aware scan | compressed state의 표현/parallelism 특성 |
| 2024 Mamba-2 [R20] | SSM과 attention 실행 효율 관계 | State Space Duality와 structured recurrence | state 크기·hardware layout을 맞춰야 함 |
| 2024 DeltaNet / Gated Delta Networks / KDA [R21–R23] | state update 능력과 long-context 효율 | fast-weight matrix state, delta correction, gates | kernel·state memory·학습 안정성 복잡 |
| 2026 DeepSeek V4 hybrid [R17] | million-token context의 cache/compute 부담 | SWA + CSA + HCA 등 압축·선택 | indexer/압축 손실/구현 metadata 비용 |

**반복해서 기억할 구분:** MQA/GQA/MLA/SWA/CSA/HCA/linear attention은 모델이 정보를 표현·접근하는 **구조적 선택**이다. FlashAttention은 수학적으로 같은 attention을 더 적은 HBM IO로 계산하는 **kernel 실행 최적화**다. PagedAttention은 KV block을 할당·공유·관리하는 **serving memory management**다. 세 최적화 계층은 조합할 수 있지만 서로 대체하지 않는다. [R7, R9]

---

# 16. 참고: P/D disaggregation

P/D disaggregation은 prefill과 decode의 실행 자원을 분리해 각 단계를 별도 worker pool에서 처리하는 serving architecture다. Prefill worker가 만든 KV/recurrent state를 decode worker에 넘기므로 계산량·SLO·routing 이점을 얻을 수 있지만, state transfer byte와 latency, topology, cache layout 호환, scheduler·failure handling 비용이 추가된다. 같은 세션에서 연결된 이전 대화 상태가 필요하므로 KV ownership과 cancellation/retry를 다뤄야 한다. 자세한 Mooncake/NIXL connector와 P/D scheduler는 이 교재의 중심 범위가 아니며 다음 발표 주제로 분리한다. [R37]

---

# 17. 발표자 자기 점검

- [ ] FLOP(s), VRAM capacity, HBM bandwidth, GPU link bandwidth, latency를 각각 다른 질문에 답하는 지표로 설명할 수 있다.
- [ ] GEMM과 Q/K/V, output, MLP projection이 `XW` 관계임을 설명할 수 있다.
- [ ] Tensor Core가 GEMM을 빠르게 하더라도 전체 GPU 계산을 대신하지 않음을 말할 수 있다.
- [ ] GPU util과 Tensor Core saturation을 구분해 profiler 지표를 선택할 수 있다.
- [ ] NVLink/NVSwitch와 IB/RDMA가 서로 다른 topology 범위에서 어떤 데이터를 옮기는지 그릴 수 있다.
- [ ] TP가 계산량을 줄이는 동시에 layer 단위 collective를 추가하는 이유를 설명할 수 있다.
- [ ] Prompt prefill이 왜 큰 GEMM을 만들고 decode가 왜 weight/KV bandwidth에 민감한지 설명할 수 있다.
- [ ] TTFT와 TPOT/ITL을 모델 계산 단계 및 서비스 queue와 연결할 수 있다.
- [ ] KV 기본식의 각 항을 단위로 검산하고, 2가 K와 V를 나타낸다고 말할 수 있다.
- [ ] GQA KV head가 TP rank에 복제될 수 있어 GPU 개수로 단순 나누지 않는 이유를 설명할 수 있다.
- [ ] MLA latent cache와 MQA/GQA의 KV head sharing이 다른 설계임을 설명할 수 있다.
- [ ] FlashAttention, PagedAttention, GQA를 각각 kernel·allocator·architecture 계층으로 분류할 수 있다.
- [ ] Hybrid model의 한 token이 모든 layer를 통과해도 attention/state 형식은 layer마다 달라질 수 있음을 설명할 수 있다.
- [ ] Qwen3.5-27B의 48 GDN/16 full 구성으로 64 KiB/token BF16 raw full-KV 계산을 재현할 수 있다.
- [ ] 모델 card의 bytes/token 주장과 allocator/scale/workspace를 포함한 실제 VRAM 사용량을 구분할 수 있다.
- [ ] 공개 자료 없는 GPT·Claude 내부 구조를 추측하지 않고 “공개 자료로 확인 불가”라고 표시할 수 있다.
- [ ] 고객의 요구를 SLO, 길이 분포, 동시성, GPU topology 질문으로 바꿔 배포안을 설명할 수 있다.

---

# 18. 주석이 포함된 참고문헌

아래 자료는 2026-11-02 확인 기준으로 선정했다. 공식 링크는 자료의 이름과 주장 범위를 확인하기 위한 원문이다. 모델 release가 바뀌면 동일 계열 이름이라도 config를 다시 확인한다.

## 기본 Attention·KV·최적화

- **[R1] Vaswani et al., “Attention Is All You Need,” NeurIPS 2017.** [논문](https://arxiv.org/abs/1706.03762). Scaled dot-product attention, Q/K/V 및 multi-head 정의를 제공한다. 공통 수식과 projection 설명의 근거.
- **[R2] Williams et al., “Roofline: An Insightful Visual Performance Model for Multicore Architectures,” 2009.** [논문](https://doi.org/10.1145/1498765.1498785). compute peak와 memory bandwidth/산술 강도를 비교하는 성능 상한 관점의 근거.
- **[R3] NVIDIA, CUDA C++ Programming Guide, 13.x.** [공식 문서](https://docs.nvidia.com/cuda/cuda-programming-guide/). thread/block/warp, memory hierarchy와 실행 모델의 근거. 세대별 세부 사양은 해당 compute capability 표 참조.
- **[R4] NVIDIA, CUDA Programming Guide: Advanced Kernel Programming / Compute Capabilities.** [공식 문서](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html). register·shared memory·SM 실행 자원과 Tensor operation capability를 설명한다.
- **[R5] NVIDIA HGX H100/H200/B200 Reference Architecture.** [공식 구성 문서](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html). HGX node의 GPU/SXM/NVLink/NVSwitch 구성 범위를 확인한다. 단일 GPU 스펙과 시스템 총량을 혼합하지 않는다.
- **[R6] NVIDIA NCCL Documentation.** [공식 문서](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/). all-reduce/all-gather/reduce-scatter 및 GPU collective의 구현 범위. 이 교재의 전송량식은 별도 단순화 산술 모델.
- **[R7] Dao et al., “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” NeurIPS 2022.** [논문](https://arxiv.org/abs/2205.14135). Attention 식을 바꾸지 않고 HBM↔SRAM IO를 줄이도록 tiling/online softmax를 사용하는 근거.
- **[R8] Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” SOSP 2023.** [논문](https://arxiv.org/abs/2309.06180). KV cache block 관리와 동적 batch/fragmentation 문제의 근거.
- **[R9] Yu et al., “Orca: A Distributed Serving System for Transformer-Based Generative Models,” OSDI 2022.** [논문](https://www.usenix.org/conference/osdi22/presentation/yu). iteration-level scheduling/continuous batching의 기원과 serving batching 문맥.
- **[R10] vLLM Documentation, Chunked Prefill.** [공식 문서](https://docs.vllm.ai/en/latest/features/chunked_prefill/). 긴 prefill scheduling과 decode interleave의 설정·버전 동작 확인용.
- **[R11] vLLM Documentation, KV Cache and Hybrid Cache Manager.** [KV cache 문서](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/). cache group/type, model-specific KV allocation의 구현 근거. 버전별로 갱신.

## MQA/GQA/MLA 및 recurrent/hybrid

- **[R12] Qwen Team, Qwen3.5-27B model card; Hugging Face Transformers Qwen3.5 model doc.** [모델 카드](https://huggingface.co/Qwen/Qwen3.5-27B), [모델 문서](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5). 64 layer, 3:1 hybrid stack, 48/16 DeltaNet heads, Q24/KV4 full attention의 공식 config/문서 근거. Qwen3.8-27B 명칭은 이 기준에서 확인되지 않음.
- **[R13] Shazeer, “Fast Transformer Decoding: One Write-Head is All You Need,” 2019.** [논문](https://arxiv.org/abs/1911.02150). MQA의 shared KV head와 decode bandwidth 동기를 제안.
- **[R14] Ainslie et al., “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints,” EMNLP 2023.** [논문](https://arxiv.org/abs/2305.13245). MHA/MQA 중간 형태와 uptraining 평가 근거.
- **[R15] DeepSeek-AI, “DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model,” 2024.** [논문](https://arxiv.org/abs/2405.04434). MLA low-rank joint KV compression, decoupled RoPE, absorption 경로의 원 보고서.
- **[R16] Jiang et al., “Mistral 7B,” 2023.** [논문](https://arxiv.org/abs/2310.06825). sliding-window attention 사례의 공개 근거. 일반 SWA 공식은 windowed KV 정의에서 유도.
- **[R17] DeepSeek-AI, “DeepSeek-V4 Technical Report,” 2026.** [공식 report entry](https://www.deepseek.com/en/transparency/), [arXiv](https://arxiv.org/abs/2606.19348). V4의 SWA/CSA/HCA architecture 및 보고서 수치의 1차 출처.
- **[R18] Hugging Face Transformers, DeepSeek-V4 implementation docs.** [공식 구현 문서](https://huggingface.co/docs/transformers/model_doc/deepseek_v4). layer type, compression rate, cache branch와 causal mask를 설명하는 implementation source. 논문 claim과 구분.
- **[R19] Gu & Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” COLM 2024.** [논문](https://arxiv.org/abs/2312.00752). selective SSM recurrence와 hardware-aware scan.
- **[R20] Dao & Gu, “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality,” ICML 2024.** [논문](https://proceedings.mlr.press/v235/dao24a.html). Mamba-2 및 SSM/attention duality.
- **[R21] Schlag et al., “Linear Transformers with Learnable Kernel Features are Better Recurrent Models,” 2021 / DeltaNet line.** [DeltaNet paper](https://arxiv.org/abs/2406.06484). Delta-rule fast-weight memory update와 recurrent attention 근거.
- **[R22] Yang et al., “Gated Delta Networks: Improving Mamba2 with Delta Rule,” ICLR 2025.** [논문](https://arxiv.org/abs/2412.06464). gating과 delta state update를 결합한 구조.
- **[R23] Moonshot AI, “Kimi Linear: An Expressive, Efficient Attention Architecture,” 2025.** [논문](https://arxiv.org/abs/2510.26692). KDA 및 MLA와의 hybrid architecture 계보.
- **[R24] DeepSeek-AI, “Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention,” 2025.** [논문](https://arxiv.org/abs/2502.11089). sparse attention 및 index-selected context 설계 계보.

## 모델 사례와 구현 문서

- **[R25] MoonshotAI, Kimi K3 official README/model card.** [공식 저장소](https://github.com/MoonshotAI/Kimi-K3). 93 layers, 69 KDA + 24 Gated MLA, 1M context와 model naming 근거. 공개 summary에 없는 latent cache dimensions는 미확정 처리.
- **[R26] Z.ai, GLM-5 official repository and GLM-5.3 announcement; GLM-5.2/5.3 model configs.** [공식 저장소](https://github.com/zai-org/GLM-5), [GLM-5.3-Flash card](https://huggingface.co/zai-org/GLM-5.3-Flash). Flash와 base checkpoint를 구분할 공식 근거. 일부 상세 decoder layout은 supporting implementation docs일 수 있음.
- **[R27] NVIDIA NeMo AutoModel, GLM-5.3-Flash model coverage.** [공식 구현 문서](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/vision-language-models/thudm/GLM-5.3-Flash). 45 decoder layer, 34 KDA + 11 KPool-DSA 등 지원 구현 config를 설명한다. 이는 Z.ai 원 논문과 증거 유형을 구분한다.
- **[R28] DeepSeek, DeepSeek-V4 Preview announcement and checkpoint cards.** [공식 발표](https://deepseek.com/en/news/v4-preview/), [공식 transparency center](https://www.deepseek.com/en/transparency/). Pro/Flash release family 및 preview의 확인 근거.
- **[R29] DeepSeek-AI, “DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression,” 2026.** [공식 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [paper page](https://huggingface.co/papers/2609.19969). CED/CSA2/Bounded Replay/890 B per token 등은 이 문서와 checkpoint scope에 한정.
- **[R30] Meta, Llama 4 Scout/Maverick official model cards/configs.** [공식 Hugging Face org](https://huggingface.co/meta-llama). Variant마다 config를 확인할 것. 브랜드 단위 일반화 금지.
- **[R31] Google, Gemma 3 technical report/model cards.** [공식 model card](https://ai.google.dev/gemma/docs/core). local/global attention 공개 내용을 확인하되 size별 config를 지정.
- **[R32] Mistral AI model cards and model configuration files.** [공식 모델 목록](https://huggingface.co/mistralai). 해당 checkpoint config에만 attention/head 구조를 귀속.
- **[R33] OpenAI public model and system documentation.** [공식 문서](https://platform.openai.com/docs/models). 제품 특성은 공개하나 내부 layer/head/cache architecture는 공개 범위가 제한됨.
- **[R34] Anthropic model documentation.** [공식 문서](https://docs.anthropic.com/en/docs/about-claude/models/overview). Claude 내부 attention/KV 구성은 공개 model config가 없어 추측하지 않음.
- **[R35] Google Gemini model documentation and technical reports.** [공식 모델 문서](https://ai.google.dev/gemini-api/docs/models). 특정 보고서 공개 범위와 production serving internals를 구분.

## CUDA Graph·Serving·P/D

- **[R36] vLLM, CUDA Graphs design docs.** [공식 문서](https://docs.vllm.ai/en/latest/design/cuda_graphs/). vLLM 버전별 decode/prefill/mixed capture strategy를 확인한다. CUDA Graph 자체 정의는 NVIDIA CUDA Runtime documentation을 함께 참조.
- **[R37] Mooncake/NIXL project documentation.** [Mooncake](https://github.com/kvcache-ai/Mooncake), [NIXL](https://github.com/ai-dynamo/nixl). KV transfer/connector context의 참고 링크. P/D는 본 교재에서 개요만 언급.
- **[R38] NVIDIA H100/H200 GPU datasheets and HGX platform specifications.** [H100](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet), [H200](https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23), [HGX](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html). Memory capacity/bandwidth, NVLink, system aggregate 정의는 SKU에 따라 교차 확인.

---

# 19. 용어집

| 용어 | 한 줄 정의 |
|---|---|
| Activation | 현재 계산 중 layer 사이를 오가는 중간 표현 tensor |
| Attention head | query/key/value projection과 attention 계산의 한 feature subspace |
| Bandwidth | 단위 시간에 옮길 수 있는 데이터 양 |
| Batch / concurrency | 한 step 또는 시스템이 함께 처리하는 요청/sequence 수; 두 정의가 항상 같지는 않음 |
| Causal mask | 미래 token이 과거 위치 계산에 들어오지 않도록 하는 mask |
| Decode | 이미 읽은 context 뒤에 token을 autoregressive하게 생성하는 단계 |
| Dense model | 모든 layer의 dense parameter path가 각 token 계산에 관여하는 모델 |
| FLOP / FLOPS | floating-point operation 횟수 / 초당 처리량 |
| GEMM | 일반적인 행렬-행렬 곱 연산 |
| HBM | GPU가 주 연산 메모리로 쓰는 고대역폭 메모리 |
| HCA / CSA | 각각 DeepSeek-V4 문서의 Heavily Compressed Attention / Compressed Sparse Attention |
| Hybrid attention | 서로 다른 attention/state layer를 하나의 model stack에서 결합 |
| ITL / TPOT | 연속 생성 token 간 지연 / 출력 token당 평균 시간; 계측 정의 확인 필요 |
| KV cache | 과거 token의 key/value를 저장하여 decode에서 재사용하는 cache |
| Latency | 요청/연산/전송의 시작부터 완료까지 걸리는 시간 |
| MLA | Multi-Head Latent Attention, low-rank latent KV 표현을 저장하는 구조 |
| MoE | router가 token마다 일부 expert를 선택하는 sparse feed-forward 구조 |
| PagedAttention | KV cache를 block 기반으로 할당·관리하는 serving 구현/알고리즘 |
| Prefill | 입력 prompt positions를 model에 넣어 context state/KV를 만드는 단계 |
| Projection | learned matrix로 hidden vector를 새 feature 공간으로 변환 |
| Recurrent state | sequence 과거를 요약해 다음 token마다 갱신하는 state tensor |
| SM | CUDA block을 실행·스케줄하는 Streaming Multiprocessor |
| SWA | Sliding Window Attention, 최근 일정 window를 보는 attention |
| Tensor Core | NVIDIA GPU의 matrix multiply-accumulate 전용 실행 유닛 |
| TTFT | 요청 시작에서 첫 출력 token까지 걸린 시간 |
| TP | Tensor Parallelism, layer tensor 연산을 여러 GPU로 분할하는 방식 |
| VRAM | GPU가 접근하는 device memory의 용량을 일컫는 실무 용어 |

