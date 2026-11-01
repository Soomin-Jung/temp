---
title: "LLM 서빙과 Attention 메모리 종합 마스터 노트"
subtitle: "Qwen3.8 27B 밀집 모델로 이해하는 계산 메모리 통신"
date: "자료 확인 기준 2026년 11월 1일"
lang: ko-KR
---

이 교재의 목적은 고객이 “Qwen3.8-27B를 배포해 주세요”라고 요청했을 때, 모델 이름과 GPU 개수 이상의 근거로 답할 수 있게 하는 것이다. 입력 길이와 동시성에 따라 필요한 메모리와 데이터 이동량이 어떻게 달라지는지 계산하고, Attention 설계가 그 비용을 어느 축에서 줄이는지 이해한다.

핵심 사례는 **Qwen/Qwen3.8-27B의 밀집형 언어 백본을 통한 기본 텍스트 추론**이다. 공개 config에는 64개 층, Gated DeltaNet 48개와 Gated Attention 16개의 혼합 구조가 명시되어 있다. 모델 전체에는 비전 인코더도 있지만, 본문의 요청 상태와 계산 예제는 텍스트 경로를 대상으로 한다. [13–15]

핵심 결론은 다음과 같다. Attention은 과거를 읽는 연산인 동시에 과거를 저장하는 규칙이다. Head를 공유하는 GQA, token 표현을 압축하는 MLA, 조회 범위를 제한하는 SWA, sequence를 압축하는 CSA/HCA, 이력을 상태로 접는 Gated DeltaNet은 서로 다른 비용을 줄인다. 어떤 방법도 모델 품질과 GPU 실행 비용을 함께 평가하는 과정을 대신하지 않는다.

문서의 수식은 벡터와 행렬의 차원을 정의한 뒤 유도한다. **공식 설정**, **공식 구현**, **수식에서 유도한 값**, **설명용 가정**을 해당 문단에서 구분한다. 산술값은 raw tensor 기준이며 실제 엔진의 할당량과 동일하지 않다. 본문 인용 번호는 끝의 주석 참고문헌과 대응한다.



# 한 페이지 핵심 요약

**고객에게 답할 근거는 모델이 만드는 tensor와 그 이동 경로다.** 같은 27B라도 긴 prompt와 높은 동시성은 짧은 단일 요청과 다른 구성이 필요하다.

| 구분 | 이 교재에서 설명할 핵심 |
|---|---|
| 계산 | Projection은 GEMM이고 attention은 score와 value 혼합이다 |
| 저장 | Weight는 모델 공유 상태이고 KV와 recurrent state는 요청별이다 |
| 이동 | HBM과 GPU 링크와 네트워크의 bandwidth는 서로 다른 경로다 |
| Prefill | 여러 prompt 위치를 처리하고 첫 출력의 logits를 만든다 |
| Decode | 새 token의 KV 또는 state를 갱신하며 다음 token을 예측한다 |
| 구조의 차이 | Head 공유와 latent 압축과 위치 선택과 recurrence는 다른 선택이다 |
| 실행의 차이 | FlashAttention과 PagedAttention과 Graph는 다른 최적화 층위다 |

Qwen3.8-27B의 64개 층 중 full attention은 16개다. BF16 full KV는 **64 KiB/token**이며 172K 예산에서 약 **10.498 GiB/request**다. 48개 GDN 층의 FP32 recurrent state는 raw 기준 **144 MiB/request**이고 convolution state는 별도다.

이 수치는 capacity 계산의 재료다. 실제 속도를 답하려면 weight/KV 읽기, GEMM, state update, GPU collective와 queueing을 함께 측정해야 한다. 메모리에 들어간다는 사실과 SLO 안에서 처리한다는 사실은 다르다.

# 목차

| 장 | 주제 |
|---:|---|
| 1 | 배포 요구를 하드웨어 질문으로 바꾸기 |
| 2 | 행렬 곱과 Projection의 기초 |
| 3 | GPU와 데이터 이동의 구조 |
| 4 | 밀집 모델의 Prefill과 Decode 실행 |
| 5 | Softmax Attention을 계산으로 이해하기 |
| 6 | KV Cache의 유도와 저장 정밀도 |
| 7 | MHA MQA GQA의 구조와 정량 비교 |
| 8 | MLA의 저랭크 표현과 Matrix Absorption |
| 9 | SWA와 Sparse와 Compressed Attention |
| 10 | Linear Attention을 Recurrent State로 유도하기 |
| 11 | Mamba와 Delta Rule과 Gated DeltaNet |
| 12 | Qwen3.8 27B의 층과 상태를 직접 계산하기 |
| 13 | FlashAttention과 PagedAttention과 CUDA Graph |
| 14 | 요청 길이와 동시성의 메모리 계산 |
| 15 | 분산 배포와 통신의 Trade off |
| 16 | 운영에서 확인할 항목과 해석 |
| 17 | Attention 최적화의 계보와 학습 지도 |
| 18 | 별도 참고 P D Disaggregation |
| 19 | 주석이 포함된 원문 참고자료 |
| 20 | 용어집과 발표자 확인 항목 |

# 1 배포 요구를 하드웨어 질문으로 바꾸기

## 1.1 고객과의 대화에서 시작하기

고객이 “좋다고 하니 이 모델을 배포해 달라”고 말하면 우리는 먼저 목표를 묻는다. 첫 답변을 빨리 받는 것이 중요한지, 긴 답변의 token 간격이 중요한지, 동시에 많은 요청을 처리하는 것이 중요한지를 구분한다. 이어서 prompt와 출력 길이의 분포, 동시 활성 요청 수, 요청 도착률, 허용 가능한 tail latency를 확인한다.

한 GPU에 모델이 들어간다고 배포 설계가 끝나는 것은 아니다. 살아 있는 요청들은 각자 KV cache와 recurrent state를 점유한다. 긴 prompt는 첫 응답 이전 계산을 늘리고, 긴 출력은 그 상태를 오래 유지하며 매 token의 조회를 반복한다. GPU를 여러 장 사용하면 한 장의 용량 제한을 넘을 수 있지만 통신과 동기화 비용이 추가된다.

| 요구 | 먼저 볼 비용 | 확인할 지표 |
|---|---|---|
| 첫 응답이 빨라야 함 | 대기열과 prompt 처리 | TTFT와 prefill 시간 |
| 생성이 매끄러워야 함 | decode step의 데이터 읽기와 실행 지연 | ITL과 TPOT의 분포 |
| 동시 처리량이 커야 함 | 상태 용량과 batch 처리 효율 | 활성 요청 수와 SLO 충족 처리량 |
| 긴 문맥을 써야 함 | KV 증가와 attention 조회량 | 실제 token 길이와 요청별 state bytes |
| 멀티 GPU를 써야 함 | shard 배치와 collective | topology와 통신 시간 |

**TTFT**는 요청부터 첫 token까지의 지연이다. **ITL**은 인접한 출력 token 사이의 간격이며, **TPOT**는 출력 token당 평균 시간을 가리키는 계측 관례다. 서비스마다 queue와 전송 시간을 포함하는 경계가 다를 수 있으므로 정의를 보고 비교한다. Throughput은 초당 처리량, goodput은 목표 SLO를 만족하는 유효 처리량이다. [27,28]

## 1.2 이 교재에서 계속 사용할 질문

계산을 볼 때는 “몇 FLOP인가”, 저장을 볼 때는 “몇 byte가 언제까지 살아 있는가”, 이동을 볼 때는 “어느 경로로 몇 byte를 몇 번 옮기는가”를 묻는다. 이 셋을 분리하면 고성능 GPU에서도 느린 이유와, 메모리는 남는데 동시성을 늘리기 어려운 이유를 설명할 수 있다.

기본 사례는 입력 170,000 token과 출력 2,000 token이다. 편의상 최대 cache 길이를 172,000으로 잡는다. 이는 길이 예산을 위한 보수적 계산이며, 마지막 출력 token을 다시 모델에 입력하지 않고 종료하는 일반 경로에서는 실제 cache 위치가 하나 적을 수 있다. 이 차이는 4장에서 실행 순서와 연결한다.

# 2 행렬 곱과 Projection의 기초

## 2.1 Token이 hidden vector가 되는 과정

Tokenizer는 텍스트를 정수 ID의 열로 만든다. Embedding은 각 ID를 길이 $d_{\mathrm{model}}$인 벡터로 바꾼다. 각 층은 이 벡터를 문맥에 맞게 변환하고, 마지막 층의 결과에서 vocabulary별 점수인 logits를 만든다. 같은 token ID라도 위치와 이전 문맥에 따라 층을 지난 hidden state는 달라진다. [1]

행렬을 읽기 위한 최소 규칙은 다음과 같다. $A\in\mathbb{R}^{m\times k}$와 $B\in\mathbb{R}^{k\times n}$은 안쪽 차원 $k$가 같으므로 곱할 수 있고 결과는 $m\times n$이다. 전치 $A^\top$는 행과 열을 바꾼다. 내적은 두 벡터를 하나의 점수로 만들고, 외적은 두 벡터로 행렬을 만든다.

$$
C_{ij}=\sum_{r=1}^{k}A_{ir}B_{rj},\qquad C\in\mathbb{R}^{m\times n}.
$$

각 $C_{ij}$를 만들려면 대략 $k$번의 곱과 덧셈이 필요하다. Multiply-add를 2 FLOP으로 세는 관례에서 큰 행렬 곱의 연산량은 약 $2mkn$ FLOP이다. 경계의 덧셈 하나 차이는 큰 행렬의 근사에서 생략한다.

## 2.2 GEMM과 GPU가 잘하는 일

GEMM은 General Matrix Multiply이며 보통 $C\leftarrow\alpha AB+\beta C$ 형태를 말한다. LLM의 선형층은 대부분 이 행렬 곱에 해당한다. 여러 token을 $X\in\mathbb{R}^{N\times d}$의 행으로 묶고, 학습된 weight $W\in\mathbb{R}^{d\times d_{\mathrm{out}}}$를 곱하면 $Y=XW$가 된다.

$$
\mathrm{FLOP}_{XW}\simeq 2Ndd_{\mathrm{out}}.
$$

Prefill은 한 요청에서도 많은 위치를 $N$에 묶을 수 있다. 한 token decode는 동시 요청이 적으면 $N$이 작다. 같은 weight를 읽더라도 큰 $N$에서는 많은 token이 weight tile을 재사용할 수 있어 연산기의 처리량을 활용하기 쉬워진다.

## 2.3 Projection은 역할을 가진 선형변환

Projection은 hidden representation을 학습된 가중치 행렬로 다른 feature 공간에 옮기는 것이다. 기하학에서 말하는 직교 투영처럼 반드시 차원을 줄이거나 어떤 축을 제거하는 연산은 아니다. 차원을 늘릴 수도 있다.

행 단위 입력 표기로 Q/K/V는 다음과 같다.

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

$W_Q$는 무엇을 찾을지, $W_K$는 어떤 기준으로 매칭할지, $W_V$는 매칭된 위치에서 어떤 정보를 전달할지를 학습한다. 구현에서는 세 선형변환을 한 큰 projection으로 묶어 계산한 뒤 view/split할 수 있다. 이 fusion은 수학적 역할을 합친다는 뜻이 아니라 실행과 메모리 접근을 효율화하는 방법이다.

Output projection은 head별 결과를 이어 붙여 residual stream으로 돌려보낸다.

$$
Y=\mathrm{Concat}(O_1,\ldots,O_{H_q})W_O.
$$

밀집형 gated MLP는 예를 들어 다음처럼 쓴다.

$$
U=XW_u,\quad G=\mathrm{SiLU}(XW_g),\quad
\mathrm{MLP}(X)=(U\odot G)W_d.
$$

$\odot$는 원소별 곱이다. 이 FFN gate와 attention output gate, recurrent state의 forgetting gate는 모두 “gate”라는 단어를 쓰지만 작용 위치와 의미가 다르다. 뒤 장에서는 무엇에 곱하는 gate인지 항상 명시한다.

## 2.4 Weight와 요청 상태를 구분하기

| 종류 | 무엇인가 | 수명과 증가 요인 |
|---|---|---|
| Weight | 학습된 projection과 정규화 등의 파라미터 | 모델을 적재한 동안 유지 |
| Activation | 현재 연산의 중간 hidden tensor | layer나 step 동안 주로 유지 |
| KV cache | attention에 재사용할 과거 위치의 K와 V | 요청 길이와 요청 수에 따라 증가 |
| Recurrent state | 과거를 누적한 제한 크기의 상태 | sequence별 유지와 갱신 |
| Convolution state | causal convolution의 최근 입력 기록 | kernel 길이만큼의 tail 유지 |
| Workspace | kernel이나 라이브러리가 쓰는 임시 공간 | shape와 backend에 따라 예약 |

Weight는 여러 요청이 같은 모델을 실행하며 공유한다. KV/state는 요청마다 다르다. Activation과 workspace는 처리 중인 batch와 prefill chunk에 따라 peak가 달라진다. 그래서 모델 parameter 수에 dtype byte를 곱한 값만으로 동시성을 계산할 수 없다.

# 3 GPU와 데이터 이동의 구조

CPU는 요청 수신과 tokenization, 스케줄링, GPU 작업 제출과 결과 전달을 주로 맡는다. GPU는 많은 token과 feature의 행렬 연산을 병렬 실행한다. 이는 일반적인 serving 배치이며 sampling 등 일부 작업의 실행 위치는 엔진마다 다르다. CPU의 제어 흐름 처리와 GPU의 대량 병렬 계산은 협력 관계다. GPU가 빨라도 host가 다음 작업을 제때 제출하지 못하면 전체 실행이 기다릴 수 있다. [19,22,28]

## 3.1 용량과 대역폭과 연산 성능

FLOP은 연산 횟수, FLOP/s 또는 FLOPS는 초당 연산 처리량이다. GB와 GiB는 저장량, GB/s와 TB/s는 데이터 이동 속도다. Latency는 전송이나 작업을 시작하고 완료하는 데 드는 시간이다. 이 지표들은 다른 제약을 설명한다.

십진수 $1\,\mathrm{GB}=10^9$ byte이고 $1\,\mathrm{GiB}=2^{30}$ byte다. GPU 제품 용량을 표기한 GB와 소프트웨어가 출력하는 GiB를 바꿔 쓸 때 단위를 함께 적는다.

Roofline 모델은 실제 처리량의 상한을 다음처럼 본다. [25]

$$
P\le\min(P_{\mathrm{peak}},\ B_{\mathrm{HBM}}\ I),\qquad
I=\frac{\mathrm{FLOP}}{\mathrm{HBM\ bytes\ moved}}.
$$

메모리가 초당 $B_{\mathrm{HBM}}$ byte를 공급하고 byte당 $I$ FLOP을 수행한다면 공급 가능한 연산은 초당 $BI$다. 큰 GEMM에서 tile을 재사용하면 $I$를 높일 수 있다. 반대로 매번 많은 weight/KV를 읽고 조금만 계산하면 연산기가 기다릴 수 있다. 예를 들어 설명용 $P_{\mathrm{peak}}=300$ TFLOP/s, $B=3$ TB/s라면 두 상한이 만나는 강도는 100 FLOP/byte다.

## 3.2 SM과 실행 단위

GPU는 여러 SM을 가진다. SM은 Streaming Multiprocessor이며 thread block을 받아 실행하는 스케줄링과 연산 자원의 묶음이다. CUDA의 논리적 계층은 grid, block, thread이고, NVIDIA warp는 32개 thread의 실행 묶음이다. 실제 SM 내부의 scheduler, 실행 유닛, register와 cache 구성은 GPU 세대와 compute capability에 따라 다르다. [19]

Register는 thread 가까이에 있는 작은 상태 공간이고, shared memory는 block이 협업하며 tile을 재사용하는 on-chip 메모리다. L1과 shared memory는 세대에 따라 일부 자원을 공유하며, L2는 GPU 전반에서 접근하는 cache 계층이다. HBM은 대형 weight와 KV/state 등을 보관하는 주 device memory다. Register/shared memory를 과도하게 쓰면 동시에 상주할 block 수가 줄어 occupancy가 낮아질 수 있다.

## 3.3 CUDA Core와 Tensor Core

Tensor Core는 지원하는 dtype과 형태의 행렬 multiply-accumulate를 높은 처리량으로 수행하는 유닛이다. GEMM kernel은 데이터를 tile로 가져오고 matrix instruction을 실행한다. 주소 계산, 분기, elementwise 연산, reduction과 정규화는 일반 SIMT 연산기와 여러 실행 파이프를 함께 사용한다. Tensor Core가 모든 GPU 작업을 대신하지 않는다. [19]

GPU-Util은 보통 샘플 구간에서 GPU가 작업을 실행 중이었는지 나타낸다. 95%라고 해도 Tensor Core가 peak FLOPS로 일한다는 뜻은 아니다. Memory stall, 작은 kernel의 반복, collective와 동기화, CPU launch overhead가 섞일 수 있다. SM active, tensor pipe, HBM bandwidth, occupancy, kernel timeline을 같이 봐야 한다.

![GPU 메모리와 노드 간 데이터 경로](master-notes-figures/hardware.svg){width=6.2in}

그림 1. HBM의 용량과 대역폭, GPU 사이 링크, 노드 사이 네트워크는 서로 다른 구간의 제약이다. Cache는 모든 load가 반드시 순서대로 통과하는 소프트웨어 pipeline을 의미하지 않는다.

## 3.4 HBM에 상주한다는 말의 의미

일반적인 GPU serving에서 KV tensor는 GPU HBM에 할당되어 decode step 사이에도 유지된다. 이때 “계속 올라와 있다”는 말은 GPU device memory에 보존된다는 뜻이다. 전체 KV가 SM의 SRAM에 상주한다는 뜻은 아니다.

Attention kernel은 필요한 일부 tile을 HBM/global memory에서 읽는다. 접근 정책과 locality에 따라 L2나 SM 근처 cache가 도움을 주고, kernel은 일부를 shared memory/register에서 계산한다. 긴 context의 KV는 on-chip cache보다 훨씬 클 수 있다. Tiling과 fusion의 효과는 이 이동 횟수와 재사용도를 개선하는 데 있다. [5,19]

CPU DRAM, NVMe, remote GPU의 KV는 별도 저장 계층이다. Offload는 GPU capacity를 확장하지만 필요한 순간 전송 또는 재계산 지연을 더한다. HBM의 KV와 remote cache의 KV를 같은 지연으로 접근한다고 보면 안 된다.

## 3.5 GPU 사이와 노드 사이 연결

NVLink는 GPU 사이의 고속 연결이고 NVSwitch는 여러 NVLink 경로를 연결하는 switching fabric이다. PCIe는 CPU, GPU, NIC 등을 잇는 범용 I/O 경로다. InfiniBand와 Ethernet/RoCE는 주로 노드 사이 fabric을 구성한다. RDMA는 NIC가 원격 메모리에 접근하는 방식이며, GPUDirect RDMA는 지원되는 환경에서 GPU memory를 직접 DMA 대상으로 사용해 host staging을 줄인다. [19–21]

| 제품 예시 | GPU당 메모리 | 명목 메모리 대역폭 | 배포 시 확인 |
|---|---|---|---|
| H100 SXM | 80 GB HBM3 | 약 3.35 TB/s | PCIe/NVL SKU와 구분 |
| H200 SXM | 141 GB HBM3e | 약 4.8 TB/s | HGX NVLink/NVSwitch topology |
| L40S | 48 GB GDDR6 | 약 864 GB/s | NVLink 미지원과 PCIe 경로 |

위 값은 공식 제품 자료의 해당 form factor 예시이며 실효 bandwidth가 아니다. GPU당 수치와 8-GPU node aggregate, 단방향과 양방향 합산을 구분한다. 예를 들어 NVLink 900 GB/s라는 aggregate 제품값을 그대로 한 pair의 단방향 payload bandwidth로 대입하지 않는다. 제품의 정확한 SKU와 실제 topology를 먼저 확인한다. [21]

# 4 밀집 모델의 Prefill과 Decode 실행

## 4.1 모든 층을 순서대로 통과한다

기본 decoder의 token hidden state는 첫 layer부터 마지막 layer까지 순서대로 진행한다. 각 층에서 attention 또는 recurrent token mixing을 거치고 residual, normalization, FFN을 수행한다. Hybrid 모델에서도 층을 건너뛰는 것이 아니라, 층마다 과거 정보에 접근하는 연산이 다를 수 있다. [1,13,15]

서로 다른 요청이나 token을 GPU에서 함께 실행할 수 있어도, 한 token의 layer $l+1$은 layer $l$의 결과에 의존한다. Pipeline parallelism은 층을 장치별 stage로 배치하며 이 dependency를 없애지 않는다.

## 4.2 Prefill과 첫 출력 token

Prompt가 $p_1,\ldots,p_N$이면 prefill은 이 위치들을 causal 조건으로 처리한다. Full-attention 층은 각 위치의 Q/K/V를 계산하고, 이후 decode에 쓸 K/V를 보존한다. Recurrent 층은 전체 sequence를 scan 또는 chunk 방식으로 처리해 마지막 상태를 얻는다. 모든 층의 모든 hidden activation을 영구 보존하는 것은 아니다.

기본적인 추론에서 **마지막 prompt 위치의 logits로 첫 출력 $y_1$을 선택할 수 있다**. 첫 출력이 나온 뒤, 다음 decode step에는 $y_1$을 입력해 $y_2$를 예측한다. 엔진의 계측 경계나 실행 분할은 다를 수 있지만, 첫 token을 만들기 위해 언제나 별도의 one-token decode를 더 해야 한다고 설명하면 off-by-one 혼동이 생긴다.

## 4.3 Decode에서 무엇이 추가되는가

Decode step에서 입력한 새 token은 각 full-attention 층에서 새 Q/K/V를 만든다. 현재 위치의 K/V를 해당 sequence cache에 추가하고, 새 Q는 유효한 과거와 현재 K/V를 참조한다. Recurrent 층은 그 token으로 자신의 state와 convolution tail을 갱신한다. 마지막 logits로 다음 token을 고른 뒤 이 과정을 반복한다.

최종 출력 $y_M$을 뽑고 종료한다면 일반적으로 $y_M$ 자체를 다시 입력하지 않으므로 cache에는 $N+M-1$개 위치가 들어갈 수 있다. 이 교재의 $T=N+M$은 최종 길이 예산을 위한 상한 가정이다. 운영 계산은 엔진의 실제 cache length와 종료 처리를 확인한다.

## 4.4 두 단계의 비용을 나눠 보기

| 항목 | Prefill | Decode |
|---|---|---|
| 처리 위치 | prompt의 여러 위치 | sequence당 보통 한 위치 |
| Projection | 큰 $N$의 GEMM | 작은 batch면 skinny GEMM |
| Full attention | 허용된 token pair 처리 | 새 query가 cache를 조회 |
| 상태 변화 | prompt KV와 최종 recurrent state 구축 | KV append와 state update |
| 흔한 압박 | FLOPs와 activation peak | weight/KV 읽기와 latency |

“Prefill은 compute-bound, decode는 memory-bound”는 출발점으로 유용하지만 항상 참은 아니다. 매우 긴 prefill의 attention IO나 작은 chunk, 큰 decode batch, 특정 kernel과 GPU 구조가 비중을 바꾼다. CPU scheduling과 GPU collective가 병목일 수도 있다.

Continuous batching은 끝난 sequence 자리를 새 요청으로 채워 빈 slot을 줄인다. Chunked prefill은 긴 prompt 작업을 나눠 다른 decode와 공존시키고 한 번에 처리하는 token과 activation peak를 제어한다. Batch를 키우면 weight reuse가 좋아질 수 있지만 요청별 cache 총량과 latency도 증가한다. [27,28]

# 5 Softmax Attention을 계산으로 이해하기

## 5.1 Q와 K의 점수에서 V의 가중합으로

한 head에서 query $q_t\in\mathbb{R}^{d_k}$, 과거 위치의 key $k_i\in\mathbb{R}^{d_k}$와 value $v_i\in\mathbb{R}^{d_v}$를 두자. 이 장의 벡터는 열벡터다. [1]

$$
s_{t,i}=\frac{q_t^\top k_i}{\sqrt{d_k}}+m_{t,i},\qquad
a_{t,i}=\frac{\exp(s_{t,i})}{\sum_{j\le t}\exp(s_{t,j})},
\qquad o_t=\sum_{i\le t}a_{t,i}v_i.
$$

$m_{t,i}$는 mask다. Causal 조건에서 미래 위치는 $-\infty$를 더해 softmax 확률을 0으로 만든다. Dot product는 query와 key의 학습된 feature가 얼마나 맞는지 점수화한다. 분모의 $\sqrt{d_k}$는 차원이 커질 때 dot product의 분산과 softmax 포화가 커지는 현상을 완화한다. Query와 key가 독립이고 각 성분 분산이 1이라는 단순 가정이면 내적 분산은 $d_k$, 표준편차는 $\sqrt{d_k}$다.

Softmax는 허용된 위치에 양수 가중치를 부여하고 합을 1로 만든다. 이는 token 하나를 무작위로 뽑는 sampling이 아니다. Sampling은 모델 마지막 vocabulary logits에서 수행하며 attention 내부 가중합과 단계가 다르다.

## 5.2 두 위치를 손으로 계산하기

이미 scale과 mask를 적용한 점수가 $[0,\ln3]$이라 하자. 지수는 $[1,3]$이므로 가중치는 $[1/4,3/4]$다. Value가 $v_1=(4,0)^\top$, $v_2=(0,8)^\top$이면:

$$
o=\tfrac14(4,0)^\top+\tfrac34(0,8)^\top=(1,6)^\top.
$$

Query가 바뀌어 두 점수의 순서가 바뀌면 같은 V를 저장해도 다른 혼합 결과가 나온다. K는 “조회하는 기준”, V는 “전달되는 내용”을 분리한다. 두 역할이 같아야 할 수학적 필요는 없고, 특정 구조에서는 저장 표현을 공유하도록 학습할 수도 있다.

## 5.3 행렬 shape로 읽기

한 sequence, 한 head의 prefill에서 $Q,K\in\mathbb{R}^{T\times d_k}$, $V\in\mathbb{R}^{T\times d_v}$다.

$$
O=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

$QK^\top$는 $T\times T$의 위치 간 점수다. 이를 그대로 HBM에 저장하면 길이 제곱에 비례하는 중간 메모리가 생긴다. 하지만 수학적으로 점수가 정의된다는 것과 구현이 전체 행렬을 HBM에 materialize해야 한다는 것은 다르다. FlashAttention은 이 지점의 IO를 개선한다.

Decode는 query 길이가 1이므로 $Q$가 $1\times d_k$, cache $K$는 $T\times d_k$다. 점수는 $1\times T$이며, 이 연산이 “새 token이 과거를 읽는다”의 구체적인 뜻이다.

## 5.4 RoPE와 위치 정보

Position 정보가 없다면 Q/K의 매칭만으로 순서를 구분하기 어렵다. RoPE는 query와 key의 일부 또는 전체 채널에 위치별 회전을 적용한다. [26]

$$
\widetilde q_t=R_tq_t,\quad \widetilde k_i=R_ik_i,\quad
\widetilde q_t^\top\widetilde k_i=q_t^\top R_t^\top R_i k_i.
$$

회전의 조합에 상대 위치 정보가 들어간다. 실제 구현은 positional dimension, base, scaling, partial rotation 등을 config로 정한다. RoPE 차원은 head dimension과 같을 필요가 없다. Qwen3.8-27B full attention은 head dimension 256 중 partial rotary factor 0.25, 즉 64차원을 회전에 사용한다. [13]

## 5.5 Gated Attention은 어디에 gate를 두는가

Attention output에 입력으로부터 만든 gate를 원소별로 곱하면, 조회한 정보 중 어떤 성분을 residual stream에 전달할지 조절할 수 있다. 설명용 표현은 다음과 같다.

$$
\widehat o_t=g(x_t)\odot o_t,\qquad y_t=W_O\widehat o_t.
$$

Gate의 activation과 적용 위치는 구조별로 다르다. Qwen의 공식 구현은 projection과 output gate 경로를 별도로 다룬다. 이름에 gated가 붙었다고 과거 KV가 fixed state로 대체되는 것은 아니다. **Qwen의 full/Gated Attention 층은 여전히 token별 K/V를 보관**하고, Gated DeltaNet 층은 다른 recurrence를 사용한다. [13,15,30]

# 6 KV Cache의 유도와 저장 정밀도

## 6.1 저장으로 재계산을 줄이는 이유

이전 token의 K/V는 그 위치까지의 causal hidden state에서 만들어진다. Model weight와 위치 조건이 고정된 기본 추론에서 미래 token이 추가되어도 과거 causal hidden state는 변하지 않는다. 따라서 과거 K/V를 저장해 다음 step에서 재사용할 수 있다.

KV cache 없이 매번 prefix 전체를 실행하면 같은 이전 위치의 projection과 layer 계산을 반복한다. $M$개 출력에 대해 처리하는 prefix token 수는 대략 $\sum_{j=0}^{M-1}(N+j)$가 된다. Cache를 쓰면 새 위치의 projection을 추가하지만, full attention의 새 query가 과거 $T$개를 읽는 비용은 남는다. “Cache가 있으니 decode가 context 길이에 무관하다”는 결론은 틀리다.

## 6.2 기본 용량식

| 기호 | 정의 | 단위 |
|---|---|---|
| $B$ | 보존하는 sequence 수 | sequence |
| $T$ | sequence당 cache token 수 | token |
| $L_f$ | 해당 KV를 저장하는 full attention 층 수 | layer |
| $H_{kv}$ | 층별 KV head 수 | head |
| $d_h$ | 동일하다고 둔 K와 V head dimension | element |
| $b$ | cache 원소당 저장 크기 | byte/element |

한 위치와 한 층의 K tensor는 $H_{kv}d_h$개 원소다. V도 같다면 둘을 합쳐 $2H_{kv}d_h$개다. Batch와 token과 층을 곱하면:

$$
M_{\mathrm{KV}}=B\,T\,L_f\,2H_{kv}d_hb.
$$

2는 K와 V 때문이다. K와 V 차원 또는 dtype이 다를 때는:

$$
M_{\mathrm{KV}}=BT\sum_{\ell\in\mathcal L_f}
H_{kv,\ell}\left(d_{k,\ell}b_{k,\ell}+d_{v,\ell}b_{v,\ell}\right).
$$

요청마다 길이가 다르면 $BT$ 대신 $\sum_{s=1}^{B}T_s$를 사용한다. 공유하지 않는 raw tensor 합이라는 가정이다. Prefix sharing은 중복된 물리 block 수를 줄일 수 있으며, logical token 합과 physical allocation은 달라진다. [6,24,33]

## 6.3 저장 byte와 계산 FLOPs는 다르다

새 query 한 head가 $T$개의 key와 점곱하면 약 $2Td_k$ FLOP, value 가중합은 약 $2Td_v$ FLOP이다. $d_k=d_v=d_h$일 때 query head 전체의 attention 본체 근사는:

$$
F_{\mathrm{decode,attn}}\simeq4BH_qTd_h.
$$

Projection, softmax의 exp와 reduction, normalization, gate, output projection은 생략한 근사다. 여기에는 $H_q$가 들어가고 cache에는 $H_{kv}$가 들어간다는 차이가 중요하다. GQA로 KV head를 6배 줄여도 query head 계산이 6배 줄어드는 것은 아니다.

Causal prefill의 유효 pair 수는 head당 $T(T+1)/2$다. 두 matmul의 pair당 약 $4d_h$ FLOP을 곱하면:

$$
F_{\mathrm{prefill,attn}}\simeq2BH_qd_hT(T+1).
$$

실제 kernel의 tile padding과 mask 처리에 따라 실행 FLOP은 달라진다. Full attention을 exact하게 유지하면서 IO를 줄여도 허용된 모든 pair 관계를 처리하는 기본 계산 구조는 남는다.

## 6.4 Weight dtype과 KV dtype의 독립성

Weight를 BF16으로 로드하고 KV cache만 FP8로 지정할 수 있다. 일반적으로 Q/K/V projection은 weight, activation, GEMM kernel의 지원 정밀도로 계산되고, 생성된 K/V는 cache에 쓸 때 FP8 저장 형식으로 quantize/cast될 수 있다. 읽을 때는 dequantize해 계산하거나 FP8을 직접 소비하는 지원 kernel 경로를 사용할 수 있다. 연산 정밀도, accumulator 정밀도, activation 정밀도, cache 저장 정밀도는 서로 다른 속성이다. [23,31]

$$
x_{\mathrm{stored}}=Q_{\mathrm{FP8}}(x/s),\qquad
\widehat x\simeq s\,x_{\mathrm{stored}}.
$$

$s$는 scale이고 위 식은 개념식이다. Per-tensor, per-head, block 또는 token/channel 단위 등 granularity와 static/dynamic scale 정책에 따라 수치 오차와 metadata 비용이 달라진다. FP8의 표현 범위와 mantissa가 BF16과 다르므로 outlier와 긴 문맥의 민감도를 평가해야 한다.

FP8 KV payload는 원소당 약 1 byte라 BF16의 절반이지만 scale, padding, alignment가 추가된다. Recurrent state를 FP32로 유지하는 hybrid 모델에서는 KV dtype만 바꿔도 recurrent state가 자동 FP8이 되지 않는다. 구체 경로는 해당 모델, engine 버전, attention backend와 옵션을 확인한다.

## 6.5 Cache의 세 가지 의미

같은 요청에서 token step 사이에 재사용하는 autoregressive KV cache, 서로 다른 요청에서 동일 prefix를 재사용하는 prefix cache, CPU/NVMe/remote 계층으로 이동하는 tiered cache는 다른 기능이다. Tokenization, position, 모델과 adapter, multimodal 조건이 맞아야 prefix 재사용이 성립한다. Recurrent layer는 prefix 끝의 state snapshot도 맞아야 한다. 하나의 cache hit율로 모든 메모리 비용을 설명하지 않는다. [24,33]

# 7 MHA MQA GQA의 구조와 정량 비교

## 7.1 공통 표기와 head 연결

Query head 수는 $H_q$, KV head 수는 $H_{kv}$다. 각 query head $h$가 연결되는 KV head를 $g(h)$로 쓰면:

$$
o_{t,h}=\sum_{i\le t}
\mathrm{softmax}_i\left(
\frac{q_{t,h}^\top k_{i,g(h)}}{\sqrt{d_h}}\right)v_{i,g(h)}.
$$

![MHA GQA MQA의 head 연결](master-notes-figures/heads.svg){width=6.2in}

그림 2. 설명용 $H_q=8$에서 MHA는 KV 8개, GQA는 KV 2개, MQA는 KV 1개를 둔다. 선은 query와 KV의 공유 관계이며 cache를 query 수만큼 복제하라는 뜻이 아니다.

## 7.2 MHA

MHA는 기본적으로 $H_q=H_{kv}=H$이며 각 query head가 독립 KV head를 갖는다. Token·layer당 cache는 $2Hd_hb$, 전체는 $BTL_f2Hd_hb$다. 서로 다른 KV projection subspace를 둘 수 있지만 저장과 읽기 byte가 크다. Head마다 문법 또는 의미 역할이 고정되어 있다고 해석할 근거는 없다. [1]

## 7.3 MQA

MQA는 $H_{kv}=1$로 두고 여러 query head가 하나의 KV를 공유한다. Token·layer당 $2d_hb$, 전체 $BTL_f2d_hb$다. 줄어드는 것은 K/V projection의 output 폭과 cache read/write다. Query head 수와 query별 softmax는 그대로 있을 수 있다. Shared KV를 kernel이 head마다 HBM에서 반복 로드한다면 잠재적 이점을 제대로 활용하지 못하므로 구현의 재사용도 중요하다. [2]

KV 공유는 학습 가능한 표현 공간에 제약을 줄 수 있다. 그러나 실제 평가 품질은 학습 규모와 데이터, 훈련 방법과 과제에 달렸다. 서로 다른 완성 모델을 단순히 MQA/MHA 명칭으로 정확도 순위화하지 않는다.

## 7.4 GQA

GQA는 query를 그룹으로 나누고 그룹마다 KV head를 공유한다. Group size를 $G=H_q/H_{kv}$로 정의한다. 이 교재에서는 group 수와 group size를 혼용하지 않는다. 예를 들어 $H_q=24,H_{kv}=4$면 group은 4개이고 group size는 6이다. [3]

Token·layer당 cache는 $2H_{kv}d_hb$, 전체 $BTL_f2H_{kv}d_hb$다. 동일한 $H_q,d_h$의 MHA 대비 byte 비율은 $H_{kv}/H_q=1/G$다. MHA와 MQA 사이의 공유 수준을 고르는 구조적 선택이다.

GQA 논문은 MHA checkpoint에서 추가 학습하는 uptraining과 품질·추론 성능의 절충을 평가했다. 학습된 MHA의 KV head 수를 실행 옵션으로 임의로 줄이면 동일한 모델이 보존된다고 볼 수 없다.

## 7.5 동일 가정에서 비교하기

$L_f=16,H_q=24,d_h=256,b=2$이고 $T=172,000,B=1$인 가상 비교다. GQA 행은 Qwen3.8-27B의 full attention 부분에 대응하지만 MHA/MQA 행은 비교를 위한 구조 변경 가정이며 실제 checkpoint가 아니다.

| 구조 | $H_{kv}$ | 전체 층 bytes/token | 172K raw KV | MHA 대비 |
|---|---:|---:|---:|---:|
| MHA | 24 | 384 KiB | 62.988 GiB | 100% |
| GQA | 4 | 64 KiB | 10.498 GiB | 16.7% |
| MQA | 1 | 16 KiB | 2.625 GiB | 4.17% |

KV head 공유는 sequence의 token 수를 줄이지 않으므로 full causal prefill의 $O(T^2)$, token당 decode 조회의 $O(T)$ 차수는 유지된다. Cache bytes의 상수항 감소와 점근 복잡도 감소를 구분한다.

## 7.6 Projection parameter와 연산 차이

Bias를 제외하고 Q/K/V projection parameter 수는 다음과 같다.

$$
P_{\mathrm{QKV}}=
d_{\mathrm{model}}\left(H_qd_k+H_{kv}d_k+H_{kv}d_v\right).
$$

같은 $H_q$에서 KV head를 줄이면 K/V projection 폭은 줄지만 Q projection 폭은 유지된다. 출력 폭과 $W_O$도 query head의 output dimension을 따른다. 따라서 model 전체 FLOPs와 속도가 cache 절감률만큼 줄어든다고 말할 수 없다.

## 7.7 Rank별 KV를 계산하기

TP에서 KV head가 rank별로 나뉘면 local head 수로 cache를 계산한다. 하지만 $H_{kv}<P_{\mathrm{TP}}$인 경우 head replication 또는 별도 분할 방식이 필요할 수 있다. 예를 들어 KV head 4개를 TP8에서 실행할 때 논리적으로 각 KV head를 두 rank가 중복 보유한다면 rank당 1 head이고 aggregate physical cache는 unsharded 논리값의 2배가 된다. 이는 설명용 배치 사례이며 실제 backend 동작을 보장하지 않는다. [16,24]

# 8 MLA의 저랭크 표현과 Matrix Absorption

## 8.1 Head 공유와 표현 압축의 차이

MLA는 K/V를 head별 원래 폭으로 모두 저장하기보다 공통 저차원 latent로 저장하도록 학습하는 구조다. 아래 식은 MLA의 content 경로를 단순화한 열벡터 표기다. [4]

$$
c_i=W_Dh_i,\qquad
k_{i,h}^{C}=U_{K,h}c_i,\qquad
v_{i,h}=U_{V,h}c_i,\qquad c_i\in\mathbb R^{r}.
$$

저장하는 $r$이 전체 expanded KV 폭보다 작으면 token당 byte를 줄일 수 있다. GQA의 $H_{kv}$ 감소와 달리 여러 head가 같은 latent에서 서로 다른 up-projection으로 정보를 읽는다. kv_lora_rank 같은 설정 이름이 있어도 runtime LoRA adapter를 사용한다는 뜻은 아니다.

## 8.2 Key projection 흡수의 유도

Query head의 content score를 보면:

$$
q_{t,h}^{C\top}k_{i,h}^{C}
=q_{t,h}^{C\top}U_{K,h}c_i
=(U_{K,h}^{\top}q_{t,h}^{C})^\top c_i.
$$

오른쪽 식에서는 $U_K^\top q$를 현재 query에 대해 한 번 계산하고 저장된 latent $c_i$와 점곱할 수 있다. 모든 과거 위치에서 expanded K를 다시 만들고 저장하는 대신 계산의 결합 순서를 바꾼 것이다. 이것이 absorption의 핵심 직관이다.

Value도 다음처럼 재배열한다.

$$
\sum_i a_{t,i,h}v_{i,h}
=\sum_i a_{t,i,h}U_{V,h}c_i
=U_{V,h}\left(\sum_i a_{t,i,h}c_i\right).
$$

가중합 후 up-projection할 수 있고 output projection과 결합하는 구현도 가능하다. 그러나 attention weight는 head마다 다를 수 있어 모든 head의 가중합이 하나로 사라지는 것은 아니다. Matrix absorption은 연산 순서를 바꾸는 수학과 이를 실현하는 kernel 양쪽이 필요하다.

## 8.3 RoPE를 분리하는 이유

Expanded K에 위치마다 $R_i$를 곱하면 score에는 $q^\top R_t^\top R_iU_Kc_i$가 들어간다. 위치 $i$마다 회전이 달라 $U_K$를 모든 위치에 대해 동일한 query 변환으로 쉽게 흡수하기 어렵다. MLA의 decoupled positional 경로는 content score와 작은 positional score를 나눠 다루는 설계다.

$$
s_{t,i,h}\propto q_{t,h}^{C\top}k_{i,h}^{C}
+q_{t,h}^{R\top}k_i^{R}.
$$

Latent $c_i$와 positional key $k_i^R$를 cache하는 전형적인 경우 token·layer당 byte는 $(r+d_R)b$다. NoPE 등 다른 변형은 이 저장식을 그대로 쓰지 않는다. [4,10]

## 8.4 Cache 공식과 설명용 수치

$$
M_{\mathrm{MLA}}=BTL_{\mathrm{MLA}}(r+d_R)b
$$

는 동일 dtype을 가정한 대표 형태다. 설명용 $r=512,d_R=64,b=2$면 한 layer당 1,152 byte/token이다. 16개 층이면 18 KiB/token이며 172K에서 약 2.953 GiB다. 이 수치는 **Qwen3.8-27B의 실제 구조가 아니라** latent cache를 설명하는 가정이다. Qwen의 실제 full attention은 4 KV head의 GQA이며 임의로 MLA cache로 바꿀 수 없다.

MLA는 token별 항목을 계속 보존하므로 global cache는 여전히 $O(T)$다. 낮은 cache byte, 추가 projection/복원 비용, batch 크기와 kernel 효율이 실제 decode 속도를 결정한다. Latent 압축 구조의 품질은 훈련과 평가 조건에서 검증해야 한다.

## 8.5 네 방식의 비교 매트릭스

| 비교 축 | MHA | MQA | GQA | MLA |
|---|---|---|---|---|
| token·layer raw 저장 | $2H_qd_hb$ | $2d_hb$ | $2H_{kv}d_hb$ | $(r+d_R)b$ 등 |
| 전체 저장 | $BTL_f$ 곱함 | $BTL_f$ 곱함 | $BTL_f$ 곱함 | layer별 latent 합 |
| 공유하거나 압축하는 대상 | 기본 head별 KV | 모든 Q의 KV | 그룹별 KV | 저랭크 공통 표현 |
| query 계산 | head별 점수와 혼합 | head별 query 유지 | head별 query 유지 | latent score와 up/absorb |
| decode 효과 | KV 읽기 큼 | KV byte 감소 큼 | 공유 비율만큼 byte 감소 | latent byte 감소와 kernel 비용 |
| 품질 해석 | 독립 KV subspace | 공유 제약 | 중간 공유 수준 | 저랭크 학습에 의존 |
| 구현 난점 | footprint | 효율적 broadcast | TP shard와 복제 | RoPE와 absorption과 dtype |

정확도의 보편적 순서는 정하지 않는다. Perplexity, 장문 exact retrieval, 코드 수정 성공률은 서로 다른 평가다. 같은 학습 예산과 데이터에서 구조만 비교한 결과인지, 서로 다른 완성 모델의 결과인지부터 확인한다.

# 9 SWA와 Sparse와 Compressed Attention

## 9.1 과거 정보의 어느 축을 줄이는가

GQA는 head 축, MLA는 token별 표현 차원을 줄인다. 이제는 **조회할 위치 수**와 **저장하는 sequence 항목 수**를 구분한다. 조회만 sparse해도 나중에 다른 위치가 선택될 수 있으므로 저장량은 여전히 길이에 비례할 수 있다. Sequence compression은 여러 원본 위치를 하나의 표현으로 합치므로 저장 항목 수도 줄일 수 있지만, 원문 세부 정보의 복원 가능성에 제약을 준다.

## 9.2 Sliding Window Attention

SWA는 현재 위치 $t$에서 최근 $W$개 위치만 참조한다. Causal 경계는 예를 들어 $\max(1,t-W+1)\le i\le t$다. $W$에 현재 token을 포함하는지에 따라 구현의 window 정의가 하나 차이 날 수 있으므로 config와 kernel의 규칙을 확인한다. [29]

Local 층 $L_s$의 raw KV는 오래된 위치를 실제로 evict한다는 가정에서:

$$
M_{\mathrm{SWA}}=B L_s\min(T,W)\,2H_{kv}d_hb.
$$

Global 층 $L_f$와 동일한 head/dtype으로 섞으면:

$$
M=B\,2H_{kv}d_hb
\left(L_fT+L_s\min(T,W)\right).
$$

예를 들어 full 4층과 local 12층, $W=4096$인 가상 stack은 긴 context에서 4개 global 층의 cache만 계속 증가한다. 모든 층을 full로 둔 경우보다 작지만 전체 모델이 $O(1)$ cache가 되는 것은 아니다.

SWA는 멀리 떨어진 원본 위치에 직접 접근할 기회를 줄인다. 여러 층을 거쳐 정보가 전달되는 효과와, 현재 query가 오래된 원문 위치를 직접 조회하는 효과는 같지 않다. Attention sink 유지, 별도 global token, 중요한 token retention 같은 정책은 해당 모델과 runtime의 설계다. 오래된 token을 무조건 버려도 품질이 보존된다고 일반화하지 않는다.

## 9.3 Selected Sparse Attention과 indexer

Query마다 관련 후보 $k$개를 고르고 본 attention을 계산하는 개념형은:

$$
\mathcal I_t=\mathrm{TopK}_{i\le t}\,\mathrm{Index}(q_t,z_i),\qquad
o_t=\mathrm{Attention}(q_t,K_{\mathcal I_t},V_{\mathcal I_t}).
$$

$z_i$는 indexer가 쓰는 표현이며 main attention의 K와 같을 필요가 없다. Top-K main attention은 $T$ 대신 $k$개 위치를 읽어 계산을 줄일 수 있지만 indexer score, 후보 data 저장, top-k reduction, gather, 동기화를 추가한다. 전체 후보를 계속 저장하면 cache는 여전히 $O(T)$다. [17,18,32]

특히 main attention의 $O(Tk)$ prefill 비용만 보고 전체 시스템도 그 차수라고 단정하면 안 된다. 각 query가 모든 index 후보를 훑는 단순 구현이라면 indexer에 다시 길이 제곱의 항이 생긴다. Index sharing, 계층적 후보 검색, block granularity는 이 비용을 줄이는 별도 선택이다.

## 9.4 CSA의 압축과 선택을 단계별로 보기

CSA는 DeepSeek-V4 문서의 **Compressed Sparse Attention**이다. 여기서는 모델 전체 배포 사례가 아니라 공개된 attention 메커니즘만 다룬다. Local raw 경로와 압축된 history의 선택 경로가 함께 있다는 점이 중요하다. [17,18]

1. 새로운 hidden input을 compressor의 window/tail에 반영한다.
2. 닫힌 compression window에서 compressed entry를 만들고 global compressed cache에 추가한다.
3. Indexer도 검색용 표현을 만들고 query별 유용한 compressed entries를 고른다.
4. 선택된 compressed entries와 최근 raw local K/V를 대상으로 attention을 계산한다.
5. 다음 token에 필요한 local tail과 compressor/indexer 상태를 유지한다.

일반적인 설명용 window 길이를 $m$, stride를 $s$라 두면 닫힌 window 수는:

$$
N_c(T)=\max\left(0,\ 1+\operatorname{floor}\left(\frac{T-m}{s}\right)\right).
$$

겹치지 않는 $m=s$의 단순 경우 $N_c\simeq T/m$이다. 논문에서 말하는 compression rate와 실제 overlapping window의 길이를 항상 같은 변수로 취급하면 안 된다. 경계와 아직 닫히지 않은 window의 상태도 별도다.

압축과 검색의 개념식은:

$$
c_j=\mathrm{Compress}(h_{(j-1)s+1:(j-1)s+m}),\quad
\mathcal I_t=\mathrm{TopK}_j\,\mathrm{Index}(q_t,c_j).
$$

오래된 정보는 compressed entry를 통해 읽고, 최근의 정밀한 raw token 정보는 local branch에서 읽는다. 압축 항목을 선택했다고 과거 원본의 full K/V를 무손실로 역변환하는 일반 경로가 생기는 것은 아니다. 특정 구조가 원본 block을 별도로 저장하고 되읽는다면 그 저장과 전송을 따로 계산해야 한다.

공개 Transformers 문서의 V4-Pro default 사례는 CSA rate 4, index top-k 1024다. 이 값은 해당 config 범위의 예이며 CSA의 정의상 상수는 아니다. 압축률과 top-k를 바꾸면 학습과 kernel 지원, 정보 손실의 양상도 달라질 수 있다. [18]

## 9.5 HCA의 강한 압축과 Dense lookup

HCA는 **Heavily Compressed Attention**이다. 이름을 Hierarchical Attention으로 풀어 tree 검색 구조라고 단정하면 잘못된 설명이 된다. 공개 V4 구현 설명에서 HCA는 recent raw window와 더 강하게 압축된 history를 함께 사용하고, global 압축 항목에는 CSA식 per-query top-k indexer 없이 접근한다. [17,18]

Non-overlap 압축률을 설명용 $m'$라 하면 global entry 수는 약 $T/m'$다. Query는 이 압축 항목 전체에 dense lookup한다. 따라서 dense attention이지만 원래 token 길이 대신 훨씬 짧은 압축 sequence에 대한 attention이다. V4 공개 config의 대표 HCA rate 128은 해당 모델의 설정이지 일반 HCA 전부의 비율이 아니다.

$$
F_{\mathrm{HCA,decode}}\propto W+T/m',\qquad
F_{\mathrm{HCA,prefill}}\propto TW+T^2/m'.
$$

이는 차원과 head를 고정한 branch-level 단순 근사다. Compressor와 projection, cache IO 및 실제 schedule 비용을 제외한다. 압축률이 높아도 원본의 구분 가능한 세부 정보는 줄 수 있으므로, local branch와 다른 층의 정보 접근이 보완한다.

## 9.6 압축 cache의 일반 회계식

층 $\ell$에서 compressed entry당 실제 저장 원소를 $E_{c,\ell}$, raw local entry당 원소를 $E_{w,\ell}$라 두면:

$$
M_{\mathrm{cache}}\simeq B\sum_\ell
\left[
N_{c,\ell}(T)E_{c,\ell}b_{c,\ell}
+\min(T,W_\ell)E_{w,\ell}b_{w,\ell}
+M_{\mathrm{index},\ell}(T)+M_{\mathrm{tail},\ell}
\right].
$$

여기에 scale와 metadata, alignment를 더한다. K와 V가 별도라면 $E$에 둘을 합산하고, 특정 구조가 K=V 또는 공통 latent를 저장한다면 2를 기계적으로 곱하지 않는다. “128:1 압축이니 모델 전체 KV가 1/128”이라는 결론은 local branch와 indexer와 tail을 빠뜨린다.

## 9.7 저장과 조회를 함께 비교하기

| 방식 | 보관 항목 | query의 조회 대상 | 길이 증가의 경향 |
|---|---|---|---|
| Full | 원본 위치별 KV | 모든 과거 위치 | 저장과 조회 모두 $T$에 비례 |
| SWA | 최근 $W$ raw KV | 최근 window | 올바른 eviction이면 bounded |
| Sparse selection | 후보 KV와 indexer | 선택 $k$와 index 검색 | 저장은 여전히 $O(T)$ 가능 |
| CSA | 압축 항목과 local과 indexer | top-k 압축 항목과 local | 저장 $T/s$와 부가 상태 |
| HCA | 강한 압축 항목과 local | 전체 압축 history와 local | 조회 길이 $T/m'$ |
| Recurrent | 행렬 등의 상태 | 현재 state | 고정 차원에서 $T$와 무관한 저장 |

품질의 손실 지점도 다르다. SWA는 직접 접근할 수 있는 거리, sparse selection은 선택 실패, compression은 여러 원본을 합치는 손실, recurrence는 제한 상태에 누적되는 간섭이 핵심이다. 실제 과제별 평가 없이 하나의 정확도 순서를 정하지 않는다.

# 10 Linear Attention을 Recurrent State로 유도하기

## 10.1 Softmax 대신 결합 가능한 유사도

Kernelized linear attention은 $\exp(q^\top k)$ 계열 softmax 유사도를 직접 계산하는 대신, feature map $\phi(q)^\top\phi(k)$ 같은 결합 가능한 형태를 사용한다. 설명을 위해 $\phi(q),\phi(k)\in\mathbb R^{d_\phi}$이고 $v\in\mathbb R^{d_v}$라 하자. [7]

$$
o_t=
\frac{\sum_{i\le t}\phi(q_t)^\top\phi(k_i)v_i}
{\sum_{i\le t}\phi(q_t)^\top\phi(k_i)}.
$$

Query와 무관한 과거 합을 먼저 모으면:

$$
S_t=\sum_{i\le t}\phi(k_i)v_i^\top,\qquad
z_t=\sum_{i\le t}\phi(k_i),
$$

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,\quad
z_t=z_{t-1}+\phi(k_t),\quad
o_t=\frac{S_t^\top\phi(q_t)}{z_t^\top\phi(q_t)}.
$$

여기서 $S_t$의 shape는 $d_\phi\times d_v$이고 $z_t$는 $d_\phi$다. 원본 token을 목록으로 다시 읽지 않아도 과거 합을 state로 보존할 수 있다. 분모의 안정화와 feature map, 정규화 방식은 구조마다 다르며 모든 linear attention이 정확히 이 정규화식을 사용하는 것은 아니다.

## 10.2 O1 state와 OT 처리 시간을 구분하기

고정된 head/state 차원에서 persistent state는 context 길이 $T$에 대해 $O(1)$이다. 그러나 $T$개 입력을 모두 읽어야 하므로 전체 prefill 계산은 적어도 $T$에 비례한다. 또한 한 step에서 대략 $d_\phi d_v$ 크기의 state를 읽고 갱신하므로 state 차원에 대한 비용은 남는다.

메모리를 $O(1)$이라고 할 때 어떤 축에 대한 차수인지 반드시 말한다. 동시 요청 $B$, 층 수, head 수와 state precision이 커지면 state 메모리도 증가한다.

## 10.3 왜 fixed state가 무한한 원문 목록과 같지 않은가

원본 KV는 각 위치를 개별적으로 address할 수 있다. Fixed state는 많은 key-value 관계를 제한된 수의 원소에 겹쳐 저장한다. 유사한 key가 반복되거나 같은 key의 value가 바뀌면 간섭이 생긴다. 문맥이 길어도 원래 모든 token을 임의 query로 정확히 복구할 수 있다는 보장은 없다.

단순 누적 메모리에서 같은 key $k$에 value $v_1$을 저장한 뒤 $v_2$를 더하면 두 기록이 중첩된다. 이 때문에 오래된 관계를 감쇠하거나 특정 key 방향의 내용을 새 value로 보정하는 update가 등장한다.

# 11 Mamba와 Delta Rule과 Gated DeltaNet

## 11.1 State라는 공통점과 다른 업데이트

Mamba는 selective state-space model 계열이고 DeltaNet은 associative memory와 delta rule을 활용한 linear attention 계열이다. Mamba를 순서대로 고친 결과가 DeltaNet이라는 단일 코드 계보는 아니다. 두 연구선은 고정 크기 state에 무엇을 쓰고 잊을지 조절한다는 문제에서 만난다. [8–12]

Selective SSM을 단순화한 형태는:

$$
h_t=\overline A_t h_{t-1}+\overline B_t x_t,\qquad
y_t=C_t h_t+D x_t.
$$

입력에 따라 선택되는 파라미터와 discretization, scan이 핵심이다. 실제 Mamba layer는 projection, short convolution, gate 등을 포함하므로 이 한 줄이 모든 tensor와 kernel을 나타내지는 않는다. Mamba-2는 structured state space duality를 통해 SSM과 attention 계열의 연결 및 효율적인 실행 방법을 제시한다. [11,12]

## 11.2 Delta Rule의 차원을 고정하기

이 장은 공식 구현과 대조하기 쉽게 **$S_t\in\mathbb R^{d_k\times d_v}$**로 쓴다. $k_t,q_t\in\mathbb R^{d_k}$, $v_t\in\mathbb R^{d_v}$는 열벡터다. State에서 key의 value 예측은 $S_{t-1}^\top k_t$다.

$$
\widehat v_t=S_{t-1}^\top k_t,\qquad
e_t=v_t-\widehat v_t,\qquad
S_t=S_{t-1}+\beta_t k_t e_t^\top.
$$

예측한 값과 새 value의 차이만 현재 key 방향에 보정한다. 재구성 손실 $\frac12\|v-S^\top k\|_2^2$의 $S$에 대한 gradient는 $k(S^\top k-v)^\top$이므로 위 update는 학습률 $\beta$인 gradient descent 형태로 유도할 수 있다. [8]

## 11.3 Key 정규화가 주는 해석

$\|k_t\|_2=1$이면 update 후 같은 key로 읽은 값은:

$$
S_t^\top k_t
=(1-\beta_t)\widehat v_t+\beta_t v_t.
$$

$\beta=1$이면 그 key 방향의 예측을 새 value로 맞추고, $0<\beta<1$이면 부분 보정한다. 단위 길이가 아니면 $\|k\|^2$가 update 효과에 들어간다. Q/K normalization과 update strength는 안정성에 관계한다. 이 등식이 다른 모든 key의 기억을 보존한다는 뜻은 아니다. 비직교 key들 사이에는 간섭이 남는다.

스칼라 예를 들면 기존 기억 $S=2$, key $k=1$, 새 value $v=5$, $\beta=0.5$일 때 예측은 2, error는 3, 새 state는 3.5다. 단순 additive update였다면 2에 5를 더해 7이 되지만 delta rule은 원하는 값 쪽으로 이동한다.

## 11.4 Gated DeltaNet의 감쇠와 보정 순서

먼저 state를 감쇠하고 그 state의 예측 오류를 보정하는 표기로 고정한다.

$$
S'_t=\alpha_tS_{t-1},\qquad
e_t=v_t-S_t'^\top k_t,\qquad
S_t=S'_t+\beta_t k_te_t^\top.
$$

동일한 식을 전개하면:

$$
S_t=\alpha_t(I-\beta_t k_tk_t^\top)S_{t-1}
+\beta_t k_tv_t^\top.
$$

$\alpha$는 전체 memory의 감쇠, $\beta$는 현재 key 방향의 보정 강도다. 예를 들어 이전 스칼라 state 2를 $\alpha=0.8$로 줄이면 1.6이다. 새 value 5에 $\beta=0.5$로 보정하면 3.3이 된다. 감쇠 전 예측 2를 error에 넣으면 다른 식이 되므로, gate와 update의 순서를 말하지 않고 비슷한 식을 섞으면 안 된다. [9,15]

Qwen 계열 구현은 입력으로부터 decay 관련 값과 beta를 만들고, Q/K 정규화, causal convolution, gated output 정규화 등을 결합한다. 위 식은 그 core recurrence를 설명하는 것이며 projection과 normalization을 모두 포함한 전체 layer 식은 아니다.

## 11.5 KDA의 채널별 forgetting

KDA는 Kimi Delta Attention이며 Gated DeltaNet보다 세밀한 forgetting을 사용한다. 같은 $d_k\times d_v$ 방향에서 $D_t=\mathrm{Diag}(\alpha_{t,1},\ldots,\alpha_{t,d_k})$라 두면:

$$
S'_t=D_tS_{t-1},\qquad
S_t=(I-\beta_t k_tk_t^\top)D_tS_{t-1}
+\beta_t k_tv_t^\top.
$$

Scalar $\alpha I$ 대신 key-feature 축의 대각 decay가 row마다 다른 기억 수명을 부여한다. 예를 들어 2행 state에 $\alpha=(1,0.1)$이면 첫 번째 key-feature 방향은 유지하고 두 번째는 강하게 줄인 뒤 delta correction한다. [10]

State를 반대로 $d_v\times d_k$로 쓰는 논문 표기를 전치하면 gate는 오른쪽에 놓인다. “어느 채널인가”는 state orientation과 함께 설명해야 한다. 이전 자료의 value-channel이라는 단정은 이 표기와 맞지 않는다. 본문은 key-feature 축의 decay로 일관되게 정의한다.

KDA는 diagonal-plus-low-rank transition을 효율적인 chunkwise 연산으로 처리하는 알고리즘도 함께 제시한다. Matrix state의 메모리가 작아지는 점과 prefill kernel이 빠른 점은 연결되지만 서로 다른 주장이다.

## 11.6 Recurrent 상태와 Convolution 상태

Causal convolution은 다음 출력을 계산하기 위해 최근 입력을 필요로 한다. Kernel 길이 $k$의 depthwise convolution을 설명용으로 쓰면:

$$
\widetilde u_t=\sum_{j=0}^{k-1}w_j\odot u_{t-j}.
$$

다음 token을 기다릴 때 과거 $k-1$개 $u$를 tail로 보존하면 된다. 새 입력을 붙이고 가장 오래된 것을 버린다. 이 convolution tail은 **최근 입력의 유한 길이 기록**이며, 긴 history의 key-value 관계를 요약한 **recurrent matrix state**와 다른 tensor다.

최소 tail은 $k-1$개여도 reference 구현은 편의상 $k$ slot을 저장할 수 있다. vLLM과 Transformers의 shape가 하나 차이 난다고 모델의 수학이 달라졌다고 판단하지 않는다. 실제 tensor shape와 dtype을 읽어 메모리를 계산한다. [15,16]

## 11.7 Prefill의 병렬 처리와 Decode의 순차 갱신

Recurrence가 있다고 GPU에서 prompt token을 하나씩 Python loop로 실행해야 하는 것은 아니다. Chunk 내 계산을 행렬 연산으로 바꾸고 chunk 사이에 state를 전달하는 알고리즘이 있다. Scan/parallel formulation은 동일한 recurrence의 효율적 실행 경로다. Decode는 현재 token 하나와 이전 state를 입력으로 바로 갱신하는 경로가 적합하다. Floating-point 연산 순서가 달라 오차가 생길 수 있으므로 backend 간 동등성은 허용오차와 실제 출력으로 확인한다. [8–10,15]

# 12 Qwen3.8 27B의 층과 상태를 직접 계산하기

## 12.1 정확한 모델과 설정 범위

**공식 config 확인:** Qwen/Qwen3.8-27B의 config revision 366a23fd246985933964e9d21ac63eb3a93190ab 및 현재 모델 카드를 기준으로 한다. Architecture class는 Qwen3_5ForConditionalGeneration이고 model_type은 qwen3_5다. 이는 구현 호환 클래스 이름이며 checkpoint가 Qwen3.5라는 뜻이 아니다. Qwen3.8-27B라는 공식 모델 명칭은 실제로 확인된다. [13,14]

| 항목 | config 값 | 이 교재에서의 해석 |
|---|---:|---|
| Hidden dimension | 5,120 | residual stream 폭 |
| 언어 backbone 층 | 64 | 4개 단위 패턴을 16번 반복 |
| Full attention 간격 | 4 | 1부터 센 4,8,…,64번 층 |
| Full attention Q heads | 24 | query projection의 head 수 |
| Full attention KV heads | 4 | GQA group 4개와 group size 6 |
| Full head dimension | 256 | $5120/24$로 추정하면 안 됨 |
| Partial rotary factor | 0.25 | rotary dimension 64 |
| GDN QK heads | 16 | Q와 K의 projection head |
| GDN V heads | 48 | recurrent state head 수와 연결 |
| GDN key와 value dimension | 각각 128 | state matrix의 두 feature 축 |
| Causal conv kernel | 4 | tail과 구현 buffer 길이를 구분 |
| SSM state dtype | float32 | config의 mamba_ssm_dtype |
| Native context | 262,144 | extension 설정과 별도 구분 |

Head dimension을 hidden size 나누기 head 수로 복원하는 관례는 이 모델에 맞지 않는다. Q projection의 attention 폭은 $24\times256=6144$이고 residual hidden 5120과 달라도 $W_Q,W_O$가 그 사이를 변환할 수 있다.

![Qwen3.8 27B의 64개 층 배치](master-notes-figures/layers.svg){width=6.2in}

그림 3. GDN 3층 뒤 full/Gated Attention 1층을 16번 반복한다. 색은 attention 종류이고 FFN은 각 층에 존재한다. Index는 1부터 센다.

## 12.2 Token 하나가 층을 통과할 때

처음 세 GDN 층에서 token hidden state는 각 층의 projection, convolution과 recurrence를 거치며 각 층 state를 갱신한다. 4번째 full attention 층에서는 그 층의 새 K/V를 append하고 해당 query가 과거 cache를 읽는다. 이후 동일한 종류의 패턴이 64번째 층까지 이어진다.

48 GDN 층이 하나의 state를 공유하는 것은 아니다. 요청마다, 각 GDN 층마다 recurrent matrix와 convolution tail이 있다. 16 full attention 층도 각 층의 서로 다른 hidden representation에서 K/V를 만들어 보관한다. Layer 간 cache를 임의로 공유하면 동일 모델의 계산을 보존하지 못한다.

GDN 경로에서 QK 16개가 V 48개에 맞춰 반복되는 reference 구현은 recurrent state의 head 축이 48임을 설명한다. Query/key projection head가 16개라는 이유로 state head를 16으로 계산하면 이 경우 3배 과소평가한다. [15,16]

## 12.3 Full attention KV bytes

BF16/FP16 cache의 raw byte는:

$$
\Delta M_{\mathrm{KV}}=
16\times2\times4\times256\times2
=65,536\ \mathrm{byte/token}=64\ \mathrm{KiB/token}.
$$

FP8 cache payload는 원소당 1 byte 가정에서 32 KiB/token이다. 64개 전체 층을 이 식에 넣으면 token-wise KV를 4배 과대평가한다. GDN의 상태는 다음 절에서 별도 계산한다.

| Cache 위치 수 | BF16 full KV | FP8 full KV payload |
|---:|---:|---:|
| 8,192 | 0.500 GiB | 0.250 GiB |
| 32,768 | 2.000 GiB | 1.000 GiB |
| 131,072 | 8.000 GiB | 4.000 GiB |
| 170,000 | 10.376 GiB | 5.188 GiB |
| 172,000 | 10.498 GiB | 5.249 GiB |
| 262,144 | 16.000 GiB | 8.000 GiB |

## 12.4 Recurrent matrix state의 raw 크기

공식 reference 구현은 state shape를 batch, V heads, key head dim, value head dim으로 설명한다. vLLM은 두 마지막 축의 순서가 반대일 수 있지만 원소 수는 동일하다. 이 사례의 한 GDN 층과 한 sequence는 $48\times128\times128$개 원소다. FP32를 가정하면:

$$
M_{\mathrm{rec,layer}}
=48\times128\times128\times4
=3,145,728\ \mathrm{byte}=3\ \mathrm{MiB}.
$$

48개 GDN 층 전체는:

$$
M_{\mathrm{rec,seq}}=48\times3\ \mathrm{MiB}=144\ \mathrm{MiB}.
$$

이는 context 길이에 선형으로 늘지 않는 state의 raw 원소 수다. Snapshot을 여러 개 보관하는 prefix 정책, allocator와 state slot 예약, TP 분할, dtype override, kernel padding은 별도다. Config의 FP32와 reference path에 근거한 계산이며 어떤 vLLM 배포에서도 항상 이 exact bytes가 관측된다는 보장은 아니다. [13,15,16]

## 12.5 Convolution state의 raw 크기

GDN의 conv에 들어가는 packed Q/K/V 채널 수는:

$$
C_{\mathrm{conv}}=2H_kd_k+H_vd_v
=2\times16\times128+48\times128=10,240.
$$

추가 상태를 두지 않는 vLLM state shape의 기본 tail은 kernel 길이에서 1을 뺀 3 slot이다. Conv state를 BF16으로 둔 설명용 기본값은:

$$
M_{\mathrm{conv,seq}}
=48\times10,240\times3\times2
=2,949,120\ \mathrm{byte}=2.8125\ \mathrm{MiB}.
$$

Transformers reference/cache convention에서 kernel 길이 전체인 4 slot을 보관하면 $48\times10,240\times4\times2=3.75$ MiB다. 이 차이는 저장 buffer convention의 예다. 실제 serving 계산에는 사용 중인 engine의 shape와 dtype을 넣는다. [15,16]

## 12.6 한 요청의 상태 회계

172K 위치의 BF16 full KV와 FP32 recurrent, BF16 3-slot conv를 합친 raw 예시는:

$$
M_{\mathrm{request,raw}}
=10.498046875\ \mathrm{GiB}
+\frac{144+2.8125}{1024}\ \mathrm{GiB}
\simeq10.6414\ \mathrm{GiB}.
$$

KV만 FP8로 바꾸면 약 5.3924 GiB다. 고정 state 항은 그대로 남는다. 따라서 짧은 context에서는 state의 비중이 상대적으로 크고 긴 context에서는 16개 global attention 층의 KV가 주로 증가한다.

![Qwen3.8 27B 요청 상태의 길이별 증가](master-notes-figures/cache_growth.svg){width=6.2in}

그림 4. BF16 full KV와 FP8 full KV는 context에 선형 증가하고, 설명용 recurrent와 conv state 합은 약 146.8 MiB로 일정하다. Metadata와 workspace 등을 포함한 실제 allocation 그래프는 아니다.

## 12.7 왜 전부 Recurrent 층으로 바꾸지 않는가

Recurrent state는 긴 이력을 제한 크기로 처리하지만 원본 위치의 개별 조회와 동일하지 않다. Full attention 층은 token별 정보를 직접 재참조하는 경로를 제공한다. Hybrid는 이 두 정보 처리 특성을 섞어 비용과 표현력을 조정한다. 3:1 비율이 모든 모델과 과제에 최적이라는 정리는 없다. 실제 학습과 장문 과제 평가로 검증되는 설계다.

# 13 FlashAttention과 PagedAttention과 CUDA Graph

## 13.1 구조와 실행과 할당을 분리하기

| 기술 | 바꾸는 층위 | 줄이려는 비용 |
|---|---|---|
| MQA와 GQA | 학습된 head 연결 | KV head의 저장과 읽기 |
| MLA | 학습된 표현과 projection | token별 KV 표현 폭 |
| SWA와 sparse와 compression | 정보 접근 구조 | 조회 위치와 저장 sequence |
| GDN과 KDA | state update 구조 | token 목록 대신 bounded state |
| FlashAttention | attention 계산과 IO 알고리즘 | 중간 행렬 materialization |
| PagedAttention | KV 주소 mapping과 block 관리 | fragmentation과 중복 할당 |
| CUDA Graph | GPU 작업의 launch와 dependency 실행 | CPU 제출 overhead |

같은 모델에서 이 기술들이 조합될 수 있다. FlashAttention을 켰다고 GQA를 학습한 모델이 MLA가 되지는 않고, PagedAttention으로 KV를 block화했다고 오래된 token 정보를 압축하는 것도 아니다.

## 13.2 FlashAttention의 Online Softmax

Naive attention은 점수 행렬, softmax 행렬을 HBM에 크게 저장했다가 다시 읽을 수 있다. FlashAttention은 key/value를 tile로 읽으며 softmax 통계를 누적해 같은 attention의 출력을 계산한다. Exact attention이라는 말은 수학적 목표가 같다는 뜻이며 floating-point 연산 순서 차이에 따른 미세한 오차까지 bitwise 동일하다는 뜻은 아니다. [5]

현재까지의 최대 점수 $m$, 지수합 $\ell$, value 누적합 $u$를 저장하고, 새로운 점수 tile $\mathcal B$를 읽으면:

$$
m'=\max\left(m,\max_{i\in\mathcal B}s_i\right),
$$

$$
\ell'=e^{m-m'}\ell+\sum_{i\in\mathcal B}e^{s_i-m'},\qquad
u'=e^{m-m'}u+\sum_{i\in\mathcal B}e^{s_i-m'}v_i.
$$

마지막에 $o=u/\ell$이다. 새로운 최대값 기준으로 이전 누적값을 다시 scale하므로 전체 점수행렬을 저장하지 않고도 softmax 가중합을 계산할 수 있다. Tiling은 on-chip memory의 작은 용량을 재사용에 활용한다.

이 방식은 prefill의 중간 $T\times T$ tensor 저장을 줄이지만 이후 decode를 위한 KV cache는 필요하다. “Attention score matrix memory”와 “persistent KV cache memory”를 구별해야 한다.

## 13.3 PagedAttention과 Block rounding

KV를 커다란 연속 구간으로 요청별 최대 길이만큼 미리 잡으면 낭비와 fragmentation이 생긴다. PagedAttention은 logical token block을 physical memory block에 매핑하고 요청이 커질 때 필요한 block을 추가한다. Prefix 공유와 copy-on-write 같은 관리도 가능하다. [6]

Block당 $P$개 token이고 현재 길이 $T$라면 단순한 raw allocation token 수는 $P\lceil T/P\rceil$이다. 예를 들어 $P=16,T=17$이면 logical은 17 token이지만 32 token 분량의 block이 필요하다. 16개 full 층의 64 KiB/token을 모두 같은 방식으로 계산하면 logical 1.0625 MiB, rounded payload 2 MiB다. 실제 hybrid allocator는 cache group과 state page 정렬을 함께 다뤄 더 복잡할 수 있다. [24]

Paged allocation은 block 안의 마지막 빈 공간을 완전히 없애지는 않는다. Prefix가 공유되어 있어도 활성 요청의 state와 ownership, eviction 정책을 관리해야 한다.

## 13.4 CUDA Graph의 정의와 실행 단계

CUDA Graph는 반복 GPU 작업과 dependency를 capture하고 executable graph로 instantiate한 뒤 replay하는 기능이다. CPU가 많은 kernel을 하나씩 launch하는 비용을 줄일 수 있다. Attention과 GEMM의 수학적 FLOPs를 제거하지는 않는다. [19,22]

1. **Capture:** 실행할 kernel, copy와 dependency 등을 기록한다.
2. **Instantiate:** 재실행할 graph executable을 준비한다.
3. **Replay:** 준비한 작업들을 반복 실행한다.

Buffer의 값은 step마다 달라져도 된다. 중요한 것은 캡처한 pointer와 shape, 실행 경로 등 replay에 필요한 조건의 안정성이다. CUDA 자체에는 parameter update 등 여러 기능이 있지만, serving 엔진이 어떤 동적 경우를 지원하는지는 별도 구현 문제다.

## 13.5 Prefill과 Decode의 Graph 전략

Decode는 한 token step이 반복되므로 고정 buffer와 batch bucket, padded slot을 활용하기 쉽다. 요청이 끝나고 새 요청이 와도 slot의 metadata와 입력을 갱신하며 같은 memory address를 유지하는 전략을 쓸 수 있다.

Prefill은 길이가 다양하므로 길이 bucket과 chunked prefill, piecewise capture, 지원되는 full graph 경로를 사용할 수 있다. Bucket과 graph 수를 늘리면 capture warmup, graph pool과 padding 비용이 증가한다. Recurrent state와 block table pointer 관리도 안정적이어야 한다.

vLLM의 FULL, PIECEWISE, FULL_AND_PIECEWISE 같은 구체 모드와 backend coverage는 버전별로 달라진다. 본문은 공식 design 문서의 개념을 설명하며, 실제 vLLM 0.30.0 환경에서는 동일 버전 config와 선택된 attention/GDN backend를 확인한다. Shape가 맞더라도 capture 제한이 있는 작업은 별도 경로로 실행될 수 있다. [22]

# 14 요청 길이와 동시성의 메모리 계산

## 14.1 170K 입력과 2K 출력의 상태

최대 cache 위치 $T=172,000$을 기준으로 BF16 full KV는 11,272,192,000 byte, 즉 11.272 GB 또는 10.498 GiB다. FP8 payload는 절반이다. 48개 GDN 층의 FP32 recurrent와 BF16 3-slot conv를 합친 raw state는 sequence당 약 146.8125 MiB다.

| 동시 sequence | BF16 full KV | FP8 full KV | 고정 raw state 합 |
|---:|---:|---:|---:|
| 1 | 10.498 GiB | 5.249 GiB | 0.143 GiB |
| 4 | 41.992 GiB | 20.996 GiB | 0.573 GiB |
| 8 | 83.984 GiB | 41.992 GiB | 1.147 GiB |
| 16 | 167.969 GiB | 83.984 GiB | 2.294 GiB |

요청들이 같은 길이이고 prefix block/state를 공유하지 않는 가정이다. 이 표는 model 전체 VRAM도, rank별 메모리도 아니다. Sequence별 실제 길이를 합산한 뒤 TP/PP/CP layout에 따라 각 rank가 보관하는 부분과 복제되는 부분을 계산해야 한다.

## 14.2 운영 VRAM 회계식

$$
M_{\mathrm{VRAM}}=M_{\mathrm{weight}}+M_{\mathrm{request}}+M_{\mathrm{execution}}+M_{\mathrm{runtime}}.
$$

요청별 상태와 실행 중 임시 메모리는 각각 다음처럼 나눈다.

$$
M_{\mathrm{request}}=M_{\mathrm{KV}}+M_{\mathrm{rec}}+M_{\mathrm{conv}}.
$$

$$
M_{\mathrm{execution}}=M_{\mathrm{activation}}+M_{\mathrm{workspace}}+M_{\mathrm{graph}}.
$$

마지막 runtime 항에 allocator reserve, cache block rounding, scale와 metadata, communication buffer, fragmentation 등을 포함해 중복 합산하지 않도록 정의한다. GPU memory utilization 설정이 곧 KV에 배정된 비율이라는 뜻은 아니다. KV pool 사용률은 그 pool 안의 사용량이며 node 전체 VRAM 점유율과 다르다. [24]

밀집형 27B weight를 단순 BF16 2 byte로 곱하면 약 54 GB다. 이는 27B라는 반올림된 parameter 수를 사용한 이론치다. 실제 checkpoint tensor 수, vision encoder, tied parameter, 부가 tensor, quantization layout과 metadata가 정확한 resident weight를 결정한다.

## 14.3 Activation peak를 별도로 계산하기

FFN intermediate dimension이 17,408이고 한 중간 tensor가 $N\times17408$ BF16이라면:

$$
M_{\mathrm{temp}}=N\times17,408\times2\ \mathrm{byte}.
$$

$N=32,768$이면 약 1.0625 GiB다. Gate와 up output이 동시에 필요하거나 fusion/reuse가 다르면 peak는 달라진다. 이 값은 특정 단일 tensor의 설명용 크기이며 전체 FFN peak가 아니다. 긴 prompt를 한 번에 크게 처리하면 KV pool에 여유가 있어도 transient activation/workspace 때문에 OOM이 날 수 있다.

## 14.4 HBM Bandwidth로 보는 Decode 하한

172K의 full KV를 step마다 한 번 읽는다고 가정하면 BF16 raw traffic는 약 11.272 GB다. H200 SXM의 명목 4.8 TB/s를 완벽히 사용한다는 극단적 이상화에서는:

$$
t_{\mathrm{KV,ideal}}\ge
\frac{11.272\ \mathrm{GB}}{4800\ \mathrm{GB/s}}
\simeq2.35\ \mathrm{ms}.
$$

별도로 54 GB weight를 완전히 한 번 읽는 가정은 약 11.25 ms다. 실제 ITL은 이 숫자와 같지 않다. TP shard와 cache 재사용, batch의 weight 공유, kernel의 반복 load, 연산 overlap, 다른 요청, recurrent state와 FFN, CPU와 collective가 결과를 바꾼다.

동일 사례의 full attention 본체 FLOPs 근사는 16층을 합쳐 $16\times4\times24\times172000\times256\simeq67.63$ GFLOP이다. Raw KV read byte로 나눈 산술 강도는 BF16에서 약 6 FLOP/byte다. 이로써 GQA가 남겨 둔 query 계산과 KV traffic의 관계를 수식으로 볼 수 있다. 이것은 attention branch의 이상화이며 모델 전체 roofline이 아니다.

## 14.5 상태 전송의 이론 하한

Full KV 약 11.272 GB만 전송하고 payload bandwidth를 단방향 50 GB/s로 **가정**하면:

$$
t_{\mathrm{transfer,ideal}}=
\frac{11.272}{50}\simeq0.2254\ \mathrm{s}.
$$

400 Gb/s line rate를 8로 나누면 50 GB/s지만 실효 payload는 더 작을 수 있다. 25 GB/s면 약 0.451 s, 12.5 GB/s면 약 0.902 s다. Recurrent/conv state, scale와 metadata, protocol, memory registration, GPU/NIC affinity, synchronization과 queueing은 제외했다. 이 계산은 통신량의 감각을 주며 end-to-end TTFT를 예측하지 않는다.

## 14.6 Capacity와 Throughput을 혼동하지 않기

상태 예산을 요청당 상태로 나누면 capacity의 상한을 근사할 수 있다.

$$
B_{\mathrm{capacity}}\lesssim
\frac{M_{\mathrm{available,state}}}{M_{\mathrm{request,state}}}.
$$

이 상한에 가까운 수의 요청을 동시에 올릴 수 있어도 TPOT SLO를 만족한다는 뜻은 아니다. 여러 요청이 각자의 큰 KV를 매 step 읽으면 HBM 경쟁이 커진다. 상태 용량에 따른 admission 한계와 지속 가능한 latency/throughput 한계는 따로 측정한다.

# 15 분산 배포와 통신의 Trade off

## 15.1 단일 GPU와 TP와 Replica

한 GPU에 필요한 weight와 상태가 모두 들어가고 낮은 latency가 목표라면 단일 GPU가 통신을 피하는 기준선이다. TP는 한 요청의 layer 내부 행렬과 head를 나눠 memory와 compute를 분산한다. Replica는 모델을 복제하고 요청을 나눠 aggregate throughput과 격리를 늘린다. 어느 쪽이 유리한지는 요청 길이와 부하와 topology에 달렸다.

| 방식 | 나누는 축 | 주된 데이터 이동 | 주요 판단 |
|---|---|---|---|
| TP | layer 내부 tensor | all-reduce와 gather 계열 | 계산 절감과 반복 통신 |
| PP | layer 구간 | stage 사이 activation | memory 분산과 pipeline bubble |
| Replica | 서로 다른 요청 | 요청 라우팅 | weight 복제와 총 처리량 |
| CP | sequence 위치 | KV나 partial attention 교환 | 긴 문맥 분산과 추가 통신 |

각 방식의 실제 sharding과 지원 범위는 engine마다 다르다. Full attention의 softmax 결과를 context shard 사이에서 결합할 때도 normalization 통계를 맞춰야 한다. Recurrent layer가 있는 hybrid 모델은 sequence shard 사이의 state 전달 규칙이 필요하다.

## 15.2 Ring All reduce의 Byte 정의

Rank 수 $P$, 각 rank가 가진 reduction input tensor 크기 $S$ byte인 단순 ring을 생각하자. Reduce-scatter와 all-gather를 합치면 한 rank가 **보내는** byte는:

$$
V_{\mathrm{send}}\simeq 2\frac{P-1}{P}S.
$$

받는 양도 같으므로 send와 receive를 모두 더한 endpoint traffic은 $4(P-1)S/P$다. Full-duplex 경로의 이상화 시간 모델은:

$$
t_{\mathrm{ring}}\simeq2(P-1)\tau
+2\frac{P-1}{P}\frac{S}{B_{\mathrm{eff}}}.
$$

$\tau$는 단계당 latency, $B_{\mathrm{eff}}$는 이 모델에서 송신 방향에 정의한 effective bandwidth다. 이 식은 단순 ring의 모델이며 NCCL이 항상 ring만 고른다는 뜻은 아니다. Tree, topology, channel 수와 overlap이 실제값을 바꾼다. [20]

**정정:** 이전 원고에서 $2(P-1)S/P$를 send와 receive의 합이라고 부른 설명은 byte 정의가 잘못됐다. 동일 식을 송신량으로 정의하고, 양방향 endpoint 합과 시간 모델을 구분한다.

## 15.3 Decode에서 작은 메시지가 비싼 이유

설명용 activation tensor가 $N\times5120$ BF16이면 $N=8$에서 80 KiB다. TP4 ring의 rank당 송신량 근사는 120 KiB다. Prefill에서 $N=8192$라면 같은 tensor는 80 MiB로 커진다. 전자는 작은 메시지 latency와 반복 동기화, 후자는 bandwidth가 상대적으로 중요해질 수 있다.

64개 layer마다 통신이 반복된다면 개별 메시지가 작아도 누적 latency가 크다. 하지만 layer당 collective 횟수를 모든 구현에 공통인 상수로 외우면 안 된다. Row/column parallel layout, sequence parallel, fusion과 backend가 달라지면 횟수와 형태도 달라진다.

## 15.4 Qwen KV의 TP 예

4개 full KV head를 TP2에서 두 개씩, TP4에서 한 개씩 나누는 이상적인 head sharding이면 BF16 172K full KV는 rank당 각각 약 5.249 GiB와 2.625 GiB다. Recurrent/conv 상태의 분할은 별도 shape calculator를 따른다. Weight와 모든 상태가 같은 비율로 나뉜다고 가정하지 않는다.

NVLink bridge가 없는 PCIe GPU node와 NVSwitch 기반 node는 같은 TP degree라도 통신 비용이 다를 수 있다. 여러 node로 확장하면 NIC 수와 대역폭, GPU-NIC 연결, RDMA, switch 경쟁과 collective 경로를 함께 봐야 한다.

# 16 운영에서 확인할 항목과 해석

## 16.1 배포 전에 고정할 값

정확한 checkpoint revision, config, tokenizer와 chat template, weight dtype, KV dtype, recurrent/conv dtype, attention backend, GDN backend, TP/PP/CP 배치를 고정한다. 동일한 모델 이름만 기록하면 나중에 저장 형식과 kernel 차이를 설명하기 어렵다.

Workload는 prompt/output 길이의 p50/p90/p99와 최대값, arrival rate, 동시성, prefix hit 조건으로 정한다. 서버의 maximum sequence 수와 실제 active batch는 다른 값이다. Prefill chunk의 token budget과 요청 전체 길이도 구분한다.

## 16.2 길이와 동시성을 함께 관측하기

짧은 prompt에서는 launch와 scheduler가 큰 비중일 수 있다. 긴 prompt에서는 attention과 FFN, activation peak가 늘어난다. 장문 cache를 가진 decode가 많으면 KV bandwidth와 capacity가 커진다. 따라서 요청을 길이 bucket으로 나눠 TTFT와 ITL을 보고, 합계 tokens/s만 보지 않는다.

| 관찰 | 가능한 가설 | 더 필요한 자료 |
|---|---|---|
| TTFT 증가 | 대기열 또는 prefill 병목 | queue time과 chunk별 실행 |
| 길이에 따라 ITL 증가 | full KV 읽기 증가 | HBM과 attention timeline |
| TP 확장 후 지연 악화 | collective와 topology | NCCL 시간과 rank 배치 |
| KV pool 여유인데 OOM | activation 또는 workspace peak | peak memory breakdown |
| FP8 KV의 품질 변화 | scale와 precision 민감도 | 장문과 짧은 문맥 평가 |
| 높은 GPU util과 낮은 TPS | memory stall 또는 launch/통신 | tensor pipe와 GPU trace |

운영 기록은 관측, 원인 가설, 검증 결과를 나눠 쓴다. Health timeout은 응답을 제때 못 받았다는 증거이며 GPU kernel이 멈췄다는 단독 증거는 아니다. 같은 원칙으로 “KV usage 90%”를 “VRAM 90%”로 바꾸어 보고하지 않는다.

## 16.3 Attention 변경의 품질을 볼 기준

Head sharing과 latent 구조는 학습된 표현을 바꾸고, KV quantization은 저장 오차를 추가하며, window/selection은 접근 가능한 정보를 바꾼다. 이 변화들은 같은 정확도 축이 아니다. 다음 token perplexity, 특정 위치의 exact retrieval, 긴 코드 수정, 여러 근거 문장 결합을 구분해 본다.

길이가 늘 때 성능이 떨어지면 positional extrapolation, 압축 간섭, indexer 선택, quantization과 runtime의 cache 관리 중 무엇이 원인인지 조건을 나눠 확인한다. “메모리가 작아서 정확도가 낮다”라는 한 문장으로 원인을 확정하지 않는다.

# 17 Attention 최적화의 계보와 학습 지도

| 시기와 자료 | 제안 또는 핵심 아이디어 | 줄이려는 병목 | 남는 비용 |
|---|---|---|---|
| 2017 Attention [1] | QKV와 multi-head | 병렬 문맥 모델링 | full pair 계산과 KV |
| 2019 MQA [2] | 한 KV head 공유 | decode KV bandwidth | query별 계산과 표현 제약 |
| 2020 Linear [7] | feature map과 결합법칙 | token history 조회 | 고정 state의 간섭 |
| 2022 FlashAttention [5] | tiling과 online softmax | HBM 중간 IO | full attention의 pair 계산 |
| 2023 GQA [3] | 그룹별 KV 공유 | memory와 품질 절충 | 구조 재학습과 kernel |
| 2023 PagedAttention [6] | block mapping | serving allocation 낭비 | raw cache bytes |
| 2023 SWA 사례 [29] | 최근 window | 장문 local KV | 먼 과거 직접 조회 |
| 2023 Mamba [11] | selective SSM | 긴 sequence 상태 처리 | state 표현과 scan |
| 2024 MLA [4] | latent KV와 absorption | token당 저장량 | projection과 global scan |
| 2024 DeltaNet [8] | error correction memory | 누적 state 간섭 | state capacity |
| 2024 Mamba 2 [12] | state space duality | recurrence 실행 효율 | 차원과 kernel 조건 |
| 2025 GDN [9] | decay와 delta update | 기억의 유지와 교체 | state dtype와 update |
| 2025 KDA [10] | 세밀한 diagonal gate | 제한 state 활용 | chunk algorithm |
| 2026 CSA와 HCA [17] | sequence 압축과 선택 | 긴 history의 저장과 조회 | local과 indexer와 압축 손실 |

표의 연도는 최초 공개/대표 논문 시점이며 모든 연구의 단일 직선 계보는 아니다. 특히 Mamba와 DeltaNet은 서로 다른 연구선이다. 구조 변화와 kernel 실행, allocator 최적화의 층위는 계속 구분해야 한다.

## 17.1 통합 과정에서 바로잡은 설명

Qwen3.8-27B는 공식 checkpoint가 확인된다. 이전 문서의 “공식 명칭 확인 불가” 문장은 검색 범위를 충분히 확인하지 못한 오류였으며, 본문은 실제 Qwen3.8 config로 대체했다. Architecture class의 qwen3_5는 구현 이름이다.

Head dimension은 hidden size 나누기 Q head 수로 항상 정해지지 않는다. Qwen의 5120과 24와 256은 서로 모순이 아니라 projection input과 output 폭이 다르다는 뜻이다.

KDA의 decay 축은 state orientation과 함께 설명해야 한다. 본문의 $d_k\times d_v$ state에서는 $D_t$가 왼쪽에서 key-feature 축을 감쇠한다. Gate를 value 축이라고 단정한 이전 설명은 고쳤다.

기존 scalar gated-delta 설명에서 decay 전후의 예측을 혼용하면 다른 recurrence가 된다. 본문은 decay한 $S'_t$로 error를 계산하도록 순서를 고정했다. Ring all-reduce도 송신량과 양방향 endpoint 합을 따로 정의했다.

# 18 별도 참고 P D Disaggregation

Prefill과 decode worker를 분리하면 서로 다른 batch와 자원 구성을 사용할 수 있다. 대신 prefill이 만든 full KV와 recurrent/conv state를 decode 쪽으로 넘기거나 복원할 수 있어야 한다. Cache layout과 state ownership, 취소와 재시도, 전송 지연이 추가된다. KV와 state를 먼저 구분해 계산해야 분리의 비용을 설명할 수 있다.

Mooncake Transfer Engine과 NIXL은 이 상태 이동에 활용할 수 있는 구성요소다. 전송 라이브러리 도입만으로 실제 RDMA나 GPU 직접 경로가 항상 선택된다고 단정하지 않는다. 이 교재에서는 14장의 byte와 전송 하한 계산까지만 연결하고 분리 serving 설계는 별도 주제로 둔다. [34]

# 19 주석이 포함된 원문 참고자료

모델 설정은 2026년 11월 1일 확인 기준이다. 링크가 main 또는 latest를 가리키는 구현 자료는 이후 바뀔 수 있으므로 실제 배포 기록에는 사용 버전과 commit을 함께 남긴다. 논문의 개선률은 그 논문의 모델과 학습 및 평가 조건에 한정된다.

## 19.1 Attention과 Cache의 원 논문

**[1] Vaswani et al. Attention Is All You Need. NeurIPS 2017.** [원문 링크](https://arxiv.org/abs/1706.03762)  
Scaled dot-product와 multi-head 정의의 근거다. QKV projection과 mask, head output을 원문 3.2절에서 대조한다.

**[2] Shazeer. Fast Transformer Decoding One Write Head is All You Need. 2019.** [원문 링크](https://arxiv.org/abs/1911.02150)  
MQA가 KV write/read의 head 중복을 줄이는 동기를 제시한다. 구조를 바꿨을 때의 품질과 추론 비용을 함께 보는 자료다.

**[3] Ainslie et al. GQA Training Generalized Multi Query Transformer Models from Multi Head Checkpoints. EMNLP 2023.** [원문 링크](https://aclanthology.org/2023.emnlp-main.298/)  
Grouped query와 uptraining의 원 논문이다. 특정 실험에서 보고한 MHA/MQA 사이의 품질과 효율을 보편 순위로 확대하지 않는다.

**[4] DeepSeek AI. DeepSeek V2 Technical Report. 2024.** [원문 링크](https://arxiv.org/abs/2405.04434)  
본 교재는 그 보고서의 MLA 부분만 참조한다. Joint low-rank KV, decoupled RoPE와 matrix absorption의 원 근거다.

**[5] Dao et al. FlashAttention Fast and Memory Efficient Exact Attention with IO Awareness. NeurIPS 2022.** [원문 링크](https://arxiv.org/abs/2205.14135)  
Tiling과 online softmax가 HBM IO를 줄이는 과정을 설명한다. Persistent KV와 score intermediate가 다른 메모리라는 근거다.

**[6] Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023.** [원문 링크](https://arxiv.org/abs/2309.06180)  
Logical block과 physical KV block의 mapping, fragmentation과 sharing의 근거다.

**[7] Katharopoulos et al. Transformers are RNNs Fast Autoregressive Transformers with Linear Attention. ICML 2020.** [원문 링크](https://arxiv.org/abs/2006.16236)  
Feature map 유사도와 결합법칙으로 recurrent state를 유도한다. 본문의 normalized linear attention 수식을 대조할 원문이다.

**[8] Yang et al. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. 2024.** [원문 링크](https://arxiv.org/abs/2406.06484)  
Delta-rule memory update와 sequence 병렬화의 근거다. 기존 원고의 다른 논문 저자 표기는 이 제목과 저자로 바로잡았다.

**[9] Yang et al. Gated Delta Networks Improving Mamba2 with Delta Rule. ICLR 2025.** [원문 링크](https://arxiv.org/abs/2412.06464)  
Decay와 delta correction을 결합한 recurrent update를 설명한다. Mamba와의 비교가 직접적인 단일 코드 계보를 뜻하지는 않는다.

**[10] Kimi Team. Kimi Linear An Expressive Efficient Attention Architecture. 2025.** [원문 링크](https://arxiv.org/abs/2510.26692)  
본 교재는 KDA recurrence와 channel-wise gate, chunkwise algorithm 부분을 참조한다. State orientation을 원문과 맞춰 확인한다.

**[11] Gu and Dao. Mamba Linear Time Sequence Modeling with Selective State Spaces. 2023 공개 COLM 2024.** [원문 링크](https://arxiv.org/abs/2312.00752)  
Selective SSM과 input-dependent state update의 근거다.

**[12] Dao and Gu. Transformers are SSMs Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML 2024.** [원문 링크](https://proceedings.mlr.press/v235/dao24a.html)  
Mamba-2와 SSM/attention의 연결 및 효율적인 실행 알고리즘을 이해하기 위한 원문이다.

## 19.2 Qwen3.8 설정과 상태 구현

**[13] Qwen Qwen3.8-27B config.json. 고정 revision 366a23fd246985933964e9d21ac63eb3a93190ab.** [원문 링크](https://huggingface.co/Qwen/Qwen3.8-27B/blob/366a23fd246985933964e9d21ac63eb3a93190ab/config.json)  
Layer types 64개, 48/16 구성, full Q24/KV4/d256, GDN K16/V48/d128, conv kernel 4와 FP32 state 설정의 1차 근거다.

**[14] Qwen Qwen3.8-27B official model card.** [원문 링크](https://huggingface.co/Qwen/Qwen3.8-27B)  
공식 checkpoint 이름과 언어 백본, vision을 포함한 범위와 native context를 확인한다. 배포 alias와 공식 artifact 이름을 구분한다.

**[15] Hugging Face Transformers Qwen3.5 compatible modeling implementation.** [원문 링크](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)  
Qwen3.8 config가 지정한 호환 architecture의 recurrence와 head repeat 경로를 확인한다. Recurrent shape와 conv 상태의 reference 구현 근거이며 runtime 실측 메모리의 보장은 아니다.

**[16] vLLM MambaStateShapeCalculator와 Qwen GDN 구현.** [State shape 코드](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_utils.py) · [Qwen GDN 코드](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)  
GDN conv 채널 계산, kernel 길이에서 1을 뺀 tail, V-head state와 TP 분할을 확인하는 공식 코드다. Main은 변하므로 적용 버전을 고정한다.

## 19.3 Sparse와 Compression과 하드웨어

**[17] DeepSeek AI. DeepSeek V4 Towards Highly Efficient Million Token Context Intelligence. 2026.** [원문 링크](https://arxiv.org/abs/2606.19348)  
본 교재는 CSA/HCA의 압축과 attention 범위를 참조한다. 다른 모델 구성이나 배포 규모는 본문의 계산 사례에 사용하지 않는다.

**[18] Hugging Face Transformers DeepSeek V4 documentation.** [원문 링크](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)  
Local raw mask와 compressor branch, HCA와 CSA cache, 대표 config의 compression rates와 top-k를 설명하는 공식 구현 자료다. 보고서와 implementation 범위를 구분한다.

**[19] NVIDIA CUDA Programming Guide와 GPUDirect RDMA Documentation.** [CUDA Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) · [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)  
SM과 warp, memory hierarchy, matrix operation, CUDA Graph 및 GPU/NIC 데이터 경로의 공식 근거다.

**[20] NVIDIA NCCL User Guide.** [원문 링크](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)  
Collective의 의미와 실행 구성을 확인한다. 본문의 ring byte와 latency 식은 명시한 ring 가정에서 따로 유도한 것이다.

**[21] NVIDIA H100 H200 L40S 제품 자료와 HGX Reference Architecture.** [H100](https://www.nvidia.com/en-us/data-center/h100/) · [H200](https://www.nvidia.com/en-us/data-center/h200/) · [L40S](https://www.nvidia.com/en-us/data-center/l40s/) · [HGX 구성](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)  
SKU와 form factor별 capacity와 bandwidth, node 구성의 근거다. Aggregate와 방향 정의를 자료 원문에서 확인한다.

## 19.4 Serving 실행과 수치 정밀도

**[22] vLLM CUDA Graph design와 v0.30.0 configuration documentation.** [Graph design](https://docs.vllm.ai/en/latest/design/cuda_graphs/) · [v0.30 config](https://docs.vllm.ai/en/v0.30.0/api/vllm/config/)  
Graph mode와 prefill/decode coverage를 확인한다. 최신 design 설명을 과거 배포 버전에 자동 적용하지 않는다.

**[23] vLLM Quantized KV Cache documentation.** [원문 링크](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)  
KV dtype과 scale, backend 지원 조건의 근거다. Weight quantization과 별개 설정임을 확인한다.

**[24] vLLM Hybrid KV Cache Manager design.** [원문 링크](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)  
서로 다른 attention/state type의 cache group과 page/allocator 문제를 설명한다.

**[25] Williams et al. Roofline An Insightful Visual Performance Model for Multicore Architectures. 2009.** [원문 링크](https://doi.org/10.1145/1498765.1498785)  
Compute peak와 bandwidth/산술 강도의 성능 상한을 연결한 원 자료다.

**[26] Su et al. RoFormer Enhanced Transformer with Rotary Position Embedding. 2021.** [원문 링크](https://arxiv.org/abs/2104.09864)  
위치 회전과 상대 위치 내적의 수학적 근거다.

**[27] Agrawal et al. Sarathi Serve. OSDI 2024.** [원문 링크](https://www.usenix.org/conference/osdi24/presentation/agrawal)  
Chunked prefill과 decode scheduling, latency/throughput trade-off를 다룬다.

**[28] Yu et al. Orca A Distributed Serving System for Transformer Based Generative Models. OSDI 2022.** [원문 링크](https://www.usenix.org/conference/osdi22/presentation/yu)  
Iteration-level scheduling과 동적으로 변하는 생성 요청 batch의 serving 문제를 설명한다.

**[29] Jiang et al. Mistral 7B. 2023.** [원문 링크](https://arxiv.org/abs/2310.06825)  
Sliding window의 실제 밀집 모델 사례와 local attention의 비용을 이해하기 위한 자료다.

**[30] Qiu et al. Gated Attention for Large Language Models Non linearity Sparsity and Attention Sink Free. NeurIPS 2025.** [원문 링크](https://arxiv.org/abs/2505.06708)  
Attention output gate의 위치와 비선형 효과를 연구한다. Recurrent forgetting gate와 구분하는 근거다.

**[31] Micikevicius et al. FP8 Formats for Deep Learning. 2022.** [원문 링크](https://arxiv.org/abs/2209.05433)  
E4M3와 E5M2의 표현 범위와 정밀도 특성을 다룬다. 모든 KV precision 정책을 하나로 보지 않는 근거다.

**[32] DeepSeek AI. DeepSeek V3.2 Technical Report. 2025.** [원문 링크](https://arxiv.org/abs/2512.02556)  
본 교재는 DSA의 indexer와 selected attention 메커니즘만 참조한다. 본체 조회와 indexer 비용을 구분한다.

**[33] vLLM Automatic Prefix Caching documentation.** [원문 링크](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)  
동일 prefix의 block 재사용과 기본 autoregressive cache의 차이를 확인한다.

**[34] Mooncake Transfer Engine과 NVIDIA NIXL 공식 저장소.** [Mooncake](https://github.com/kvcache-ai/Mooncake) · [NIXL](https://github.com/ai-dynamo/nixl)  
별도 참고 장의 상태 이동 구성요소를 확인하는 자료다.

# 20 용어집과 발표자 확인 항목

## 20.1 용어를 실행과 연결하기

| 용어 | 이 교재에서의 의미 |
|---|---|
| Activation | 현재 연산의 중간 표현 |
| Projection | 학습된 행렬로 feature 공간 변환 |
| Head | attention의 분리된 projection과 조회 subspace |
| GEMM | 일반적인 행렬 곱 |
| HBM capacity | GPU memory에 동시에 저장할 수 있는 양 |
| HBM bandwidth | 단위 시간에 읽고 쓸 수 있는 byte |
| KV cache | 과거 위치의 key와 value를 재사용하는 상태 |
| Latent cache | 원래 KV 대신 저장하는 저차원 표현 |
| Recurrent state | token마다 갱신하는 제한 차원의 history |
| Conv state | causal convolution의 최근 입력 tail |
| Indexer | 본 attention이 읽을 후보를 고르는 경로 |
| Tile | on-chip 자원에 맞춰 나눈 계산과 데이터 묶음 |
| Block table | logical KV와 physical block의 mapping |
| TP rank | tensor parallel group의 한 실행 process |
| Collective | 여러 rank가 공동 수행하는 통신 연산 |
| TTFT와 ITL | 첫 token 지연과 token 사이 간격 |

## 20.2 스스로 설명할 수 있는지 확인하기

☐ 고객의 속도 요구를 TTFT와 ITL, 처리량과 동시성으로 구체화할 수 있다.

☐ GEMM의 shape와 FLOP 근사를 읽고 projection과 연결할 수 있다.

☐ Q/K의 score와 softmax, V 가중합을 두 위치 예제로 계산할 수 있다.

☐ Prefill의 마지막 prompt logits와 첫 출력 token의 관계를 설명할 수 있다.

☐ Weight와 activation과 KV와 recurrent와 conv와 workspace의 수명을 구분한다.

☐ HBM에 상주하는 KV와 SM에서 계산 중인 tile을 구분한다.

☐ MQA/GQA의 KV 감소와 query head 계산의 차이를 설명할 수 있다.

☐ MLA의 key/value absorption을 결합법칙으로 유도할 수 있다.

☐ SWA와 sparse와 compression이 줄이는 축을 각각 설명할 수 있다.

☐ CSA의 indexer 비용과 HCA의 dense compressed lookup을 구분한다.

☐ Linear attention의 $S,z$ recurrence와 그 shape를 쓸 수 있다.

☐ Delta rule에서 예측 오류를 보정하는 의미와 gate 순서를 설명할 수 있다.

☐ GDN의 scalar decay와 KDA의 key-feature decay를 구분한다.

☐ Qwen3.8-27B의 full KV 64 KiB/token과 48층 recurrent 144 MiB를 계산할 수 있다.

☐ Conv state의 최소 tail과 실제 buffer slot의 차이를 반영할 수 있다.

☐ FP8 KV가 recurrent dtype과 전체 VRAM을 자동으로 절반으로 만들지 않음을 설명한다.

☐ FlashAttention과 PagedAttention과 CUDA Graph를 서로 다른 층위로 설명한다.

☐ TP의 rank별 KV 복제 가능성과 ring 송수신 byte 정의를 구분한다.

☐ Capacity 계산과 bandwidth 하한과 실제 tokens/s를 구분해 보고할 수 있다.
