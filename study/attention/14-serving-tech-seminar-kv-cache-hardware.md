# LLM 추론 서빙의 Attention과 KV Cache
## GPU 연산 메모리 통신과 현대 Attention 구조

- 작성 기준: 2026-10-30
- 성격: 기술교류회 발표용 마스터 노트 / 발표 대본·수식·계산 사례·실험 체크리스트·원 논문
- 대상: AI 플랫폼, LLM Serving, 인프라, 엔지니어링 팀
- 권장 발표: 본편 90~120분 + 토론; 상세 내용은 발표자 심화 노트
- 원칙: **확인된 제품 스펙, 모델 config, 설명용 계산, 실제 운영 관측을 섞어 말하지 않는다.**

> 발표의 중심 논제: LLM serving의 아키텍처는 알고리즘적 아이디어의 전시장이라기보다, **계산량(FLOPs), HBM 용량(capacity), HBM 대역폭(bandwidth), GPU 간 fabric, scheduler overhead, 정확도와 정보 보존 능력** 사이의 교환 관계다.

---

# 0. 발표 진행 지도

| 장 | 주제 | 발표의 질문 | 시간 |
|---|---|---|---:|
| 1 | CPU와 GPU, GPU 내부 구조 | GPU가 빠르다는 말은 정확히 무엇이 빠르다는 뜻인가? | 12분 |
| 2 | FLOPS·HBM·Interconnect·Roofline | 연산기가 더 빨라도 token/s가 안 늘어나는 이유는? | 13분 |
| 3 | Prefill/Decode와 KV Cache | KV는 왜 필요하고 무엇을 버리지 않고 보존하는가? | 15분 |
| 4 | MHA/MQA/GQA/MLA와 Flash/PagedAttention | 각 기술은 무슨 바이트와 계산을 줄이는가? | 20분 |
| 5 | Qwen3.8-27B 메모리·통신 계산 | 170K 입력 한 건은 실제로 얼마나 비싼가? | 15분 |
| 6 | TP/PP/DP/CP/EP와 IB | GPU를 늘리는 선택이 왜 새로운 병목을 만드는가? | 15분 |
| 7 | P/D Disaggregation + NIXL | 하나의 모델을 두 단계로 찢어 배포하면 무엇이 바뀌는가? | 15분 |
| 8 | Mamba·Gated DeltaNet·KDA·Hybrid | 고정 크기 상태로 긴 문맥을 다룰 수 있는가? | 20분 |
| 9 | 운영 사례와 토론 | 설정, 메트릭, 장애를 어떻게 연결해서 판단하는가? | 15분 |

**발표 첫 문장:** “GPU 8장이 있다고 모델이 반드시 8배 빨라지지는 않습니다. 메모리와 링크, 그리고 모델이 과거 정보를 저장하는 방식이 다르기 때문입니다.”

**10개의 메시지:**
1. GPU는 CPU보다 무조건 빠른 장치가 아니라, 높은 데이터 병렬성과 메모리 대역폭에 특화된 장치다.
2. 연산 성능(FLOPS), 메모리 용량(GB), 메모리 대역폭(GB/s), 링크 대역폭(GB/s), latency는 서로 다른 지표다.
3. Prefill과 Decode는 같은 모델을 실행하지만 다른 병목을 갖는다.
4. KV cache는 이전 토큰의 K·V를 재계산하지 않기 위한 저장 공간이며, 일반적으로 토큰 수에 선형 증가한다.
5. MQA/GQA는 **KV head 수**, MLA는 **토큰별 KV 표현 차원**, recurrent attention은 **과거 토큰을 보관하는 방식 자체**를 바꾼다.
6. FlashAttention은 기본적으로 정확한 softmax attention의 실행/I/O 최적화, PagedAttention은 KV 할당/주소 관리 최적화다. 둘은 MQA 같은 학습된 모델 구조와 다른 층위다.
7. Tensor Parallelism은 가중치·일부 KV를 shard하지만 매 layer 집단통신이 새로운 비용이 된다.
8. InfiniBand는 노드 간 데이터 전송을 개선할 수 있어도 모델의 계산 복잡도나 전송할 바이트 자체를 제거하지는 못한다.
9. P/D 분리는 단계별 독립 최적화를 가능하게 하지만 KV 이동과 운영 복잡성을 추가한다.
10. Hybrid architecture는 메모리 절약과 정확한 위치 기반 검색을 동시에 얻으려는 절충이며, prefix caching과 state rollback은 더 어려워진다.

---

# 1. CPU vs GPU: 병렬 실행과 데이터 이동

## 1.1 CPU는 GPU의 약한 버전이 아니다

**CPU:** 상대적으로 적고 복잡한 코어, 큰 캐시, 강한 분기 예측, 짧은 지연(latency), OS·네트워크·제어 작업에 적합. 토크나이징, HTTP, 스케줄링, RPC, KV metadata, kernel submission 등이 CPU 경로에 남는다.

**GPU:** 다수의 SM(Streaming Multiprocessor), warps, FP32/FP16/BF16/FP8용 연산기와 Tensor Core, 높은 aggregate throughput, HBM 대역폭을 이용해 다량의 병렬 연산을 처리한다. 호스트→디바이스 발행과 GPU 내 실행 사이에는 launch overhead도 있다.

그래서 동일한 GPU에서 **작은 배치 decode**는 CPU dispatch/launch에 막힐 수 있지만 **큰 prefill GEMM**은 Tensor Core가 포화될 수 있다. GPU 이용률(`nvidia-smi`의 GPU Util)이 100%라고 Tensor Core FP throughput이 100%라는 뜻은 아니다.

## 1.2 GPU 1장 내부의 데이터 경로

```text
CPU / DRAM (Host)
   │ PCIe 또는 CPU–GPU 전용 fabric
   ▼
GPU HBM (가중치 + KV cache + 상태 + activation)
   │ global-memory load/store
   ▼
GPU L2 Cache
   │
   ▼
SM ── register / shared memory(SRAM) ── CUDA core / Tensor Core
   │
   └── 결과를 다시 L2/HBM으로 저장
```

원칙: Tensor Core가 계산할 수 있는 속도와, 필요한 데이터가 연산기에 공급되는 속도가 모두 중요하다. 연산량을 줄이는 것뿐 아니라 **HBM 접근 횟수를 줄이는 것**도 주요 최적화다.

## 1.3 다섯 가지 구별

| 지표 | 단위 | 질문 | 대표 병목 |
|---|---|---|---|
| Peak compute | TFLOPS | 초당 이론적으로 몇 번 연산할 수 있는가? | 대규모 GEMM |
| HBM capacity | GB/GiB | 모델·KV·activation을 보유할 수 있는가? | OOM, 동시성 한계 |
| HBM bandwidth | TB/s | GPU 메모리를 얼마나 빨리 읽고 쓰는가? | decode weight/KV streaming |
| Fabric bandwidth | GB/s | GPU 사이·노드 사이 데이터를 얼마나 빨리 옮기는가? | TP collective, KV transfer, EP all-to-all |
| Latency/overhead | µs/ms | 소량 작업을 시작하고 동기화하는 비용은? | small batch, many collectives, RPC |

**H200 SXM 공식 스펙**: 141 GB HBM3e, 4.8 TB/s, NVLink 총 900 GB/s(스펙 기준 aggregate·방향 정의 주의), BF16 Tensor Core 1,979 TFLOPS(구조적 sparsity 조건 포함). Dense/non-sparse 벤치마크에 1,979을 그대로 적용하지 않는다. H100 SXM은 80 GB, 3.35 TB/s. 실제 노드에서 PCIe/NVSwitch/NIC 토폴로지는 `nvidia-smi topo -m` 등으로 확인한다.

출처: [NVIDIA HGX reference architecture](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html), [H200 datasheet](https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23?xs=443995)

## 1.4 Roofline model

단순화한 GPU 성능 상한:

\[
P_{\mathrm{achievable}}\leq\min\left(P_{\mathrm{compute\ peak}},\ B_{\mathrm{HBM}}\times I\right)
\]

여기서 **Arithmetic Intensity** \(I=\frac{\mathrm{FLOPs}}{\mathrm{Bytes\ moved}}\)이다. 데이터 재사용도가 높으면 compute roof에, 낮으면 memory roof에 막힌다. 서로 다른 레벨의 메모리를 분석할 때는 각 레벨별 roof가 존재한다.

예시: 한 step에서 가중치 54 GB를 전부 읽고 이를 재사용할 충분한 batch가 없다면 H200 4.8 TB/s를 단순 나눗셈해도 최소 약 11 ms가 필요하다. **이는 데이터 전송에 대한 낙관적 하한이지 실제 11 ms/token을 보장하는 값이 아니다.** 실제는 캐시 효과, 여러 rank의 shard, 압축 형식, GEMM, KV read, 통신, kernel launch, batching이 개입한다.

### 질문 유도
- H200 FLOPS가 H100보다 높아도 왜 토큰이 같은 속도로 나오지 않을까?
- `GPU Util=100%`이면서 Tensor Core 활성도가 낮다면 무엇을 의심할까?
- 256K context에서 prompt prefill과 1-token decode의 arithmetic intensity가 같을까?

---

# 2. LLM 토큰 생성: Prefill과 Decode

```text
입력 prompt:  [t1 t2 t3 ... t170000]
                 │
                 ▼
Prefill: 여러 입력 토큰을 처리해 attention KV / recurrent state 구성
                 │  첫 응답 토큰 확정
                 ▼
Decode:       [y1] → [y2] → [y3] → ... → [y2000]
                 │    매 step에서 이전 context에 접근
                 ▼
Completion
```

**Prefill:** 긴 prompt 처리, 큰 GEMM, context length에 따른 attention, prefill chunk, activation peak가 중요. TTFT(Time to First Token)에 민감하다. 긴 입력의 softmax global attention에는 시퀀스 길이 \(T\)에 대해 대체로 \(O(T^2)\) attention 연산 구조가 있다. FFN과 projection은 \(O(T)\)이므로 총 prefill 비용이 곧바로 \(O(T^2)\)만인 것은 아니다.

**Decode:** 요청당 한 iteration에 보통 새 토큰 하나, 작은/skinny GEMM, 가중치 반복 로드, 이전 KV 접근, scheduler/CPU overhead, TP 동기화가 중요. TPOT(Time Per Output Token)과 ITL(Inter-Token Latency)에 민감하다. 병렬 batch가 크면 weight 재사용과 arithmetic intensity가 개선되므로 무조건 memory-bound는 아니다.

**Three latencies:**
- Queueing: 실행 시작 전 대기. TTFT의 일부가 될 수 있다.
- TTFT: 첫 토큰까지의 사용자 체감 지연; 큐·prefill·transfer·첫 decode 등이 포함될 수 있고 시스템 정의에 따라 경계가 다르다.
- TPOT/ITL: 생성 단계의 토큰 간 지연. 전체 요청 지연 = 대기 + prefill + transfer(해당 시) + decode + 후처리.

**동시 실행의 위험:** Integrated scheduler에서 긴 prefill chunk가 들어오면 다른 요청의 decode step이 지연되어 tail ITL이 튈 수 있다. Chunked prefill과 PD disaggregation이 이 문제에 대한 서로 다른 대책이다.

원 논문: [Sarathi-Serve, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal), [DistServe, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)

---

# 3. Attention을 처음부터 도출한다

## 3.1 토큰 벡터에서 Q/K/V가 만들어지는 과정

입력 hidden states \(X\in\mathbb R^{T\times d_{model}}\)에서 학습된 선형 변환으로

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

를 만든다. 개념적으로:
- **Q(Query):** 지금 필요한 정보에 대한 질문 표현
- **K(Key):** 어떤 과거 위치가 그 질문과 관련 있는지를 판단할 검색 키
- **V(Value):** 관련 위치를 찾았을 때 실제로 가져올 정보

한 attention head에서:

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+\mathrm{mask}\right)V
\]

Q와 K의 내적은 **관련성 점수**, softmax는 과거 위치에 분배한 **가중치**, 가중합 V는 **컨텍스트 정보**다. Causal mask는 현재 위치가 미래 토큰을 읽지 못하게 한다. 여러 head는 서로 다른 projection으로 표현을 나눠 학습한다.

### Decode에서 KV Cache를 안 쓰면?
생성한 다음 토큰의 attention을 계산하려고 과거 모든 토큰의 K/V projection을 재계산할 필요가 생긴다. **KV Cache는 과거의 K/V를 계속 저장하고 매 step에 새로운 K/V만 append**하여 반복 계산을 제거한다. 이것이 autoregressive inference의 기본적인 최적화다.

주의: **decode 내부 K/V 재사용과 Automatic Prefix Caching(APC)은 다르다.** APC는 서로 다른 요청에서 완전히 같은 prefix 부분을 재사용한다. 모델이 지원하는 KV state 관리와 prefix matching이 전제된다.

출처: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/v0.30.0/features/automatic_prefix_caching/)

## 3.2 KV Cache의 정확한 크기 공식

Full softmax attention을 구현하는 일반 MHA/GQA 레이어에서, 배치 내 한 sequence의 논리적 KV 크기:

\[
M_{\mathrm{KV,seq}}=2\,T\sum_{\ell\in\mathrm{KV\ layers}}H_{KV,\ell}\,d_{h,\ell}\,b_{\ell}
\]

레이어가 균일하면:

\[
\boxed{M_{KV}=2\times T\times L\times H_{KV}\times d_h\times b}
\]

- 2: K와 V 각각
- \(T\): 이전 prompt + 이미 생성한 token의 실제 저장 길이
- \(L\): **전체 모델 층 수가 아니라 KV를 보유하는 해당 attention 층 수**
- \(H_{KV}\): layer별 KV head 개수
- \(d_h\): head dimension(일반 hidden_size/Q head수와 반드시 같지는 않음)
- \(b\): element bytes(FP16/BF16=2, FP8≈1, FP32=4)

이 식은 **이론적인 raw tensor bytes**다. Paged KV block의 부분 사용, TP에서 KV head replication, kernel alignment, vLLM memory pool reservation, prefix cache block sharing, MTP lookahead, GDN state, cuda graph overhead는 포함하지 않는다.

## 3.3 세 개의 서로 다른 cache 개념

| 종류 | 저장 내용 | 목적 | 같은 것인가? |
|---|---|---|---|
| Autoregressive KV cache | 한 요청의 이전 토큰 K/V | 같은 요청에서 중복 재계산 방지 | 기본 상태 관리 |
| Automatic Prefix Cache | 다른 요청과 공유 가능한 prefix blocks/state | 공유 prefix prefill 재계산 회피 | 별도 정책/구현 |
| KV offload / remote cache | 사용 빈도가 낮은 KV를 CPU/SSD/다른 GPU 보관 | HBM capacity 증대·preemption 처리 | 데이터 이동비용 추가 |

**중요한 반례:** KV Cache는 매번 과거 전체를 재계산할 때보다 계산량을 줄이지만, cache 때문에 VRAM 용량과 HBM read bandwidth가 크게 필요해진다. 최적화는 무료가 아니다.

---

# 4. Attention 메커니즘 네 가지: MHA → MQA → GQA → MLA

## 4.1 Multi-Head Attention (MHA)

```text
Q:  q0 q1 q2 q3 q4 q5 q6 q7
K:  k0 k1 k2 k3 k4 k5 k6 k7
V:  v0 v1 v2 v3 v4 v5 v6 v7
```

각 Q head에 독립적인 K/V head를 대응시킨다. 서로 다른 표현 공간/기능을 풍부하게 학습할 수 있지만 \(H_{KV}=H_Q\)라 cache가 크다. 오래된 decoder 구조에서 중요한 기준점이다.

## 4.2 Multi-Query Attention (MQA)

```text
Q:  q0 q1 q2 q3 q4 q5 q6 q7
          \    |    /
K/V:            1개 공유
```

모든 query head가 **하나의 KV head**를 공유한다. \(H_{KV}=1\). decode에서 KV memory traffic을 획기적으로 줄이지만, 키/값의 독립적인 표현 능력 제약으로 일부 과제에서 품질 손실 가능성이 있다. 반드시 크게 나빠진다는 뜻은 아니다.

원 논문: [Shazeer, Fast Transformer Decoding, 2019](https://arxiv.org/abs/1911.02150)

## 4.3 Grouped-Query Attention (GQA)

```text
Q:       q0 q1 q2 | q3 q4 q5 | q6 q7 q8
K/V:      kv0   |   kv1    |    kv2
```

Q를 여러 그룹으로 묶어서 그룹마다 하나의 KV head를 공유한다. \(1<H_{KV}<H_Q\). MHA와 MQA 사이의 절충이다. 원 GQA 논문은 추가 학습을 통해 MHA에 근접한 성능과 MQA에 가까운 추론 효율을 보였지만 **모든 데이터셋·모델에서 정확도가 똑같다는 보장은 아니다.**

원 논문: [GQA, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.298/)

## 4.4 Multi-head Latent Attention (MLA)

GQA는 **몇 개의 KV head를 저장하느냐**를 줄인다. MLA는 다른 방향으로 **K/V를 저차원 latent 표현에 압축해 저장하고, 필요한 계산을 projection matrix에 흡수(matrix absorption)하는 방법**을 쓴다. DeepSeek-V2/V3가 대표적이다.

개념도:

```text
Token hidden x_t
    │ low-rank down projection
    ▼
Compressed latent c_t(작은 차원) + positional component
    │  ← cache에서는 압축 표현만 보유할 수 있음
    │
    └── query와 흡수된 projection을 통해 attention 수행
```

간단한 설명용 기본형:

\[
M_{\mathrm{MLA}}\approx T\times L_{MLA}\times (d_{latent}+d_{rope})\times b
\]

- 정확한 식은 모델의 decoupled RoPE, position KV, cache layout, kernel/backend에 따라 달라진다.
- MLA가 **모든 모델에서 KV를 동일한 비율로 압축**하는 것은 아니다.
- latent을 저장한다고 전통적인 softmax attention의 긴 문맥 전체 조회 비용이 자동으로 선형 시간이 되는 것은 아니다.
- 추가 down/up projection과 최적화된 MLA kernel의 구현 효율을 함께 고려해야 한다.

예시로 24개 MLA layer가 각각 latent 512 + RoPE 64 원소를 BF16으로 캐시하는 구현을 가정하면 토큰당 \(24\times(512+64)\times2=27\,648\) bytes ≈27 KiB다. **이건 임의로 명시한 차원의 가상 시나리오로 Kimi K3의 확정 메모리 수치가 아니다.**

원 논문: [DeepSeek-V2, 2024](https://arxiv.org/abs/2405.04434). 해당 논문에서는 특정 설정/비교 기준에서 KV를 93.3% 줄인 결과를 보고한다. 이를 모든 모델에 일반화하지 않는다.

## 4.5 동일한 24 Q heads, head_dim 256, 16 layers의 비교

BF16, 170,000 tokens, 기타 state/오버헤드 제외:

| 종류 | KV head | byte/token | 한 요청 170K KV | MHA 대비 |
|---|---:|---:|---:|---:|
| MHA | 24 | 384 KiB | 62.26 GiB | 100% |
| GQA | 4 | 64 KiB | 10.38 GiB | 16.7% |
| MQA | 1 | 16 KiB | 2.59 GiB | 4.2% |
| MLA | 해당 없음 | latent 차원·RoPE에 종속 | 모델별로 산정 | 일반화 금지 |

**주의:** MHA·GQA·MQA 행은 같은 \(L,H_Q,d_h\)를 놓고 **가상 모델을 비교한 실험 설계**다. 서로 다른 실제 checkpoint의 품질과 속도를 비교한 benchmark가 아니다.

## 4.6 FLOPs/정확도와 Byte 비용의 차이

- MQA/GQA: K/V projection과 KV cache byte 및 decode attention read 감소. 그러나 global softmax token-to-token lookup은 남아 있으므로 prefill attention의 긴 문맥 \(O(T^2)\) 구조는 본질적으로 남는다.
- MLA: token-wise stored representation을 압축하므로 cache 용량과 대역폭을 줄인다. 역시 global softmax attention이면 조회 대상 토큰 수 문제는 남는다.
- 정확도는 attention 이름만으로 선형적으로 순위 매길 수 없다. 모델 학습 품질, width/head 설정, 데이터, RoPE, kernel 및 task에 따라 달라진다.

---

# 5. FlashAttention과 PagedAttention은 다른 층위다

| | FlashAttention | PagedAttention |
|---|---|---|
| 위치 | Attention 연산 kernel/알고리즘 | Serving KV cache 관리·접근 |
| 핵심 아이디어 | tiling + online softmax, 중간 행렬의 HBM materialization 최소화 | OS paging처럼 논리 KV와 물리 block 매핑 분리 |
| 바꾸는 것 | HBM↔SRAM I/O, kernel 효율 | allocator, fragmentation, cache sharing, block table |
| 바꾸지 않는 것 | 기본적으로 exact softmax 의미와 \(O(T^2)\) 계산의 점근 차수 | 학습된 모델의 attention 메커니즘 |
| 시너지 | 같은 모델에서 함께 활용 가능 | 같은 모델에서 함께 활용 가능 |

**중요한 반박:** “FlashAttention을 쓰면 170K의 KV cache가 없어지나요?” → 아니다. prefill에서 중간 attention score matrix를 거대하게 저장하지 않도록 해도 **미래 decode에 필요한 K/V는 살아 있어야 한다.**

원 논문: [FlashAttention, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html), [PagedAttention/vLLM, SOSP 2023](https://doi.org/10.1145/3600006.3613165)

---

# 6. 운영 수치: Qwen3.8-27B, 170K 입력 + 2K 출력

## 6.1 공식 config 기준

[Qwen3.8-27B 공식 README](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md), [config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json):

- Dense model 약 27B parameters; BF16 weights 이론상 약 54 GB(10진수; 실제 checkpoint/부가 텐서 상이)
- hidden size 5,120
- language model layers 64
- Attention 구성: **48 Gated DeltaNet + 16 Gated Full Attention** (3:1)
- Gated Full Attention: Q heads 24, KV heads 4, head_dim 256
- native max_position_embeddings 262,144
- Gated DeltaNet: Q/K 관련 16 heads, V 48 heads, head_dim 128, convolution kernel=4; state dtype과 kernel 구현은 별도 점검
- 별도 vision encoder와 multi-token prediction 관련 항목이 있어서 serving mode/추가 overhead에 주의

**무엇을 계산하나?** 위의 64개 층 전체를 일반적인 GQA KV Cache 공식에 넣으면 **4배 과대평가**한다. 이 모델에서 일반적인 token-wise full-attention KV는 16개 층에서만 발생하고, 나머지 48층에는 recurrent/conv state가 유지된다.

## 6.2 Full Attention raw KV memory

\[
2\times16\times 4\times256\times2 = 65\,536\ \mathrm{bytes/token}=64\,\mathrm{KiB/token}
\]

| 캐시 길이 | 계산된 KV 크기 (단일 sequence) |
|---:|---:|
| 8,192 | 0.50 GiB |
| 32,768 | 2.00 GiB |
| 131,072 | 8.00 GiB |
| 170,000 | 10.376 GiB |
| 262,144 | 16.00 GiB |
| 172,000 (입력 170K + 출력 2K 가정) | 10.498 GiB |

FP8 KV dtype이고 실제 kernel이 그 형식을 지원한다면 raw KV byte/element만 약 절반일 수 있다. 정밀도에 따른 정확도·스케일링·kernel 지원을 반드시 확인한다.

## 6.3 GPU당 메모리 budget

논리적인 GPU당 사용량은 다음처럼 분리한다:

\[
M_{GPU}=M_{weights/shard}+M_{KV/shard}+M_{GDN/state}+M_{activation}+M_{CUDA\ Graph}+M_{workspace}+M_{allocator}+M_{framework}
\]

**예시 — BF16 weights 약 54 GB, TP=2, KV 완전 균등 shard 가정:**
- Weights: rank당 대략 27 GB(이론상; 일부 replicated tensor/embedding/vision은 별도)
- 170K Full Attention KV: rank당 대략 5.19 GiB(이상적으로 4 KV heads가 정확하게 TP2로 분할될 때)
- 추가: GDN/conv state, graph/activation, runtime fragmentation, prefix-state, temporary workspaces

**TP=4:** 4 KV heads가 정확히 각 rank에 하나씩 대응되면 Full Attention KV가 rank당 2.594 GiB가 될 수 있다. **TP=8:** KV heads 4개보다 TP가 크기 때문에 head replication 또는 다른 partition 전략이 등장할 수 있어 단순히 TP8로 8등분된다고 주장하면 안 된다. 모델의 config와 해당 vLLM 버전의 sharding 코드를 확인한다.

**동시성:** 10.38 GiB/request × 동시에 존재하는 10건 = 약 104 GiB의 aggregate logical Full Attention KV다. 그러나 실제로는 요청별 길이, prefix 공유, TP/PP/CP, KV block 충전율, non-KV state, 스케줄러 admission, TPOT 제약 때문에 VRAM 총량으로만 동시 처리 가능 건수를 확정할 수 없다.

## 6.4 가중치와 activation OOM의 차이

Hybrid model에서는 KV usage가 88%여도 activation/FFN 임시 텐서가 커져 CUDA OOM이 날 수 있다. 운영에서 유사한 사례: FFN intermediate 17,408개를 갖는 임시 tensor \([T,17\,408]\)가 BF16이면

\[
M_{temp}\approx T\times17{,}408\times2\ \mathrm{bytes}
\]

- \(T=32\,768\): 약 1.06 GiB
- \(T=65\,536\): 약 2.13 GiB
- \(T=131\,072\): 약 4.25 GiB

이건 **특정 단일 텐서의 계산값**이다. 실제 동시 실행 tensor, graph capture/compile buffer, temporary memory reuse에 따라 peak는 더 복잡하다.

발표 메시지: `gpu-memory-utilization`은 **사용하려는 메모리 budget**과 관계가 있고, `kv_cache_usage_perc`는 **할당된 KV pool의 occupancy**다. KV 90%라고 HBM 점유율이 90%인 것은 아니다.

---

# 7. 멀티 GPU 분산: TP·PP·DP·CP·EP

## 7.1 병렬화의 차이

| 방법 | 분할 대상 | 단일 요청의 footprint | 주된 통신 | 쓰는 이유 |
|---|---|---|---|---|
| TP | layer 내부 weights/heads/연산 | weights 및 일부 KV를 여러 rank에 shard | **매 layer collective** | 한 요청 분산·capacity·GEMM |
| PP | 층 구간(stage) | stage마다 weights/KV layers 나눔 | stage 간 activation send | 아주 큰 모델·multi-node |
| DP | request/replica | replica별 weights/KV 독립 | 일반적으로 inference 요청별 TP collective 없음 | aggregate throughput·격리 |
| CP | 하나의 sequence/context | KV/token 축 shard | attention query/partial result 교환 | 긴 context capacity·prefill |
| EP | MoE experts | expert weights 분산 | **token dispatch/combine all-to-all** | MoE에서 가중치 수용·expert throughput |

**주의:** 실제 프레임워크마다 PP의 pipeline 스케줄링, CP 구현, TP attention head replication, EP token dispatch가 다르다. 용어가 동일하다고 실행 패턴이 동일한 것은 아니다.

## 7.2 TP의 함정: 계산은 나누는데 통신은 추가된다

간단한 ring all-reduce에서 \(N\)개 rank에 메시지 크기 \(S\) bytes가 있을 때 이상화한 rank당 링크 데이터량은

\[
B_{ring,rank}\approx 2\frac{N-1}{N}S
\]

이다. TP transformer layer에서는 row/column parallel linears 등에 따라 보통 반복적인 collective가 필요하다. 최적화 방식에 따라 all-reduce, reduce-scatter, all-gather 등의 구성·횟수가 바뀐다. **“TP=4면 레이어당 무조건 all-reduce 두 번”이라는 암기는 특정 구현을 넘어서면 틀릴 수 있다.**

통신 소요 시간은 메시지 크기와 지연의 조합이다:

\[
T_{comm}\gtrsim T_{latency}+\frac{\mathrm{bytes}}{B_{effective}}
\]

수십~수백 번 반복되는 작은 collectives는 bandwidth보다 **동기화 latency**가 중요해질 수 있다. NVLink가 없고 PCIe만 있는 H100 4-GPU 노드는, NVSwitch로 연결된 H200 8-GPU 노드와 TP trade-off가 다르다.

## 7.3 Node-local vs Multi-node 통신

```text
같은 노드 (GPU0 ↔ GPU1)
GPU HBM ── NVLink/NVSwitch 혹은 PCIe P2P ── GPU HBM

다른 노드 (GPU0 ↔ GPU9)
GPU HBM ─ PCIe ─ NIC/HCA ─ InfiniBand/RoCE fabric ─ NIC/HCA ─ PCIe ─ GPU HBM
                       └─ GPUDirect RDMA 가능 시 host staging 회피
```

- **NVLink/NVSwitch:** GPU 사이 scale-up 고대역폭 fabric. GPU 이름만 보고 실제 연결 존재를 단정하지 않는다.
- **PCIe:** GPU↔CPU/NIC 및 일부 GPU↔GPU P2P 전송 경로. root complex, PCIe 세대/링크 폭/ACS/IOMMU 등이 영향을 준다.
- **InfiniBand:** 노드 간 fabric. ConnectX 계열 HCA, Quantum switch, subnet manager 등의 실제 배치 필요.
- **RDMA:** 원격 메모리에 NIC가 직접 접근하는 전송 방식; GPUDirect RDMA는 가능할 때 GPU memory를 직접 대상으로 쓰며 CPU staging을 줄일 수 있다.
- **RoCE:** Ethernet 기반 RDMA. IB와 동일한 물리적 네트워크는 아니다.
- **NCCL:** 여러 GPU의 collective communication library. 링크 하드웨어 자체가 아니라 collective를 구성하는 소프트웨어 스택이다.

GPUDirect RDMA가 동작한다고 모든 통신이 무조건 NVLink처럼 빠르지는 않다. HCA link speed, PCIe topology, registration, message size, QP/CQ, congestion, orchestration과 scheduler가 관여한다.

## 7.4 이론적 KV transfer 하한 예제

Qwen3.8-27B 170K prompt의 **Full Attention raw KV 11.141 GB(10진수)**를 P에서 D로 그대로 복사한다고 가정한다.

- **400 Gb/s 단일 링크:** 이상적인 단방향 line rate 50 GB/s → 11.141/50≈**0.223초**
- **200 Gb/s 단일 링크:** 25 GB/s → 약 **0.446초**
- **100 Gb/s 단일 링크:** 12.5 GB/s → 약 **0.891초**

위 수치는 **payload only / protocol overhead 0 / 링크 병목만 고려 / 다른 전송 경쟁 0**이라는 비현실적 이상적 하한이다. KDA recurrent state나 metadata, layout conversion/reshard, registration, RPC, 네트워크 contention, synchronization, P/D decode admission, 최종 TTFT를 반영하지 않는다. 여러 HCA가 실제 aggregate되어 있으면 모델이 달라진다.

**중요:** NVLink 900 GB/s 수치는 스펙의 aggregate 양방향 표현과 per-direction 규칙을 따져야 한다. IB 한 링크 50 GB/s와 단순히 나눠 `18배 빠름`을 end-to-end 성능으로 주장하면 안 된다.

## 7.5 NCCL/IB 검증 명령

```bash
nvidia-smi topo -m
nvidia-smi -q -d MEMORY,UTILIZATION
nvidia-smi nvlink -s         # GPU/driver에 따라 지원 여부 다름
ibv_devinfo                 # rdma-core 설치 필요
ibstat                      # infiniband-diags 등 설치 필요
# nccl-tests 빌드/권한 보유 시:
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 4
# 필요할 때 NCCL_DEBUG=INFO 및 NCCL_DEBUG_SUBSYS=INIT,NET,COLL
```

측정은 반드시 **topology→baseline NCCL p2p/collective→IB port counters→serving TTFT/ITL** 순으로 연결한다. `nccl-tests` 수치 하나로 inference TP 성능을 대표하지 않는다.

---

# 8. P/D Disaggregation: 분리하면 빨라지는가?

## 8.1 Integrated 모델의 고민

```text
Client → Router → [vLLM 1 instance: Prefill + Decode]
                             │
             ┌────── compute-heavy Prefill
             └────── HBM/latency-sensitive Decode
```

같은 GPU에서 두 성격의 작업이 resource 경쟁을 한다. 큰 prefill 때문에 decode tail ITL이 튈 수 있고, 둘의 MBT, batch, TP 전략을 독립적으로 최적화하기 어렵다.

## 8.2 P/D 분리 구조

```text
Client
  │
  ▼
Router/Orchestrator
  │
  ├── Prefill pool (P) ─ long prompt, large GEMM, chunk, TTFT
  │          │
  │          │ KV + recurrent/conv state + metadata handoff
  │          ▼
  └── Decode pool (D) ─ token generation, TPOT, small GEMM, CUDA Graph
             │
             ▼
      streamed response
```

장점:
- P와 D에 다른 TP/PP, number of engines, `max-num-batched-tokens`, `max-num-seqs`, kernel/graph 전략 적용
- long prefill이 running decode에 직접 간섭하지 않도록 격리
- 입력 길이와 생성량에 맞춰 P:D GPU 비율 조절

대가:
- P→D state 전송 시간 및 대역폭
- 두 pool에 각각 가중치를 올리는 중복 비용(공유 GPU가 아니라 분리된 engine이면)
- compatible KV layout·TP resharding, memory registration, metadata, transfer retry, admission handoff
- P 또는 D 실패 시 recovery/timeout/stream semantics 복잡

**P/D가 항상 integrated보다 빠르다고 말하면 안 된다.** 짧은 prompt/적은 부하/아주 낮은 TTFT 요구에서는 분리 비용이 이득보다 클 수 있다. 목표는 latency 하나가 아니라 TTFT·TPOT SLO 안의 **goodput**이다.

주요 논문: [DistServe OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin), [Mooncake paper](https://arxiv.org/abs/2407.00079), [TaiChi aggregation vs disaggregation](https://arxiv.org/abs/2508.01989)

## 8.3 Mooncake와 NIXL의 정확한 위치

- Mooncake 논문은 **KV-centric scheduling + disaggregated KV cache**를 포함하는 serving system 아키텍처를 설명한다.
- 오픈소스 Mooncake Transfer Engine은 P/D 또는 KV tier 간 데이터 이동에도 사용된다.
- **NIXL (NVIDIA Inference Xfer Library)**은 VRAM/DRAM/파일 등 이기종 메모리 사이 point-to-point transfer를 통일된 API로 노출하는 데이터 이동 라이브러리. UCX/RDMA 및 여러 backend를 사용한다.
- vLLM의 **NixlConnector**는 KVConnector 계층에서 전송을 통합하는 구현이다.
- NIXL을 도입한다고 자동으로 RDMA, GPU Direct, NVLink 각각의 실제 경로가 선택됐다고 단정하면 안 된다. UCX transport/버전/장치 토폴로지/로그로 확인한다.

출처: [NIXL overview](https://docs.nvidia.com/nixl/getting-started/overview/), [NIXL architecture](https://docs.nvidia.com/nixl/getting-started/architecture/), [vLLM NixlConnector](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)

## 8.4 실무 사례 — 무엇을 근거로 설명할 것인가

본 사례는 실제 팀 내부 운영 맥락을 활용하지만 아래 '관측·가설·대책'은 구분한다:

**상황:** Qwen3.8-27B hybrid model, H200 8GPU 노드, TP2 기반 Prefill/Decode, 긴 입력(p90 약 170K)과 평균 약 2K 출력. 초기 P3:D1 같은 구성에서도 Prefill 측 queue가 병목이 될 수 있다.

**관측 (운영 기록):** TP2 P3:D1에서 prefill 병목; TP1 prefill×2 + TP2 decode×1(총 4GPU) 구성 추가 벤치 계획. 다른 운영 시점에 decode `/health`가 timeout을 반복했고 liveness probe에 의해 kubelet이 container를 재시작한 기록이 있다.

**추론:**
- P queue가 늘어난다고 반드시 “P GPU가 FLOPS 상한에 달했다”는 뜻은 아니다. MBT, long attention, MNS, CUDA Graph, activation/compute, memory pressure, transfer 대기, scheduler admission을 분리해야 한다.
- Liveness 8회 timeout 후 Kill은 *Kubernetes가 실패한 health 응답에 따라 컨테이너를 재시작했다는 증거*다. 원인이 GPU kernel hang, GIL/CPU, event loop stall, KV transfer, 메모리 압력인지는 이 이벤트만으로 알 수 없다.
- H200 NVSwitch 환경의 TP2와 H100 NVLink bridge 미탑재 4GPU 서버의 TP2 성능을 단순 비교하면 안 된다.

**실험 설계:**
1. Integrated TP4 기준선 확보.
2. P1:D1 TP2, P2:D2 TP2, P3:D1 TP2, TP1 P×2 + TP2 D×1을 가능한 동일한 workload/total GPU budget에 맞춰 비교.
3. QPS sweep, arrival distribution, prompt 8K/32K/128K/170K/256K, output 512/2K/4K, 동일 prefix hit 조건.
4. 단독 P 처리량과 D 처리량, P queue time, PD transfer bytes/time, TTFT p50/p95/p99, TPOT/ITL p95, completed goodput, errors, OOM, preemption.
5. `KV/cache pool utilization`과 node VRAM allocation을 별도로 측정.
6. 동일 요청 256K에서 CPU/activation/workspace peak 및 state-aware prefix cache 경로 검증.

**발표에 사용할 실제 내부 자료:**
- [기존 Serving Optimization Study](https://github.com/Soomin-Jung/temp/blob/main/study/llm-serving-optimization/README.md)
- [Serving Engineer Attention View](https://github.com/Soomin-Jung/temp/blob/main/study/attention/90-serving-engineer-view.md)
- [GPU Communication Study](https://github.com/Soomin-Jung/temp/blob/main/study/gpu-architecture/README.md)
- [P/D Disaggregation 기록](https://github.com/Soomin-Jung/temp/blob/main/vllm-stack/pd-disaggregation/README.md)

> 주의: private `temp` 저장소 URL은 발표팀의 접근 권한을 확인해 배포한다. 공개 논문 링크와 내부 사고/로그 링크를 혼합한 자료를 외부로 공유하지 않는다.

---

# 9. Mamba → DeltaNet → Gated DeltaNet → KDA

## 9.1 근본적인 문제: 매 토큰의 과거를 영원히 보관해야 할까?

Softmax full attention은 query가 과거의 개별 토큰을 직접 찾아가는 **content-addressable memory**에 가깝다. 장점은 긴 문맥에서 정확한 위치와 정보를 조회할 수 있다는 점이고, 대가로 token-wise KV가 증가한다.

Recurrent/linear attention은 history를 고정된 크기의 **state**에 압축하고 새 토큰이 들어올 때 state를 업데이트한다. 메모리 크기는 context 길이 자체에 무관하지만, state 용량이 제한되므로 임의의 과거 토큰을 항상 손실 없이 복원할 수는 없다.

## 9.2 Basic linear attention

softmax가 없는 kernelized attention의 한 형태는 행렬 곱 결합법칙을 이용해

\[
S_t=S_{t-1}+\phi(k_t)v_t^\top
\]

같은 상태를 만들고 \(\phi(q_t)^\top S_t\)로 읽는다(정규화 state는 별도). 여기서 state 크기는 \(d_k\times d_v\)에 비례하지만, sequence length에는 무관하다. **가중치 \(\phi\)와 정규화에 따라 정확한 구현 식은 다르다.**

한계: 과거 key→value 매핑이 단순 누적되므로 같은 key의 정보를 교체/삭제하거나 충돌을 해결하기 어려울 수 있다.

## 9.3 Mamba / Mamba-2 — 다른 계보의 state-space 모델

단순화된 selective SSM:

\[
h_t=A_t h_{t-1}+B_t x_t,\qquad y_t=C_t h_t
\]

입력에 따라 transition/selection이 달라지는 recurrent 모델이다. Mamba-2는 structured state space duality를 통해 Transformer 변형과 SSM 사이의 수학적 연결을 정리하고 효율적 구현을 제시한다.

**표현상 주의:** Mamba가 Gated DeltaNet의 직접적인 전신이라거나 `Mamba → DeltaNet`이 코드 계보상 순차적 진화라는 뜻은 아니다. 둘은 서로 연관되는 **recurrent state 기반 대안 계열**이며 Gated DeltaNet 논문은 Mamba-2와 비교/연결해 설명한다.

원 논문: [Mamba](https://arxiv.org/abs/2312.00752), [Mamba-2/SSD, ICML 2024](https://proceedings.mlr.press/v235/dao24a.html)

## 9.4 DeltaNet의 Delta Rule

설명용 상태 정의를 \(S\in\mathbb R^{d_v\times d_k}\)라고 두면:

\[
\hat v_t = S_{t-1}k_t
\]
\[
e_t=v_t-\hat v_t
\]
\[
S_t=S_{t-1}+\beta_t e_t k_t^\top
\]

- 기존 state가 현재 key에 대해 예측한 value \(\hat v_t\)를 계산
- 실제 써야 할 value와의 차이(error)를 얻음
- 그 error만큼 key에 대응하는 연관 기억을 수정

즉, **무작정 더하는 기억이 아니라 예측 오류를 보정하는 기억**이다. 그러나 state 용량이 한정돼 있기 때문에 정확한 random-access attention과 동등하다는 뜻은 아니다.

## 9.5 Gated DeltaNet

Delta correction에 **gate/forgetting**을 결합한다. 업데이트 개념:

\[
\widetilde S_{t-1}=\alpha_t S_{t-1},\quad
S_t=\widetilde S_{t-1}+\beta_t(v_t-\widetilde S_{t-1}k_t)k_t^\top
\]

위 수식은 **원리를 설명하는 단순화된 형태**이며 실제 논문의 gate normalization, kernel, update ordering과 1:1로 동일한 구현 수식이 아닐 수 있다.

해석: gate는 오래되거나 불필요한 state를 약화시키고, delta correction은 새 정보를 **특정 key 연관성에 맞춰 덮어쓰는 역할**을 한다.

원 논문: [Gated Delta Networks, ICLR 2025](https://arxiv.org/abs/2412.06464)

## 9.6 Kimi Delta Attention (KDA)

KDA는 Gated DeltaNet을 확장해 **더 세밀한 channel-wise gating**을 포함한 표현력 높은 recurrent attention을 지향한다. Kimi Linear 기술보고서는 specialized chunkwise 계산으로 GPU 효율을 높였다고 설명한다.

차이의 핵심:
- Gated DeltaNet: gate를 통한 memory erasure + error correction
- KDA: 채널별·차원별 더 세밀한 forgetting/transition 유연성
- Softmax MLA: 여전히 token-wise global latent KV를 기반으로 특정 과거 위치 조회

원 논문: [Kimi Linear: An Expressive, Efficient Attention Architecture (2025)](https://arxiv.org/abs/2510.26692)

## 9.7 Linear attention = 정확도 하락, 이라는 단정은 피한다

원 연구들은 특정 학습 조건에서 좋은 결과를 보고한다. 하지만 모델 전체의 품질은 architecture 외에도 dataset, scale, training recipe, RL, context extension 등에 좌우된다. 엄밀히 말하면 **“KV가 작을수록 정밀도가 떨어진다”**는 단일 그래프가 없다. 메모리와 표현력 사이에 architectural pressure는 있지만 품질의 실측 비교는 통제된 실험이 필요하다.

---

# 10. Hybrid Architecture: 둘 중 하나가 아니라 섞는다

## 10.1 구조 비교

| 계열 | 토큰별 메모리 성장 | 긴 과거의 조회 | 장점 | 핵심 제약 |
|---|---|---|---|---|
| Global MHA/GQA | \(O(T)\) | explicit token-wise softmax | 위치별 정확한 접근 | KV/HBM bandwidth |
| MLA | \(O(T)\), 작은 표현 | explicit latent token-wise | 낮은 KV byte/token | global token search 비용은 유지 |
| Sliding Window | \(O(W)\) (local layer) | 최근 W개만 | bounded attention cache | 먼 문맥의 직접 조회 제한 |
| Mamba / GDN / KDA | \(O(1)\) context-length-dependent state | 고정 recurrent state | 긴 context에서 bounded state | 정보 압축/잊기, state snapshot/rollback |
| Hybrid GDN+Attention | \(O(T)\) global layer + fixed state | 선택된 층에서 global lookup | 메모리·quality 절충 | 다양한 state manager 필요 |

\(O(1)\)은 **시퀀스 길이가 증가할 때 state size가 더 커지지 않는다**는 뜻이지 모든 요청의 state가 0바이트라는 뜻이 아니다. 동시에 수백 요청을 처리하면 고정 크기 recurrent state도 요청 수에 비례해 누적될 수 있다.

## 10.2 Qwen3.8-27B: 48 GDN + 16 Gated Attention

```text
[ GDN + FFN ]
[ GDN + FFN ]
[ GDN + FFN ]
[ Gated Full Attention + FFN ]
       × 16 = 64 layers
```

75%의 층은 token-wise KV cache 대신 recurrent/conv state를 유지한다. 25%의 층에서 global softmax attention을 사용한다. **16개 Full Attention layer가 있으므로 170K KV가 0이 되는 것이 아니다.**

실무 영향:
- 긴 문맥에서 64-layer full GQA 모델 대비 token-wise KV 저장 층수가 크게 줄어든다.
- Prefix caching은 KV block 외에 GDN state snapshot을 맞춰야 할 수 있다.
- Speculative decoding rejected tokens의 recurrent state rollback/replay가 추가로 필요하다.
- CUDA Graph와 hybrid KV manager가 state pointers/shape에 대응해야 한다.
- Layer별 실행 backend가 달라 단순한 64-layer full-attention 모델보다 runtime 검증 항목이 많다.

## 10.3 Kimi K3: 69 KDA + 24 Gated MLA

Kimi K3 공식 공개 자료 기준:
- total 2.8T parameters, activated 약 104B/token, 93 layers
- **69 KDA + 24 Gated MLA**
- 1,048,576 token context (1M)
- 896 routed experts 중 token당 16개 + shared experts 2개
- native MXFP4 weights 및 MXFP8 activations, MoE EP/communication이 중요한 시스템 조건

```text
Long-context sequence axis
  69 × KDA layers       → context-length-independent recurrent state
  24 × Gated MLA layers → token-wise compressed latent KV

Model-width / MoE axis
  large expert bank → expert weight placement + EP all-to-all

Depth axis
  attention residual design → deep model representation flow
```

**K3의 계산은 H200 8장 TP만 늘리면 되는 문제가 아니다.** 2.8T weights, activated expert compute, EP all-to-all, multi-node fabric, KV/state, 1M context scheduler가 동시에 중요하다. KDA+MLA로 attention memory가 절약되어도 MoE weights와 activation transport는 다른 병목일 수 있다.

공식 자료: [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3), [model card](https://huggingface.co/moonshotai/Kimi-K3), [Kimi Linear](https://arxiv.org/abs/2510.26692)

## 10.4 최종적으로 모델의 관점이 변화했다

과거 질문: “컨텍스트가 256K니까 KV cache가 얼마나 큰가?”

현대의 질문: “이 모델은 **어떤 층에서 어떤 state를 얼마나 오랫동안 보존**하며, 이를 **어떤 kernel, cache manager, transport, scheduler**로 실행하는가?”

KV tensor 한 개 공식만으로 모든 hybrid model을 설명하지 못한다. 실제 실행의 **state contract**를 이해해야 한다.

---

# 11. 운영에 대응시키기: bottleneck diagnosis

| 증상 | 가능 원인(확정 아님) | 첫 확인 |
|---|---|---|
| TTFT p95 상승, Prefill waiting 누적 | P compute/attention, MBT 제한, admission, prefix miss, transfer waiting | QPS, token budget, P tokens/s, cache hit, queue reason |
| TTFT는 양호, ITL p95 상승 | D KV reads, weight bandwidth, TP collective, decode batch/graph | D batch, HBM active, NCCL/links, graph replay |
| GPU Util 높고 Tensor Active 낮음 | HBM/CPU/launch/stall/collective | SM/Tensor/DRAM/CPU 동시 관측 |
| TP 늘린 뒤 TPS 불변 | 통신 비용·작은 GEMM·replication | `topo -m`, NCCL traces, per-rank throughput |
| KV usage 낮은데 OOM | activation peak, CUDA Graph, FFN temp, framework reservations | allocated/reserved/peak, MBT sweep, memory snapshots |
| KV usage 90%, waiting 증가 | KV admission 압박, preemption 또는 scheduler budget | KV pool count, running/waiting, preemption |
| P/D transfer 오래 걸림 | UCX fallback, registration, HCA binding, reshard, capacity, concurrency | transport logs, transfer-bytes/time, IB counters |
| P/D Pod 갑자기 죽음 | liveness timeout 또는 OOMKilled 등 | pod events, probe timeout, container termination reason, GPU/CPU logs |
| Hybrid prefix hit 낮음 | block alignment, recurrent-state checkpoint, changing prompts | prefix query/hit tokens, mamba/hybrid state rules |

**관측 설정:** vLLM version별 실제 `/metrics` 확인. v0.30.0 계열에서는 `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `vllm:prefix_cache_hits`, `vllm:prefix_cache_queries`, `vllm:time_to_first_token_seconds` 등을 확인하고, 현재 문서와 실제 version 차이를 검증한다. Prometheus counter의 자동 `_total` 접미사 및 metric rename 가능성을 고려해야 한다.

메트릭 공식 문서: [vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/)

## 11.1 실험 설계 원칙

**A. Workload를 고정한다**
- request arrival rate (open-loop/closed-loop 구별)
- 입력 길이 분포(p50/p90/p99), output 길이
- prefix reuse ratio / prompt uniqueness
- reasoning/tool/nonthinking 및 streaming/nonstreaming 동작

**B. H/W를 고정한다**
- 동일 H200/H100 type, GPU placement, power profile, MIG 설정
- 동일 NIC, IB port/link, topology, NUMA affinity
- CPU cores/limits, host memory, vLLM build/CUDA/driver

**C. 한 번에 하나의 topology 축만 바꾼다**
- Baseline Integrated TP1/2/4
- P1:D1 TP2 → P2:D2 TP2 → P3:D1 TP2
- TP1 Prefill ×2 + TP2 Decode ×1
- MTP off→on, graph mode 등은 별도 실험

**D. 종료 조건은 SLO와 goodput이다**
- p95 TTFT, p95 TPOT/ITL, p99 tail, error rate, request/s
- prompt tok/s, output tok/s, latency per request length bucket
- GPU당 **SLO-qualified completed requests/sec**와 cost/token

**E. 긴 입력의 특이점**
- P/D state transfer의 byte와 실제 transfer time 분리
- 256K max context는 prompt + generated + reasoning/history + auxiliary tokens의 총합일 수 있음
- Prefill MBT가 커질수록 activation buffer가 KV 영역을 침범할 수 있음
- `max-num-batched-tokens`는 한 scheduling iteration token budget이지 요청당 최대 입력 길이와 동일하지 않음

---

# 12. 발표 시 나올 법한 반문과 정답

### Q1. MQA가 MHA보다 무조건 정확도가 나쁜가?
아니다. KV head 공유가 표현력 제약을 추가할 수 있고 논문은 경향/트레이드오프를 보여주지만, checkpoint 학습과 작업에 따라 실측 결과가 달라진다.

### Q2. GQA가 full attention의 \(O(T^2)\)를 줄이나?
KV cache byte와 일부 projection/IO를 줄일 수 있지만, 전체 과거 토큰에 softmax attention을 계산한다면 시퀀스 차원의 2차 계산 구조는 유지된다.

### Q3. MLA와 GQA는 같은 기술 아닌가?
둘 다 memory-efficient softmax attention 계열이지만, GQA는 KV head를 공유하고 MLA는 K/V representation 자체를 저차원으로 압축한다.

### Q4. FlashAttention을 쓰면 KV가 필요 없나?
아니다. prefill attention의 중간 score matrix를 적게 materialize하게 해도 decode용 과거 KV cache는 필요하다.

### Q5. PagedAttention이 FlashAttention보다 정확도가 높은가?
정확도 알고리즘의 종류가 다른 비교가 아니다. 하나는 runtime IO kernel, 다른 하나는 memory management다.

### Q6. `num_key_value_heads=4`면 TP8에서 KV가 GPU 8장에 나뉘나?
반드시 아니다. 물리 head 분할 제약으로 replication 또는 다른 분할/실행 방식이 나타날 수 있다. 실제 모델 backend 확인.

### Q7. 8-GPU H200 1.1TB VRAM이면 1TB 모델이 올라가나?
필수 weight 외에 per-rank overhead, sharding과 replication, activation, KV, experts, memory allocator가 필요하다. 합산 VRAM만으로 가능하다고 확정할 수 없다.

### Q8. P/D 분리하면 TTFT도 TPOT도 무조건 개선되나?
아니다. phase interference를 줄이고 최적화 자유도를 늘리지만 transfer overhead와 capacity duplication이 있다. workload에 따라 integrated가 더 나을 수 있다.

### Q9. NVLink가 없으면 GPU 간 통신이 불가능한가?
아니다. PCIe P2P 또는 CPU staging 경로가 가능할 수 있으나, topology/driver 설정과 사용 kernel에 따라 성능이 달라진다.

### Q10. KDA를 쓰면 long context에서 정보 손실이 0인가?
아니다. 고정 차원 state는 압축 표현이고, 임의의 토큰을 모두 정확하게 별도 저장하는 구조가 아니다. Kimi K3는 주기적으로 MLA를 함께 배치한다.

### Q11. GDN이 있으면 KV cache가 0인가?
순수 recurrent layer의 token-wise KV는 필요 없을 수 있지만 recurrent/conv state가 있다. Hybrid 모델의 일부 global attention 층에는 KV가 남는다.

### Q12. KDA와 Mamba는 같은 메커니즘인가?
둘 다 state를 업데이트한다는 큰 공통점이 있지만, selective SSM과 delta-rule associative memory는 업데이트 의미와 매개변수화가 다르다.

### Q13. 170K prompt에 왜 10GiB KV가 실제 VRAM 전체가 아닌가?
이 계산은 Full Attention raw KV tensor만 센다. weights, recurrent state, activation, CUDA Graph, workspace, allocator, reserve/paging이 별도다.

### Q14. IB 400Gbps면 10GiB를 0.2초에 실전 전송하나?
그건 이상적인 물리 링크 payload-only 하한일 뿐이다. effective payload bandwidth와 KV reshard/registration/metadata/동시 요청을 측정해야 한다.

### Q15. KV cache가 줄면 모든 지표가 좋아지는가?
Capacity가 늘어도 compute-bound prefill, CPU launch, TP collective, FFN/MoE가 병목이면 TPS 또는 latency는 개선되지 않을 수 있다.

---

# 13. 추천 발표 슬라이드 18장

1. 타이틀: 왜 GPU가 많아도 LLM은 느린가? (오늘의 질문 3개)
2. AI inference 요청 구조: Prefill→KV/State→Decode
3. CPU vs GPU 구조: CPU↔PCIe↔HBM↔L2↔SM/Tensor Core
4. Roofline: FLOPS는 빠른데 메모리가 느리면?
5. H200/H100 비교: GB / TB/s / TFLOPS / fabric
6. KV cache가 없는 autoregressive decode의 중복 계산
7. Q/K/V와 softmax attention 원리
8. KV cache 기본 공식과 토큰당 byte 계산
9. MHA vs MQA vs GQA head 배치 (가장 중요한 도식)
10. MLA latent cache와 GQA와의 구조적 차이
11. FlashAttention vs PagedAttention vs Prefix Cache (층위 구분)
12. Qwen3.8-27B 실제 config: 64 layers, 48 GDN + 16 attention
13. 170K 한 요청 KV 10.38 GiB, 256K에서 16 GiB
14. TP/PP/DP/CP/EP와 NVLink/PCIe/IB 통신 지형도
15. 400Gb/s에서 170K raw KV의 최소 전송시간 0.223초
16. Integrated vs P/D + NIXL, P/D가 없애는 간섭과 새로 만든 비용
17. Mamba·Delta Rule·GDN·KDA의 state update 비교, Kimi K3 69 KDA+24 MLA
18. 운영 사례 / SLO-oriented experimental design / 최종 메시지

**시간이 60분이라면:** 3~5장을 합치고, 11·14·17장을 압축하며, 실무 사례 12~16을 중심으로 진행.

---

# 14. 발표 대상 모델에서 확인할 Attention 구조

이 장은 발표 준비일인 **2026년 10월 30일**에 공개된 모델 카드·기술 보고서·공식 설정을 기준으로 한다. 모델 API 제품명과 공개 가중치 checkpoint는 이름이 비슷해도 구조가 다를 수 있다. 특히 `GLM-5.3`과 `GLM-5.3-Flash`는 같은 구조로 묶으면 안 된다.

| 모델 | 공개 자료에서 확인되는 구조 | KV/상태 관점 | 해석할 때 주의할 점 |
|---|---|---|---|
| Kimi K3 | 93 layers, 69 Kimi Delta Attention(KDA) + 24 Gated MLA, 약 1M context | KDA layers는 recurrent state, Gated MLA layers는 token-wise compressed latent cache | KLA가 아니라 공식 용어는 KDA다. “3:1”은 3 KDA 뒤 MLA를 둔 공개 layer pattern의 요약이다. |
| GLM-5.3 | GLM-5.2와 같은 base model이며, 공개 설명상 차이는 post-training | GLM-5.2 base가 쓰는 sparse attention 계열을 참고하되, GLM-5.3 자체에 새 hybrid architecture를 귀속하지 않는다 | GLM-5.3-Flash의 architecture 변경을 GLM-5.3에 잘못 적용하지 않는다. |
| GLM-5.3-Flash | 새 base, sparse + linear hybrid. 공개 config/model documentation은 45 layers 중 34 KDA + 11 DeepSeek Sparse Attention(DSA)을 명시 | KDA recurrent state + DSA의 compressed latent KV/indexer state 혼합 | 표준 GLM-5.3과 다른 모델이다. 정확한 layer pattern, config, runtime support는 checkpoint/version으로 확인한다. |
| DeepSeek-V4-Pro | CSA(Compressed Sparse Attention)와 HCA(Heavily Compressed Attention)를 결합한 hybrid, 1M context | CSA는 압축 history에서 indexer가 고른 항목을 sparse lookup; HCA는 더 압축한 history에 dense lookup; 최근 raw local branch 병행 | HCA는 “hierarchical”이 아니라 **Heavily Compressed Attention**이다. CSA/HCA는 모두 압축이지만 sparse/dense 조회 방식이 다르다. |
| DeepSeek-V4.1-Flash | 공개 보고서는 Causal Encoder-Decoder(CED), 20-layer causal encoder + 20-layer decoder, CSA2 등을 설명 | decoder global KV가 encoder 최종 hidden states에서 투영되는 구조와 replay 기반 SWA 관리가 들어감 | V4-Pro의 단순 소형판으로 가정하지 않는다. 보고된 byte/token 수치는 그 모델의 포맷·구현에만 적용한다. |

## 13.1 DeepSeek V4의 CSA와 HCA

V4는 한 가지 방식으로 모든 거리를 처리하지 않는다. 최근 문맥은 세밀한 raw KV로 보고, 장거리 문맥은 더 작은 sequence representation으로 변환한다.

### CSA

CSA는 긴 이력을 compression window로 압축한 뒤, Lightning Indexer가 현재 query와 관련이 큰 compressed entries를 선택해 attention한다. 공개 구현 설명에서 대표 기본값으로 압축률 `m=4`와 index top-k가 제시되지만 실제 모델 config를 기준으로 확인해야 한다.

기호로 단순화하면:

\[
N_{compressed}\approx\left\lceil\frac{T}{m}\right\rceil,\qquad
\mathcal I_t=\operatorname{TopK}(s(q_t,c_1),\ldots,s(q_t,c_{N_{compressed}}))
\]

\[
o_t=\operatorname{Attention}(q_t,K_{\mathcal I_t},V_{\mathcal I_t})
\]

- `T`: 누적 token 수
- `m`: compression stride 또는 compression ratio를 나타내는 모델별 계수
- `c_j`: compressed history entry
- `s`: indexer scoring function
- `\mathcal I_t`: query t가 선택한 compressed entry 위치 집합

이것은 아키텍처를 이해하기 위한 추상식이다. 실제 compression projection과 indexer score 수식은 DeepSeek-V4 보고서의 세부 정의를 따라야 한다. **Core attention에서 top-k만 읽는다는 사실만으로 전체 계산량이 O(k)라고 말하면 안 된다.** Indexer 후보 검색, compression, gather, prefill의 모든 query 처리도 비용에 포함된다.

### HCA

HCA는 훨씬 큰 단위로 sequence를 압축해 모든 compressed entries를 dense attention으로 읽는다. 공개 자료의 대표 compression ratio는 `m′=128` 수준이며, CSA처럼 per-query top-k indexer를 거치지 않는다.

\[
N_{HCA}\approx\left\lceil\frac{T}{m'}\right\rceil
\]

한 query의 장거리 경로는 이 압축된 항목 전체에 대한 softmax attention을 수행한다. 별도 local window가 최근 토큰의 정밀한 관계를 보완한다. 따라서 HCA는 sparse attention이 아니라 **compressed sequence에 대한 dense attention**이다.

### 메커니즘 비교

| 구조 | 저장하는 장거리 표현 | 현재 query가 조회하는 범위 | 장점 | 지불하는 비용 |
|---|---|---|---|---|
| GQA | token별 K/V, KV head 공유 | 전체 과거 token | 위치별 상세 retrieval 유지 | context에 선형인 KV, decode HBM read |
| MLA | token별 compressed latent | 전체 latent token | token당 cache byte 감소 | latent token 수와 global scan은 남음 |
| DSA | token별/latent 표현과 indexer | 선택된 token/entry | 불필요한 core attention lookup 감소 | indexer 및 불규칙 gather 비용 |
| CSA | sequence-compressed KV entries | 선택된 compressed entries | 압축과 sparse lookup을 함께 적용 | compressor, indexer, local raw KV |
| HCA | 더 강하게 압축된 sequence entries | 압축된 전체 entries | 매우 작은 장거리 sequence 길이 | 압축 과정과 정보 손실 위험, dense lookup |
| KDA/GDN | token별 KV 대신 recurrent state | 현재 state 전체 | context 길이에 무관한 state 크기 | 고정 state 용량, snapshot/rollback 난이도 |

V4 보고서는 1M context 조건에서 V4-Pro의 single-token inference FLOPs와 KV cache를 DeepSeek-V3.2의 각각 27%, 10%로 보고한다. 이것은 논문의 모델·비교 설정에서 나온 결과이지 모든 HCA/CSA 구현의 보장 비율이 아니다.

## 13.2 DeepSeek-V4.1-Flash를 별도 행으로 두는 이유

V4.1-Flash의 공개 모델 보고서는 encoder-decoder 형태를 사용한다. 이 설계에서는 decoder의 global KV가 각 decoder layer만으로부터 만들어지는 대신 encoder 최종 hidden states를 투영해 생성된다. 또 CSA2 layer mode 공유와 SWA Bounded Replay, FP4 main KV를 결합했다고 설명한다. 해당 보고서가 제시한 `890 bytes/token` global KV는 모델 전용 저장 방식과 구성값이므로 이를 일반 Transformer의 `2LTH_{kv}d_hb` 공식에 대입하거나 다른 V4 모델에 그대로 옮기지 않는다.

운영적으로는 다음이 중요하다.

1. 이름의 `.1`이 architecture가 unchanged라는 뜻은 아니다.
2. 일반 Decoder-only LLM의 KV 산식만으로 CED의 전체 state를 추정할 수 없다.
3. encoder output, decoder state, CSA indexer, replay window, quantization scale을 함께 확인해야 한다.
4. 공개 model report와 실제 serving engine이 지원하는 구현은 별도로 확인한다.

## 13.3 발표에서 반드시 분리해서 말할 것

- **표준 GLM-5.3:** 기존 base 구조 위에 post-training을 적용한 모델이라고 공식 README가 설명한다.
- **GLM-5.3-Flash:** 새 base와 sparse+linear hybrid 구조를 사용한다. 공개 모델 문서/config에는 KDA와 DSA layer 구성이 나온다.
- **DeepSeek-V4-Pro:** CSA/HCA 계열의 compressed softmax attention이다.
- **DeepSeek-V4.1-Flash:** CED, CSA2 등 V4-Pro와 다른 변경이 공개되어 있다.
- **Kimi K3:** KDA와 Gated MLA를 섞은 hybrid다.

이 차이는 단지 모델 홍보 이름의 구별이 아니다. 요청 하나당 어떤 cache/state를 저장하고, prefill/decode 중 어떤 데이터를 GPU 메모리와 fabric으로 이동하는지 결정한다.

---

# 15. 공식 논문·기술문서: 권장 읽기 순서

## A. Foundation

1. Vaswani et al., **Attention Is All You Need** (NeurIPS 2017): https://arxiv.org/abs/1706.03762
   - 왜 Q/K/V와 multi-head attention인가? \(QK^T\)와 softmax의 의미.
2. Williams et al., **Roofline: An Insightful Visual Performance Model for Multicore Architectures** (2009): https://doi.org/10.1145/1498765.1498785
   - FLOPS 대 memory bandwidth, arithmetic intensity.
3. NVIDIA **HGX AI Factory Reference Architecture**: https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html
   - 하드웨어 스펙 확인. GPU 모델/폼팩터/노드 구성 구별.

## B. Attention과 메모리

4. Shazeer, **Fast Transformer Decoding: One Write-Head is All You Need** (2019): https://arxiv.org/abs/1911.02150
   - Multi-Query Attention, KV memory bandwidth에 대한 원문.
5. Ainslie et al., **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.298/
   - MHA/MQA 사이의 공유 수준, uptraining.
6. DeepSeek-AI, **DeepSeek-V2** (2024): https://arxiv.org/abs/2405.04434
   - MLA와 low-rank KV compression.
7. Dao et al., **FlashAttention** (NeurIPS 2022): https://arxiv.org/abs/2205.14135
   - I/O aware exact attention의 핵심.
8. Kwon et al., **Efficient Memory Management for LLM Serving with PagedAttention** (SOSP 2023): https://doi.org/10.1145/3600006.3613165
   - Paged KV allocation, prefix sharing, fragmentation.

## C. LLM Serving과 분산

9. Agrawal et al., **Sarathi-Serve** (OSDI 2024): https://www.usenix.org/conference/osdi24/presentation/agrawal
   - chunked prefill, stall-free scheduling, latency/throughput.
10. Zhong et al., **DistServe** (OSDI 2024): https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
    - P/D 분리, TTFT/TPOT goodput, network-aware placement.
11. Qin et al., **Mooncake** (2024 preprint / 후속 출판): https://arxiv.org/abs/2407.00079
    - KV-centric scheduler, disaggregated cache, remote KV tiers.
12. NVIDIA **NIXL Overview**: https://docs.nvidia.com/nixl/getting-started/overview/
    - data movement abstraction과 UCX/RDMA backend.
13. vLLM **NixlConnector Usage Guide**: https://docs.vllm.ai/en/latest/features/nixl_connector_usage/
    - 실제 vLLM KV transfer 연결과 호환성.
14. Wang et al., **Prefill-Decode Aggregation or Disaggregation? (TaiChi)** (2025): https://arxiv.org/abs/2508.01989
    - P/D가 항상 이기는 게 아닌 이유, SLO trade-off.

## D. State-space / Linear / Hybrid

15. Gu & Dao, **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (2023): https://arxiv.org/abs/2312.00752
    - state-space recurrence와 selective update.
16. Dao & Gu, **Transformers are SSMs / Mamba-2** (ICML 2024): https://proceedings.mlr.press/v235/dao24a.html
    - structured state space duality.
17. Yang et al., **Gated Delta Networks** (ICLR 2025): https://arxiv.org/abs/2412.06464
    - delta-rule + gating, comparison with Mamba-2.
18. Moonshot AI, **Kimi Linear** (2025): https://arxiv.org/abs/2510.26692
    - Kimi Delta Attention(KDA), chunkwise kernel, hybrid KDA+MLA.
19. Qwen **Qwen3.8-27B Official README/config**: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md / https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
    - 실제 48 GDN+16 Gated Attention, 24 Q/4 KV, head dimension.
20. DeepSeek-AI, **DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence** (2026): https://arxiv.org/abs/2606.19348
    - CSA와 HCA의 압축·조회 방식, 1M context 비용 비교. HCA는 Heavily Compressed Attention.
21. DeepSeek-AI, **DeepSeek-V4.1-Flash Model Report/Card**: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
    - CED, CSA2, FP4 KV, SWA Bounded Replay. 모델 전용 memory 수치는 일반식과 구분.
22. Z.ai, **GLM-5 official repository and GLM-5.3-Flash model card**: https://github.com/zai-org/GLM-5 / https://huggingface.co/zai-org/GLM-5.3-Flash
    - 표준 GLM-5.3과 Flash의 base/architecture 차이 및 공개 config 확인.
23. Moonshot AI **Kimi-K3 Official Report & Model Card**: https://github.com/MoonshotAI/Kimi-K3 / https://huggingface.co/moonshotai/Kimi-K3
    - 93 layers, 69 KDA + 24 gated MLA, 2.8T MoE, 1M.
21. PyTorch, **Hybrid Models as First-Class Citizens in vLLM**: https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/
    - recurrent state + paged KV의 serving 운영 복잡도.

---

# 16. 사용자 temp 저장소의 기존 심화 자료

사용자가 이미 정리해 둔 상세 내용은 아래 순서로 참고. GitHub private repository일 수 있으므로 접근권한을 확인한다.

- [Attention Architecture 전체 목차](https://github.com/Soomin-Jung/temp/blob/main/study/attention/README.md)
- [01 MHA/MQA/GQA](https://github.com/Soomin-Jung/temp/blob/main/study/attention/01-full-attention.md)
- [02 MLA / Low Rank](https://github.com/Soomin-Jung/temp/blob/main/study/attention/02-mla-low-rank-attention.md)
- [03 Linear/Recurrent Attention](https://github.com/Soomin-Jung/temp/blob/main/study/attention/03-linear-recurrent-attention.md)
- [10 Kimi K3 Attention](https://github.com/Soomin-Jung/temp/blob/main/study/attention/10-kimi-k3.md)
- [90 Serving Engineer View](https://github.com/Soomin-Jung/temp/blob/main/study/attention/90-serving-engineer-view.md)
- [GPU Architecture 00~13](https://github.com/Soomin-Jung/temp/blob/main/study/gpu-architecture/README.md)
- [GPU NVLink/NVSwitch](https://github.com/Soomin-Jung/temp/blob/main/study/gpu-architecture/10-pcie-nvlink-nvswitch.md)
- [RDMA/GPUDirect RDMA](https://github.com/Soomin-Jung/temp/blob/main/study/gpu-architecture/11-rdma-and-gpudirect-rdma.md)
- [IB/RoCE hardware](https://github.com/Soomin-Jung/temp/blob/main/study/gpu-architecture/12-infiniband-roce-and-hardware.md)
- [NCCL 실습](https://github.com/Soomin-Jung/temp/blob/main/study/gpu-architecture/13-nccl-collectives-observability-labs.md)
- [Serving optimization, scheduler/cache/cuda graphs/P-D/TP](https://github.com/Soomin-Jung/temp/blob/main/study/llm-serving-optimization/README.md)
- [Mooncake/NIXL P/D 비교](https://github.com/Soomin-Jung/temp/blob/main/vllm-stack/pd-disaggregation/README.md)
- [Kimi K3 모델별 Deep Dive](https://github.com/Soomin-Jung/temp/blob/main/models/kimi-k3/README.md)

---

# 17. 맺음말: 발표의 철학적 결론

**과거를 모두 기억하는 것은 비싸고, 압축해서 기억하는 것은 정보를 잃을 수 있으며, 여러 장치에 나누어 기억하는 것은 전달비용을 만든다.**

MHA/GQA/MLA/KDA의 차이는 결국 **무엇을 기억하고, 무엇을 버리고, 언제 어떤 값을 다시 읽는가**의 차이다. GPU가 가진 FLOPS와 HBM, NVLink/IB는 이 정보 흐름을 실행하기 위한 물리적 자원이다. 따라서 serving architecture는 ‘빠른 엔진 옵션을 고르는 일’이 아니라 **서비스 SLO를 지키는 데 가장 비싼 자원을 찾아 비용을 다른 여유 자원으로 옮기는 일**이다.

**슬라이드 마지막 한 문장:** “LLM 최적화는 GPU를 더 쓰는 방법이 아니라, 어떤 정보를 어디에 저장하고 얼마나 자주 이동할지를 설계하는 일이다.”
