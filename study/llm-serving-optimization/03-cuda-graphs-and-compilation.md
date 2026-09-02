# 03. CUDA Graph, Compilation, Capture Size

## 1. CUDA Graph가 해결하는 문제

CUDA Graph는 GPU 계산량 자체를 줄이는 기술이 아니다.

일반 stream execution에서는 host가 kernel을 하나씩 제출한다.

~~~text
CPU
  launch kernel A
  launch kernel B
  launch kernel C
  launch kernel D
~~~

각 launch에는 driver/runtime overhead가 있다.

kernel이 큰 Prefill에서는 이 overhead 비율이 작다.

반대로 Decode처럼 kernel 하나하나가 짧으면:

~~~text
GPU kernel execution  짧음
CPU launch overhead   상대적으로 큼
~~~

이 될 수 있다.

CUDA Graph는 반복되는 launch sequence를 한 번 capture한 뒤 replay한다.

~~~text
capture once
A -> B -> C -> D

runtime
graph launch
graph launch
graph launch
~~~

즉 핵심 효과는 kernel execution cost 감소가 아니라 host submission / launch overhead 감소다.

NVIDIA CUDA Programming Guide도 graph의 핵심 효과를 반복 workflow의 CPU launch cost를 줄이는 것으로 설명한다.

## 2. 언제 효과가 큰가

CUDA Graph 효과가 커지는 조건:

~~~text
small batch
short-running kernels
many kernels per forward
high model layer count
CPU dispatch overhead 큼
Decode-heavy workload
~~~

효과가 작아지는 조건:

~~~text
large Prefill GEMM
very long attention kernel
HBM-bound long-context decode
large communication-bound TP
~~~

즉 graph가 이미 작은 비중인 경우에는 capture size를 아무리 세밀하게 잡아도 큰 gain이 없다.

## 3. Capture가 어려운 이유

Graph replay는 capture 당시와 runtime의 실행 구조가 충분히 같아야 한다.

변하는 것:

~~~text
batch size
number of tokens
request count
KV block table
sequence length
attention mode
speculative query length
workspace pointer
control flow
~~~

Serving engine은 이 dynamic state를 indirection buffer, static workspace, dispatcher, graph bucket 등으로 감춘다.

그래서 CUDA Graph 최적화는 단순 on/off가 아니라:

~~~text
어떤 workload shape를 graph key로 분류하는가?
어떤 op까지 capture 가능한가?
어떤 batch가 eager fallback 되는가?
~~~

의 문제다.

## 4. vLLM V1의 CUDA Graph mode

현재 주요 mode:

~~~text
NONE
PIECEWISE
FULL
FULL_DECODE_ONLY
FULL_AND_PIECEWISE
~~~

중요:

- NONE / PIECEWISE / FULL은 concrete runtime mode다.
- FULL_DECODE_ONLY와 FULL_AND_PIECEWISE는 workload 종류에 따라 runtime mode를 바꾸는 dual mode다.

## 5. NONE

~~~text
Decode        eager
Prefill/mixed eager
~~~

장점:

- 가장 깨끗한 control
- graph pool 없음
- compatibility issue 분석에 좋음
- startup capture time 없음

단점:

- small-batch Decode에서 launch overhead가 그대로 드러남

사용 목적:

~~~text
correctness baseline
performance control
graph-related regression isolation
~~~

## 6. PIECEWISE

모델 전체를 하나의 graph로 capture하지 않는다.

graph-compatible partition을 capture하고 unsupported/custom op는 graph 밖에서 실행한다.

~~~text
compiled partition A -> graph
attention/custom op  -> eager
compiled partition B -> graph
~~~

장점:

- attention backend compatibility가 가장 좋음
- dynamic Prefill/mixed workload에 유연함
- hybrid attention 모델에서 안전한 기본 후보

단점:

- graph boundary마다 host/runtime overhead가 일부 남음
- full graph보다 Decode latency 개선 폭이 작을 수 있음
- piecewise graph pool memory 사용

Prefill 전용 engine에서는 보통 NONE과 PIECEWISE가 먼저 비교할 가치가 있다.

## 7. FULL

전체 forward를 full CUDA Graph로 capture한다.

이론적으로:

~~~text
model forward 전체
-> 하나의 graph replay
~~~

장점:

- launch overhead 최소화 가능

제약:

- 모든 backend가 full graph를 지원하지 않음
- dynamic Prefill shape는 capture compatibility가 어렵다.
- attention backend가 uniform decode만 지원하는 경우 requested FULL이 다른 mode로 downgrade될 수 있다.

따라서 설정 파일의 requested mode보다 startup의 resolved mode가 더 중요하다.

## 8. FULL_DECODE_ONLY

~~~text
uniform Decode
  -> FULL graph

Prefill / mixed
  -> eager
~~~

P/D disaggregation의 Decode engine에 매우 자연스럽다.

이유:

~~~text
Decode engine은 거의 uniform decode만 수행
Prefill graph pool을 만들 이유가 적음
~~~

따라서:

~~~text
FULL decode latency benefit
+
PIECEWISE pool memory 절약
~~~

을 노릴 수 있다.

vLLM 공식 CUDA Graph 문서도 이 mode를 P/D Decode instance에 적합한 사용 예로 설명한다.

## 9. FULL_AND_PIECEWISE

~~~text
uniform Decode
  -> FULL

Prefill / mixed
  -> PIECEWISE
~~~

범용 serving engine에서 가장 공격적인 mode다.

장점:

- Decode full graph
- Prefill/mixed piecewise graph
- 하나의 engine에서 다양한 workload 대응

비용:

- graph pool 가장 큼
- capture startup time 증가
- role-specific P/D에서는 사용하지 않는 graph까지 가질 수 있음

따라서 P/D에서는:

~~~text
Prefill-only
  PIECEWISE

Decode-only
  FULL_DECODE_ONLY
~~~

가 더 경제적일 수 있다.

## 10. Attention backend capability가 최종 mode를 결정한다

모델 architecture 이름만으로 full graph 가능 여부를 결정할 수 없다.

실제 결정 요소:

~~~text
attention backend
model-specific custom op
KV connector
speculative decode path
compile mode
GPU architecture
~~~

vLLM 공식 문서의 일반적인 capability 분류에서도 FlashAttention 3처럼 broad full-graph support를 가진 backend와, FlashInfer/FlashMLA/Mamba처럼 uniform Decode 중심으로 support하는 backend가 구분된다.

그래서 반드시 startup에서 다음을 기록한다.

~~~text
requested cudagraph mode
resolved cudagraph mode
attention backend
capture size list
~~~

## 11. Capture size의 진짜 의미

capture size는 context length bucket이 아니다.

중요한 dispatch dimension은 execution batch shape다.

특히 Decode에서:

~~~text
MTP OFF
query length = 1

MTP K3
query length = 4
~~~

active request가 32개면:

~~~text
K0
num_tokens ~= 32

K3
num_tokens ~= 128
~~~

가 된다.

즉 동일 concurrency라도 MTP K가 바뀌면 graph shape가 바뀐다.

## 12. 왜 capture size를 촘촘히 잡나

runtime shape가 정확한 graph bucket과 맞지 않으면 engine은 더 큰 captured shape로 padding할 수 있다.

예:

~~~text
captured:
64
128

actual:
70

-> 128 graph로 padding
~~~

그러면 58 token slot의 padded work가 생길 수 있다.

capture size를 더 촘촘히:

~~~text
64
80
96
112
128
~~~

으로 잡으면 padding waste를 줄일 수 있다.

하지만 graph 개수가 늘어나면:

~~~text
capture startup time 증가
graph memory 증가
KV/state pool 감소 가능
~~~

이라는 반대 비용이 생긴다.

## 13. Capture density의 최적화 목표

capture bucket은 가장 촘촘한 것이 최적이 아니다.

목표는 실제 batch-size distribution의 높은 확률 영역에 충분한 density를 배치하는 것이다.

예:

~~~text
runtime active seq distribution

1~8      5%
9~16    15%
17~32   55%
33~64   20%
65+      5%
~~~

이면 16~40 부근을 촘촘히 하고 큰 size는 드문 bucket만 두는 방식이 합리적이다.

## 14. max_cudagraph_capture_size

이 값은 graph coverage의 상한이다.

vLLM은 explicit capture size list가 없으면 일정 패턴으로 size를 자동 생성한다.

기본 max는 대략:

~~~text
min(max_num_seqs * 2, 512)
~~~

방향으로 제한된다.

이것은 다음 trade-off를 반영한다.

~~~text
larger max
-> more high-concurrency graph coverage

but
-> more capture memory
-> longer startup
-> diminishing latency gain
~~~

## 15. MTP와 capture size

MTP K에서는 query length가 1+K이므로 capture size alignment가 달라진다.

개념적으로:

~~~text
target concurrent seqs = N
MTP K

required token-shape coverage
~= N * (1 + K)
~~~

예:

| Active seq | K | token shape |
|---:|---:|---:|
| 32 | 0 | 32 |
| 32 | 1 | 64 |
| 32 | 3 | 128 |
| 64 | 3 | 256 |

그래서 MTP K를 바꾸면서 capture size를 고정하면 graph hit pattern도 바뀐다.

## 16. max-num-seqs와 graph는 연결돼 있다

max-num-seqs는 단순 scheduler ceiling이 아니다.

graph warmup/capture envelope, recurrent metadata buffer에도 영향을 줄 수 있다.

특히 hybrid recurrent model에서는:

~~~text
FULL decode graph
+
max-num-seqs
+
available recurrent-state blocks
~~~

사이에 compatibility constraint가 생길 수 있다.

따라서 graph를 켰더니 initialization이 실패했다면 GPU graph 기능보다 cache geometry를 먼저 확인해야 할 수 있다.

## 17. Graph memory는 KV capacity와 경쟁한다

~~~text
HBM
├─ weights
├─ CUDA Graph pools
├─ workspaces
├─ communication buffers
└─ KV/state pool
~~~

capture size 증가:

~~~text
graph hit rate 상승 가능
TPOT 감소 가능

but
KV/state pool 감소
max concurrency 감소 가능
~~~

따라서 최종 선택 metric:

~~~text
TPOT
output tok/s
graph coverage
startup time
graph memory
maximum concurrency
~~~

를 함께 본다.

## 18. Prefill과 Decode의 권장 비교

### Prefill

첫 A/B:

~~~text
NONE
PIECEWISE
~~~

이유:

- Prefill은 non-uniform/dynamic shape가 많다.
- FULL graph compatibility가 낮을 수 있다.
- large compute가 launch overhead를 이미 희석할 수 있다.

### Decode

첫 A/B:

~~~text
NONE
FULL_DECODE_ONLY
~~~

후속:

~~~text
FULL_AND_PIECEWISE
~~~

MTP와 hybrid backend에서 FULL_DECODE_ONLY 대비 차이를 본다.

## 19. Graph가 효과 없는 패턴

~~~text
Graph ON
TPOT 거의 동일

SM Active 높음
DRAM Active 매우 높음
CPU 낮음
~~~

이면 launch-bound가 아니었다.

Graph tuning을 중단하고 HBM/KV/TP를 본다.

## 20. Graph가 효과 큰 패턴

~~~text
Graph OFF
CPU core 높음
SM Active 낮음
small decode batch
kernel duration 짧음

Graph ON
CPU 감소
SM Active 증가
TPOT 감소
~~~

이면 graph가 정확한 bottleneck을 제거한 것이다.

## 21. 운영에서 반드시 기록할 것

~~~text
requested mode
resolved mode
compile mode
attention backend
capture size list
max capture size
capture startup time
graph memory delta
KV capacity delta
runtime batch-size distribution
MTP K
~~~

capture configuration만 기록하고 runtime batch distribution을 기록하지 않으면 왜 효과가 있었는지 설명할 수 없다.

## 22. 주요 원문

- NVIDIA CUDA Programming Guide — CUDA Graphs
  - https://docs.nvidia.com/cuda/cuda-programming-guide/
- vLLM CUDA Graphs Design
  - https://docs.vllm.ai/en/latest/design/cuda_graphs/
- vLLM CompilationConfig
  - https://docs.vllm.ai/en/latest/api/vllm/config/

> CUDA Graph 최적화의 본질은 graph를 많이 만드는 것이 아니라 반복되는 실행 shape에서 host submission cost를 제거하고, 그 대가로 쓰는 graph memory가 다른 더 중요한 자원을 침범하지 않게 하는 것이다.
