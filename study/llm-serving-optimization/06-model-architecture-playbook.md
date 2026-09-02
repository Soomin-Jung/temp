# 06. 모델 아키텍처별 운영 Playbook

## 1. 같은 serving knob도 모델에 따라 의미가 바뀐다

한 모델에서 잘 먹힌 MBT, TP, KV dtype, CUDA Graph, MTP 값을 다른 모델에 그대로 복사하면 안 된다.

이 장에서는 architecture를 다음처럼 분류한다.

~~~text
A. Dense Full Attention / GQA
B. MLA
C. Recurrent / Linear Attention
D. Hybrid Recurrent + Full Attention
E. Sparse / Compressed Attention
F. MoE
G. MTP / speculative-head integrated model
~~~

실제 모델은 여러 범주를 동시에 가진다.

~~~text
Qwen3.6
  Dense FFN
  GDN recurrent
  Full Attention hybrid
  MTP

DeepSeek-V3 계열
  MLA
  MoE
  MTP

Kimi-K3
  KDA / Gated MLA hybrid
  MoE

DeepSeek-V4 계열
  compressed / sparse attention
  MoE
  speculative decoding 계열
~~~

## 2. Dense Full Attention / GQA

가장 중요한 식:

~~~text
KV bytes proportional to context length
~~~

long context에서는:

~~~text
context 증가
-> KV capacity 감소
-> Decode read bytes 증가
-> TPOT 증가 가능
~~~

### Prefill

주요 병목:

~~~text
large GEMM
full attention
HBM write
~~~

민감한 knob:

~~~text
MBT
chunked prefill
KV dtype
TP
attention backend
~~~

### Decode

주요 병목:

~~~text
weights read
KV read
small GEMM
launch overhead
~~~

민감한 knob:

~~~text
CUDA Graph
batch/concurrency
KV dtype
TP
MTP/SD
~~~

### 운영 눈썰미

~~~text
context 증가
TPOT 증가
DRAM Active 증가
KV utilization 증가
~~~

가 같이 보이면 token-growing KV 영향 가능성이 높다.

## 3. MLA 계열

MLA는 token당 cache representation을 줄인다.

~~~text
KV bytes/token 감소
Decode memory traffic 감소
long-context capacity 증가
~~~

하지만 context-linear scaling은 남는다.

KV가 줄어들면 오히려 다른 병목이 빨리 보일 수 있다.

~~~text
MoE all-to-all
small GEMM
launch overhead
TP collective
~~~

즉 cache-efficient architecture일수록 다음 bottleneck이 더 빨리 드러날 수 있다.

## 4. Recurrent / Linear Attention

Mamba, GDN, KDA 계열은 history를 recurrent state로 압축한다.

~~~text
token-wise KV history
->
fixed-size recurrent state
~~~

장점:

~~~text
long-context memory scaling 개선
Decode history read 감소
~~~

새로운 비용:

~~~text
state update kernel
state copy
rollback
prefix state snapshot
speculative state 관리
~~~

### 운영 눈썰미

context가 크게 늘었는데 cache usage 증가가 완만하면 recurrent state 비중이 큰 모델일 수 있다.

반대로 short-context high concurrency에서 maximum concurrency가 기대보다 낮으면 request당 fixed state가 큰지 확인한다.

## 5. Hybrid Recurrent + Full Attention

Qwen3.5/3.6 계열이 중요한 예다.

~~~text
recurrent layers
+
full-attention layers
~~~

request memory:

~~~text
fixed recurrent state
+
context-growing full-attention KV
~~~

### Short context

~~~text
recurrent state 비중 큼
MTP speculative state overhead 비율 큼
~~~

### Long context

~~~text
full-attention KV 비중 증가
~~~

### Prefill

~~~text
GDN recurrent scan
+
full attention
+
custom kernel/metadata
~~~

때문에 CPU dispatch, piecewise graph boundary, state metadata가 conventional Full Attention보다 중요할 수 있다.

### Decode

Mamba/GDN backend가 uniform Decode full graph에 강한 경우 FULL_DECODE_ONLY가 자연스러운 후보가 된다.

### 운영 눈썰미

~~~text
MTP K 증가
TPOT 개선
but
maximum concurrency 급감
~~~

이면 speculative recurrent-state footprint를 본다.

또:

~~~text
max-num-seqs 큼
FULL decode graph startup 실패
~~~

이면 recurrent-state block count와 graph envelope 관계를 확인한다.

## 6. Sparse Attention

Sparse Attention은 모든 history token을 읽지 않는다.

~~~text
T history
->
K selected history
K << T
~~~

장점:

~~~text
Decode read bytes 감소
long-context compute 감소
~~~

새 비용:

~~~text
indexer
top-k selection
irregular gather
metadata
~~~

따라서 sparse top-k가 작다고 자동으로 빠른 것은 아니다.

DRAM traffic이 줄어도 indexer latency와 irregular access가 새 bottleneck이 될 수 있다.

## 7. Compressed Attention

compressed attention은 여러 raw token을 더 적은 cache entry로 축약한다.

~~~text
raw history
-> compressed entries
-> optional sparse selection
~~~

장점:

~~~text
cache capacity 증가
history read 감소
~~~

비용:

~~~text
compression compute
boundary handling
prefix-cache complexity
transfer representation 복잡도
~~~

P/D에서는 무엇을 transfer하는지 확인해야 한다.

~~~text
raw KV
compressed KV
auxiliary index state
~~~

architecture-specific transfer format이 P/D 효율을 결정할 수 있다.

## 8. DeepSeek-V4류 compressed / sparse hybrid

Compressed Sparse Attention 계열에서는:

~~~text
history compression
+
sparse selection
~~~

을 같이 본다.

운영 metric:

~~~text
compression ratio
selected top-k
compressed cache bytes
indexer latency
sparse gather latency
~~~

MBT를 크게 했을 때 large GEMM이 좋아져도 compressor/indexer path가 새 병목이 될 수 있다.

long-context KV capacity 계산도 conventional Full Attention 공식만 적용하면 틀릴 수 있다.

## 9. MoE

MoE layer cost:

~~~text
attention
+
routing
+
all-to-all
+
expert GEMM
+
all-to-all
~~~

attention optimization 후 MoE path가 bottleneck으로 튀어나오기 쉽다.

### Prefill

많은 token이 expert로 들어가므로 expert GEMM은 커질 수 있지만 all-to-all burst와 expert imbalance가 커질 수 있다.

### Decode

token batch가 작으면:

~~~text
small expert GEMM
communication latency
expert imbalance
~~~

비중이 커질 수 있다.

### 운영 눈썰미

~~~text
GPU Util 높음
Tensor Active 들쭉날쭉
rank별 step time 편차
NVLink/NIC all-to-all 높음
~~~

이면 expert imbalance 또는 EP communication을 본다.

## 10. Dense와 MoE의 parallelism 차이

Dense 모델:

~~~text
TP를 늘리면 대부분 layer compute가 shard
~~~

MoE 모델:

~~~text
TP
+
EP
~~~

조합이 가능하다.

예:

~~~text
TP8
versus
TP2 x EP4
~~~

는 GPU 수는 같아도 communication graph가 완전히 다르다.

따라서 parameter count보다 다음을 함께 본다.

~~~text
expert size
tokens/expert
all-to-all topology
load balance
~~~

## 11. MTP integrated model

checkpoint 안에 MTP head가 있다는 사실과 runtime에서 SD가 이득이라는 사실은 다르다.

분리해서 본다.

~~~text
checkpoint capability
runtime speculative activation
accepted length
draft overhead
state/KV overhead
graph shape
~~~

### Dense Full Attention target

candidate KV가 주요 memory cost다.

### Recurrent Hybrid target

candidate recurrent state까지 증가할 수 있다.

### MoE target

verification token이 expert compute와 all-to-all을 더 소비한다.

즉 같은 K라도 architecture별 cost가 다르다.

## 12. External Draft와 Integrated MTP

External draft:

~~~text
extra model weights
extra draft KV
draft model execution
~~~

Integrated MTP:

~~~text
smaller auxiliary module
shared target representation
architecture-specific state integration
~~~

따라서 memory budget과 graph compatibility가 다르다.

## 13. Qwen 계열을 보는 눈

전통적인 Qwen GQA 모델:

~~~text
Full Attention / GQA serving rules
~~~

Qwen hybrid GDN generation 모델:

~~~text
GDN state
+
full-attention KV
+
MTP
+
hybrid graph backend
~~~

추가로 봐야 하는 값:

~~~text
recurrent state bytes/request
MTP state multiplier
resolved hybrid block size
full Decode graph compatibility
~~~

기존 상세 문서:
[../attention/11-qwen.md](../attention/11-qwen.md)

## 14. DeepSeek 계열을 보는 눈

MLA + MoE + MTP 계열:

~~~text
latent KV capacity
EP all-to-all
expert balance
MTP acceptance
TP/EP topology
~~~

compressed/sparse attention이 더해지는 계열에서는:

~~~text
compressed cache
indexer
sparse gather
~~~

까지 포함한다.

기존 상세 문서:
[../attention/12-deepseek.md](../attention/12-deepseek.md)

## 15. Kimi 계열을 보는 눈

KDA / attention hybrid 모델:

~~~text
recurrent or linear state
+
global attention branch
+
MoE
~~~

운영 질문:

~~~text
state bytes/request?
global branch는 context에 따라 얼마나 증가?
TP에서 state는 어떻게 shard?
EP all-to-all이 attention gain을 상쇄?
prefix cache는 state boundary를 어떻게 저장?
~~~

기존 상세 문서:
[../attention/10-kimi-k3.md](../attention/10-kimi-k3.md)

## 16. 모델을 처음 받았을 때 먼저 볼 config

### Attention

~~~text
num_hidden_layers
num_attention_heads
num_key_value_heads
head_dim
layer_types
sliding/local window
latent dimension
sparse top-k
compression ratio
~~~

### Recurrent

~~~text
state dimension
key/value heads
conv kernel
state dtype
recurrent layer count
~~~

### MoE

~~~text
num experts
top-k experts
expert intermediate size
shared experts
~~~

### Speculative

~~~text
MTP head 존재
num speculative heads/tokens
draft architecture
~~~

### Context

~~~text
max position
rope scaling
native context
~~~

## 17. Config를 자원 식으로 번역한다

처음 계산하거나 추정할 것:

~~~text
weights/rank
KV bytes/token/rank
fixed state/request/rank
expert weights/rank
TP divisibility
EP traffic
MTP state multiplier
~~~

deployment topology를 고르기 전에 이 resource model부터 만든다.

## 18. Architecture별 anti-pattern

### Full Attention

~~~text
long context인데 KV capacity 계산 없이 max-seqs만 키움
~~~

### Recurrent

~~~text
state는 context O(1)이니 memory 문제 없다고 생각
~~~

### Hybrid

~~~text
Full Attention 공식만 적용
~~~

### MoE

~~~text
parameter count만 보고 TP 결정
~~~

### Sparse

~~~text
top-k가 작으니 무조건 빠르다고 생각
~~~

### MTP

~~~text
acceptance rate만 보고 K 결정
~~~

## 19. 처음 보는 모델의 진단 순서

~~~text
1. layer type 분해
2. token-growing state와 fixed state 분리
3. dense와 MoE 분리
4. TP/EP shard geometry 확인
5. context scaling 식 작성
6. Prefill/Decode kernel path 확인
7. CUDA Graph support 확인
8. SD/MTP state cost 확인
9. hardware topology에 mapping
10. benchmark factor 선정
~~~

## 20. 본질

> 모델 architecture를 안다는 것은 layer 이름을 외우는 것이 아니라 context, batch, GPU 수가 늘어날 때 어떤 compute, memory, communication 항이 커지는지 예측할 수 있다는 뜻이다.
