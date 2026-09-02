# 01. Scheduler, Token Budget, MBT와 max-num-seqs

## 1. Scheduler의 본질

Serving scheduler는 매 iteration마다 다음을 결합한다.

~~~text
runnable requests
new requests
KV admission
token budget
sequence budget
policy
~~~

vLLM V1에서 특히 중요한 값:

~~~text
max-num-batched-tokens
max-num-scheduled-tokens
max-num-seqs
~~~

## 2. max-num-batched-tokens

한 iteration의 token budget 상한이다.

~~~math
T_{step} \le B_{token}
~~~

중요:

> MBT는 GPU가 반드시 처리하는 token 수가 아니다.

## 3. Prefill에서 MBT

170K prompt와 MBT=8K:

~~~text
8K
8K
8K
...
~~~

MBT 증가 효과:

~~~text
larger Prefill batch
→ GEMM efficiency 상승 가능
→ Tensor Core utilization 상승 가능
→ prompt tok/s 상승 가능
~~~

비용:

~~~text
single step duration 증가
queue fairness 악화 가능
tail TTFT 악화 가능
workspace 증가 가능
~~~

Prefill MBT는 batch efficiency와 scheduling fairness의 trade-off다.

## 4. Decode에서 MBT

Pure decode는 request당 step마다 대략 1 token이다.

~~~text
active requests N
→ Tstep ~= N
~~~

따라서 Decode에서 32K MBT는 대부분 불필요하다.

## 5. MTP가 있으면 token envelope가 바뀐다

MTP K:

~~~text
uniform query length ~= 1 + K
~~~

따라서:

~~~math
T_{step} \approx N(1+K)
~~~

예:

| Active seqs | K | logical envelope |
|---:|---:|---:|
| 512 | 0 | 512 |
| 512 | 1 | 1024 |
| 512 | 2 | 1536 |
| 512 | 3 | 2048 |

그래서 Decode MBT=2K는 max-seqs=512, K3의 자연스러운 full envelope가 된다.

이것이 MBT와 MTP를 joint tuning해야 하는 이유다.

## 6. max-num-scheduled-tokens

일부 speculative method는 scheduler가 발행한 token과 model runner 내부에서 확장되는 token 수가 다르다.

~~~text
scheduler-issued work
+
speculative expansion
=
actual model-side token work
~~~

그래서 내부 scheduled budget과 execution budget이 분리될 수 있다.

SD 방법마다 이 관계가 다르므로 source-level 확인이 중요하다.

## 7. max-num-seqs

한 iteration에서 처리할 sequence 수의 상한.

~~~math
N_{step} \le N_{max}
~~~

MBT와 직교한다.

~~~text
max-seqs=1024
MBT=512
pure decode

→ token budget이 먼저 512 request 부근을 제한 가능
~~~

반대로:

~~~text
max-seqs=64
MBT=8192

→ sequence budget이 먼저 제한
~~~

## 8. max-num-seqs는 scheduler에만 영향을 주지 않는다

persistent metadata가 max sequence envelope를 기준으로 잡힐 수 있다.

~~~text
seq_lens
query metadata
block tables
slot mapping
attention workspace
recurrent state index
CUDA Graph envelope
~~~

따라서 무작정 크게 잡는 것은 무료가 아니다.

## 9. Hybrid recurrent model + full CUDA Graph

vLLM 0.26 계열의 Mamba/GDN full decode graph에서는 각 decode sequence가 recurrent-state cache block을 필요로 한다.

따라서 다음 조건이 문제가 될 수 있다.

~~~text
max_num_seqs
>
available recurrent-state blocks
~~~

실제 workload concurrency가 낮아도 configuration 상한 때문에 graph initialization이 막힐 수 있다.

## 10. non-binding threshold를 찾는다

예:

| max seqs | output tok/s |
|---:|---:|
| 256 | 20K |
| 512 | 27K |
| 1024 | 27.2K |

512 이후 gain이 거의 없다면 512를 고정한다.

그 뒤에는 max-seqs가 아니라 다른 병목을 찾는다.

## 11. MBT도 knee point를 찾는다

Prefill:

| MBT | prompt tok/s | TTFT p95 |
|---:|---:|---:|
| 8K | 40K | 2.5s |
| 16K | 57K | 2.1s |
| 32K | 63K | 2.2s |
| 64K | 64K | 2.8s |

32K 이후 throughput gain은 작고 tail latency가 악화될 수 있다.

Decode:

| MBT | output tok/s | TPOT |
|---:|---:|---:|
| 512 | 8K | 22ms |
| 1K | 11K | 18ms |
| 2K | 12K | 17ms |
| 4K | 12.1K | 17ms |

2K가 practical non-binding 후보가 된다.

## 12. KV admission이 마지막 문지기다

실제 scheduled set은 개념적으로:

~~~text
min(
  runnable work,
  sequence budget,
  token budget,
  KV/state admission
)
~~~

long context에서는 KV admission이 가장 먼저 binding할 수 있다.

## 13. partial prefill도 본다

~~~text
max-num-partial-prefills
max-long-partial-prefills
long-prefill-token-threshold
~~~

MBT만 키우고 partial-prefill concurrency가 1이면 여러 긴 prompt를 충분히 interleave하지 못할 수 있다.

반대로 너무 많이 열면 active KV set이 늘어난다.

## 14. 실전에서의 해석 순서

### Prefill이 느릴 때

~~~text
1. waiting queue가 쌓이나?
2. 실제 scheduled tokens가 MBT를 채우나?
3. prompt tok/s가 MBT 증가에 따라 상승하나?
4. Tensor Active가 함께 상승하나?
5. TTFT tail이 악화되나?
6. KV admission이 먼저 막히나?
~~~

### Decode가 느릴 때

~~~text
1. active seqs가 max-seqs를 채우나?
2. actual token batch가 MBT를 채우나?
3. MTP K를 고려한 budget이 충분한가?
4. CUDA Graph가 해당 batch shape를 덮나?
5. HBM/TP collective가 새 bottleneck인가?
~~~

## 15. 본질

> Scheduler parameter는 GPU 성능을 직접 만드는 knob가 아니라 GPU에게 어떤 모양의 일을 공급할지 결정하는 traffic shaper다.

가장 큰 값이 아니라 GPU, cache, latency SLO가 scheduler에 더 이상 막히지 않는 최소 non-binding 값을 찾는다.
