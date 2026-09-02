# Scheduler Budget, Speculative Decoding & CUDA Graph

업데이트: 2026-09-03 KST

이 문서는 vLLM에서 `max_num_batched_tokens`, speculative decoding, CUDA Graph가 왜 서로 독립적인 knob가 아닌지 설명한다.

## 1. Scheduler의 token budget

vLLM V1 scheduler에서 `max_num_batched_tokens`는 한 iteration의 raw token capacity다.

개념적으로:

```text
iteration
  ├─ running decode
  ├─ running partial prefill
  └─ waiting prefill

전체 scheduled token
  <= scheduler budget
```

chunked prefill이 켜져 있으면 긴 prompt는 남은 budget에 맞춰 잘린다.

따라서 MBT는 request size limit가 아니라 **iteration-level resource allocator**다.

## 2. 왜 decode는 request당 1 token이라고 말하는가

normal autoregressive decode에서 request 하나는 한 iteration에 보통 다음 token 하나를 생성한다.

batch size가 B이면 target model query token 수는 대략 B다.

```text
B requests
 × 1 query token
 = B decode query tokens
```

하지만 이것은 speculative decoding이 없는 경우의 mental model이다.

## 3. SD가 켜지면 scheduler slot이 늘어난다

draft/verification 기반 speculative decoding에서는 target model이 한 번에 여러 candidate position을 확인한다.

uniform speculative decode의 query length는 개념적으로:

```text
1 + K
```

에 가까워진다.

- 1: 기존 decode position
- K: speculative candidate positions

다만 실제 slot reservation은 method마다 다르며, vLLM source의 `max_num_new_slots_for_drafting`를 기준으로 판단해야 한다.

vLLM의 config는 speculative decoding이 사용할 추가 slot을 고려해 `max_num_scheduled_tokens`를 별도로 계산한다.

공개 이슈/소스에서 확인되는 핵심 형태:

```text
max_num_scheduled_tokens
  =
max_num_batched_tokens
  - max_num_new_slots_for_drafting × max_num_seqs
```

method에 따라 drafting slot 수 계산은 달라질 수 있다.

따라서 "decode 요청당 정확히 K+1만큼 MBT를 항상 차지한다"라고 단순화하면 안 된다.

정확한 mental model은:

> SD는 각 active request가 미래 draft position을 위해 추가 scheduler/KV slot을 요구하므로, 동일 MBT에서 normal decode보다 실제 issue 가능한 token budget이 줄어들 수 있다.

vLLM은 이 값이 작아지면 startup warning으로 MBT를 올리거나 K/max_num_seqs를 낮추라고 안내한다.

## 4. 왜 K와 MBT를 같이 sweep해야 하는가

K를 늘리면 좋은 점:

- acceptance가 충분하면 한 target step에서 더 멀리 전진
- model weight read / launch 횟수를 줄일 수 있음
- low-BS memory-bound decode에서 특히 유리

나쁜 점:

- verification query가 커짐
- rejected draft compute 증가
- batch가 커질수록 compute saturation이 빨리 옴
- scheduler reserved slot 증가
- graph shape/capture 종류 증가
- drafter 자체 compute 증가

따라서:

```text
K optimum
  != model constant

K optimum
  = f(batch size, acceptance, GPU, MBT, target/drafter cost)
```

vLLM의 Dynamic SD도 concurrency가 커질수록 K를 줄이는 방향을 지원한다.

## 5. P/D에서는 더 명확하게 분리한다

Prefill engine에는 SD verification workload가 거의 없다.

Decode engine에는 SD가 직접 scheduler shape를 바꾼다.

따라서 예:

```text
Prefill:
  MBT 8K / 16K / 32K sweep

Decode:
  K 0 / 3 / 5 / 7
  ×
  MBT 2K / 4K / 8K / 16K
  ×
  max_num_seqs
```

처럼 실험 axis를 분리한다.

단 hybrid Mamba/GDN/KDA model은 runtime state block constraint 때문에 decode MBT의 최소 실용값이 별도로 생길 수 있다.

## 6. MBT를 너무 낮출 때 생기는 문제

### Scheduler

- long chunk를 지나치게 잘게 나눔
- iteration 수 증가
- queue turnover overhead 증가

### Kernel

- GEMM/token dimension 축소
- Tensor Core 활용률 저하
- launch overhead 비중 증가

### Hybrid state model

- state block/alignment보다 작은 budget이 runtime invariant를 깨거나 비효율 유발 가능

### SD

- reserved draft slot이 raw MBT를 잠식
- effective schedulable budget 급감

따라서 decode MBT 2K가 어떤 모델에서 좋았다는 결과를 다른 모델에 복사하지 않는다.

## 7. MBT를 너무 높일 때 생기는 문제

- prefill chunk가 decode를 오래 방해
- mixed batch가 커짐
- tail ITL/TPOT 악화
- CUDA Graph capture/memory pressure
- temporary workspace 증가
- burst에서 large prefill이 budget을 독점

vLLM upstream에서도 adaptive prefill budget을 연구하는 이유가 이 때문이다.

정적 MBT 하나가 모든 scheduling pressure에서 최적일 수 없다.

## 8. CUDA Graph mode

vLLM 0.28 계열의 `cudagraph_mode`:

| Mode | 의미 | 주 사용 |
|---|---|---|
| `NONE` | eager | correctness/debug |
| `PIECEWISE` | graph 가능한 부분만 capture | 호환성 우선 |
| `FULL` | full graph 중심 | backend가 mixed/prefill까지 지원할 때 |
| `FULL_DECODE_ONLY` | uniform decode만 full graph | P/D Decode 유력 |
| `FULL_AND_PIECEWISE` | decode full + others piecewise | 일반 성능 default |

### FULL_DECODE_ONLY

P/D decode instance에서 특히 중요하다.

- pure/uniform decode graph만 유지
- prefill/mixed graph pool을 줄일 수 있음
- capture time/graph memory 절감 가능
- decode launch overhead 감소

### FULL_AND_PIECEWISE

- 일반 mixed workload에 강함
- 가장 넓은 graph coverage
- capture/memory cost 큼

## 9. Backend compatibility가 mode보다 먼저다

CUDA Graph mode를 강제로 선택해도 attention/state backend가 지원하지 않으면 runtime이 downgrade할 수 있다.

vLLM은 backend capability를 대략 다음처럼 본다.

- mixed batch까지 full graph 가능
- uniform batch만 가능
- single-token uniform decode만 가능
- full graph 불가

Mamba 계열은 full graph capability가 decode 쪽으로 제한되는 대표적 사례다.

따라서 Qwen hybrid GDN/Mamba 계열에서는 `FULL_DECODE_ONLY`가 자연스러운 후보가 될 수 있다.

## 10. Spec decode와 Graph shape

normal decode:

```text
uniform query length = 1
```

spec decode:

```text
uniform query length = 1 + K
```

따라서 K가 바뀌면 graph dispatch key와 kernel shape가 달라질 수 있다.

즉 K sweep에서 다음도 기록해야 한다.

- graph capture count
- replay hit
- eager fallback
- captured size
- graph pool memory
- warm-up/capture time

SD throughput만 보면 원인을 놓친다.

## 11. 실험 matrix

### Decode

| Axis | 후보 |
|---|---|
| K | 0 / 3 / 5 / 7 |
| MBT | 2K / 4K / 8K / 16K |
| sequences | workload concurrency에 맞춰 sweep |
| graph | eager / FULL_DECODE_ONLY / FULL_AND_PIECEWISE |
| context | short / p50 / p90 |
| output | short / production mean / long-tail |

### 관측

- acceptance rate / accepted length
- scheduler running/waiting
- effective scheduled tokens
- GPU util
- HBM BW
- SM/Tensor Core activity
- TTFT
- ITL/TPOT
- throughput
- graph replay/fallback
- HBM/KV/state usage

## 12. 결과 해석

### K↑, ITL↓, throughput↑

SD가 spare compute를 잘 활용하는 구간일 가능성.

### K↑, ITL↑, throughput↓

verification compute가 saturation을 넘었을 가능성.

### MBT↓, ITL↓, TTFT↑

decode 보호는 좋아졌지만 prefill chunk progress가 느려진 것.

P/D 분리 환경에서는 이 trade-off가 integrated보다 훨씬 관리하기 쉽다.

### MBT↑, GPU util↑, p99 ITL↑

GPU efficiency는 좋아졌지만 scheduler interference가 증가.

### graph mode 변경 후 HBM↑, latency↓ 

launch overhead를 메모리로 산 전형적인 trade-off.

## 13. 운영 권고

1. eager correctness baseline을 먼저 만든다.
2. P/D role별 MBT를 분리한다.
3. Decode에서 graph mode를 먼저 고정한다.
4. K와 MBT를 joint sweep한다.
5. max_num_seqs를 반드시 같이 기록한다.
6. context distribution을 production p50/p90/p99로 반복한다.
7. 평균보다 p95/p99 ITL을 promotion 기준에 포함한다.
8. GPU util 하나로 판단하지 않는다.
9. state/KV capacity와 compute saturation을 같이 본다.
10. 최종 profile은 숫자가 아니라 workload envelope와 함께 저장한다.

## 14. 공식 근거

- vLLM SchedulerConfig  
  https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py
- vLLM speculative budget config  
  https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py
- Dynamic Speculative Decoding  
  https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/dynamic_speculative_decoding.md
- CUDA Graph design  
  https://github.com/vllm-project/vllm/blob/v0.28.0/docs/design/cuda_graphs.md
