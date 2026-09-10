# vLLM v0.26.0 -> v0.29.0 Deployment Option Matrix

업데이트: 2026-09-10 KST

## 1. 목적

이 문서는 현재 v0.26 계열에서 실제로 명시해 운영/벤치마크하는 option을 v0.29로 가져갈 때 **KEEP / PIN / ADD / REMOVE / RETEST / A/B** 판정을 내리는 실전 migration matrix다.

핵심 원칙은 다음이다.

> 같은 CLI 이름이 v0.29에도 존재한다고 해서 같은 execution contract라고 가정하지 않는다.

특히 scheduler, MRV2, CUDA Graph, hybrid cache, speculative decoding, collective backend는 option 이름이 유지돼도 내부 path가 달라졌다.

---

## 2. 빠른 결론

| 영역 | Option / config | v0.26 baseline intent | v0.29 핵심 변화 | 판정 |
|---|---|---|---|---|
| Model | `--max-model-len` | 200K/256K long context를 의도적으로 고정 | model family의 actual supported context와 cache layout 변화 가능 | **KEEP + RETEST** |
| Memory | `--gpu-memory-utilization` | 0.94~0.97 수준에서 KV/non-KV 균형 | MRV2 CUDA Graph memory profiling이 KV auto-sizing에 반영 | **KEEP + RETEST** |
| KV | `--kv-cache-dtype fp8` | KV capacity 확보 | KV layout/model-specific dtype path 확대 | **KEEP + RETEST** |
| KV | `--block-size 256` | long-context/P-D transfer granularity | hybrid group와 `prefix_match_unit`, Mamba block alignment가 중요해짐 | **KEEP initially + RETEST** |
| Scheduler | `--max-num-batched-tokens` | P/D role별 명시 튜닝 | HW-tier default drift; SD scheduled budget 변화 | **PIN** |
| Scheduler | `--max-num-seqs` | decode concurrency 제한/확장 | v0.29에서 compile hash factor에도 포함 | **PIN + RETEST** |
| Scheduler | `--enable-chunked-prefill` | long prompt memory/throughput 제어 | scheduler internals 변화, hybrid checkpoint와 상호작용 | **KEEP + A/B** |
| Scheduler | `--no-enable-chunked-prefill` | full prefill 실험 | model이 공식 지원하지 않으면 warning/correctness risk | **A/B only** |
| Scheduler | `max_num_partial_prefills` | legacy chunked-prefill fairness | v0.27에서 제거 | **REMOVE** |
| Scheduler | `max_long_partial_prefills` | legacy long-prefill cap | v0.27에서 제거 | **REMOVE** |
| Scheduler | `long_prefill_token_threshold` | long request 분류 | field는 남지만 위 legacy concurrency knobs 제거 | **RETEST / usually omit** |
| Scheduler | `async_scheduling` | 기존 baseline scheduler semantics | draft model에서 modern async path가 적극 사용됨 | **A/B + explicit during migration** |
| Admission | `--max-num-queued-reqs` | 없음 | v0.29 신규 engine-local overload valve | **ADD optional** |
| Admission | `--max-num-queued-tokens` | 없음 | long-prefill backlog QoS valve | **ADD recommended for P pool after validation** |
| Prefix | `--enable-prefix-caching` | hybrid는 명시적 opt-in 성격 | v0.29은 supported hybrid model도 default-enabled path | **PIN during migration** |
| Prefix | `--prefix-cache-retention-interval` | 없음 | Mamba/SWA checkpoint retention policy 신규 | **ADD for hybrid tests** |
| Mamba | `--mamba-backend` | Triton/FlashInfer 선택 | CPU backend 및 SSU selector 추가 | **PIN when model uses state cache** |
| Mamba | `--mamba-ssu-algorithm` | 없음 | FlashInfer auto/simple/vertical/horizontal | **A/B optional** |
| Mamba | ReplaySSM controls | 없음/미사용 | v0.28+ available | **OFF in compatibility phase; evaluate later** |
| Compile | `--compilation-config` | FULL_DECODE_ONLY / FULL_AND_PIECEWISE 등 | MRV2 graph manager/memory profiling 변화 | **PIN + RETEST** |
| Compile | `--max-cudagraph-capture-size` | 대부분 implicit | 일반 512, datacenter Blackwell 1024 default cap | **ADD explicit per HW profile** |
| Compile | `--cudagraph-capture-sizes` | implicit/default | graph coverage/memory policy 변화 | **RETEST** |
| SD | `--speculative-config` | MTP/EAGLE/DSpark | multi-layer MTP, DSpark adaptive scheduling, MRV2 dependencies 확대 | **KEEP syntax, RETEST semantics** |
| SD | per-request spec metrics | 없음 | v0.29 observability 추가 | **ADD staging** |
| Parallel | TP | 모델/장비별 명시 | FlashInfer all-reduce default-on | **KEEP + communicator A/B** |
| Parallel | DP | throughput/replica layout | DCP/PCP/EPD integration 확대 | **KEEP + RETEST** |
| Parallel | EP | MoE serving | collective/backend/kernel path 변화 | **KEEP + RETEST** |
| Collective | `VLLM_ALLREDUCE_USE_FLASHINFER` | baseline off/default false 계열 | v0.29 default true | **PIN=0 for compatibility, then A/B** |
| Frontend | `--reasoning-parser` | model-specific parser | parser-engine adapters/registries 변화 | **KEEP name + RETEST output** |
| Frontend | `--tool-call-parser` | model-specific tool parser | Kimi K3 등 신규 adapter, shared engine | **KEEP name + RETEST output** |
| Frontend | `--enable-auto-tool-choice` | tool calling enabled | structured-output/reasoning adapter coupling 강화 | **KEEP + RETEST** |
| Frontend | `--renderer-num-workers` | tokenizer/render concurrency | renderer path 발전, endpoint별 사용 방식 차이 | **KEEP initially + CPU/profile RETEST** |
| KV transfer | `--kv-transfer-config` | Mooncake/NIXL P/D | top-level schema 유사, lifecycle/layout는 크게 변화 | **KEEP shape + PIN dependencies + RETEST** |
| KV transfer | Mooncake package | custom 0.3.10.post2 | vLLM >=0.3.12, Mooncake 0.3.13 x86 wheel에 INTRA_NVLINK 포함 | **MIGRATE via controlled A/B** |
| KV transfer | NIXL | exact 1.3.1 baseline | v0.28/0.29 requirement 1.3.2 + push/pull compat hash | **PIN 1.3.2 candidate** |
| Deprecated | `calculate_kv_scales` | legacy | v0.28 제거 | **REMOVE** |
| Deprecated | `override_attention_dtype` | ROCm-centric legacy | v0.28 제거 | **REMOVE if present** |
| Entrypoint | `python -m vllm.entrypoints.openai.api_server` | possible legacy launch | v0.29 deprecated | **MIGRATE to `vllm serve`** |

---

## 3. Scheduler / memory

## 3.1 `max_num_batched_tokens`: 무조건 explicit pin

### v0.26 source

`EngineArgs.get_batch_defaults()`는 GPU memory와 `UsageContext`에 따라 default를 설정한다.

H100/H200-class (`>=70 GiB`, non-A100):

```text
LLM_CLASS           16384
OPENAI_API_SERVER    8192
max_num_seqs         1024
```

기타 GPU:

```text
LLM_CLASS            8192
OPENAI_API_SERVER    2048
```

Source:
- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/engine/arg_utils.py

### v0.29 source

B200/B300-class (`>=160 GiB`) branch가 추가된다.

```text
>=160 GiB
  LLM_CLASS           16384
  OPENAI_API_SERVER   16384

>=70 GiB, non-A100
  LLM_CLASS           16384
  OPENAI_API_SERVER    8192
```

Source:
- https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/engine/arg_utils.py
- PR #51726: https://github.com/vllm-project/vllm/pull/51726

### Platform decision

P/D serving에서는 default를 사용하지 않는다.

권장 profile 개념:

```yaml
prefill:
  maxNumBatchedTokens: <benchmark-certified>

decode:
  maxNumBatchedTokens: <SD/concurrency-certified>
```

B300라고 해서 16384를 자동 채택하지 않는다. `>=160 GiB` default는 generic heuristic이지, 170K-class prefill workload에 대한 최적값이 아니다.

---

## 3.2 `max_num_seqs`: scheduler knob + compiled-shape input

v0.26 `SchedulerConfig.compute_hash()`는 `max_num_batched_tokens`를 graph hash에 포함한다.

v0.29은 여기에 `max_num_seqs`도 추가한다.

이유는 PLE 등 model component가 per-request static buffer를 `max_num_seqs` 기반으로 할당하고 compiled graph에 shape가 들어갈 수 있기 때문이다.

Source:
- v0.26: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/scheduler.py
- v0.29: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/scheduler.py

### Implication

다음 식으로 값을 자주 바꾸는 benchmark에서는:

```text
50 -> 100 -> 256 -> 1024
```

단순 concurrency 변화만 일어나는 것이 아니라 compile artifact/cache reuse도 달라질 수 있다.

따라서 benchmark record에 반드시 포함한다.

---

## 3.3 `gpu_memory_utilization`: 숫자가 같아도 KV capacity가 같지 않다

CacheConfig default는 v0.26/v0.29 모두 0.92 계열이지만, 실제 운영에서는 더 높은 explicit 값을 사용한다.

문제는 v0.29 MRV2가 CUDA Graph memory profiling을 KV auto-sizing에 반영한다는 점이다.

같은:

```text
--gpu-memory-utilization 0.94
```

에서도:

```text
weights
+ non-KV workspace
+ graph reservation
+ model-specific buffers
```

추정 결과가 바뀌어 KV block count가 달라질 수 있다.

### Required measurement

startup 단계별로 다음을 로그/metric으로 남긴다.

```text
after model load
before memory profile
after memory profile
after KV allocation
after CUDA Graph capture
steady-state idle
peak prefill
peak decode
```

### Isolation

두 번째 A/B에서 `kv_cache_memory_bytes`를 직접 고정하여 runner/kernel 차이와 auto-sizing 차이를 분리한다.

---

## 3.4 Chunked Prefill

v0.26과 v0.29 모두 `enable_chunked_prefill` 자체는 modern generation model에서 일반적인 path다.

하지만 다음이 달라졌다.

- legacy partial-prefill concurrency knobs 제거
- scheduler implementation 변화
- Mamba internal checkpoints
- P/D-specific prompt truncation ordering
- speculative scheduling budget

### Recommendation

170K-class prompt에서는 compatibility phase에서 기존 policy를 유지한다.

```text
128K MBT + chunked prefill
```

같은 baseline을 먼저 0.29에서 재현한 뒤, no-chunked/full-prefill을 별도 실험으로 비교한다.

`--no-enable-chunked-prefill`을 migration과 동시에 적용하면 version effect와 scheduling policy effect가 섞인다.

---

## 3.5 Removed partial-prefill options

v0.26에는:

```text
max_num_partial_prefills
max_long_partial_prefills
```

가 `SchedulerConfig`에 존재한다.

v0.27에서 제거되었고 v0.29 `SchedulerConfig`에는 없다.

Action:

```text
REMOVE from Helm values / generated argv / docs / CI validation
```

`long_prefill_token_threshold`는 남지만 과거 두 concurrency option과 함께 사용하던 semantics를 그대로 기대하면 안 된다.

---

## 3.6 Queue admission control

v0.29 신규:

```text
--max-num-queued-reqs
--max-num-queued-tokens
```

`max_num_queued_tokens` source 설명은 prefill-phase prompt-token backlog를 제한하여 TTFT QoS를 보호하는 목적이다.

특히 P/D에서는 prefill pool capacity와 직접 연결된다.

권장 ownership:

```text
API Gateway / platform proxy
  -> tenant/user rate limit

Router
  -> model/replica/P-D routing

vLLM engine-local admission
  -> last-resort saturation guard
```

초기 production 값은 benchmark의 measured prefill throughput과 target TTFT로 산정한다.

단, vLLM 자체 설명대로 partially-prefilled request와 prefix hit에 대한 token count는 보수적으로 overestimate될 수 있다.

---

## 4. Prefix / hybrid / Mamba

## 4.1 `enable_prefix_caching`

v0.26 `EngineArgs`는 hybrid model에서 prefix caching을 지원하더라도 **default opt-in**으로 남겼다.

실제 source:

```python
default_prefix_caching = (
    model_config.is_prefix_caching_supported
    and not model_config.is_hybrid
)
```

v0.29:

```python
default_prefix_caching = model_config.is_prefix_caching_supported
```

즉 hybrid model에서 option을 생략한 결과가 달라진다.

### Decision

Migration compatibility phase에서는 명시한다.

```text
--enable-prefix-caching
```

또는 baseline이 off였다면 명시적으로 off.

`omitted`를 사용하지 않는다.

---

## 4.2 `prefix_cache_retention_interval`

v0.29 신규.

```text
0     = semantic checkpoints only
N     = semantic + periodic N-token checkpoints
None  = dense retention
```

Mamba/SWA cache group에 적용된다.

Action:

- dense transformer: usually omit
- Qwen3.5+/Kimi K3 hybrid: **ADD to test matrix**
- MTP/EAGLE: effective retention override 여부까지 확인

---

## 4.3 `mamba_backend` / SSU algorithm

v0.26:

```text
triton
flashinfer
```

v0.28+:

```text
triton
flashinfer
cpu
```

그리고 FlashInfer backend에:

```text
mamba_ssu_algorithm = auto | simple | vertical | horizontal
```

가 추가된다.

Compatibility phase에서는 기존 backend를 pin한다. 신규 algorithm은 optimization phase에서 A/B한다.

---

## 4.4 ReplaySSM

v0.28+ CacheConfig에 `use_replayssm`, `replayssm_buffer_len`이 존재한다.

이 기능은 standard Mamba2 decode에서 per-step full-state store를 줄이는 성격이다.

Migration 초기에는 off 상태를 유지한다.

이유:

- P/D
- prefix cache
- speculative decode
- Mamba state lifetime

와 동시에 바뀌면 regression 원인 격리가 어려워진다.

---

## 5. CUDA Graph / compilation

## 5.1 Compilation mode

현재 역할별로 사용한:

```text
FULL_DECODE_ONLY
FULL_AND_PIECEWISE
```

등은 그대로 가져가되 동일 behavior라고 가정하지 않는다.

MRV2 graph manager, model-specific breakable regions, graph memory profiling이 달라졌다.

### Required A/B

```text
EAGER
 -> FULL_DECODE_ONLY
 -> PIECEWISE/FULL_AND_PIECEWISE
```

각 단계에서 correctness + memory + TTFT + TPOT를 비교한다.

---

## 5.2 `max_cudagraph_capture_size`

PR #49390:

```text
general GPU              cap 512
datacenter Blackwell     cap 1024
```

Action:

hardware profile마다 명시적으로 관리하는 것을 권장한다.

예:

```yaml
h200:
  maxCudagraphCaptureSize: 512

b300:
  maxCudagraphCaptureSize: 512   # compatibility phase
  # 1024는 optimization benchmark 후 채택
```

즉 B300에서 default 1024를 바로 production contract로 받아들이지 않는다.

---

## 6. Speculative decoding

## 6.1 `speculative_config`

JSON/CLI surface는 계속 유지되지만 내부 implementation은 크게 달라졌다.

v0.27~0.29:

- multi-layer MTP
- quantized DSpark Markov heads
- DSpark adaptive scheduling
- DFlash2
- MRV2-only features
- async scheduling integration
- Mamba/GDN speculative state handling

Action: **KEEP syntax, RETEST semantics**.

### Required record

```text
method
num_speculative_tokens
draft checkpoint/revision
draft quantization
max_num_scheduled_tokens
max_num_batched_tokens
max_num_seqs
async_scheduling
MRV version
acceptance rate
accepted tokens/request
```

---

## 6.2 `max_num_scheduled_tokens`

SchedulerConfig 설명상 speculative model이 batch에 token을 append할 수 있는 경우 `max_num_batched_tokens`보다 작은 scheduler issue budget을 사용할 수 있다.

이 값이 automatic이면 benchmark에서 resolved value를 반드시 기록한다.

DSpark의 MBT minimum 문제를 분석할 때:

```text
MBT != always scheduler-issued target tokens
```

이라는 구분을 유지한다.

---

## 7. Parallelism / collectives

## 7.1 TP/DP/EP

TP/DP/EP 값 자체는 explicit KEEP.

하지만 implementation은 version-specific이다.

특히:

- DSV4 / Kimi K3 MoE kernels
- DeepEP / FlashInfer collectives
- DCP/PCP
- MRV2 sampling

이 바뀌므로 같은 TP4/TP8도 동일 compute/communication path가 아니다.

---

## 7.2 `VLLM_ALLREDUCE_USE_FLASHINFER`

v0.29 PR #52998:

```text
default 0 -> 1
```

batch-invariant mode에서는 fixed reduction order 문제로 FlashInfer all-reduce를 사용하지 않도록 gate한다.

### Migration policy

Phase 1:

```bash
VLLM_ALLREDUCE_USE_FLASHINFER=0
```

으로 0.26과 hidden variable을 최대한 맞춘다.

Phase 2:

```text
0 vs 1
```

A/B.

측정:

- TPOT
- GPU utilization
- all-reduce time
- deterministic/greedy output
- mixed-dtype norm/quantized model correctness

---

## 8. Frontend / parser

## 8.1 `reasoning_parser`

이름이 동일해도 adapter implementation이 변한다.

v0.26와 0.29 registry를 비교하면:

- `deepseek_v4`, `qwen3`, `glm47` 등 기존 parser 유지
- `kimi_k3`, `ling3`, `hy_v4`, `muse_glimmer` 등 추가
- 일부 parser가 legacy class에서 Parser Engine adapter로 변경

Action: **KEEP parser name, RETEST semantic event stream**.

---

## 8.2 `tool_call_parser`

v0.29에는 `kimi_k3`, `ling3`, `hy_v4`, `muse_glimmer` 등 새로운 parser가 추가된다.

기존 Qwen/DSV4/GLM parser도 shared parser engine 및 structured-output와의 연결이 달라질 수 있다.

Action:

```text
parser existence test X
actual tool-call regression corpus O
```

---

## 8.3 `renderer_num_workers`

v0.29에서도 `renderer_num_workers` surface는 존재하고 default 1이다.

Renderer worker pool은 blocking tokenizer/render preprocessing의 host-side concurrency를 늘린다.

기존 4-worker profile은 처음에는 유지하되 다음을 다시 본다.

- API process CPU utilization
- tokenizer contention
- stream latency
- multimodal preprocessing
- model load/startup threads

GPU benchmark만 보고 renderer worker 수를 결정하지 않는다.

---

## 9. KV transfer

## 9.1 `kv_transfer_config`

v0.26→0.29 top-level field는 상당 부분 유지된다.

```text
kv_connector
engine_id
kv_buffer_device
kv_buffer_size
kv_role
kv_rank
kv_parallel_size
kv_ip
kv_port
kv_connector_extra_config
kv_connector_module_path
kv_load_failure_policy
```

따라서 chart schema를 대폭 갈아엎을 필요는 없다.

하지만 다음은 반드시 재검증한다.

- connector name
- transfer mode
- cache-group layout
- block size
- DCP/PCP
- Mamba state
- speculative config compatibility
- external router metadata

---

## 9.2 Mooncake

Baseline custom build를 그대로 복사하지 않는다.

후속 판단은 [`kv-transfer-mooncake.md`](kv-transfer-mooncake.md)에 상세히 정리한다.

Migration matrix상 결론:

```text
0.3.10.post2 custom
 -> 0.3.13 official x86 CUDA wheel A/B
 -> 필요 시 0.3.13 custom-minimal
```

---

## 9.3 NIXL

v0.26 requirement:

```text
nixl == 1.3.1
```

v0.28/0.29 계열:

```text
nixl == 1.3.2
```

그리고 NIXL connector compatibility hash가 push/pull transfer mode를 포함하게 된다.

Action: candidate는 exact `1.3.2` pin + producer/consumer transfer-mode validation.

---

## 10. Removed/deprecated cleanup

### REMOVE

```text
max_num_partial_prefills
max_long_partial_prefills
calculate_kv_scales
override_attention_dtype
```

### MIGRATE

```text
python -m vllm.entrypoints.openai.api_server
 -> vllm serve
```

Deprecated option을 chart template에 계속 남겨 두면 새 version에서 unknown option failure가 나거나, 더 나쁘게는 wrapper가 swallow해서 사용자가 적용됐다고 오해할 수 있다.

CI에 argv compatibility smoke test를 추가한다.

---

## 11. 권장 compatibility profile

0.29 첫 candidate는 '최신 default'가 아니라 아래처럼 **0.26 intent에 가까운 explicit profile**로 만든다.

```text
max-model-len                 explicit
max-num-batched-tokens        explicit per role
max-num-seqs                  explicit per role
gpu-memory-utilization        explicit
kv-cache-dtype                explicit
block-size                    explicit
chunked-prefill               explicit
prefix-caching                explicit
compilation mode              explicit
max cudagraph capture size    explicit
async scheduling              explicit during A/B
FlashInfer allreduce           OFF initially
speculative config            OFF -> method-by-method enable
KV transfer                    OFF -> backend-by-backend enable
```

이 profile이 correctness baseline을 통과한 후 0.29 optimization을 하나씩 제거/enable한다.

---

## 12. CI validation 제안

새 image/tag를 올릴 때 다음을 machine-readable manifest로 dump한다.

```json
{
  "vllm_version": "0.29.0",
  "resolved_model_runner": "v2",
  "max_model_len": 262144,
  "max_num_batched_tokens": 8192,
  "max_num_scheduled_tokens": 8192,
  "max_num_seqs": 100,
  "chunked_prefill": true,
  "prefix_caching": true,
  "prefix_cache_retention_interval": 0,
  "cudagraph_mode": "FULL_DECODE_ONLY",
  "max_cudagraph_capture_size": 512,
  "allreduce_flashinfer": false,
  "kv_connector": null,
  "spec_method": null
}
```

Golden manifest와 diff하여 **unintended default drift를 CI에서 잡는 방식**을 권장한다.

---

## 13. Related

- [`v0.26.0-to-v0.29.0.md`](v0.26.0-to-v0.29.0.md)
- [`model-runner-v1-v2.md`](model-runner-v1-v2.md)
- [`hybrid-mamba-gdn.md`](hybrid-mamba-gdn.md)
- [`kv-transfer-mooncake.md`](kv-transfer-mooncake.md)
- upstream v0.26 scheduler: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/scheduler.py
- upstream v0.29 scheduler: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/scheduler.py
- PR #51726: https://github.com/vllm-project/vllm/pull/51726
- PR #49390: https://github.com/vllm-project/vllm/pull/49390
- PR #52998: https://github.com/vllm-project/vllm/pull/52998
