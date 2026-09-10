# KV Transfer / Mooncake / NIXL Migration Deep Dive

업데이트: 2026-09-10 KST

## 1. 목적

vLLM v0.26 -> v0.29 migration에서 P/D disaggregation은 `KVTransferConfig` JSON이 parse되는지만 확인해서는 안 된다.

실제 data path는 최소 다음 세 층으로 나뉜다.

```text
1. Control / routing layer
   - kv_role
   - connector type
   - producer/consumer metadata
   - router-visible transfer parameters

2. Scheduler / cache layer
   - cache groups
   - block hashes
   - block/page sizes
   - prefix match boundary
   - Mamba/GDN state checkpoint
   - speculative state ownership

3. Native transport layer
   - Mooncake / NIXL
   - CUDA IPC / NVLink / RDMA / TCP
   - memory registration
   - native wheel build features
```

세 층이 모두 호환돼야 P/D가 정상 동작한다.

---

## 2. v0.26 -> v0.29 dependency boundary

v0.26 `requirements/kv_connectors.txt`:

```text
lmcache >= 0.3.9
nixl == 1.3.1
mooncake-transfer-engine >= 0.3.8
```

v0.29:

```text
lmcache >= 0.3.9
nixl == 1.3.2
mooncake-transfer-engine >= 0.3.12
```

그리고 CUDA 13 image에서는 Mooncake의 CUDA13 variant로 swap한다.

Sources:
- https://github.com/vllm-project/vllm/blob/v0.26.0/requirements/kv_connectors.txt
- https://github.com/vllm-project/vllm/blob/v0.29.0/requirements/kv_connectors.txt

### 핵심

`>=0.3.12`를 그대로 production package resolution에 맡기지 않는다.

P/D artifact는 다음을 exact pin한다.

```text
vLLM image digest
NIXL exact version
Mooncake exact version
Mooncake package variant
Mooncake wheel SHA256
Mooncake build feature manifest
```

---

## 3. KVTransferConfig 자체는 생각보다 안정적이다

v0.26과 v0.29의 `KVTransferConfig` top-level surface는 매우 유사하다.

공통 주요 field:

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
enable_permute_local_kv
kv_load_failure_policy
```

v0.29에는 MultiConnector 안의 child connector까지 검사할 수 있는 helper가 추가되지만 config schema 자체가 전면 재설계된 것은 아니다.

Sources:
- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/kv_transfer.py
- https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/kv_transfer.py

### 의미

Migration risk는 대체로:

```text
config schema
```

보다:

```text
connector lifecycle
cache layout
model-specific state
router metadata
transport ABI/features
```

에 있다.

---

## 4. Connector invocation을 lifecycle로 이해한다

대략적인 P/D request lifecycle은 다음처럼 본다.

### Producer / Prefill

```text
request arrival
 -> local prefix lookup
 -> external connector lookup metadata
 -> scheduler decides new tokens
 -> prefill execute
 -> transferable block/state selection
 -> connector save/push/register
 -> transfer metadata emitted to router
```

### Consumer / Decode

```text
request arrival with transfer params
 -> connector lookup/load request
 -> local allocation/reservation
 -> remote data transfer
 -> cache/state materialization
 -> decode scheduling
 -> generation
```

실제 connector에 따라 push/pull/store semantics는 다르지만 **scheduler와 connector hook의 순서가 correctness를 결정한다.**

Mamba truncation-order bugs가 대표적인 사례다.

---

## 5. NIXL push와 pull은 서로 호환되는 두 표현이 아니다

PR #50620은 NIXL transfer mode를 compatibility hash에 넣는다.

Source:
- https://github.com/vllm-project/vllm/pull/50620

이 PR이 필요한 이유는:

```text
NixlConnector       = pull / READ protocol
NixlPushConnector   = push / WRITE protocol
```

이 서로 다른 transfer protocol인데 과거에는 producer/consumer mismatch를 handshake 전에 막지 못했기 때문이다.

PR에서:

```text
compatibility hash factor += transfer_mode
connector version 6 -> 7
kv_transfer_params += transfer_mode
```

가 된다.

### Platform implication

Router는 connector name만 보면 안 된다.

최소 route identity:

```text
backend
transfer_mode
model/cache compatibility identity
producer endpoint
```

를 가져야 한다.

그리고 producer/consumer startup 단계에서 compatibility handshake failure를 fatal하게 처리한다.

---

## 6. Hybrid/Mamba P/D에서는 request mutation ordering도 contract다

v0.29 근처에서 NIXL과 Mooncake가 각각 거의 같은 종류의 bug를 수정했다.

NIXL:
- PR #53523: https://github.com/vllm-project/vllm/pull/53523

Mooncake:
- PR #53663: https://github.com/vllm-project/vllm/pull/53663

문제:

```text
local prefix lookup
 -> producer-side N-1 prompt truncation
```

순서로 처리하면 cache hit 계산 후 request length가 바뀌어 zero-new-token assertion이나 잘못된 scheduling이 발생할 수 있다.

Fix:

```text
on_new_request
 -> N-1 truncation
 -> local prefix lookup
 -> external matching
 -> scheduling
```

### Lesson

P/D connector는 단순 DMA plugin이 아니다.

request preprocessing과 scheduler semantics에도 영향을 준다.

---

## 7. Mooncake source-build history를 정확히 분리한다

기존 custom Mooncake 0.3.10.post2 build에는 서로 다른 목적이 섞여 있었다.

### A. 반드시 필요한 native feature 확보

```text
USE_INTRA_NVLINK=ON
```

공식 wheel에 `nvlink_intra`가 compile되지 않은 상태에서 same-node NVLink transport를 사용하기 위해 필요했다.

### B. 사용하지 않는 feature 제거 / deterministic artifact

예:

```text
USE_MNNVL=OFF
Store OFF
EP OFF
불필요 dependency 제거
```

즉:

```text
"custom build가 필요했음"
```

이라는 과거 사실을 하나의 이유로 뭉개면 안 된다.

상세 historical build pattern:
- `vllm-stack/pd-disaggregation/kv-transfer-backends/mooncake-transfer-engine/source-build-airgap.md`

---

## 8. vLLM PR #51067이 해결한 것과 해결하지 않은 것

vLLM PR #51067은 release image에서 private/custom Mooncake wheel override를 제거하고 official wheel을 사용하게 바꿨다.

Source:
- https://github.com/vllm-project/vllm/pull/51067

PR이 설명하는 기존 custom wheel 사유:

1. `WITH_NVIDIA_PEERMEM=OFF`가 과거 build-time only였음
2. MNNVL 지원 artifact 문제
3. CUDA13 `libcudart.so.12` 문제

이들은 upstream/runtime option 및 CUDA13 package variant로 개선됐다.

하지만 이것만 보고:

> official 0.3.12.post1 x86 wheel이면 `nvlink_intra`까지 지원된다

라고 결론내리면 안 된다.

`nvlink_intra` compile flag는 별도로 확인해야 한다.

---

## 9. Mooncake 0.3.12.post1 x86 official wheel

Mooncake v0.3.12.post1 release workflow의 x86 CUDA CMake args:

```text
BUILD_UNIT_TESTS=OFF
USE_HTTP=ON
USE_ETCD=ON
USE_CUDA=ON
WITH_EP=ON
STORE_USE_ETCD=ON
...
```

여기에:

```text
USE_INTRA_NVLINK=ON
```

은 없다.

반면 arm64 profile에는 `USE_MNNVL=ON`이 명시된다.

Source:
- https://github.com/kvcache-ai/Mooncake/blob/v0.3.12.post1/.github/workflows/release.yaml

### 결론

**x86 same-node `nvlink_intra` 사용이 목적이라면 0.3.12.post1 official wheel만으로는 기존 custom build를 제거한다고 판단할 근거가 부족하다.**

따라서 기존 `0.3.12.post1 source-build candidate` 판단은 맞았다.

---

## 10. Mooncake 0.3.13에서 packaging contract가 바뀐다

v0.3.13 `_build-wheel.yaml`은 build profile을 variant/architecture별로 중앙화했다.

x86 CUDA profile:

```text
-DUSE_CUDA=ON
-DUSE_INTRA_NVLINK=ON
-DWITH_EP=ON
...
```

x86 CUDA13 profile도:

```text
-DUSE_CUDA=ON
-DUSE_INTRA_NVLINK=ON
-DWITH_EP=ON
...
```

이다.

Source:
- https://github.com/kvcache-ai/Mooncake/blob/v0.3.13/.github/workflows/_build-wheel.yaml

### 이것이 의미하는 것

v0.3.13부터는 **x86 official CUDA wheel 자체가 `USE_INTRA_NVLINK=ON` profile로 빌드된다.**

따라서 과거 custom build 이유 A:

```text
nvlink_intra transport가 official x86 wheel에 없어서 직접 rebuild
```

는 0.3.13부터 제거 가능한 후보가 된다.

---

## 11. 그렇다고 custom build를 무조건 폐기할 필요는 없다

0.3.13 official x86 wheel은:

```text
USE_HTTP=ON
USE_ETCD=ON
USE_CUDA=ON
USE_INTRA_NVLINK=ON
WITH_EP=ON
STORE_USE_ETCD=ON
```

등의 broad profile이다.

현재 direct node-local P/D만 필요하다면 custom minimal artifact는 여전히 다음 장점이 있다.

```text
사용하지 않는 Store 제거
EP 제거
ETCD 제거
MNNVL 등 불필요 transport 제거
폐쇄망 dependency closure 축소
artifact behavior 단순화
```

따라서 결정은:

```text
official wheel이 기능적으로 부족한가?
```

뿐 아니라:

```text
minimal/supply-chain deterministic artifact를 유지할 운영 가치가 있는가?
```

까지 포함한다.

---

## 12. 권장 Mooncake migration decision

### Candidate A — Official 0.3.13

```text
Mooncake = 0.3.13 official x86 CUDA/CUDA13 wheel
```

목적:

- upstream packaging path 채택
- custom build maintenance 제거 가능성 평가
- nvlink_intra 실제 runtime 확인

### Candidate B — Custom minimal 0.3.13

```text
USE_CUDA=ON
USE_INTRA_NVLINK=ON
USE_MNNVL=OFF
USE_TCP=<fallback policy>
WITH_STORE=OFF
WITH_EP=OFF
```

목적:

- 기존 deterministic/minimal artifact 철학 보존
- official vs custom의 순수 feature-surface 차이 측정

### 추천

먼저 **A를 기본 candidate**로 검증한다.

A가 동일 기능/correctness/performance를 만족하면 custom build의 주된 기능적 이유는 사라진다.

그 후 supply-chain/air-gap simplicity를 비교해 B 유지 여부를 결정한다.

---

## 13. Official vs custom A/B acceptance

동일 vLLM 0.29 image/base에서 Mooncake wheel만 바꾼다.

### 기능

```text
import
TransferEngine init
nvlink_intra protocol available
same-node registration
P/D transfer
```

### Correctness

```text
100% cold requests
repeated-prefix requests
long-context 170K
2K decode
hybrid model
MTP off/on where supported
```

### Performance

```text
Avg/P90 transfer latency
GB/s
number of descriptors
TTFT
prefill tokens/s
CPU utilization
```

### Stability

```text
100 / 1000 request soak
prefill restart
decode restart
connector reconnect
aborted request
KV eviction/preemption
```

### Artifact

```text
wheel SHA
linked .so
transport feature probe
CMake/build manifest where available
```

---

## 14. hostPID / CUDA IPC는 별도 유지 검증

기존 same-node Mooncake P/D에서 `hostPID: true` 적용 후 `cuIpcOpenMemHandle 201` 유형 문제가 사라진 operational evidence가 있다.

v0.29 + Mooncake 0.3.13으로 올린다고 해서 이 Kubernetes-level requirement를 자동 제거하지 않는다.

Migration order:

```text
0.29 + 0.3.13 + hostPID=true
 -> certify

then isolated test:
0.29 + 0.3.13 + hostPID=false
 -> whether requirement still exists?
```

즉 connector package upgrade와 Pod IPC namespace policy change를 한 번에 하지 않는다.

같은 이유로 `/dev/shm` sharing/mount policy도 처음에는 유지한다.

---

## 15. Mooncake Store와 direct Transfer Engine을 분리한다

v0.29 Mooncake에는 direct P/D transport뿐 아니라 Store path도 크게 발전해 있다.

Store 관련:

- decode KV save
- tenant/group semantics
- external prefix persistence
- Mamba exact boundary-state persistence
- hybrid DCP prefix caching

하지만 현재 node-local direct P/D requirement와 Mooncake Store requirement는 동일하지 않다.

### Recommendation

```text
Direct P/D certification
```

과:

```text
Mooncake Store / external KV tier certification
```

을 별도 program으로 둔다.

필요하지 않은 Store를 단순히 최신 기능이라는 이유로 production artifact에 활성화하지 않는다.

---

## 16. Mamba Store correctness: PR #51358

PR #51358은 Mamba `align` mode의 exact state persistence를 수정했다.

Source:
- https://github.com/vllm-project/vllm/pull/51358

기존 positional block lookup은 sparse/mutable Mamba block table에서:

- NULL block
- freed block
- reassigned block
- speculative block

을 올바른 prefix hash에 잘못 저장할 수 있었다.

결과적으로 external prefix hit 시 **garbled/runaway output**이 가능했다.

Fix는 core가 exact:

```text
(group_id, block_id, boundary_tokens)
```

를 넘기고 async store가 끝날 때까지 physical block lifetime을 pin하는 방향이다.

### P/D platform lesson

External cache correctness는:

```text
hash/key correctness
+ source block correctness
+ lifetime correctness
```

세 조건이 모두 필요하다.

---

## 17. Sparse Mamba NULL block: PR #51362

PR #51362는 sparse Mamba block table의 `NULL_BLOCK_ID`를 physical block 0으로 오인하는 save 문제를 수정한다.

Source:
- https://github.com/vllm-project/vllm/pull/51362

이 사례 때문에 신규 KV backend qualification에 다음 test를 추가한다.

```text
sparse cache group
NULL block
partial tail
copy-on-write
speculative scratch
block reuse
```

---

## 18. NIXL vs Mooncake를 어떻게 비교할 것인가

단순히 평균 transfer GB/s만 비교하지 않는다.

### Control plane

| 항목 | NIXL | Mooncake |
|---|---|---|
| push/pull distinction | explicit connector/mode | protocol/backend path |
| compatibility hash | strong version/mode factors | connector-specific validation |
| router metadata | kv_transfer_params | connector metadata |

### Data plane

비교 대상:

```text
intra-node NVLink
CUDA IPC
RDMA
TCP fallback
registration overhead
transfer descriptor overhead
```

### Model semantics

```text
attention-only
MLA
Mamba/GDN hybrid
heterogeneous block sizes
speculative state
```

### Operational behavior

```text
worker restart
request abort
transfer timeout
partial failure
recompute/fail policy
```

즉 backend 선택은 microbenchmark 하나가 아니라 E2E workload별로 한다.

---

## 19. `kv_load_failure_policy`

`KVTransferConfig`에는:

```text
fail
recompute
```

policy가 있다.

Production P/D에서 이 값을 implicit default로 두지 말고 failure domain에 따라 결정한다.

### `fail`

장점:
- latency predictability
- hidden recomputation 없음

단점:
- transient connector failure가 user-visible failure

### `recompute`

장점:
- availability 향상 가능

단점:
- decode/prefill 역할 분리 환경에서 unexpected recompute load
- TTFT tail 증가
- prefill pool capacity 계획 교란

### 권장

초기 migration/benchmark는 `fail` 성격을 유지해 connector failure를 숨기지 않는다.

Production resilience 실험에서만 recompute를 별도 평가한다.

---

## 20. P/D 서비스 discovery와 compatibility identity

Router가 endpoint를 단순:

```text
model name + role
```

로 묶으면 부족하다.

권장 identity:

```json
{
  "model": "qwen3.6-27b",
  "vllm": "0.29.0",
  "role": "prefill",
  "connector": "NixlPushConnector",
  "transfer_mode": "push",
  "tp": 2,
  "block_size": 256,
  "cache_layout": "...",
  "spec_method": null,
  "compatibility_hash": "..."
}
```

Mooncake에서도 equivalent transport/build identity를 metadata로 보존하는 것이 좋다.

---

## 21. P/D observability

필수 metric:

```text
lookup requests
external hit/miss
transfer success/failure
transfer bytes
transfer descriptors
transfer latency p50/p90/p95
register latency
load/save queue
expired KV requests
fallback/recompute count
```

Hybrid 모델에서는 가능하면:

```text
cache group
attention vs Mamba state
checkpoint boundary
```

까지 debug telemetry를 제공한다.

---

## 22. Failure-injection test

### Network/transport

- producer unavailable
- consumer unavailable
- transfer timeout
- connection reset

### Cache lifecycle

- transfer 중 block eviction pressure
- request abort
- preemption
- prefix hit + partial tail

### Pod lifecycle

- prefill restart
- decode restart
- same-node reschedule
- hostPID/shm configuration mismatch

### Router

- push producer -> pull consumer mismatch
- wrong model/cache compatibility hash
- stale endpoint

Expected behavior를 명시한다.

```text
reject early
recompute
fail request
retry another engine
```

중 하나여야 하며 silent hang은 허용하지 않는다.

---

## 23. 폐쇄망 artifact policy

Official wheel을 쓰더라도 air-gap에서는 최종 artifact를 mirror하기 전에 provenance를 고정한다.

```text
upstream release tag
wheel filename
SHA256
Python ABI
CUDA variant
linked native libs
feature probe output
```

Custom source build라면 추가:

```text
Mooncake commit
submodule/source lock
CMake flags
builder base image
compiler/CMake/CUDA version
```

기존 source-build 문서의 `SOURCE_LOCK.env` 방식은 계속 유효하다.

---

## 24. 권장 migration 순서

```text
1. vLLM 0.29 + P/D off
2. NIXL 1.3.2 same-node P/D
3. Mooncake 0.3.13 official + hostPID/shm existing policy
4. Mooncake 0.3.13 custom-minimal A/B if needed
5. hybrid Qwen/Kimi
6. MTP/DSpark
7. DCP/PCP
8. failure/restart soak
9. optional hostPID policy relaxation test
```

이 순서를 지키면:

```text
vLLM core
connector version
native transport
hybrid state
speculative state
Kubernetes IPC policy
```

를 한꺼번에 바꾸지 않게 된다.

---

## 25. 현재 권장 결론

### NIXL

0.29 candidate에서는 `1.3.2`를 exact pin하고 push/pull compatibility identity를 route metadata에 포함한다.

### Mooncake

`0.3.12.post1` official x86 wheel은 `nvlink_intra` 제거 판단의 근거로 부족하다.

**Mooncake 0.3.13 official x86 wheel부터 `USE_INTRA_NVLINK=ON`이 upstream build profile에 명시되므로, 이 버전부터 custom source build를 제거할 수 있는 현실적인 candidate가 된다.**

다만 custom minimal build가 제공하던 dependency/feature minimization 장점은 별도의 운영 결정이다.

### Kubernetes

hostPID/shared-memory policy는 connector version 변경과 분리해서 검증한다.

---

## Sources

- v0.26 connector requirements: https://github.com/vllm-project/vllm/blob/v0.26.0/requirements/kv_connectors.txt
- v0.29 connector requirements: https://github.com/vllm-project/vllm/blob/v0.29.0/requirements/kv_connectors.txt
- KVTransferConfig v0.26: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/kv_transfer.py
- KVTransferConfig v0.29: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/kv_transfer.py
- vLLM official Mooncake wheel migration PR #51067: https://github.com/vllm-project/vllm/pull/51067
- NIXL transfer-mode compatibility PR #50620: https://github.com/vllm-project/vllm/pull/50620
- Mooncake exact Mamba state PR #51358: https://github.com/vllm-project/vllm/pull/51358
- Mooncake sparse Mamba fix PR #51362: https://github.com/vllm-project/vllm/pull/51362
- Mooncake Mamba truncation PR #53663: https://github.com/vllm-project/vllm/pull/53663
- NIXL Mamba truncation PR #53523: https://github.com/vllm-project/vllm/pull/53523
- Mooncake 0.3.12.post1 release workflow: https://github.com/kvcache-ai/Mooncake/blob/v0.3.12.post1/.github/workflows/release.yaml
- Mooncake 0.3.13 build profiles: https://github.com/kvcache-ai/Mooncake/blob/v0.3.13/.github/workflows/_build-wheel.yaml
- existing air-gap build guide: `../../vllm-stack/pd-disaggregation/kv-transfer-backends/mooncake-transfer-engine/source-build-airgap.md`
