# NIXL Debugging, Validation, and Production Certification

기준: vLLM `v0.27.1`, NIXL `1.3.1`

이 문서는 `NixlConnector`가 안 될 때 **vLLM / NIXL / UCX / hardware 중 어느 계층 문제인지 빠르게 분리**하기 위한 runbook이다.

---

# 1. 문제를 6개 Gate로 분리

```text
Gate 0: package / ABI
Gate 1: vLLM config + router contract
Gate 2: local memory registration
Gate 3: NIXL handshake / remote agent metadata
Gate 4: transfer descriptor / READ-WRITE submit
Gate 5: UCX/backend physical data path
Gate 6: semantic correctness / block lifetime
```

앞 Gate가 실패했는데 뒤 Gate를 튜닝하지 않는다.

---

# 2. Gate 0 — package / ABI

## 확인

```bash
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
import nixl
print("nixl", nixl)
PY

python3 -m pip freeze | grep -Ei 'nixl|ucx'
```

### 실패 유형

- `nixl` meta package는 있는데 native CUDA package 없음
- CUDA 12/13 mismatch
- `libplugin_UCX.so` load 실패
- bundled UCX와 system UCX collision
- missing verbs/CUDA runtime library

### 추가

```bash
ldd <nixl native library>
ldd <libplugin_UCX.so>
```

---

# 3. Gate 0.5 — standalone NIXL agent

vLLM을 빼고:

```bash
python3 -c "import nixl; a=nixl.nixl_agent('smoke')"
```

를 먼저 통과한다.

UCX backend initialization이 여기서 깨지면 vLLM 문제로 보지 않는다.

---

# 4. Gate 1 — vLLM config

확인:

```json
{
  "kv_connector": "NixlConnector or NixlPushConnector",
  "kv_role": "kv_producer or kv_consumer",
  "kv_buffer_device": "cuda",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "backends": ["UCX"]
  }
}
```

### 흔한 오해

`NixlConnector`는 v0.27.1에서 Pull alias다.

Push를 시험하려면:

```text
NixlPushConnector
```

를 명시한다.

---

# 5. Gate 1 — Router `kv_transfer_params`

NIXL은 Mooncake와 다른 metadata contract를 가진다.

remote prefill request에서 대표적으로 필요한 정보:

```text
remote engine ID
remote request ID
remote side-channel host/port
remote block IDs
remote TP/PP size
```

Router가 P response에서 받은 metadata를 D request에 정확히 전달하는지 확인한다.

### stale engine

P restart 후 old engine ID/side-channel endpoint를 계속 쓰면 handshake가 실패한다.

---

# 6. Gate 1 — Scheduler 상태

D Scheduler에서:

```text
get_num_new_matched_tokens
 -> external tokens > 0
 -> load_kv_async = true
 -> allocate D destination blocks
 -> WAITING_FOR_REMOTE_KVS
```

가 되는지 본다.

이 단계가 안 되면 NIXL `transfer()`는 호출되지 않는다.

---

# 7. Gate 2 — KV memory registration

vLLM startup log에서:

```text
Initializing NIXL wrapper
Detected attention backend
Detected kv cache layout
Registering KV_Caches
```

을 확인한다.

### direct GPU path

```text
kv_buffer_device=cuda
nixl_memory_type=VRAM
use_host_buffer=false
```

여야 한다.

### host staging path

```text
kv_buffer_device=cpu
use_host_buffer=true
```

라면 GPU-direct benchmark가 아니다.

---

# 8. Memory registration failure 원인

- NIXL platform에서 current device type 지원 안 됨
- NIXL backend가 VRAM type 지원 안 함
- CUDA context/device mismatch
- UCX가 CUDA support 없이 build됨
- shared/packed KV layout 계산 오류
- Mamba/HMA unsupported combination

---

# 9. Gate 3 — side-channel port

vLLM NIXL side channel은 기본 base port 5600을 사용한다.

DP rank가 더해질 수 있다.

Kubernetes에서 확인:

- P/D Pod/container port collision
- hostNetwork 여부
- multi-process 동일 host collision
- NetworkPolicy
- Service를 통과시키는지 direct Pod IP인지

NIXL data plane port와 vLLM ZMQ metadata side channel을 같은 것으로 보지 않는다.

---

# 10. Gate 3 — handshake thread CUDA context

vLLM source가 background handshake thread에서 `set_device()`하는 이유는 UCX의 CUDA IPC capability detection 때문이다.

증상:

```text
same-node인데 UCX가 cuda_ipc를 못 쓰고 느린 path
```

라면:

- connector source/version
- CUDA device selection
- process-visible GPU IDs
- UCX CUDA component

을 확인한다.

---

# 11. Gate 3 — compatibility hash mismatch

대표 error:

```text
NIXL compatibility hash mismatch
```

확인:

```text
P/D vLLM version
model revision
model config
KV dtype
dtype
attention backend
KV layout
block geometry
speculative decoding configuration
```

### 원칙

초기에는 `enforce_handshake_compat=false`로 우회하지 않는다.

raw KV bytes를 incompatible layout에 넣는 것보다 fail하는 것이 안전하다.

---

# 12. Gate 3 — engine ID mismatch

handshake metadata의:

```text
metadata.engine_id
```

와 router가 기대한 engine ID가 다르면 fail한다.

이것은 network보다 service discovery/lifecycle 문제다.

---

# 13. Gate 4 — descriptor build

Pull에서는:

```text
remote_block_descs_ids
local_block_descs_ids
```

길이가 같아야 한다.

Push도 같은 원칙이다.

문제가 나면:

- local/remote block count
- KV cache group 수
- logical/physical block ratio
- TP mapping
- MLA replication
- Mamba state regions

을 확인한다.

---

# 14. Gate 4 — cold handshake vs steady state

첫 remote engine request:

```text
ZMQ GET_META
compat hash
load agent metadata
remote descriptor preparation
endpoint setup
```

비용이 들어간다.

후속 request는 cached remote agent state를 재사용한다.

성능을 볼 때:

```text
cold TTFT
warm TTFT
```

를 분리한다.

---

# 15. Gate 4 — Pull submit

source breakpoint/logical checkpoint:

```text
NixlPullConnectorWorker._read_blocks
 -> nixl_wrapper.make_prepped_xfer("READ", ...)
 -> nixl_wrapper.transfer(handle)
```

여기까지 오면 vLLM scheduler/router/memory registration은 대부분 통과한 것이다.

---

# 16. Gate 4 — Push submit

```text
D PUSH_REG sent
 -> P receives/matches registration
 -> _do_start_push_kv
 -> _xfer_blocks
 -> make_prepped_xfer("WRITE")
 -> transfer(handle)
```

### Push가 안 시작될 때

두 map을 본다.

```text
_push_finished_blocks
_pending_d_registrations
```

한쪽만 계속 쌓이면 request ID matching/router lifecycle 문제 가능성이 높다.

---

# 17. Gate 5 — UCX capability inventory

```bash
ucx_info -v
ucx_info -d
```

확인:

- `cuda` memory support
- `cuda_ipc`
- `cuda_copy`
- verbs/RDMA transport
- mlx5 device
- TCP

설치한 UCX가 CUDA-aware가 아니면 NIXL 설정을 아무리 바꿔도 GPU-direct path가 안 나온다.

---

# 18. Gate 5 — UCX logging

진단 시 일시적으로:

```bash
UCX_LOG_LEVEL=info
```

필요하면 debug/detail로 올린다.

관찰:

- selected transport
- selected device
- endpoint creation
- memory registration
- CUDA IPC use
- RDMA lane

production 상시 debug log는 overhead가 있으므로 certification/debug에만 사용한다.

---

# 19. `UCX_TLS` A/B isolation

## same-node

가능한 CUDA lane만 허용하는 테스트로 "왜 TCP를 탔는가"를 분리할 수 있다.

## Network A

RDMA-capable transport + CUDA components로 좁혀 TCP fallback을 배제하는 certification test를 만든다.

정확한 UCX transport names는 설치된 UCX version의 `ucx_info` 결과를 기준으로 한다.

---

# 20. `UCX_NET_DEVICES`

multi-HCA에서 intended HCA를 명시한다.

잘못된 device를 잡으면:

- NUMA penalty
- link down
- wrong RoCE interface
- lower bandwidth

가 발생할 수 있다.

GPU↔NIC topology:

```bash
nvidia-smi topo -m
```

과 함께 본다.

---

# 21. NCCL env는 별개

```text
NCCL_IB_HCA
NCCL_SOCKET_IFNAME
```

등은 NCCL collective 경로를 제어한다.

NIXL UCX KV transfer는 **UCX 설정**을 본다.

같은 Pod에서 NCCL은 정상인데 NIXL RDMA는 실패할 수 있고 반대도 가능하다.

---

# 22. UAR exhaustion / RDMA resource contention

vLLM source가 `num_threads`를 기본 4로 제한하는 배경이다.

증상:

- NIXL/UCX가 먼저 올라간 뒤 NVSHMEM/DeepEP가 UAR allocation 실패
- 반대로 다른 RDMA runtime 후 NIXL init 실패

확인:

- NIXL UCX threads
- mlx5 DevX/UAR logs
- Pod 내 다른 RDMA consumers
- NIC firmware/driver resource

무작정 NIXL thread 수를 크게 늘리지 않는다.

---

# 23. Registration cache

vLLM은 NIXL import 전에:

```text
UCX_RCACHE_MAX_UNRELEASED=1024
```

를 자동 설정하려 한다.

custom startup에서 NIXL을 먼저 import하면 이 safeguard가 적용되지 않을 수 있다.

log warning을 확인한다.

---

# 24. Same-node physical proof

NIXL transfer 성공 후에도:

```text
UCX CUDA IPC
 -> CUDA P2P
```

가 NVLink/NVSwitch인지 PCIe인지 확인한다.

도구:

- `nvidia-smi topo -m`
- DCGM/NVML link counters
- Nsight Systems
- peer bandwidth microbenchmark

---

# 25. Network A physical proof

RDMA certification:

```text
[ ] UCX selected RDMA-capable transport
[ ] correct mlx5 device
[ ] GPU VRAM registration
[ ] HCA TX/RX counters increase
[ ] expected bandwidth
[ ] CPU staging signature 없음/최소
```

필요 도구:

```bash
ibv_devinfo
ibstat
ucx_info -d
```

및 fabric counter.

---

# 26. Transfer completed but output wrong

가장 위험한 case다.

의심:

- P/D model mismatch
- descriptor offset mismatch
- TP mapping bug
- block ID stale/reused
- local/remote physical block-size mismatch
- MLA replication bug
- Mamba state transfer incomplete

compatibility hash가 통과했다고 모든 semantic bug가 불가능한 것은 아니다.

baseline output + deterministic small test가 필요하다.

---

# 27. Lease expiry

증상:

```text
Potentially invalid KV blocks
expired request
remote read declined
```

원인:

- D가 waiting queue에 너무 오래 머묾
- heartbeat 전달 실패
- scheduler/process clock offset 문제
- lease duration이 workload queueing보다 짧음

단순히 timeout 값을 키우기 전에 **D scheduling delay와 heartbeat path**를 본다.

---

# 28. Notification failure

READ data가 도착해도 P가 completion notif를 못 받으면 P block이 timeout까지 hold될 수 있다.

결과:

- memory leak처럼 보이는 delayed free
- KV capacity 감소

따라서 data plane 성공률과 notification 성공률을 별도 metric으로 본다.

---

# 29. `kv_load_failure_policy=recompute`

장점:

- service availability 향상

단점:

- connector 문제를 숨길 수 있음
- P/D를 켰는데 D가 prompt를 local recompute해도 사용자 응답은 성공

### 운영 전략

- certification/test: `fail`
- production: SLO/availability 요구에 따라 `recompute` 평가
- 항상 recompute/fallback metric을 별도 경보

---

# 30. Benchmark matrix

최소:

| Dimension | Values |
|---|---|
| Connector | Pull / Push |
| Topology | same-node / cross-node |
| TP | same / heterogeneous |
| Prompt | 1K / 8K / 32K+ |
| Concurrency | 1 / medium / saturation |
| KV hit | no local hit / partial / full |
| Model cache | standard attention / MLA / hybrid if applicable |

측정:

```text
handshake cold latency
prepared xfer creation
post latency
transfer completion latency
bytes
number of descriptors
effective GB/s
TTFT
P/D GPU utilization
CPU utilization
```

---

# 31. Recommended first Network B experiment

```text
same Pod/node
P GPU set and D GPU set explicitly separated
NixlPushConnector
UCX backend
kv_buffer_device=cuda
kv_load_failure_policy=fail
```

그 뒤 Pull과 동일 workload 비교.

목표는 "NIXL이 된다"가 아니라:

```text
actual CUDA IPC/P2P path
+ no host staging
+ stable block lifetime
+ lower/equal handoff overhead
```

이다.

---

# 32. Recommended first Network A experiment

```text
P/D on separate nodes
NixlConnector (Pull)
UCX
correct UCX_NET_DEVICES
fail policy
```

단계:

1. NIXL standalone VRAM RDMA
2. vLLM 1P1D same TP
3. concurrency
4. heterogeneous TP
5. multi-node model parallel와 결합

---

# 33. Production dashboard

최소 패널:

```text
KV xfer requests/s
KV xfer bytes/s
KV xfer latency p50/p95/p99
failed transfers
failed notifications
handshake failures
lease expirations
recompute fallback count
external KV hit tokens
TTFT by P/D route
UCX/RDMA/NVLink hardware counters
```

---

# 34. 최종 certification definition

다음 모두가 true여야 한다.

```text
[ ] correct vLLM/NIXL version
[ ] direct VRAM path enabled
[ ] P/D router metadata correct
[ ] compatibility handshake passes
[ ] intended backend selected
[ ] actual READ/WRITE submitted
[ ] data completion confirmed
[ ] notification/lifetime correct
[ ] D prompt recompute absent for transfer test
[ ] physical link matches intended topology
[ ] failure/restart recovers without stale engine/memory
[ ] metrics expose transfer health
```

---

## References

- vLLM NIXL usage: https://github.com/vllm-project/vllm/blob/v0.27.1/docs/features/nixl_connector_usage.md
- vLLM NIXL source: https://github.com/vllm-project/vllm/tree/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/nixl
- NIXL architecture: https://github.com/ai-dynamo/nixl/blob/v1.3.1/docs/nixl.md
- NIXL UCX plugin: https://github.com/ai-dynamo/nixl/tree/v1.3.1/src/plugins/ucx
