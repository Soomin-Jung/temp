# 2026-08-28 Mooncake `nvlink_intra` — Kubernetes Container GPU Namespace Incident

> Status: **Root-cause boundary isolated; mitigation design implemented in `vllm-production-stack-custom` PR #4; GPU-node runtime certification pending**
>
> Baseline under investigation:
>
> - Kubernetes: k3s
> - NVIDIA Device Plugin: `v0.18.0`
> - NVIDIA driver: pre-580 generation
> - vLLM: `v0.26.0`
> - Mooncake Transfer Engine: `0.3.10.post2`
> - P/D placement: **one P/D Cell Pod on one GPU node**
> - target data path: Mooncake `nvlink_intra`
> - MIG: not used
> - time-slicing: not used

---

# 1. Executive summary

P/D Cell을 한 Pod 안의 여러 vLLM engine container로 배치하면서 각 container가
`resources.limits["nvidia.com/gpu"]`를 직접 요청하도록 구성하면, NVIDIA Device
Plugin은 **container별로 서로 다른 physical GPU set을 inject**한다.

예:

```text
Host physical GPUs
A B C D

Prefill container
  NVIDIA_VISIBLE_DEVICES=A,B
  CUDA local ordinals:
    cuda:0 -> A
    cuda:1 -> B

Decode container
  NVIDIA_VISIBLE_DEVICES=C,D
  CUDA local ordinals:
    cuda:0 -> C
    cuda:1 -> D
```

Mooncake `nvlink_intra`는 generic network transport가 아니라 remote GPU allocation을
CUDA IPC handle로 export/import하는 구현이다.

```text
Decode KV allocation
 -> cudaIpcGetMemHandle()
 -> serialized cudaIpcMemHandle_t
 -> metadata exchange
 -> Prefill cudaIpcOpenMemHandle()
 -> remote allocation mapping
 -> cudaMemcpy / P2P data movement
```

실제 runtime에서 control-plane과 KV descriptor 생성까지 진행되었으나
`cudaIpcOpenMemHandle()`에서:

```text
invalid argument
```

또는 실험 조건에 따라:

```text
invalid device context
```

로 실패했다.

`hostIPC=true`, `shareProcessNamespace=true`, 공용 `/dev/shm`은 이 문제를
해결하지 못했다.

현재 가장 강한 결론은:

> **Linux IPC namespace 문제가 아니라, legacy Device Plugin의 per-container GPU
> device namespace/ordinal model과 Mooncake 0.3.10 `nvlink_intra` CUDA IPC path가
> 현재 P/D multi-container topology에서 충돌하는 것이 핵심 blocker다.**

단, CUDA documentation상 invisible device의 IPC handle open이 항상 금지되는 것은
아니므로, 단순히 `NVIDIA_VISIBLE_DEVICES` 문자열이 다르다는 사실 하나만으로
CUDA 자체의 일반 제약이라고 결론내리지는 않는다. Mooncake/driver/container runtime
조합에서 실제 실패한 boundary를 대상으로 설계를 바꾼다.

---

# 2. 최초 P/D control-plane 문제와 이번 문제는 별개다

이번 incident 이전에는 LMStack Router와 MooncakeConnector 사이의 orchestration
contract mismatch가 있었다.

대표 로그:

```text
Router:
  Prefill responses did not contain kv_transfer_params

Prefill:
  Missing transfer_id in kv_transfer_params from router!
```

이 문제는 Cell-local orchestrator를 `vllm-project/router`로 고정하고 Mooncake-aware
`transfer_id + bootstrap + engine_id` metadata를 생성하도록 하면서 control-plane
단계가 진전되었다.

이번 incident에서는:

```text
Router transfer_id             PASS
Prefill request execution      PASS
Decode destination metadata    PASS
Mooncake descriptor creation   PASS
cudaIpcOpenMemHandle           FAIL
actual GPU copy                NOT REACHED
```

이다.

따라서 Router 문제와 CUDA IPC data-plane 문제를 섞어 해석하면 안 된다.

---

# 3. MTP mismatch 문제도 별개의 첫 번째 blocker였다

초기 비대칭 실험:

```text
Prefill MTP OFF
Decode  MTP ON
```

에서는 Decode metadata에 MTP KV layer가 존재하지만 Prefill local layer spec에는 없는
상태가 되어:

```text
mtp.layers.0.self_attn.attn
```

KeyError가 발생했다.

이후 P/D MTP를 대칭으로 맞추자 layer-name mismatch를 지나 실제 Mooncake transport
단계까지 진행했고, 그 다음에 `cudaIpcOpenMemHandle()` 실패가 노출되었다.

즉 장애 층위는:

```text
1. P/D KV layer symmetry
       ↓ fixed/bypassed for test
2. Router/Mooncake metadata
       ↓ verified
3. CUDA IPC remote mapping
       ↓ current blocker
4. actual NVLink copy
       ↓ not reached
```

이다.

---

# 4. 실제 runtime evidence

## 4.1 Prefill-side import failure

계측한 Mooncake 0.3.10 code에서:

```text
[NVLINK_IPC_DEBUG][CTX]
where=relocateSharedMemoryAddress:before-ipc-open
ctx=<non-null>
ctx_rc=0
ctx_device=0
dev_rc=0

[NVLINK_IPC_DEBUG][IMPORT]
target_id=1
requested_dest=0x...
request_length=1605632
remote_base=0x...
remote_region_length=6633291776
handle_sig=0x...

[NVLINK_IPC_DEBUG][IMPORT_FAIL]
err_code=1
err_string=invalid argument
target_id=1
remote_base=0x...
handle_sig=0x...
```

중요한 사실:

- current CUDA context는 NULL이 아니었다.
- `cuCtxGetCurrent()` 성공.
- `cuCtxGetDevice()` 성공.
- 즉 "Mooncake sender thread에 CUDA context가 전혀 없다"는 가설은 이 run에서는 배제.
- 실패 지점은 실제 copy 이전의 remote IPC allocation import.

## 4.2 address 관계

관측값에서:

```text
requested_dest - remote_base
= 0x188000
= 1,605,632 bytes
= request_length
```

이었다.

따라서 `requested_dest`가 startup `EXPORT_ALLOC.base_ptr`와 그대로 같지 않은 것은
오류 증거가 아니다.

Mooncake는 allocation base 안의 KV block offset을 target address로 사용한다.

```text
remote allocation base
      +
block/layer offset
      =
requested destination
```

이 관계가 정상 범위 안에 있었다.

## 4.3 Decode runtime log가 적은 이유

Decode는 request마다 `cudaIpcGetMemHandle()`을 새로 실행하지 않는다.

KV cache startup registration 단계에서:

```text
registerLocalMemory()
 -> cudaIpcGetMemHandle()
 -> BufferDesc metadata
```

를 만들어 둔다.

요청 시 Decode는 transfer completion을 기다리는 consumer이고, actual P→D WRITE는
Prefill-side Mooncake Transfer Engine이 수행한다.

따라서 runtime Decode 로그가:

```text
pulling kv_caches for [...] failed:
Mooncake transfer engine returned -1
```

정도로 보이는 것은 call flow와 일치한다.

---

# 5. Mooncake 0.3.10 `nvlink_intra` source path

Source:

```text
mooncake-transfer-engine/
  src/transport/intranode_nvlink_transport/
    intranode_nvlink_transport.cpp
```

## 5.1 export / registration

`IntraNodeNvlinkTransport::registerLocalMemory()`:

```text
cudaPointerGetAttributes(addr)
  ↓
cuMemGetAddressRange(addr)
  ↓
true cudaMalloc allocation base/size
  ↓
cudaIpcGetMemHandle(base_ptr)
  ↓
serialize cudaIpcMemHandle_t
  ↓
BufferDesc.shm_name
```

PyTorch caching allocator의 tensor view가 allocation 중간을 가리킬 수 있으므로
Mooncake는 `cuMemGetAddressRange()`로 backing allocation base를 찾는다.

## 5.2 import / relocation

`relocateSharedMemoryAddress()`:

```text
remote BufferDesc
  ↓
deserialize cudaIpcMemHandle_t
  ↓
cudaIpcOpenMemHandle(
  handle,
  cudaIpcMemLazyEnablePeerAccess
)
  ↓
local mapped base
  ↓
mapped_base + (remote_requested - remote_base)
```

현재 실패는 이 `cudaIpcOpenMemHandle()` 호출이다.

## 5.3 actual copy

mapping 성공 후:

```cpp
cudaMemcpy(..., cudaMemcpyDefault)
```

계열로 이동한다.

따라서 현재 오류는:

```text
NVLink bandwidth 부족
PCIe/NVLink physical path 오류
copy throughput 문제
```

보다 앞선 단계다.

---

# 6. 왜 `/dev/shm`, hostIPC, PID namespace로 해결되지 않았는가

## 6.1 `/dev/shm`

CUDA IPC handle은 POSIX shared-memory file 이름을 상대 process가 찾아서 mmap하는
형태가 아니다.

Mooncake metadata에 serialize된 opaque `cudaIpcMemHandle_t`를 NVIDIA CUDA
runtime/driver API가 import한다.

따라서 P/D engine이 같은 hostPath `/dev/shm`을 mount했다고 해서 peer GPU
allocation 접근 권한이나 CUDA device mapping이 합쳐지는 것은 아니다.

## 6.2 `hostIPC: true`

Kubernetes `hostIPC`는 Pod가 host Linux IPC namespace를 사용하도록 한다.

영향 대상:

- SysV shared memory
- POSIX IPC objects
- semaphore/message queue 계열

하지만 다음을 합치지 않는다.

```text
NVIDIA_VISIBLE_DEVICES
/dev/nvidia* device injection
OCI device allow list
CUDA logical ordinals
NVIDIA device-plugin allocation
```

실제 `hostIPC=true` test에서도 동일 오류가 재현되었다.

## 6.3 `shareProcessNamespace: true`

이는 Pod container 사이 PID namespace를 공유한다.

CUDA device allocation scope를 바꾸지 않는다.

실제 test에서도 해결되지 않았다.

결론:

> 이번 문제를 Linux IPC/PID namespace 문제로 취급하지 않는다.

---

# 7. NVIDIA Device Plugin v0.18.0과 현재 allocation model

현재 환경은 NVIDIA Device Plugin `v0.18.0`을 사용한다.

legacy Kubernetes Device Plugin model에서 GPU extended resource request는
**container-scoped**다.

예:

```yaml
prefill:
  resources:
    limits:
      nvidia.com/gpu: 2

decode:
  resources:
    limits:
      nvidia.com/gpu: 2
```

이면 kubelet/device-plugin/runtime은 각 container에 별도 device allocation을
적용한다.

결과:

```text
Prefill -> UUID A,B only
Decode  -> UUID C,D only
```

그리고 CUDA는 visible set을 다시 local ordinal 0..N-1로 enumerate한다.

Pod 단위 shared GPU claim은 이 legacy API의 native abstraction이 아니다.

DRA shared ResourceClaim은 장기적인 정식 방향일 수 있으나 현재 환경에서는:

- device-plugin upgrade 부담
- Kubernetes/GPU Operator integration 변경
- NVIDIA driver <580

때문에 즉시 적용 대상으로 삼지 않는다.

MIG와 time-slicing도 사용하지 않는다.

---

# 8. 검토했지만 채택하지 않은 대안

## 8.1 단일 container에 P/D process 모두 실행

기능적으로 동일 device namespace를 쉽게 만들 수 있으나:

- 기존 P/D Cell container lifecycle 구조를 크게 변경
- health/restart/resource attribution 복잡화
- chart architecture 변경 범위가 지나치게 큼

때문에 거부.

## 8.2 `hostIPC` / `shareProcessNamespace`

실제 test 실패.

## 8.3 shared `/dev/shm`

이미 사용 중이고 실패 지속.

## 8.4 Mooncake `nvlink` / MNNVL

`nvlink`는 `nvlink_intra`와 별도 구현이다.

Fabric Memory를 사용할 때:

```text
cuMemRetainAllocationHandle
cuMemExportToShareableHandle(CU_MEM_HANDLE_TYPE_FABRIC)
cuMemImportFromShareableHandle(CU_MEM_HANDLE_TYPE_FABRIC)
cuMemAddressReserve
cuMemMap
cuMemSetAccess
```

를 사용한다.

일반 PyTorch/cudaMalloc allocation이면 Fabric export가 성립하지 않을 수 있고,
vLLM CuMem allocator/Fabric capability/IMEX 조건이 추가된다.

실제 환경에서 MNNVL path도 성공하지 않았으며, 단일-node P/D Cell이라는 product
contract에 비해 요구조건이 과도하다.

따라서 same-node P/D Cell baseline은 `nvlink_intra`로 고정한다.

---

# 9. NIXL은 같은 문제를 갖는가?

## 9.1 결론

> **NIXL connector에 이번 Mooncake workaround를 자동 적용하지 않는다.**

vLLM `NixlConnector`는 Mooncake처럼 connector source에서
`cudaIpcOpenMemHandle()`을 직접 호출하는 고정 implementation이 아니다.

vLLM `v0.26.0`:

```text
NixlConnector / NixlPullConnector / NixlPushConnector
  ↓
NIXL Agent API
  ↓
register_memory(backends=[...])
  ↓
make_prepped_xfer(READ/WRITE)
  ↓
NIXL backend
  ↓
UCX / LIBFABRIC / ...
```

default backend 후보는 UCX지만 `kv_connector_extra_config.backends`로 변경 가능하다.

## 9.2 same-node UCX에서는 CUDA IPC를 쓸 수 있다

NIXL UCX backend가 CUDA memory를 UCX에 등록하면 actual lane selection은 UCX로
내려간다.

same-node NVIDIA 환경에서는 UCX가:

```text
cuda_ipc
cuda_copy
shared-memory/other CUDA-aware lanes
```

중 적절한 path를 선택할 수 있다.

실제 vLLM issue #52607에서는 same-node NIXL/UCX에서 `cuda_ipc` path가 선택되었고,
peer GPU accessibility 조건이 맞지 않을 때:

```text
cudaIpcOpenMemHandle -> invalid argument (1)
```

이 재현되었다.

하지만 이 사실은:

```text
NixlConnector == always CUDA IPC
```

를 의미하지 않는다.

cross-node에서는 UCX RDMA/TCP 등이 선택될 수 있고 LIBFABRIC backend도 존재한다.

따라서 NIXL은 **backend + operation direction + UCX transport selection**까지 포함해서
별도 certification해야 한다.

## 9.3 Chart 정책

현재 PR #4:

```text
MooncakeConnector
  -> pod-local shared GPU namespace workaround
  -> forced nvlink_intra

NixlConnector / NixlPullConnector / NixlPushConnector
  -> 기존 per-container nvidia.com/gpu request 유지
  -> 기존 command path 유지
  -> NIXL side-channel env 유지

MoRIIOConnector
  -> 기존 path 유지
```

한다.

NIXL을 향후 node-local `cuda_ipc`로 고정 운영하기로 결정할 경우 그때 별도
GPU namespace policy를 설계한다.

---

# 10. MoRIIO는 왜 제외하는가

vLLM `MoRIIOConnector`는 AMD ROCm/MoRI IO 계열이다.

관심 transport가:

- ROCm device memory
- RDMA
- xGMI

계열이며 현재 NVIDIA CUDA IPC incident와 동일한 implementation contract가 아니다.

따라서 이번 workaround를 적용하지 않는다.

---

# 11. 채택한 mitigation: Pod-local GPU Reservation Bridge

목표:

```text
1. Kubernetes scheduler GPU accounting 유지
2. P/D container 분리 유지
3. P/D compute GPU는 절대 overlap하지 않음
4. 모든 Mooncake engine이 같은 Cell GPU UUID set/ordinal namespace를 봄
5. values operator가 GPU index/range를 직접 계산하지 않음
6. device-plugin v0.18.0 / current driver 유지
```

## 11.1 values contract

운영자는 기존 값만 선언한다.

예:

```yaml
prefill:
  count: 2
  requestGPU: 2

decode:
  count: 1
  requestGPU: 4
```

Chart 계산:

```text
P total = 2 * 2 = 4
D total = 1 * 4 = 4
Cell GPU total = 8
```

그리고:

```text
P0 -> CNTR_GPU_IDX=0,1
P1 -> CNTR_GPU_IDX=2,3
D0 -> CNTR_GPU_IDX=4,5,6,7
```

를 자동 생성한다.

별도 GPU range values는 없다.

## 11.2 reservation sidecar

```text
gpu-reservation
  resources:
    nvidia.com/gpu: <Cell total>
```

만 Kubernetes GPU extended resource를 직접 요청한다.

sidecar는 자신의 NVIDIA runtime-visible GPU를:

```bash
nvidia-smi --query-gpu=pci.bus_id,uuid
```

로 읽고 PCI bus 기준 deterministic sort 후 UUID만:

```text
/var/run/pd-gpu/gpus
```

에 atomic write한다.

shared `emptyDir`을 사용한다.

HTTP API server는 사용하지 않는다.

이유:

- 별도 port 불필요
- curl dependency 불필요
- startup retry server 불필요
- localhost API health surface 불필요
- atomic file 하나면 dependency barrier가 충분

## 11.3 P/D engine container device injection

Mooncake P/D engine에는 GPU resource request를 직접 붙이지 않는다.

manifest에는:

```yaml
env:
  - name: NVIDIA_VISIBLE_DEVICES
    value: all
```

을 Chart가 강제한다.

중요:

> `NVIDIA_VISIBLE_DEVICES`는 container creation 시 NVIDIA runtime이 소비하므로
> launcher script 안에서 나중에 설정하면 늦다.

따라서 manifest env로 들어간다.

## 11.4 launcher

각 Mooncake engine은 기존 image ENTRYPOINT 대신 Chart ConfigMap의 wrapper를
`command`로 사용한다.

```text
/opt/pd-cell-gpu/launch-vllm.sh
```

wrapper:

1. reservation file wait
2. expected GPU count validation
3. duplicate/empty UUID validation
4. `nvidia-smi`로 reservation GPU가 engine container에서 실제 visible인지 확인
5. 모든 P/D engine에서 동일한 CVD 설정
6. `CNTR_GPU_IDX` range validation
7. selected UUID logging
8. `vllm serve --device-ids ...` exec

공통 CVD:

```bash
CUDA_VISIBLE_DEVICES=<reservation 전체 UUID list>
```

실제 engine 선택:

```bash
vllm serve --device-ids "${CNTR_GPU_IDX}" ...
```

---

# 12. 왜 integer `--device-ids`가 안전한가

vLLM `v0.26.0`의 `EngineArgs._resolve_device_ids()`는:

- string 값이면 physical GPU UUID/ID로 resolve
- integer 값이고 CVD가 있으면 integer를 **CVD-visible set의 index**로 취급

한다.

개념:

```text
CUDA_VISIBLE_DEVICES=
  GPU-A,GPU-B,GPU-C,GPU-D

--device-ids 2,3
       ↓
CVD index 2,3
       ↓
physical GPU C,D
```

그 뒤 worker는 assigned physical GPU ID를 다시 visible ordinal로 변환해
`torch.device("cuda:N")`를 선택한다.

vLLM unit test에도:

```text
CVD=4,5
--device-ids 0,1
-> physical 4,5
```

및 UUID CVD mapping이 포함되어 있다.

따라서 Chart가 actual UUID subset을 직접 만들어 engine별 args에 넣을 필요가 없다.

`CNTR_GPU_IDX`만 자동 생성하면 된다.

---

# 13. 왜 모든 P/D engine의 CVD는 같아야 하는가

잘못된 구성:

```text
P  CVD=A,B
D  CVD=C,D
```

이면:

```text
P cuda:0=A
D cuda:0=C
```

처럼 ordinal namespace가 다시 분리된다.

채택 구성:

```text
P  CVD=A,B,C,D
D  CVD=A,B,C,D
```

이면:

```text
cuda:0=A
cuda:1=B
cuda:2=C
cuda:3=D
```

가 P/D 양쪽에서 동일하다.

실제 compute는:

```text
P  --device-ids 0,1
D  --device-ids 2,3
```

으로 non-overlap을 보장한다.

---

# 14. 다른 GPU workload와 공존

예:

```text
8 GPU node

P/D Cell reservation: 4 GPUs -> A,B,C,D
standalone vLLM:      4 GPUs -> E,F,G,H
```

standalone vLLM은 기존처럼:

```yaml
resources:
  limits:
    nvidia.com/gpu: 4
```

를 사용하면 된다.

scheduler/device-plugin은 reservation sidecar가 차지한 A-D를 정상 workload에 다시
할당하지 않는다.

따라서 **cluster 전체 GPU workload를 reservation pattern으로 바꿀 필요는 없다.**

Mooncake P/D Cell만 특수 path를 사용한다.

---

# 15. isolation caveat

P/D engine은 `NVIDIA_VISIBLE_DEVICES=all`로 device injection을 받으므로 node에
다른 workload GPU가 있으면 Linux device layer에서는 그 GPU도 접근 가능할 수 있다.

launcher가:

```bash
CUDA_VISIBLE_DEVICES=<Cell reservation only>
```

로 좁혀 정상 CUDA runtime에서는 다른 workload GPU를 숨긴다.

즉:

| 계층 | 상태 |
|---|---|
| Kubernetes scheduler accounting | 유지 |
| reservation GPU exclusive allocation | 유지 |
| P/D compute GPU non-overlap | 유지 |
| P/D 동일 CUDA ordinal namespace | 유지 |
| CUDA runtime에서 Cell 외 GPU 숨김 | 유지 |
| Linux device-cgroup hard isolation | **미제공** |

현재 trusted vLLM/Mooncake workload 운영을 위한 bridge다.

hard isolation이 mandatory가 되면 DRA shared ResourceClaim 또는 NRI/custom runtime
계층으로 전환한다.

---

# 16. Helm policy in PR #4

exact `MooncakeConnector`일 때만:

```text
gpu-reservation sidecar                       ON
engine per-container nvidia.com/gpu request  OFF
engine NVIDIA_VISIBLE_DEVICES=all            ON
shared GPU UUID file                         ON
Chart launcher command                       ON
CNTR_GPU_IDX auto assignment                 ON
vLLM --device-ids                            ON
mooncake_protocol=nvlink_intra               FORCED
```

NIXL/MoRIIO:

```text
기존 per-container GPU resource allocation 유지
기존 command 유지
Mooncake launcher 없음
```

Mooncake에서 다음 사용자 override는 reject:

```text
kv_connector_extra_config.mooncake_protocol
--device-ids
MIG/time-slicing/shared GPU requestGPUType
```

`requestGPUType`은 `nvidia.com/gpu`만 허용한다.

---

# 17. `requestGPU`의 새로운 의미

Mooncake path에서 `requestGPU`는 더 이상 engine container resource field 자체가 아니다.

다음 세 용도의 source-of-truth이다.

```text
1. engine local worker GPU count
2. CPU/memory default sizing
3. Cell total reservation 및 automatic GPU index partition
```

따라서 반드시:

> `requestGPU == 해당 engine profile이 실제 생성하는 local GPU worker 수`

여야 한다.

예:

```text
TP4, local single engine -> requestGPU=4
TP2                    -> requestGPU=2
```

profile 내부 TP/PP/DP와 requestGPU가 어긋나면 vLLM `--device-ids` assignment와 실제
worker topology가 맞지 않아 startup/distributed init이 실패할 수 있다.

---

# 18. restart semantics

reservation UUID file은 Pod-local `emptyDir`에 있다.

- engine container restart: file 유지
- reservation sidecar restart: 동일 Pod device allocation을 다시 enumerate 후 file rewrite
- Pod recreate: 새로운 GPU allocation/UUID list 생성

vLLM/Mooncake engine restart에는 별도의 stale `engine_id` 문제가 있으므로 GPU map
문제와 별개로 Cell whole-restart policy를 유지해서 검증해야 한다.

---

# 19. Runtime certification plan

## Gate 0 — rendered manifest

Mooncake P1D2 예:

```text
P requestGPU=4, count=1
D requestGPU=2, count=2
total=8
```

확인:

```text
gpu-reservation:
  nvidia.com/gpu requests=8
  nvidia.com/gpu limits=8

prefill-0:
  no nvidia.com/gpu resource
  NVIDIA_VISIBLE_DEVICES=all
  CNTR_GPU_IDX=0,1,2,3

decode-0:
  no nvidia.com/gpu resource
  NVIDIA_VISIBLE_DEVICES=all
  CNTR_GPU_IDX=4,5

decode-1:
  no nvidia.com/gpu resource
  NVIDIA_VISIBLE_DEVICES=all
  CNTR_GPU_IDX=6,7
```

최종 KVTransfer JSON:

```text
mooncake_protocol = nvlink_intra
```

확인.

NIXL render에는 위 reservation/launcher가 없어야 한다.

## Gate 1 — reservation sidecar

로그:

```text
[pd-gpu] reservation ready: GPU-...,GPU-...
```

확인.

```bash
kubectl exec <pod> -c gpu-reservation -- cat /var/run/pd-gpu/gpus
```

UUID count와 expected total 일치.

## Gate 2 — common CUDA namespace

모든 P/D engine에서:

```bash
echo "$NVIDIA_VISIBLE_DEVICES"
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
python - <<'PY'
import torch
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i).uuid)
PY
```

확인.

Expected:

```text
NVIDIA_VISIBLE_DEVICES=all
CVD = reservation UUID set only
torch ordinal -> UUID mapping identical across every P/D engine
```

## Gate 3 — vLLM compute assignment

engine startup log 또는 temporary debug:

```text
P0 selected UUIDs = expected first range
D0 selected UUIDs = next range
D1 selected UUIDs = next range
no overlap
```

launcher 자체가 selected indices/UUIDs를 log한다.

## Gate 4 — Mooncake control plane

한 request에서:

```text
same transfer_id
P do_remote_decode=true
D do_remote_prefill=true
remote_engine_id present
remote_bootstrap_addr present
```

확인.

## Gate 5 — CUDA IPC mapping

기존 실패 로그:

```text
cudaIpcOpenMemHandle failed: invalid argument
cudaIpcOpenMemHandle failed: invalid device context
```

가 사라져야 한다.

필요 시 instrumentation 유지:

```text
current context/device
source pointer device
remote base
handle signature
```

## Gate 6 — actual KV bytes

반드시:

```text
P transfer submit success
P bytes/descriptors transferred
D receive/load complete
Decode prompt recompute 없음
```

을 증명한다.

HTTP 200/text generation만으로 PASS 처리하지 않는다.

## Gate 7 — MTP

plain MTP OFF/OFF를 먼저 통과시킨다.

그 뒤:

```text
P MTP ON
D MTP ON
```

대칭 구성으로 별도 correctness test.

asymmetric MTP는 기존 layer-spec mismatch 때문에 독립 문제로 관리한다.

## Gate 8 — P1D2 distribution

consistent-hash 때문에 한 Decode만 계속 선택될 수 있으므로 hash/session 입력을
변경하거나 임시 round-robin으로 D0/D1 모두 실제 KV receive를 증명한다.

## Gate 9 — coexistence

동일 node에 standalone vLLM 등 다른 GPU workload가 있는 상태에서:

- scheduler가 Cell reservation GPU를 재할당하지 않는지
- Cell launcher CVD에 다른 workload GPU가 들어오지 않는지
- standalone workload의 정상 GPU ownership이 유지되는지

확인.

## Gate 10 — performance

위 모든 gate 이후에만:

- TTFT
- TPOT/ITL
- transfer BW
- NVLink counters
- long prompt
- concurrency
- soak

를 측정한다.

---

# 20. Rollback

새 GPU namespace path에서 문제가 발생하면 connector를 NIXL로 바꾼다고 자동으로
같은 semantics가 유지되는 것이 아니다.

현재 Chart는 connector별로 완전히 분리한다.

```text
MooncakeConnector
 -> shared reservation contract

NIXL/MoRIIO
 -> existing per-engine resource contract
```

따라서 rollback은 connector semantics까지 포함하여 명시적으로 수행한다.

Mooncake 자체에서 과거 per-container resource path로 돌아가는 것은
`nvlink_intra` CUDA IPC 실패를 다시 만들 수 있으므로 production fallback으로
간주하지 않는다.

---

# 21. Source evidence map

## vLLM 0.26.0

Mooncake connector:

- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py

NIXL connector:

- https://github.com/vllm-project/vllm/tree/v0.26.0/vllm/distributed/kv_transfer/kv_connector/v1/nixl
- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/distributed/nixl_utils.py

`--device-ids`:

- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/engine/arg_utils.py
- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/platforms/interface.py
- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/platforms/cuda.py
- https://github.com/vllm-project/vllm/blob/v0.26.0/tests/engine/test_arg_utils.py

Thread-local CUDA device bug/fix reference:

- https://github.com/vllm-project/vllm/pull/39548

Related Mooncake same-node issue:

- https://github.com/vllm-project/vllm/issues/51518

NIXL/UCX same-node cuda_ipc issue:

- https://github.com/vllm-project/vllm/issues/52607

## Mooncake

- https://github.com/kvcache-ai/Mooncake/blob/v0.3.10/mooncake-transfer-engine/src/transport/intranode_nvlink_transport/intranode_nvlink_transport.cpp
- https://github.com/kvcache-ai/Mooncake/blob/v0.3.10/mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp
- https://github.com/kvcache-ai/Mooncake/pull/1622
- https://github.com/kvcache-ai/Mooncake/pull/2578

## NVIDIA Device Plugin / Kubernetes

- https://github.com/NVIDIA/k8s-device-plugin/tree/v0.18.0
- https://github.com/NVIDIA/k8s-device-plugin/blob/v0.18.0/internal/rm/allocate.go
- https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/

## NIXL

- https://github.com/ai-dynamo/nixl/tree/v1.3.1
- https://github.com/ai-dynamo/nixl/tree/v1.3.1/src/plugins/ucx

## MoRIIO

- https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py

---

# 22. Final decision

현재 node-local P/D Cell에서 exact `MooncakeConnector`의 production contract는:

```text
single Pod / single node
+
exclusive nvidia.com/gpu reservation sidecar
+
all Mooncake engines see peer GPU devices
+
identical Cell-wide CUDA_VISIBLE_DEVICES
+
automatic non-overlapping vLLM --device-ids
+
Mooncake protocol forced to nvlink_intra
```

이다.

NIXL/MoRIIO에는 이 정책을 자동 상속하지 않는다.

이 설계는 현재 driver/device-plugin을 유지하면서 Kubernetes scheduler accounting과
P/D container lifecycle을 보존하고, Mooncake CUDA IPC가 필요로 하는 공통 GPU
namespace를 제공하기 위한 **현실적인 bridge**다.

최종 PASS는 PR #4의 manifest render 성공이 아니라, 위 Runtime Certification
Gate 5~6에서 `cudaIpcOpenMemHandle` 성공과 실제 KV bytes handoff가 증명된 시점이다.
