# Mooncake Transfer Engine — vLLM P/D KV Transfer Deep Dive

기준:

- 재현 기준: vLLM `v0.27.1` + Mooncake `0.3.10.post2`
- 차기 validation: vLLM `v0.28.0-cu129` + Mooncake `0.3.12.post1`
- 실무 source-build reference: `Soomin-Jung/vllm-production-stack-custom` PR #5
- 0.3.12.post1 official x86 wheel에도 `USE_INTRA_NVLINK=ON`이 없으므로 `nvlink_intra` source-build requirement는 유지

이 디렉터리는 **MooncakeConnector 자체보다 한 단계 아래인 Mooncake Transfer Engine을 중심으로** 설명한다.

---

## 1. 한 문장 정의

Mooncake Transfer Engine은 application이 등록한 local/remote memory region 사이에서 `READ/WRITE` 요청을 받아, topology와 compiled transport에 맞춰 **RDMA, intra-node NVLink/CUDA IPC, MNNVL/Fabric Memory, TCP 등으로 실제 bytes를 옮기는 native transfer runtime**이다.

vLLM의 `MooncakeConnector`는 이 runtime의 client/adaptor다.

```text
vLLM Scheduler / Worker lifecycle
          |
          v
MooncakeConnector
  - request/block orchestration
  - bootstrap/ZMQ side channel
  - src/dst pointer construction
          |
          v
Mooncake TransferEngine Python binding
          |
          v
TransferEngineImpl
          |
          v
MultiTransport
   |-- RdmaTransport
   |-- NvlinkTransport          # MNNVL / cross-node NVLink fabric 계열
   |-- IntraNodeNvlinkTransport # same-node CUDA IPC/P2P
   `-- TcpTransport
          |
          v
GPU/NIC/fabric
```

---

## 2. 왜 vLLM connector와 Transfer Engine을 분리해서 봐야 하나

Mooncake 문제를 겪을 때 실패 지점이 서로 다르다.

```text
A. Router contract 실패
B. vLLM Scheduler block orchestration 실패
C. Mooncake bootstrap/ZMQ peer discovery 실패
D. GPU memory registration 실패
E. Transfer Engine transport initialization 실패
F. transport 선택이 기대와 다름
G. 실제 transfer submit/completion 실패
H. transfer는 됐지만 wrong block/offset/layout
```

`import mooncake`가 성공했다고 E~H가 해결되는 것이 아니다.

우리에게 실제로 발생했던 대표 사례도 **wheel은 설치되었지만 `nvlink_intra` 구현이 artifact에 포함되지 않은 packaging 문제**였다. 이 경계는 0.3.12.post1 x86 official release build에서도 그대로 확인되므로 version upgrade만으로 해결됐다고 보지 않는다.

---

## 3. Mooncake Transfer Engine core objects

### `TransferEngine`

vLLM Python에서 직접 생성하는 binding object다.

vLLM v0.27.1:

```python
self.engine = TransferEngine()
self.engine.initialize(
    self.hostname,
    "P2PHANDSHAKE",
    protocol,
    device_name,
)
```

vLLM 관점에서는 이것이 native runtime 경계다.

### `TransferEngineImpl`

Mooncake C++ core implementation.

책임:

- RPC/P2P metadata endpoint 초기화
- topology/HCA discovery
- transport install
- local memory registration
- remote segment metadata 조회
- batch transfer lifecycle

Source:

- `mooncake-transfer-engine/src/transfer_engine_impl.cpp`

### `MultiTransport`

compiled transport implementation을 보유하고 요청별 target segment protocol에 맞춰 transport를 선택한다.

Source:

- `mooncake-transfer-engine/src/multi_transport.cpp`

중요:

```text
local vLLM config의 mooncake_protocol
        !=
각 request가 실제 사용할 transport를 단독 결정하는 값
```

`MultiTransport::selectTransport()`는 target segment metadata에 기록된 `protocol`을 읽고, local `transport_map_`에 동일 구현이 설치되어 있는지 검사한다.

---

## 4. 두 개의 metadata/control plane

Mooncake를 이해하기 어려운 이유 중 하나는 **vLLM 자체 side channel과 Mooncake Transfer Engine metadata가 모두 존재**하기 때문이다.

### 4.1 vLLM Mooncake side channel

목적:

- 어느 P engine/TP/PP worker가 request를 보유하는지
- D의 destination block IDs/base address가 무엇인지
- request `transfer_id`가 무엇인지

전달한다.

사용 기술:

- Prefill bootstrap server: HTTP `/register`, `/query`
- D -> P worker request: ZMQ DEALER/ROUTER

### 4.2 Mooncake Transfer Engine metadata/RPC

목적:

- registered memory segment
- remote segment protocol
- transport-specific remote access metadata
- Transfer Engine RPC endpoint

을 관리한다.

vLLM은 `P2PHANDSHAKE` mode로 Transfer Engine을 초기화한다.

따라서 실제 흐름은:

```text
Router
  -> vLLM request metadata

D MooncakeConnector
  -> P bootstrap HTTP
  -> P worker ZMQ

P MooncakeConnector
  -> Mooncake Transfer Engine
  -> Mooncake remote segment metadata/RPC
  -> selected transport
  -> D registered memory
```

이다.

---

## 5. Memory registration

vLLM worker가 model KV cache를 만든 직후 `register_kv_caches()`를 호출한다.

Mooncake connector는 layer별 tensor/storage를 조사해:

- base address
- block stride/length
- transferable KV region length
- layer name/index
- KV cache group index

를 기록한다.

그 뒤 unique storage pointer에 대해:

```python
self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
```

를 호출한다.

### 왜 storage pointer를 보나

Torch tensor는 allocation 전체의 view일 수 있다.

```text
CUDA allocation/storage
+-----------------------------------+
| tensor A view | tensor B view ... |
+-----------------------------------+
^
storage.data_ptr()
```

transport가 remote-accessible memory를 등록할 때는 tensor view의 논리 범위와 backing allocation을 구분해야 한다.

`nvlink_intra`도 caching allocator sub-allocation 때문에 실제 CUDA allocation base를 다시 찾는 로직을 가진다.

---

## 6. vLLM에서 실제 data-plane operation

MooncakeConnector의 중요한 특징:

> **D가 P에게 전송을 요청하지만, P가 D 메모리로 WRITE한다.**

순서:

```text
1. D Scheduler가 destination KV block 할당
2. D worker가 자신의 registered KV base address + block IDs를 준비
3. D가 P bootstrap에서 P worker ZMQ 주소를 찾음
4. D -> P: MooncakeXferMetadata
5. P가 request의 local source blocks가 ready될 때까지 대기
6. P가 source/destination raw pointer + lengths 계산
7. P: TransferEngine.batch_transfer_sync_write(...)
8. Mooncake transport가 D memory로 bytes WRITE
9. P -> D ZMQ response: request transfer completion
10. D가 finished_recving_reqs에 request 등록
11. vLLM Scheduler가 Decode 진행
```

따라서 traffic 분석에서도:

- ZMQ request direction: D → P
- KV data direction: P → D

를 구분한다.

---

## 7. Pointer 계산

vLLM은 block IDs를 그대로 Transfer Engine에 넘기지 않는다.

개념적으로:

```text
src_ptr = P_region_base
        + P_block_id * P_block_stride
        + P_TP_slice_offset

dst_ptr = D_region_base
        + D_block_id * D_block_stride
        + D_TP_slice_offset

length = transferable bytes for this TP mapping
```

v0.27.1은 추가로:

- contiguous block run coalescing
- logical vs kernel physical block mapping
- HMA group identity
- PP layer alignment
- heterogeneous TP
- replicated MLA KV

까지 처리한다.

그 결과 `src_ptrs`, `dst_ptrs`, `lengths` arrays가 만들어지고 Transfer Engine batch API에 전달된다.

---

## 8. Transport install vs request selection

두 단계를 구분한다.

### initialization: 어떤 transport object가 설치되는가

`TransferEngineImpl::init()`:

```text
topology discover
 -> compile-time branch
 -> MC_* environment branch
 -> installTransport(...)
 -> transport_map_[protocol]
```

### request: 어떤 installed transport를 쓰는가

`MultiTransport::selectTransport()`:

```text
request.target_id
 -> target segment descriptor
 -> descriptor.protocol
 -> transport_map_[protocol]
```

즉 양쪽 endpoint의 segment metadata와 local compiled feature set이 일치해야 한다.

---

## 9. Compile-time feature matrix

Mooncake `multi_transport.cpp`는 macro가 있을 때만 구현 class를 compile/include한다.

대표:

| CMake / macro | runtime transport name | 의미 |
|---|---|---|
| 기본 RDMA support | `rdma` | verbs 기반 network transfer |
| `USE_MNNVL=ON` | `nvlink` | MNNVL / Fabric Memory 계열 |
| `USE_INTRA_NVLINK=ON` | `nvlink_intra` | same-node CUDA IPC/P2P |
| `USE_TCP=ON` | `tcp` | TCP socket + staging path |

구현이 compile되지 않았는데 `installTransport()`를 요청하면:

```text
Unsupported transport <name>, please rebuild Mooncake
```

으로 실패한다.

이것이 **왜 source build feature manifest가 운영 artifact의 일부여야 하는지**를 보여준다.

---

## 10. 문서 구성

### [Transport and Runtime Paths](transport-and-runtime-paths.md)

- transport selection source branch
- RDMA
- `nvlink`
- `nvlink_intra`
- TCP
- 실제 GPU/NIC physical path
- 어떤 로그/카운터로 증명할지

### [Air-gapped Source Build](source-build-airgap.md)

- PR #5를 최신 vLLM base image에 일반화
- source closure/submodule lock
- CUDA 12/13 ABI
- build profile별 CMake flags
- runtime image slimming
- build vs runtime certification

### [vLLM Connector Flow & Debugging](vllm-connector-flow-debugging.md)

- Router → Scheduler → Worker → Transfer Engine end-to-end
- request state
- stale engine/timeout/failure
- TP/layout mismatch
- 로그 위치
- certification checklist

---

## 11. 우리 환경에서 권장하는 build profile 사고방식

하나의 "모든 기능 ON" image만 관리하기보다 **목적과 fallback semantics를 명시한 profile**을 관리하는 편이 안전하다.

예:

```text
profile: nvidia-node-local
  USE_CUDA=ON
  USE_INTRA_NVLINK=ON
  USE_MNNVL=OFF or deliberately controlled
  USE_TCP policy explicitly decided

profile: nvidia-rdma
  USE_CUDA=ON
  USE_MNNVL=OFF
  USE_INTRA_NVLINK=OFF
  USE_TCP=ON
  RDMA deps present

profile: nvidia-mixed-validation
  USE_CUDA=ON
  USE_MNNVL=ON
  USE_INTRA_NVLINK=ON
  USE_TCP=ON
  -> runtime selection branch must be understood precisely
```

현재 PR #5는 마지막에 가까운 **NVLink-enabled validation build**다.

---

## 12. 핵심 source map

Mooncake `0.3.10.post2`:

```text
mooncake-transfer-engine/
├── src/transfer_engine_impl.cpp
├── src/multi_transport.cpp
├── src/transfer_metadata.cpp
├── src/transport/
│   ├── rdma_transport/
│   ├── nvlink_transport/
│   ├── intranode_nvlink_transport/
│   └── tcp_transport/
└── include/...
```

vLLM `v0.27.1`:

```text
vllm/distributed/kv_transfer/kv_connector/v1/mooncake/
├── mooncake_connector.py
├── mooncake_utils.py
├── rdma_utils.py
└── stats.py
```

두 저장소를 동시에 봐야 end-to-end call chain이 완성된다.

---

## References

- https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
- https://github.com/kvcache-ai/Mooncake/blob/v0.3.10.post2/mooncake-transfer-engine/src/transfer_engine_impl.cpp
- https://github.com/kvcache-ai/Mooncake/blob/v0.3.10.post2/mooncake-transfer-engine/src/multi_transport.cpp
- https://github.com/Soomin-Jung/vllm-production-stack-custom/pull/5


## Version Migration Note — vLLM 0.28 / Mooncake 0.3.12

0.3.10.post2의 source-level 설명은 transport architecture와 최초 incident 재현 기준으로 보존한다. 최신 runtime migration에서는 다음을 다시 diff한다.

- source/submodule lock
- CMake option/target
- Python build dependency
- vLLM Mooncake connector interface
- hybrid KV/state cache handling
- transport selection log
- CUDA/PyTorch/glibc ABI

현재 migration 기준은 [vLLM 0.28.0 Migration & KV Connector Compatibility](../../../2026-09-03-vllm-0.28-migration.md)를 우선한다.
