# KV Transfer Backends for vLLM P/D Disaggregation

업데이트: 2026-08-25 KST

이 디렉터리는 vLLM P/D disaggregation에서 **KV cache가 실제로 어떤 소프트웨어 계층과 하드웨어 경로를 통과해 Prefill(P)에서 Decode(D)로 이동하는지**를 소스 코드 기준으로 정리한다.

단순히 `MooncakeConnector` 또는 `NixlConnector` 사용법을 기록하는 것이 목적이 아니다. 다음 질문에 답할 수 있어야 한다.

- Router가 어떤 `kv_transfer_params`를 만들고 P/D에 어떻게 주입하는가?
- vLLM Scheduler가 remote KV load/save를 언제 결정하는가?
- Worker가 어느 시점에 KV tensor를 등록하고 실제 transfer를 시작하는가?
- Connector 아래의 transfer runtime은 어떤 메모리 descriptor를 만들고 어떤 backend/protocol을 선택하는가?
- GPU VRAM에서 출발한 bytes가 NVLink/NVSwitch, PCIe, HCA, InfiniBand/RoCE, TCP를 어떤 조건에서 통과하는가?
- 같은 `RDMA`, `NVLink`라는 단어가 Mooncake와 NIXL에서 각각 어느 계층을 의미하는가?
- wheel import 성공과 실제 GPU-direct KV transfer 성공을 어떻게 구분하는가?
- 폐쇄망 image에서 어떤 dependency를 source build해야 하며 무엇을 반드시 pin해야 하는가?

---

## 1. 기준 버전

### vLLM runtime 기준

2026-08-25 현재 vLLM 최신 release는 `v0.27.1`이다.

- vLLM: `v0.27.1`
- upstream Dockerfile 기본: Python 3.12 / CUDA 13.0.3
- release에는 CUDA 12.9 계열 artifact도 존재하므로 **실제 base image의 CUDA/PyTorch ABI를 image 내부에서 확인**한다.
- `requirements/kv_connectors.txt`
  - `nixl == 1.3.1`
  - `mooncake-transfer-engine >= 0.3.8`

즉 두 backend의 version policy가 다르다.

- **NIXL**: vLLM이 exact pin한다. v0.27.1 분석은 `NIXL 1.3.1`을 runtime compatibility baseline으로 본다.
- **Mooncake**: lower bound만 있으므로 reproducible deployment를 위해 별도 pin이 필요하다. 현재 검증 이력은 `0.3.10.post2`다.

NIXL upstream 최신 release는 `1.4.0`이지만 vLLM 0.27.1의 공식 pin보다 새 버전이다. 따라서 이 문서에서 NIXL 1.4.0은 **upstream architecture/delta 참고용**으로만 사용하고, production image에 무검증 upgrade하지 않는다.

---

## 2. 용어 계층 — 가장 먼저 구분해야 하는 것

```text
Application / Router
        |
        v
vLLM KV Connector
  MooncakeConnector
  NixlPullConnector / NixlPushConnector
        |
        v
Transfer Runtime / Library
  Mooncake Transfer Engine
  NIXL Agent
        |
        v
Transport / Backend Plugin
  Mooncake: rdma / nvlink / nvlink_intra / tcp ...
  NIXL: UCX / LIBFABRIC / GDS / POSIX / OBJ ...
        |
        v
Low-level mechanism
  CUDA IPC / CUDA P2P / GPUDirect RDMA / verbs / UCX PUT/GET
  libfabric provider / socket / cuFile ...
        |
        v
Physical fabric
  NVLink / NVSwitch / PCIe / InfiniBand / RoCE / Ethernet / NVMe
```

### Connector

vLLM이 Scheduler/Worker lifecycle에 맞춰 외부 KV를 load/save하도록 연결하는 adapter다.

예:

- `MooncakeConnector`
- `NixlPullConnector`
- `NixlPushConnector`

### Transfer runtime

Connector가 실제 데이터 이동을 맡기는 별도 라이브러리다.

- MooncakeConnector → `mooncake-transfer-engine`
- NixlConnector → `NIXL agent`

### Backend / transport

Transfer runtime 내부에서 데이터 이동을 수행하는 구현체다.

- Mooncake는 구현체를 `Transport`라고 부르는 편이다.
- NIXL은 modular `backend/plugin` 추상화를 사용한다.

### Physical fabric

실제 GPU/NIC/스위치/케이블 수준의 경로다. `nvlink`라는 설정 문자열이 있다고 해서 반드시 물리적으로 NVLink를 탔다고 단정하면 안 된다. topology와 peer mapping까지 확인해야 한다.

---

## 3. vLLM 공통 경로

vLLM v0.27.1에서는 Scheduler와 Worker가 각자 별도 connector instance를 가진다.

```text
KVTransferConfig
   |
   v
KVConnectorFactory
   |-- role=SCHEDULER -> connector scheduler object
   `-- role=WORKER    -> connector worker object

Scheduler
   | get_num_new_matched_tokens()
   | allocate_slots()
   | update_state_after_alloc()
   | build_connector_meta()
   v
SchedulerOutput.kv_connector_metadata
   |
   v
GPUModelRunner / ActiveKVConnector.pre_forward()
   | bind_connector_metadata()
   | start_load_kv()
   v
Connector Worker
   | memory descriptors / handshake / transfer
   v
post_forward()
   | get_finished()
   | invalid_block_ids
   | connector stats
   v
Scheduler state update
```

자세한 source-level call path는 [vLLM KV Transfer Runtime Path](vllm-kv-transfer-path.md)를 본다.

---

## 4. Mooncake와 NIXL의 핵심 차이

| 항목 | MooncakeConnector | NIXL Pull | NIXL Push |
|---|---|---|---|
| vLLM connector | `MooncakeConnector` | `NixlConnector` = `NixlPullConnector` | `NixlPushConnector` |
| 데이터 주도권 | P가 실제 bytes WRITE | D가 P에서 READ | P가 D로 WRITE |
| control side channel | Mooncake bootstrap HTTP + ZMQ | vLLM NIXL ZMQ handshake + NIXL notif | 동일 + PUSH_REG notif |
| transfer runtime | Mooncake Transfer Engine | NIXL | NIXL |
| backend selection | Mooncake compile flags + env + topology + remote segment protocol | `kv_connector_extra_config.backends`, 기본 `UCX` | 동일 |
| memory abstraction | registered Mooncake memory region | NIXL registered descriptors/sections | 동일 |
| same-node NVLink 접근 | `nvlink_intra` transport를 명시적으로 build/선택 | 보통 UCX가 CUDA IPC/P2P를 사용할 수 있음 | push가 same-node/non-RDMA에 더 자연스러운 경우가 있음 |
| cross-node RDMA | Mooncake RDMA transport | NIXL UCX 또는 LIBFABRIC 등 | 동일 |

Mooncake에서 Decoder가 Prefill에 ZMQ 요청을 보내므로 겉보기에는 pull처럼 보이지만 **실제 data plane operation은 Prefill의 `batch_transfer_sync_write()`**다.

NIXL Pull은 이름 그대로 Decode가 `make_prepped_xfer("READ", ...)` 후 transfer를 발행한다.

NIXL Push는 Decode가 destination registration을 notification으로 알려주고 Prefill이 `make_prepped_xfer("WRITE", ...)`를 발행한다.

---

## 5. 디렉터리 읽는 순서

1. [vLLM KV Transfer Runtime Path](vllm-kv-transfer-path.md)
2. [Backend Selection and Comparison](backend-selection.md)
3. [Mooncake Transfer Engine](mooncake-transfer-engine/README.md)
4. [Mooncake transport와 실제 GPU data path](mooncake-transfer-engine/transport-and-runtime-paths.md)
5. [Mooncake 폐쇄망/source build](mooncake-transfer-engine/source-build-airgap.md)
6. [Mooncake vLLM lifecycle와 debugging](mooncake-transfer-engine/vllm-connector-flow-debugging.md)
7. [NIXL Core Architecture](nixl/README.md)
8. [vLLM NIXL Integration](nixl/vllm-integration.md)
9. [NIXL backend/plugin data path](nixl/backends-data-path.md)
10. [NIXL 폐쇄망/source build](nixl/source-build-airgap.md)
11. [NIXL debugging/validation](nixl/debugging-validation.md)

---

## 6. Network A / Network B 기준 사고방식

### Network B: cross-node IB/RDMA를 전제로 할 수 없는 환경

우선 목표는 **P/D가 동일 node 안에서 GPU P2P 경로를 사용하도록 cell을 배치**하는 것이다.

후보:

- Mooncake `nvlink_intra`
- NIXL Push/UCX same-node CUDA IPC/P2P

검증해야 할 것은 단순 initialization이 아니라 다음이다.

```text
P GPU KV address
  -> registered/exported GPU memory
  -> peer GPU mapping
  -> actual transfer
  -> D GPU KV address
```

CPU host staging이 발생하지 않았는지 확인해야 한다.

### Network A: IB/RDMA 사용 가능 환경

후보:

- Mooncake `rdma`
- NIXL `UCX`
- 환경에 따라 NIXL `LIBFABRIC`

검증 항목:

- HCA discovery
- GPU memory registration
- GPUDirect RDMA 조건
- `nvidia-peermem`/DMA-BUF/driver 조건
- memlock
- GID/port/MTU
- NUMA/GPU/NIC affinity
- 실제 NIC counter와 transfer throughput

---

## 7. 절대 혼동하지 않을 체크포인트

### `mooncake_protocol` != 실제 physical transport 증명

vLLM Mooncake worker는 `mooncake_protocol`을 `TransferEngine.initialize()`에 넘긴다. 그러나 Mooncake 0.3.10.post2의 실제 auto-installed transport는 compile flag, `MC_*` env, HCA discovery에 의해 결정된다.

### `NixlConnector` != 별도 network protocol

NIXL은 abstraction library다. 기본 backend가 UCX일 뿐이며 `kv_connector_extra_config.backends`로 다른 plugin을 지정할 수 있다.

### NCCL tuning != KV connector tuning

NIXL/UCX KV transfer는 NCCL data path가 아니다. `NCCL_IB_HCA` 같은 env만 바꿔서는 NIXL 경로가 바뀌지 않는다.

Mooncake 역시 자체 transfer engine을 사용한다.

### import 성공 != transfer 성공

다음은 서로 다른 Gate다.

1. wheel install
2. Python import
3. connector initialize
4. memory registration
5. peer/bootstrap handshake
6. data descriptor 생성
7. 실제 transfer submit
8. transfer completion
9. D가 받은 KV로 decode하고 prompt recompute가 발생하지 않음

---

## 8. 핵심 upstream source

### vLLM v0.27.1

- `vllm/config/kv_transfer.py`
- `vllm/distributed/kv_transfer/kv_connector/factory.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu/kv_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`
- `requirements/kv_connectors.txt`

### Mooncake 0.3.10.post2

- `mooncake-transfer-engine/src/transfer_engine_impl.cpp`
- `mooncake-transfer-engine/src/multi_transport.cpp`
- `mooncake-transfer-engine/src/transport/rdma_transport/`
- `.../nvlink_transport/`
- `.../intranode_nvlink_transport/`
- `.../tcp_transport/`

### NIXL

- upstream: `ai-dynamo/nixl`
- architecture: `docs/nixl.md`
- public API: `src/api/cpp/`
- plugins: `src/plugins/`

---

## 9. 현재 실무 기준

- 기존 Mooncake source build 경험은 `Soomin-Jung/vllm-production-stack-custom` PR #5를 canonical implementation reference로 사용한다.
- PR #5의 `v0.26.0-cu129` 고정 package 이름을 v0.27.1 image에 그대로 복사하지 않는다. 먼저 base image의 CUDA runtime/devel ABI를 확인한다.
- 새 image는 tag만이 아니라 digest와 build manifest를 기록한다.
- Network B는 먼저 node-local transfer를 인증한 뒤 cross-node와 분리한다.
- Network A는 RDMA initialization 성공과 GPU-direct 성공을 분리해서 인증한다.

이 디렉터리의 문서는 최종적으로 **"어느 설정을 넣었나"가 아니라 "어느 code branch가 실행되어 어떤 bytes가 어떤 물리 경로로 이동했나"**를 설명하는 것을 기준으로 한다.
