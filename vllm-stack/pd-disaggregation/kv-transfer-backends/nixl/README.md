# NIXL — NVIDIA Inference Xfer Library Deep Dive

기준:

- vLLM: `v0.27.1`
- vLLM runtime pin: `nixl == 1.3.1`
- NIXL upstream latest 참고: `v1.4.0` (production baseline 아님)

이 디렉터리는 NIXL을 **vLLM connector가 아니라 독립적인 inference data movement library**부터 이해한 뒤 vLLM P/D integration으로 올라간다.

---

## 1. NIXL이 정확히 무엇인가

NIXL = **NVIDIA Inference Xfer Library**.

분산 inference framework가 직접 UCX, RDMA verbs, GPUDirect Storage, object storage client 등 서로 다른 data movement API를 각각 다루지 않도록:

```text
memory / storage descriptor
        +
remote agent metadata
        +
READ / WRITE operation
        |
        v
      NIXL Agent
        |
        v
backend/plugin selection
        |
        +--> UCX
        +--> LIBFABRIC
        +--> GDS
        +--> POSIX
        +--> OBJ
        +--> HF3FS
        +--> Mooncake plugin
        `--> ...
```

형태의 공통 abstraction을 제공한다.

NIXL은 "RDMA protocol"도 아니고 "UCX의 다른 이름"도 아니다.

---

## 2. vLLM과의 관계

```text
vLLM P/D Router
      |
      v
vLLM NixlPullConnector / NixlPushConnector
      |
      v
NIXL Python API / NixlWrapper
      |
      v
NIXL Agent (native core)
      |
      v
NIXL backend plugin
      |
      v
UCX / libfabric / storage API
      |
      v
NVLink / PCIe / IB / RoCE / Ethernet / storage
```

vLLM은 NIXL의 conductor 역할 일부를 담당한다.

- KV tensor allocation: vLLM
- local/remote block mapping: vLLM connector
- metadata side channel: vLLM ZMQ
- transfer descriptor/handle/backend: NIXL
- actual physical movement: NIXL backend + underlying library/hardware

---

# Part A. NIXL의 핵심 객체

## 3. Transfer Agent

NIXL 공식 architecture에서 중심 객체다.

Agent는 한 process 안에서:

- backend/plugin instances
- registered memory/storage sections
- remote agent metadata cache
- transfer request/handle
- notification

을 관리한다.

각 agent는 globally unique한 name/ID를 가진다.

vLLM connector에서는 worker별 NIXL agent가 생성되며, remote engine metadata를 handshake로 받아 local agent에 추가한다.

---

## 4. Memory Section / Segment Descriptor

NIXL은 raw pointer 하나만 전달하지 않는다.

공식 개념상 memory section은 address range/segment들의 집합이며, memory type을 추상화한다.

예:

```text
VRAM
DRAM
File
NVMe / storage
Object
```

vLLM GPU KV path에서는 주로:

```text
CUDA KV tensor
 -> VRAM descriptor
```

가 된다.

`kv_buffer_device=cpu`면 host buffer를 만들고 DRAM을 NIXL에 등록할 수 있다.

---

## 5. Registered Memory vs Transfer Descriptor

둘을 구분해야 한다.

### registration

"이 memory range를 backend가 remote access 가능한 resource로 준비해라."

```text
NIXL agent.register_memory(...)
```

### transfer descriptor

"등록된 memory 중 이번 transaction에서 어떤 subranges/block들을 움직일 것인가."

vLLM은 KV block ID를 NIXL descriptor index로 바꾸고 prepared handle을 재사용한다.

이 구분 덕분에 매 request마다 전체 KV allocation을 다시 register하지 않고 **초기 registration + request별 index selection**을 할 수 있다.

---

## 6. Backend Plugin Interface

NIXL core는 특정 transport에 고정되지 않는다.

backend plugin이 다음을 구현한다.

- memory registration/deregistration
- backend-specific metadata serialize/load
- endpoint/connection management
- transfer capability check
- READ/WRITE post/progress/status

v1.3.1 source tree에는 UCX, LIBFABRIC, GDS, POSIX, object/storage 계열 등 다양한 plugin이 존재한다.

vLLM `NixlConnector` 기본:

```python
backends = ["UCX"]
```

이다.

---

## 7. Metadata Handler

remote memory를 one-sided operation으로 접근하려면 단순 IP만으로 부족하다.

backend에 따라:

- endpoint address
- registered memory key/handle
- device identity
- remote segment metadata

가 필요하다.

NIXL은 이 backend metadata를 serialize한다.

공식 설계는 **metadata exchange 방식 자체를 application/conductor에 맡긴다.**

가능 방식:

```text
side channel
ETCD
Redis-like central service
```

vLLM은 자체 ZMQ handshake를 사용한다.

---

# Part B. NIXL transfer lifecycle

## 8. 일반적인 초기화

```text
1. create agent
2. initialize backend(s)
3. allocate application memory
4. register memory with agent/backend(s)
5. obtain local agent metadata
6. exchange metadata out of band
7. load remote agent metadata
8. optional proactive connection
```

여기까지는 **control/setup path**다.

---

## 9. 실제 transfer

initiator는:

```text
operation = READ or WRITE
local descriptors
remote descriptors
remote agent
optional notification
```

을 사용해 transfer request/handle을 만든다.

NIXL API는 async handle을 반환하고 application이 progress/status를 확인한다.

vLLM에서는 optimized prepared-list API를 사용해 block descriptor selection overhead를 줄인다.

---

## 10. READ / WRITE 의미

### READ

initiator local memory가 destination.

```text
Initiator local <- Remote memory
```

P/D NIXL Pull:

```text
D initiates READ
D KV <- P KV
```

### WRITE

initiator local memory가 source.

```text
Initiator local -> Remote memory
```

P/D NIXL Push:

```text
P initiates WRITE
P KV -> D KV
```

---

## 11. Notification

one-sided operation은 remote CPU/application이 data arrival을 자동으로 알 수 없을 수 있다.

NIXL은 transfer와 연결된 notification mechanism을 제공한다.

vLLM은 이를 이용해:

- P block free 가능 여부
- D receive completion
- Push registration
- lease heartbeat

같은 lifecycle을 구현한다.

---

# Part C. NIXL과 UCX

## 12. 왜 기본 backend가 UCX인가

UCX는 HPC/AI 환경에서 여러 transport를 공통 API 아래 노출한다.

CUDA-aware UCX는 환경에 따라:

- CUDA IPC
- GPU copy
- InfiniBand/RoCE RDMA
- shared memory
- TCP

등의 lane을 사용할 수 있다.

즉:

```text
NIXL UCX backend
       |
       v
      UCX
       |
       v
actual UCX transport/lane
```

이다.

NIXL에서 `backends=["UCX"]`라고 했다고 actual physical path가 항상 RDMA라는 뜻은 아니다.

---

## 13. 같은 host의 GPU-to-GPU

가능한 개념 경로:

```text
P VRAM
 -> NIXL UCX registered memory
 -> UCX detects same-host CUDA peer
 -> CUDA IPC / peer-copy capable path
 -> D VRAM
```

실제 lane은 UCX config/topology에 따라 달라진다.

확인:

```bash
ucx_info -d
UCX_LOG_LEVEL=info/debug
UCX_TLS=...
UCX_NET_DEVICES=...
```

---

## 14. cross-node GPU-to-GPU

가능한 경로:

```text
P VRAM
 -> UCX memory registration
 -> GPUDirect RDMA capable lane
 -> HCA
 -> IB/RoCE
 -> remote HCA
 -> D VRAM
```

조건이 맞지 않으면 host staging 또는 다른 UCX lane이 선택될 수 있으므로 physical proof가 필요하다.

---

# Part D. NIXL과 vLLM에서 특히 중요한 descriptor 구조

## 15. KV block은 descriptor index가 된다

vLLM은 model 전체 KV cache를 처음부터 NIXL에 등록하고 request마다 필요한 block ID만 descriptor index로 선택한다.

개념:

```text
layer0 block0 -> descriptor 0
layer0 block1 -> descriptor 1
...
layer1 block0 -> descriptor N
...
```

실제 v0.27.1 NIXL connector는:

- layer/region
- block index
- HMA group
- Mamba state sub-region
- TP split/replicate

때문에 더 복잡한 descriptor map을 만든다.

---

## 16. Prepared descriptor list/handle

vLLM에서는 매 request마다 raw descriptor를 처음부터 reconstruct하는 비용을 줄이기 위해 local/remote transfer descriptor list를 미리 준비한다.

이후 request block IDs를 descriptor indices로 바꿔:

```python
make_prepped_xfer(
    "READ" or "WRITE",
    local_prepped_handle,
    local_desc_indices,
    remote_prepped_handle,
    remote_desc_indices,
)
```

형태로 transaction을 만든다.

이 구조 때문에 **block size가 너무 작아 descriptor count가 폭증하면 actual wire bandwidth 외의 CPU/submission latency도 커질 수 있다.**

---

# Part E. Heterogeneous TP

## 17. 단순 rank_i ↔ rank_i가 아닌 경우

예:

```text
P TP=2
D TP=4
```

Full Attention KV가 head-sharded라면:

```text
P rank0 KV
 -> D rank0 slice
 -> D rank1 slice

P rank1 KV
 -> D rank2 slice
 -> D rank3 slice
```

같은 mapping이 필요할 수 있다.

NIXL connector는 `TPMapping`을 계산해 source rank set과 descriptor slice를 만든다.

---

## 18. SPLIT vs REPLICATE

### Full Attention / GQA 계열

KV head가 TP에 따라 shard되면 `SPLIT`.

### MLA

latent KV가 rank별 동일하게 replicated되는 경우 `REPLICATE`.

NIXL connector는 region별 MLA 여부를 기록하고 descriptor split logic을 달리한다.

---

# Part F. Hybrid SSM / Mamba / GDN

## 19. 왜 일반 Attention보다 어렵나

KV cache라는 이름 아래 실제 memory semantics가 다르다.

```text
Attention
 -> K/V token pages

Mamba/GDN
 -> convolution state
 -> temporal SSM state
```

vLLM NIXL connector는 Mamba state를 별도 descriptor sub-region으로 분해한다.

### DS layout 요구

현재 path에서는 conv state가 contiguous sub-projection으로 transfer 가능하도록 특정 dimension-first layout을 요구한다.

따라서 Kimi Linear/KDA 계열처럼 hybrid state가 있는 model에서는 **NIXL support matrix와 actual model cache spec**을 함께 봐야 한다.

---

# Part G. Lease와 lifetime

## 20. 왜 remote memory lifetime protocol이 필요한가

one-sided READ에서 P application은 D가 언제 read를 끝냈는지 모르면 P block을 free할 수 없다.

vLLM NIXL connector는 producer block에 lease를 둔다.

기본:

```text
kv_lease_duration = 30s
```

D가 queue에서 오래 기다리면 heartbeat를 보내 lease를 연장한다.

read completion notification이 오면 P가 block을 해제할 수 있다.

heterogeneous TP에서는 한 P rank의 block을 여러 D rank가 읽을 수 있어 **모든 consumer notification count**를 기다린다.

---

# Part H. Compatibility handshake

## 21. Compatibility hash

vLLM NIXL connector는 remote agent metadata를 받아들이기 전에 compatibility hash를 검사한다.

목적:

- 서로 다른 vLLM/config
- model/dtype/layout
- attention backend
- KV geometry

같은 incompatibility로 raw bytes를 잘못 꽂는 것을 방지한다.

실패 시:

```text
NIXL compatibility hash mismatch
```

가 발생한다.

초기 production에서는 이 check를 끄지 않는 것을 원칙으로 한다.

---

# Part I. NIXL version policy

## 22. vLLM v0.27.1은 1.3.1을 pin

따라서 production baseline:

```text
vLLM v0.27.1
NIXL 1.3.1
```

이다.

### upstream 1.4.0

1.4.0에는:

- compressed/stride descriptors
- `make_prepped_xfer` overhead 개선
- tracing/NVTX
- telemetry 개선
- UCX 1.22.x 관련 개선
- LIBFABRIC correctness fixes

등이 들어왔다.

매력적인 변경이지만 **vLLM pin보다 앞서므로 별도 compatibility test 없이 production image에 override하지 않는다.**

---

# Part J. 이 디렉터리 읽는 순서

1. 이 문서 — NIXL core mental model
2. [vLLM Integration](vllm-integration.md)
3. [Backends and Actual Data Paths](backends-data-path.md)
4. [Air-gapped Source Build](source-build-airgap.md)
5. [Debugging and Validation](debugging-validation.md)

---

# Part K. Source map

NIXL `v1.3.1`:

```text
src/
├── api/
│   ├── cpp/
│   ├── python/
│   └── gpu/
├── core/
├── plugins/
│   ├── ucx/
│   ├── libfabric/
│   ├── cuda_gds/
│   ├── gds_mt/
│   ├── mooncake/
│   ├── posix/
│   ├── obj/
│   ├── hf3fs/
│   └── ...
└── bindings/
```

vLLM `v0.27.1`:

```text
vllm/distributed/kv_transfer/kv_connector/v1/nixl/
├── connector.py
├── base_scheduler.py
├── base_worker.py
├── pull_scheduler.py
├── pull_worker.py
├── push_scheduler.py
├── push_worker.py
├── metadata.py
├── tp_mapping.py
├── stats.py
└── utils.py
```

---

## References

- NIXL v1.3.1 architecture: https://github.com/ai-dynamo/nixl/blob/v1.3.1/docs/nixl.md
- NIXL v1.3.1 source: https://github.com/ai-dynamo/nixl/tree/v1.3.1
- vLLM v0.27.1 NIXL connector: https://github.com/vllm-project/vllm/tree/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/nixl
- vLLM requirement pin: https://github.com/vllm-project/vllm/blob/v0.27.1/requirements/kv_connectors.txt
