# Mooncake vs NIXL: Backend Selection and Architecture Comparison

기준: vLLM v0.27.1, Mooncake Transfer Engine 0.3.10.post2, NIXL 1.3.1 runtime pin / 1.4.0 upstream reference

이 문서는 "무엇이 더 좋다"를 단정하기보다 **어느 계층에서 어떤 선택이 일어나고, 우리 P/D topology에서 무엇을 검증해야 하는지**를 정리한다.

---

## 1. 가장 중요한 결론

Mooncake와 NIXL은 동일한 계층의 제품이 아니다.

```text
vLLM MooncakeConnector
   -> Mooncake Transfer Engine
      -> Mooncake Transport(rdma/nvlink/nvlink_intra/tcp)

vLLM NixlPull/PushConnector
   -> NIXL Agent
      -> NIXL Backend Plugin(UCX/LIBFABRIC/GDS/...)
         -> UCX transport / libfabric provider / cuFile ...
```

NIXL 자체에는 `MOONCAKE` plugin도 존재한다. 이것은 **vLLM MooncakeConnector를 사용한다는 뜻이 아니다.**

```text
vLLM NixlConnector -> NIXL -> MOONCAKE plugin
```

과

```text
vLLM MooncakeConnector -> Mooncake Transfer Engine
```

은 완전히 다른 integration path다.

---

## 2. Data ownership 관점

### MooncakeConnector

control flow는 Decoder가 Prefill에 요청하지만, data operation은 producer가 수행한다.

```text
D: "내 destination block 주소는 이것이다"
        |
        v
P: src/dst pointer list 계산
        |
        v
P: batch_transfer_sync_write(remote_session, src, dst, len)
        |
        v
D VRAM
```

따라서 **P-push / WRITE**라고 보는 것이 정확하다.

### NIXL Pull

```text
D: remote P descriptor 확보
D: make_prepped_xfer("READ")
D: transfer(handle)
D: completion polling
```

P는 registered memory를 보유한 채 passive target이 된다.

### NIXL Push

```text
D: destination registration 정보를 P에 notification
P: P block 완료 + D registration match
P: make_prepped_xfer("WRITE")
P: transfer(handle)
D: completion notif
```

Mooncake와 더 비슷한 data ownership이다.

---

## 3. Control plane 비교

### Mooncake

vLLM connector가 Mooncake Transfer Engine의 peer metadata만으로 모든 orchestration을 해결하지 않는다.

vLLM은 별도 control plane을 둔다.

```text
Prefill TP/PP workers
   -> local ZMQ addresses 등록
   -> MooncakeBootstrapServer HTTP /register

Decoder
   -> remote_bootstrap_addr /query
   -> remote_engine_id -> TP/PP worker ZMQ address map 확보
   -> ZMQ MooncakeXferMetadata 전송
```

이후 P가 Mooncake Transfer Engine을 호출한다.

### NIXL

NIXL upstream 자체는 metadata exchange 방식을 application/conductor에 맡긴다.

vLLM은 이를 ZMQ side channel로 구현한다.

```text
NIXL worker
 -> local registered memory + NIXL agent metadata 생성
 -> scheduler side handshake listener가 GET_META 응답

remote worker
 -> ZMQ GET_META
 -> compatibility hash 검증
 -> NIXL agent metadata decode
 -> add_remote_agent
```

NIXL 자체는 ETCD 같은 centralized metadata 방법도 제공하지만 vLLM NixlConnector의 P/D path가 ETCD를 사용하는 것은 아니다.

---

## 4. Backend/transport 선택 비교

### Mooncake 0.3.10.post2

실제 설치 transport 선택은 `transfer_engine_impl.cpp`에서 일어난다.

NVLink transport가 build되어 있을 때:

```text
MC_INTRANODE_NVLINK exists
  -> nvlink_intra
else if MC_FORCE_MNNVL exists OR HCA list empty
  -> nvlink
else
  -> rdma
```

`USE_MNNVL` / `USE_INTRA_NVLINK`를 build하지 않은 branch에서는:

```text
HCA exists && MC_FORCE_TCP absent
  OR MC_FORCE_HCA exists
  -> rdma
else
  -> tcp
```

즉 compile profile이 runtime fallback semantics까지 바꾼다.

### NIXL

vLLM NIXL worker는 기본값:

```python
backends = ["UCX"]
```

을 사용한다.

다른 backend를 사용하려면:

```json
{
  "kv_connector_extra_config": {
    "backends": ["LIBFABRIC"]
  }
}
```

처럼 선택한다.

NIXL Agent는 동일 memory를 여러 backend에 등록할 수 있고, local/remote descriptor가 공유하는 backend 중 가능한 경로를 선택하는 abstraction을 제공한다.

즉 Mooncake의 transport auto-discovery와 NIXL plugin selection은 구조가 다르다.

---

## 5. Same-node NVIDIA GPU path

### Mooncake `nvlink_intra`

목표가 명확하다.

```text
P torch KV tensor
 -> CUDA allocation base 추적
 -> cudaIpcGetMemHandle
 -> peer process cudaIpcOpenMemHandle
 -> destination offset 계산
 -> cudaMemcpyDefault
 -> CUDA P2P
 -> NVLink/NVSwitch/PCIe peer path
```

장점:

- same-node P/D cell 목적과 transport semantic이 직접 맞는다.
- runtime log에서 `nvlink_intra`를 명확히 확인할 수 있다.

주의:

- `cudaMemcpyDefault`가 있다는 것만으로 physical NVLink를 증명하지 않는다.
- `nvidia-smi topo -m`, peer-access, NVLink counters로 실제 topology를 확인한다.

### NIXL + UCX

NIXL의 same-node GPU path는 보통 UCX의 CUDA support가 담당한다.

가능 경로:

```text
CUDA memory descriptor
 -> NIXL UCX backend
 -> UCX CUDA IPC / cuda_copy / other selected lane
 -> peer GPU memory
```

즉 `NIXL`이라는 이름 자체가 NVLink transport는 아니다.

UCX가 어떤 transport lane을 선택했는지 `UCX_LOG_LEVEL`, `UCX_TLS`, topology/UCX info로 검증해야 한다.

NixlPushConnector는 same-node/non-RDMA 환경에서 producer WRITE 방식이 더 적합할 수 있으나, 실제 v0.27.1 환경에서 Pull 대비 성능을 벤치마크해야 한다.

---

## 6. Cross-node NVIDIA GPU + RDMA path

### Mooncake RDMA

```text
P/D GPU memory
 -> Mooncake batch memory registration
 -> RDMA transport MR registration
 -> HCA lkey/rkey
 -> verbs QP/endpoint
 -> GPUDirect RDMA if platform permits
 -> remote GPU memory
```

Mooncake는 HCA discovery 결과를 transport selection에도 사용한다.

### NIXL UCX

```text
P/D GPU memory
 -> NIXL VRAM descriptor
 -> NIXL UCX backend registration
 -> UCX memory handle / endpoint
 -> CUDA-aware UCX
 -> rc/dc/other lane over IB/RoCE
 -> GPUDirect RDMA when available
```

NIXL에서는 NCCL env가 아니라 UCX env를 봐야 한다.

예:

```bash
UCX_TLS=rc,cuda_copy,cuda_ipc
UCX_NET_DEVICES=mlx5_0:1
```

실제 값은 환경 topology에 맞춰 검증한다.

---

## 7. Heterogeneous TP와 KV layout

P/D에서 TP가 다르면 단순 rank-to-rank copy가 아니다.

예:

```text
P TP=2
D TP=4
```

head-sharded KV라면 P rank 하나의 region을 두 D rank가 나눠 받아야 할 수 있다.

### Mooncake

v0.27.1 connector 내부에서:

- `TransferTopology.handshake_target_ranks()`
- `_validate_asymmetric_region_lengths()`
- `_compute_sender_transfer_plan()`
- region offset/length 계산

으로 sender가 어느 slice를 어느 D rank에 write할지 결정한다.

MLA처럼 producer cache가 replicated된 경우에는 중복 write를 줄이는 별도 branch가 있다.

### NIXL

NIXL connector는 `TPMapping`을 계산하고 descriptor를 split/replicate한다.

- Full attention: head slice SPLIT
- MLA: REPLICATE
- Mamba/GDN state: 별도 sub-region
- block-size ratio: physical descriptor 재계산

NIXL의 descriptor abstraction이 이 부분에서 특히 강하다.

---

## 8. Hybrid KV Cache / Mamba/GDN

최신 vLLM connector를 평가할 때 Dense/standard attention만 보면 안 된다.

### Mooncake v0.27.1

- `SupportsHMA`
- Sliding Window group clipping
- Mamba/GDN special handling
- P side prompt last token truncate / D recompute
- KV group identity 유지

### NIXL v0.27.1

- `SupportsHMA`
- Mamba conv/SSM state를 여러 descriptor region으로 분해
- DS conv state layout 요구
- attention/SSM group별 descriptor semantics 분리
- heterogeneous TP mapping

KDA/MLA 같은 hybrid model을 다룰 때는 NIXL의 이 계층을 반드시 이해해야 한다.

---

## 9. Failure semantics 비교

### Mooncake

대표 failure point:

- bootstrap `/query` 실패
- stale engine_id
- ZMQ timeout
- P ready timeout
- registered layer/region mismatch
- TP region length mismatch
- TransferEngine return code != 0
- sender timeout 후 P block expiry

Mooncake successful transfer metric은 P worker에 기록되는 비대칭이 있다. P가 WRITE 주체이기 때문이다.

### NIXL

대표 failure point:

- compatibility hash mismatch
- handshake timeout
- remote agent registration 실패
- backend plugin 미존재
- memory registration 실패
- prepared descriptor mismatch
- `make_prepped_xfer` 실패
- async transfer 실패
- notification 실패
- lease expiry

v0.27.1은 invalid block ID를 Scheduler로 돌려보낼 수 있어 recompute policy와 연결된다.

---

## 10. Packaging 관점

### Mooncake

우리 경험에서 가장 큰 함정은 **공식 wheel의 feature set이 runtime 기대와 다를 수 있다는 것**이었다.

`USE_CUDA=ON`과 `nvlink_intra`가 포함됐다는 것은 같은 말이 아니다.

`USE_MNNVL`, `USE_INTRA_NVLINK` compile option을 켜야 해당 class가 `MultiTransport::installTransport()`에 존재한다.

따라서 source build가 현실적인 선택이 된다.

### NIXL

NIXL은 plugin architecture라 packaging 확인 포인트가 다르다.

- `nixl` Python meta package
- CUDA major별 native package (`nixl-cu12`, `nixl-cu13` 계열)
- UCX shared libraries/plugin
- 선택한 NIXL backend `.so`

vLLM upstream은 NIXL을 exact pin하고 Dockerfile에서 KV connector dependency 설치를 optional build arg로 둔다.

NIXL을 source build할 경우 NIXL 본체만이 아니라 **UCX build/packaging까지 하나의 closure**로 봐야 한다.

---

## 11. 우리 환경의 우선 검증 순서

### Network B

1. Mooncake `nvlink_intra` actual transfer 인증
2. NIXL `NixlPushConnector + UCX` same-node 경로 비교
3. NIXL Pull도 동일 workload로 측정
4. TTFT, transfer latency, CPU usage, descriptor count 비교

### Network A

1. NIXL UCX cross-node RDMA
2. Mooncake RDMA
3. 필요 시 NIXL LIBFABRIC 별도 평가
4. heterogeneous TP / multi-node P/D 확장

이는 제품 우열이 아니라 현재 플랫폼 topology와 upstream maturity를 기준으로 한 시험 순서다.

---

## 12. Benchmark에서 반드시 분리할 시간

P/D latency를 하나의 TTFT 숫자로만 보면 원인을 못 찾는다.

```text
T_total handoff
 = Router P dispatch
 + Prefill compute
 + P response/orchestration
 + D request admission
 + handshake/bootstrap
 + descriptor construction
 + data transfer
 + completion notification
 + D scheduling gap
 + first decode compute
```

특히 NIXL은 실제 data transfer가 빠른데 descriptor construction/step alignment가 병목일 수 있다.

Mooncake도 ZMQ bootstrap/ready wait와 actual Transfer Engine time을 분리해야 한다.

---

## 13. 선택 원칙 요약

### Mooncake가 자연스러운 경우

- node-local `nvlink_intra`를 명시적으로 통제하고 싶을 때
- Mooncake-aware router contract를 이미 보유할 때
- 자체 source build와 transport feature matrix를 관리할 수 있을 때

### NIXL이 자연스러운 경우

- UCX/RDMA 중심의 범용 transport abstraction이 필요할 때
- heterogeneous memory/backend 확장을 고려할 때
- vLLM upstream이 적극적으로 발전시키는 NIXL P/D path를 따라가고 싶을 때
- pull/push semantics를 선택하고 싶을 때

### 둘 다 공통으로 필요한 것

- connector version pin
- router contract pin
- memory registration 검증
- control/data plane 분리 관측
- real hardware path 검증
- prompt local recompute 부재 확인

---

## References

- vLLM KV connector factory: https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/factory.py
- vLLM MooncakeConnector: https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
- vLLM NIXL connector: https://github.com/vllm-project/vllm/tree/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/nixl
- Mooncake Transfer Engine selection: https://github.com/kvcache-ai/Mooncake/blob/v0.3.10.post2/mooncake-transfer-engine/src/transfer_engine_impl.cpp
- Mooncake MultiTransport: https://github.com/kvcache-ai/Mooncake/blob/v0.3.10.post2/mooncake-transfer-engine/src/multi_transport.cpp
- NIXL architecture: https://github.com/ai-dynamo/nixl/blob/v1.4.0/docs/nixl.md
