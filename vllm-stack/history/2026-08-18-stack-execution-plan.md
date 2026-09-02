# vLLM Stack 진행 계획

작성일: 2026-08-18  
최종 업데이트: 2026-09-03 KST

## 1. 현재 결론

- custom vLLM Production Stack 0.1.8은 현재 운영 기준선이다.
- PR #1 baseline은 `main`에 merge됐다.
- 현재 운영 우선순위에 따라 0.1.12+ 이관보다 Node-local P/D Cell을 먼저 구현한다.
- 단기 P/D는 기존 `servingEngineSpec.modelSpec[]`을 수정하지 않는 top-level `pdCellSpec.models[]` additive extension이다.
- Prefill/Decode는 별도 모델이 아니라 한 served model의 실행 phase다.
- Cell 하나는 Router + Prefill×N + Decode×M을 담은 Pod 하나다.
- PR #2 이전 커밋에서 P1:D1 배포와 LiteLLM/Anthropic 경로는 성공했다.
- PR #4는 실제 운영에 필요한 values/container/router 계약을 보완했고 Qwen3.6-27B로 확대 검증 중이다.
- Mooncake `0.3.10.post2` source-build overlay는 PR #5로 merge되어 폐쇄망 image build 경로를 확보했다. 현재 blocker는 **실제 `nvlink_intra` data path 인증과 vLLM 0.28.0용 Mooncake 0.3.12 계열 재빌드/호환성 검증**이다.
- vLLM `0.27.1-cu129`은 비교 기준선으로 유지하고, **`v0.28.0-cu129`을 차기 validation candidate**로 올린다.
- custom Production Stack은 특정 0.1.8 snapshot에 기능을 계속 누적하지 않고 **upstream fork + 최소 overlay**로 운영한다.
- Node-local P/D가 안정화되는 동안에도 upstream sync와 vLLM 0.28 runtime validation을 병행하고, 이후 independent P/D pool / multi-node / native multiprocess 트랙으로 확장한다.

상세 문서:

- [P/D 문서 인덱스](../pd-disaggregation/README.md)
- [Node-local P/D Cell 현재 구현 계획](../pd-disaggregation/node-local-pd-cell-vllm-stack-0.1.8.md)
- [Mooncake 0.3.10-post2 폐쇄망 Source Build](../pd-disaggregation/history/2026-08-24-mooncake-0.3.10-post2-offline-build.md)
- [Model Serving Validation Contract](../model-serving-validation.md)
- [vLLM 0.28.0 Migration & KV Connector Compatibility](../migrations/vllm-0.28.md)
- [API Routing Contract](../api-routing-contract.md)
- [Serving Optimization Mental Model](../../study/inference-serving-optimization/README.md)

---

## 2. 진행 현황

| Track | 상태 | 완료 조건 |
|---|---|---|
| 0.1.8 baseline | 완료 | PR #1 merge |
| P/D renderer 1차 | 구현 완료 / Draft | PR #2의 Helm/template/test 계약 |
| P1:D1 최초 runtime | 완료 | `36e45f0c277d4206ce233c8057a383e387c616c1` 배포, LiteLLM, Anthropic 호출 |
| PR #4 template audit | 구현 완료 / runtime 테스트 중 | Qwen3.6-27B P1:D1 재통과 |
| Variable P:D | 대기 | P2:D1 → P2:D2 → P1:D3 |
| Cell lifecycle | 대기 | replica 0↔1, scheduling, restart/ejection/rejoin |
| Mooncake 0.3.10 image | 완료 / runtime Gate 별도 | PR #5 merge, 폐쇄망 source-build overlay 확보 |
| Mooncake 0.3.12 image | 신규 validation | vLLM 0.28.0-cu129 ABI에서 `nvlink_intra` source build와 transfer bench |
| vLLM 0.28.0 | 신규 validation | integrated baseline → P/D connector → scheduler/graph → SD → soak |
| Observability | 후순위 | 모든 engine/router port와 namespace scrape 검증 |
| Production Stack upstream sync | 병행 | fork + thin overlay 원칙으로 current upstream 변화 흡수 |

---

## 3. 실제 작업 순서

```text
1. vLLM 0.27.1 + PR #4 Qwen3.6-27B P1:D1 재현 기준선 고정
        ↓
2. vLLM 0.28.0-cu129 integrated correctness baseline
        ↓
3. Mooncake 0.3.12.post1 source-built nvlink_intra overlay image
        ↓
4. nvlink_intra standalone + P1:D1 actual transfer validation
        ↓
5. P/D role별 scheduler/graph baseline
        ↓
6. P2:D1 → P2:D2 → P1:D3
        ↓
7. Decode SD K × MBT × max_num_seqs joint sweep
        ↓
8. replica 0↔1 / failure recovery / Prometheus scrape
        ↓
9. PR #2/#4 정리와 upstream fork sync
        ↓
10. independent P/D pool / multi-node / Native MP-LWS
```

Mooncake image source build와 PR #4 Helm audit은 변경 책임이 다르므로 별도 PR로 유지한다. vLLM version bump도 chart topology 변경과 한 번에 묶지 않고 runtime validation revision으로 분리한다.

---

## 4. 단기 P/D API

```yaml
pdCellSpec:
  enabled: true

  router:
    type: lmstack
    repository: registry.example/lmstack-router
    tag: validated-version

  models:
    - name: qwen36-pd-p1d1
      servedModelNames:
        - Qwen3.6-27B-PD
        - standard
      repository: registry.example/vllm-openai
      tag: validated-version
      replicaCount: 1

      kvTransfer:
        connector: MooncakeConnector
        config:
          kv_buffer_device: cuda
          kv_load_failure_policy: fail
          kv_connector_extra_config:
            mooncake_protocol: nvlink_intra

      prefill:
        count: 1
        requestGPU: 4
        profile: /profiles/qwen36/pd-prefill-tp4.yaml

      decode:
        count: 1
        requestGPU: 4
        profile: /profiles/qwen36/pd-decode-tp4.yaml
```

의미:

```text
models[].name       = Kubernetes topology identity
servedModelNames[]  = primary model ID + aliases
replicaCount        = Cell Pod 수
prefill.count       = Cell 내부 Prefill engine 수
decode.count        = Cell 내부 Decode engine 수
requestGPU          = engine container의 GPU 예약량
```

장기적으로는 하나의 `modelSpec` 아래 `deploymentMode: disaggregated`로 이동하지만, 0.1.8에서는 기존 integrated/Ray renderer regression을 피하기 위해 root를 분리한다.

---

## 5. Runtime 구조

```text
LiteLLM / Global Router
        ↓
model-specific Service
        ↓
Cell Router :8000
   ├─ Prefill :8101+
   └─ Decode  :8201+
```

Cell Pod 전체 GPU request:

```text
cellGpu =
  prefill.count × prefill.requestGPU
  + decode.count × decode.requestGPU
```

예:

| Topology | Engine GPU | Cell GPU |
|---|---:|---:|
| P1:D1 | P4 + D4 | 8 |
| P2:D1 | P2×2 + D4 | 8 |
| P2:D2 | P2×2 + D2×2 | 8 |
| P1:D3 | P2 + D2×3 | 8 |

Pod는 여러 Node로 분할되지 않으므로 Node-locality가 구조적으로 보장된다. Node 이름 pinning은 기본값이 아니며 scheduler/affinity로 배치한다.

---

## 6. Router 계약

### Global Router

- 모델 endpoint discovery
- Cell Service/Router 선택
- Cell 내부 P/D topology를 알지 않음

### Cell Router

- Prefill/Decode phase orchestration
- localhost static backend membership
- 다른 Cell로 KV가 넘어가지 않도록 failure domain 고정

지원 type:

| Type | 핵심 args |
|---|---|
| `lmstack` | static discovery, static model/alias/role label, `disaggregated_prefill_orchestrated` |
| `vllm` | `--vllm-pd-disaggregation`, 반복형 `--prefill`/`--decode`, connector/policy |
| `custom` | 사용자가 제공한 command/args |

image 계열마다 CLI를 따로 검증하고 tag/digest를 pin한다.

---

## 7. KV connector와 transport

`kvTransfer`는 각 model block이 직접 소유한다. 같은 chart에서도 model/profile/topology별로 NIXL과 Mooncake를 다르게 선택할 수 있다.

Failure policy:

- production 기본: `fail`
- `recompute`: 제한된 fallback 실험에만 사용
- 장기-context에서는 Decode가 Prefill을 재수행하는 비용이 너무 커 기본값으로 사용하지 않음

망별 기준:

| 범위 | 우선 후보 | 비고 |
|---|---|---|
| Network B H200 same-node | Mooncake `nvlink_intra` 또는 검증된 NIXL/UCX local path | 0.28에서는 Mooncake 0.3.12 source build 필요 |
| Network A cross-node IB | NIXL/UCX RDMA 또는 Mooncake `rdma` | 실제 HCA/GDRDMA 검증 필요 |
| cross-node without RDMA | TCP 계열 | 장기-context P/D 성능 병목 가능 |

Mooncake:

- `nvlink` = MNNVL
- `nvlink_intra` = 동일 Node NVLink/NVSwitch
- 둘 다 compile-time option이며 별도 flag가 필요
- 0.3.12.post1 official x86 wheel release build에도 `USE_INTRA_NVLINK=ON`이 없으므로 same-node direct path는 source-build 대상
- requested protocol은 vLLM log, 실제 installed transport는 Mooncake log로 확인

---

## 8. Qwen3.6-27B 검증 Matrix

### 순서

1. P1:D1
2. P2:D1
3. P2:D2
4. P1:D3
5. replica `0 → 1 → 0 → 1`
6. 비대칭 GPU/TP

### API

- `/v1/models`
- chat completions non-streaming
- chat completions streaming delta
- Anthropic wrapper
- long context
- cancellation
- timeout
- model alias

### Data path

- Router가 올바른 P와 D 선택
- bootstrap/side-channel handshake
- producer KV 생성
- GPU memory까지 KV transfer
- consumer load 성공
- 실패 시 `fail`이 transfer 오류를 숨기지 않음

### Scheduler / CUDA Graph / SD

Prefill과 Decode를 같은 profile로 튜닝하지 않는다.

- Prefill: 큰 MBT 후보, chunked prefill, prompt throughput/TTFT
- Decode: ITL/TPOT, KV/recurrent state capacity, `FULL_DECODE_ONLY` 후보
- SD 사용 시 `num_speculative_tokens × max_num_batched_tokens × max_num_seqs` joint sweep
- hybrid GDN/Mamba/KDA model은 state block/alignment invariant를 MBT 하한과 함께 확인

### 성능

- TTFT p50/p95/p99
- ITL p50/p95/p99
- P/D queue
- P/D GPU utilization
- Decode KV usage
- transfer bytes/latency/error

---

## 9. Observability와 failure

Pod 하나에 여러 vLLM process가 있으므로 `instance=PodIP`만으로는 시계열이 충돌한다.

필수 identity:

```text
pd_deployment
pd_cell
pd_role
container
node
PodIP:port
```

초기 장애 정책:

```text
Router Ready
AND all Prefill Ready
AND all Decode Ready
→ Cell Ready
```

검증 항목:

- Prefill kill
- Decode kill
- Router kill
- Pod delete
- endpoint 제거 시간
- 다른 Cell의 계속 서비스 여부
- model reload 후 자동 재가입

Prometheus namespace/scrape rule 변경은 data path와 connector 검증 뒤에 진행한다.

---

## 10. Merge Gate

P/D PR은 다음이 충족되기 전 merge하지 않는다.

- 기존 integrated/Ray renderer regression 없음
- PR #4 HEAD의 P1:D1 runtime 성공
- variable P:D 최소 1종 성공
- Cell replica/lifecycle 성공
- LiteLLM/Global Router/model discovery 성공
- engine별 metrics 수집
- failure ejection/rejoin 성공
- connector와 image provenance 고정
- same-node 실제 transport가 `nvlink_intra`임을 로그와 bench로 증명

장기 목표:

```text
0.1.8 pdCellSpec
  → 0.1.12+ disaggregated serving
  → independent P/D pools
  → fabric/multi-node P/D
  → Native Multiprocess / LWS
```
