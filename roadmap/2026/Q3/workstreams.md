# 2026 Q3 Workstreams

세부 이슈를 전부 로드맵에 올리지 않고, Q3는 아래 6개 Program 단위로 추적한다.

상태 표기:
- `ACTIVE` — 현재 직접 진행 중
- `VALIDATING` — 구현은 존재하며 실제 runtime/운영 Gate 검증 중
- `NEXT` — 선행 과제 완료 후 바로 진행
- `DESIGN` — 아키텍처/검증 단계
- `ONGOING` — 운영과 함께 상시 수행

---

## 1. Serving Foundation

**상태: ACTIVE**

범위:
- vLLM 0.27.x comparison baseline → 0.28.0 validation
- vLLM Production Stack upstream fork + thin overlay
- LiteLLM
- Agentic API Responses lane
- Global LMRouter / model discovery
- Prometheus / serving health contract

핵심 개념:
- upstream-first / fork sync
- Helm overlay / profile-driven deployment
- API semantics별 path routing
- routing responsibility separation
- runtime version × connector ABI compatibility
- engine-aware health

Q3 Gate:
1. 0.1.8 baseline regression 검증
2. vLLM 0.28.0-cu129 integrated baseline과 주요 model regression 검증
3. P/D additive extension이 기존 integrated deployment를 변경하지 않는지 검증
4. Chat/Completions → LiteLLM, Messages → LiteLLM Anthropic, Responses → Agentic API → LMStack Router path contract 확정
5. upstream Production Stack sync/rebase workflow와 thin overlay 경계 확정
6. model identity / profile / topology / metrics contract 확정
7. base URL / auth / model discovery / chat / responses / streaming / reasoning / tool-call golden test 계약 확정

운영 자동화 포인트:
- endpoint 등록/제외
- health 기반 route drain
- configuration drift detection
- rollout 검증

상세:
- [`vllm-stack/2026-08-18-진행계획.md`](../../../vllm-stack/2026-08-18-%EC%A7%84%ED%96%89%EA%B3%84%ED%9A%8D.md)
- [`vllm-stack/model-serving-validation.md`](../../../vllm-stack/model-serving-validation.md)

---

## 2. Distributed Inference

**상태: ACTIVE / Q3 최우선**

범위:
- Node-local P/D Disaggregation
- Kimi-K3 Multi-node
- Network A IB / GPUDirect RDMA
- 향후 Multi-node P/D
- Native Multiprocess / LWS 검토

핵심 개념:
- Prefill / Decode phase separation
- P:D ratio
- KV transfer
- failure domain
- TP / PP / DP / EP / TEP
- NCCL collective
- InfiniBand / RDMA / GPUDirect RDMA
- topology-aware scheduling

현재 확정 상태:
- Network A H200 환경에서 vLLM NCCL 로그를 통해 IB/GDRDMA channel 확인
- NVIDIA Network Operator 기반 RDMA resource 사용
- Network B는 IB가 없어 Node-local P/D가 우선 topology
- 0.1.8에서는 한 Pod = 한 P/D Cell 구조로 시작
- 0.1.8 baseline PR #1은 `main`에 merge 완료
- PR #2는 기존 renderer를 건드리지 않는 `pdCellSpec.models[]` additive extension으로 구현
- PR #2 이전 커밋에서 P1:D1 배포, LiteLLM, Anthropic 경로 성공
- PR #4는 router type, model alias, 비대칭 GPU, model-local KV config와 container 상속을 보완했으며 runtime 테스트 중
- Network B Mooncake same-node 경로에는 `nvlink_intra`가 필요하다. 0.3.10 source-build overlay는 확보했으며 vLLM 0.28에서는 0.3.12.post1 source-build compatibility를 재검증한다.
- Prefill/Decode는 scheduler workload가 다르므로 MBT/graph mode를 역할별로 분리하고, Decode SD는 K×MBT×max_num_seqs joint tuning한다.

Q3 Gate:

| Gate | 상태 |
|---|---|
| PR #2 이전 커밋 P1:D1 + LiteLLM/Anthropic | 완료 |
| PR #4 HEAD Qwen3.6-27B P1:D1 | 검증 중 |
| Mooncake 0.3.10-post2 source-build overlay | 완료 |
| vLLM 0.28.0 + Mooncake 0.3.12.post1 `nvlink_intra` | 신규 validation |
| P2:D1 → P2:D2 → P1:D3 | 대기 |
| Cell replica 0↔1 / scale-out | 대기 |
| 모든 P/D engine metric 수집 | 후순위 |
| Router / Prefill / Decode failure recovery | 대기 |
| Kimi-K3 H200 multi-node 재현 가능한 배포 | 병행 과제 |
| Multi-node soak / failure / performance benchmark | 대기 |
| Multi-node P/D feasibility 비교 | 장기 |

운영 자동화 포인트:
- serving group 단위 health 판단
- 부분 engine failure 시 route ejection
- node-set lifecycle / recovery
- P:D capacity imbalance 감지
- multi-node admission 전에 RDMA/NCCL qualification

상세:
- [`vllm-stack/pd-disaggregation/`](../../../vllm-stack/pd-disaggregation/)

---

## 3. Model Enablement & Performance

**상태: ONGOING**

범위:
- Kimi-K3
- DeepSeek-V4 Flash 0731 / Pro 0813 계열
- Qwen3.8-27B 계열
- 신규 open-weight 모델 검증
- 모델 배포 오류 / kernel compatibility 분석
- long-context / MoE / speculative decoding tuning

핵심 개념:
- model architecture capability
- MoE / Expert Parallel
- MLA / KDA / GDN 계열 attention
- speculative decoding / MTP / draft-based decoding
- CUDA Graph
- quantization / KV dtype
- scheduler / chunked prefill
- TTFT / ITL / throughput trade-off

원칙:
- 모델별 일회성 실행 명령을 남기지 않는다.
- `Model Profile + Runtime Profile + Validation Result` 형태로 축적한다.
- KV capacity만으로 concurrency를 추정하지 않고 compute / HBM / collective 병목을 함께 본다.

Q3 Gate:
1. 주요 신규 모델의 지원 vLLM version / image / kernel matrix 정리
2. GPU topology별 safe deployment profile 확보
3. streaming / tool / reasoning / API contract 검증
4. long-context benchmark와 capacity envelope 기록
5. 장애 원인을 모델 문제 / vLLM regression / CUDA-kernel / topology 문제로 분류 가능하게 정리

운영 자동화 포인트:
- model capability metadata 기반 deployment validation
- unsafe topology 사전 차단
- model-specific health / performance baseline 비교
- 배포 후 golden test 자동화

상세:
- [`models/`](../../../models/)
- [`study/`](../../../study/)

---

## 4. Cache Plane

**상태: ACTIVE / VALIDATING**

범위:
- local prefix caching
- KV offloading
- P→D KV transfer
- shared remote KV
- cache-aware routing

핵심 개념:
- KV block lifecycle
- prefix reuse
- cache locality
- memory hierarchy
- transfer latency vs recompute cost
- connector compatibility

주요 구현 후보:
- vLLM native connector
- LMCache `>=0.3.9` (0.28 baseline)
- NIXL `1.3.2` (0.28 baseline)
- Mooncake `>=0.3.12` / MooncakeStore

설계 원칙:
- `Offloading`, `P/D Transfer`, `Shared KV Reuse`를 하나의 기능으로 묶지 않는다.
- Network B와 Network A의 network capability 차이를 cache architecture에 반영한다.
- 중앙 remote cache를 기본값으로 가정하지 않는다.
- Mooncake `nvlink`는 MNNVL, `nvlink_intra`는 same-node NVLink/NVSwitch로 구분한다.
- connector 이름뿐 아니라 wheel/image에 compile된 transport feature까지 compatibility axis로 관리한다.

현재 초점:

- 0.3.10.post2 기준 source-built overlay와 air-gap build skeleton 확보
- vLLM 0.28.0의 connector baseline인 Mooncake 0.3.12 / NIXL 1.3.2로 compatibility 이동
- Mooncake 0.3.12.post1 official x86 wheel에도 `USE_INTRA_NVLINK=ON`이 없음을 확인
- `USE_CUDA=ON + USE_INTRA_NVLINK=ON` source-built artifact 재검증; 필요 시 `USE_MNNVL=ON` 비교
- Helm/Router 이전에 standalone transfer bench로 GPU memory data path 검증

Q3 Gate:
1. connector × model × topology compatibility matrix
2. hit rate뿐 아니라 lookup / transfer / initialization / eviction cost 측정
3. P/D KV transfer와 shared KV reuse의 책임 분리
4. cache locality + live load + health를 조합하는 routing 기준 설계

운영 자동화 포인트:
- backend cache pressure 감지
- cache tier health / capacity 관리
- cache-aware routing policy
- connector 실패 시 recompute fallback

장기-context production에서는 `recompute`를 기본 failure policy로 사용하지 않는다. 기본은 `fail`로 두고 transfer 오류를 명시적으로 노출한다.

---

## 5. Stateful Serving

**상태: DESIGN / CANDIDATE VALIDATION**

범위:
- `/v1/responses` 계열 server-side state
- Codex Responses HTTP/SSE/WebSocket compatibility
- conversation store
- response / tool lineage
- session-aware routing

핵심 개념:
- durable semantic state
- conversation_id / response_id / previous_response_id
- session affinity
- context reconstruction
- lifecycle / retention

설계 원칙:
- conversation store와 KV cache를 같은 저장소 문제로 취급하지 않는다.
- semantic state는 engine lifecycle 밖에서 durable하게 관리한다.
- router/cache와 연결할 때는 session identity만 공유하고 storage contract는 분리한다.
- Responses wire protocol, durable state, tool execution을 서로 다른 capability로 검증한다.
- `affinity != durability`, `state hydration != model token reduction`을 기본 계약으로 둔다.

현재 확인:
- current Codex는 Responses wire protocol만 허용하며 Chat Completions-only provider에 직접 연결할 수 없다.
- vLLM 0.28.0 native Responses store는 opt-in process-local memory 구현이므로 production store로 채택하지 않는다.
- pinned LMStack Router 0.1.9는 `/v1/responses`를 model-routing 후 backend에 전달하지만 `/v1/messages`는 제공하지 않는다.
- LiteLLM은 Chat/Messages compatibility와 model registry를 담당하되 durable state owner로 사용하지 않는다.
- vLLM Agentic API를 `State Facade + PostgreSQL + tool orchestration`의 POC 우선 후보로 검증한다.
- Agentic API는 하나의 logical `LLM_API_BASE`를 가지므로 multi-model backend selection은 LMStack Router가 담당한다.
- Agentic API의 tenant/persisted-state authorization과 retention/production hardening은 채택 blocker다.

Q3 Gate:
1. Codex → Agentic API → vLLM direct HTTP/SSE/WebSocket golden test
2. Codex → Agentic API → LMStack Router → multi-model vLLM path 검증
3. canonical conversation / response state schema와 PostgreSQL storage contract
4. cross-replica/restart/rolling-update 이후 previous-response continuation
5. LiteLLM 포함/우회 path의 typed item, event, tool-call fidelity 비교
6. session routing과 cache locality 연결 방식 정의
7. retention / compaction / failure / idempotency semantics 정리
8. principal별 response/conversation object authorization 검증
9. long-context에서 client bytes, hydrated input tokens, prefix-cache 효과를 분리 측정

운영 자동화 포인트:
- state backend health / capacity
- orphan session cleanup
- retention / compaction policy
- stateful endpoint failover

---

## 6. Operations Control — MOC

**상태: CORE AVAILABLE / BUSINESS LOGIC BACKLOG**

MOC는 위 5개 Program을 대체하지 않는다. 각 Plane이 제공하는 primitive를 운영 정책으로 조율하는 cross-plane controller다.

Core Framework:

```text
State Collector
  → Normalizer / Canonical State
  → Policy Engine
  → Planner
  → Safety Guard
  → Adapter / Executor
  → Verify / Reconcile
  → Audit
```

현재 Core Framework와 다음 business capability를 구분한다.

| Capability | 상태 |
|---|---|
| Production Stack discovery/adoption | BACKLOG |
| vLLM runtime adapter | BACKLOG |
| Kubernetes Service endpoint 동적 관리 | BACKLOG |
| LiteLLM model-info 동기화 | BACKLOG |
| workload-aware placement / time·event scale | BACKLOG |
| donor/victim GPU reclaim | BACKLOG |
| TP8 admission을 위한 eviction/repacking | BACKLOG |

후속 Controller 후보:
- EndpointDrainController
- HealthSentinel
- PDGroupController
- MultiNodeLifecycleController
- GPUNodeReclaimController
- CachePolicyController
- ConversationPolicyController
- ScheduledCapacityController

Q3 원칙:
- 지금 당장 MOC 기능 구현을 위해 P/D / Kimi 납기를 늦추지 않는다.
- 대신 각 과제가 MOC가 나중에 제어할 수 있도록 **identity / health / metric / lifecycle / adapter boundary**를 남긴다.
- Core Framework를 business capability 완료로 표기하지 않는다. 상세 상태는 [`moc/README.md`](../../../moc/README.md)에서 관리한다.

운영 지능의 목표:
- 사람이 수동으로 하는 `관찰 → 판단 → drain → 이동 → 재배포 → 검증` 절차를 안전한 reconcile plan으로 코드화한다.
- 단순 autoscaler가 아니라 GPU node 단위 placement와 inference-specific health를 이해하는 운영 controller로 발전시킨다.

---

# Dependency View

```text
Serving Foundation
      │
      ├──────────────┐
      ▼              ▼
Distributed       Model Enablement
Inference            │
      │               │
      ├──────┬────────┘
      ▼      ▼
 Cache    Stateful Serving
 Plane
      └──────┬───────────────┐
             ▼               │
      Operations Control     │
            (MOC)            │
             ▲               │
             └─ Metrics / Performance Feedback
```

실제 일정은 `Distributed Inference + Model Enablement`가 가장 앞에 있지만, 장기 구조는 `Serving Foundation`의 contract를 공유해야 한다.
