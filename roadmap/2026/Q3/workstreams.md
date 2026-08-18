# 2026 Q3 Workstreams

세부 이슈를 전부 로드맵에 올리지 않고, Q3는 아래 6개 Program 단위로 추적한다.

상태 표기:
- `ACTIVE` — 현재 직접 진행 중
- `NEXT` — 선행 과제 완료 후 바로 진행
- `DESIGN` — 아키텍처/검증 단계
- `ONGOING` — 운영과 함께 상시 수행

---

## 1. Serving Foundation

**상태: ACTIVE**

범위:
- vLLM Production Stack custom baseline
- 0.1.8 → 0.1.12+ semantic migration
- LiteLLM
- Global LMRouter / model discovery
- Prometheus / serving health contract

핵심 개념:
- upstream-first
- Helm overlay / profile-driven deployment
- API compatibility contract
- routing responsibility separation
- engine-aware health

Q3 Gate:
1. 0.1.8 baseline regression 검증
2. P/D additive extension이 기존 integrated deployment를 변경하지 않는지 검증
3. LiteLLM / Global Router / P/D Cell Router의 책임 경계 확정
4. 0.1.12+ 이관 시 재사용할 model identity / profile / topology / metrics contract 확정

운영 자동화 포인트:
- endpoint 등록/제외
- health 기반 route drain
- configuration drift detection
- rollout 검증

상세:
- [`vllm-stack/2026-08-18-진행계획.md`](../../../vllm-stack/2026-08-18-%EC%A7%84%ED%96%89%EA%B3%84%ED%9A%8D.md)

---

## 2. Distributed Inference

**상태: ACTIVE / Q3 최우선**

범위:
- Node-local P/D Disaggregation
- Kimi-K3 Multi-node
- 망A IB / GPUDirect RDMA
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
- 망A H200 환경에서 vLLM NCCL 로그를 통해 IB/GDRDMA channel 확인
- NVIDIA Network Operator 기반 RDMA resource 사용
- 망B는 IB가 없어 Node-local P/D가 우선 topology
- 0.1.8에서는 한 Pod = 한 P/D Cell 구조로 시작

Q3 Gate:
1. P1:D1 + orchestrated LMRouter + KV transfer E2E
2. Variable P:D
3. Cell replica scale-out
4. 모든 P/D engine metric 수집
5. Router / Prefill / Decode failure recovery
6. Kimi-K3 H200 multi-node 재현 가능한 배포
7. Multi-node soak / failure / performance benchmark
8. Multi-node P/D feasibility 비교

운영 자동화 포인트:
- serving group 단위 health 판단
- 부분 engine failure 시 route ejection
- node-set lifecycle / recovery
- P:D capacity imbalance 감지
- multi-node admission 전에 RDMA/NCCL qualification

상세:
- [`pd-disaggregation/`](../../../pd-disaggregation/)

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

**상태: DESIGN**

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
- LMCache
- NIXL
- Mooncake / MooncakeStore

설계 원칙:
- `Offloading`, `P/D Transfer`, `Shared KV Reuse`를 하나의 기능으로 묶지 않는다.
- 망B와 망A의 network capability 차이를 cache architecture에 반영한다.
- 중앙 remote cache를 기본값으로 가정하지 않는다.

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

---

## 5. Stateful Serving

**상태: DESIGN**

범위:
- `/v1/responses` 계열 server-side state
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

Q3 Gate:
1. canonical conversation / response state schema
2. storage backend abstraction
3. multi-replica 환경의 previous-response continuation 설계
4. session routing과 cache locality 연결 방식 정의
5. retention / compaction / failure semantics 정리

운영 자동화 포인트:
- state backend health / capacity
- orphan session cleanup
- retention / compaction policy
- stateful endpoint failover

---

## 6. Operations Control — MOC

**상태: DESIGN / 구현 후순위**

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
