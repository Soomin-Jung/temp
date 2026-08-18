# MOC — LLM E2E Operations Control Plane

업데이트: 2026-08-19 KST

## 정체성

MOC는 이전 WADM 구상과 분리된 독립적인 LLM E2E Operations Control Plane이다. 요청 data path를 대신하지 않고 Kubernetes, vLLM runtime, Router, LiteLLM, GPU topology의 actual state를 모아 안전한 운영 변경을 계획·실행·검증한다.

## 현재 구현 경계

현재 구현된 범위는 **Core Framework**다.

```text
State Collection
  → Canonical State / Normalization
  → Policy Evaluation
  → Plan Generation
  → Safety Guard
  → Adapter / Executor
  → Verify / Reconcile
  → Audit
```

Core Framework가 있다는 사실과 실제 운영 business capability가 완성됐다는 사실을 혼동하지 않는다. 아래 기능은 명시적으로 RFC/backlog에 남겨야 한다.

## Business Capability Backlog

| Capability | 현재 상태 | 완료 기준 |
|---|---|---|
| Kubernetes Service endpoint 동적 관리 | BACKLOG | readiness/health에 따라 endpoint 등록·drain·복귀를 idempotent하게 수행 |
| LiteLLM model-info 동적 동기화 | BACKLOG | served model identity, endpoint, capability 변경을 안전하게 반영하고 drift 탐지 |
| Workload-aware 동적 model placement | BACKLOG | model/runtime profile, GPU topology, 현재 부하를 함께 사용해 placement 계획 생성 |
| 시간·이벤트 기반 scale rule | BACKLOG | schedule/event, cooldown, protected workload, rollback을 포함한 reconcile |
| Donor/Victim GPU 회수·재할당 | BACKLOG | 우선순위와 disruption budget을 지키며 GPU를 회수하고 대상 workload 검증 |
| TP8 공간 확보를 위한 eviction/repacking | BACKLOG | 여러 조각 난 GPU 배치를 안전하게 재구성하고 중간 capacity 손실을 제한 |
| vLLM runtime adapter | BACKLOG | profile, replicas, topology, health, metrics를 canonical action/state로 변환 |
| Production Stack discovery/adoption | BACKLOG | 기존 배포를 재생성하지 않고 발견·adopt하며 ownership 충돌과 drift를 처리 |

## 우선 구현 순서

1. Production Stack discovery/adoption과 vLLM runtime adapter
2. Kubernetes endpoint management와 LiteLLM synchronization
3. workload-aware placement와 time/event scaling
4. donor/victim reclaim
5. eviction/repacking과 TP8 admission

앞 단계가 뒤 단계의 identity, health, ownership, rollback contract를 제공해야 한다. GPU 이동 로직부터 만들면 안전하게 drain하거나 원복할 기준이 없다.

## 필요한 공통 계약

- `served_model_name`은 string/list 입력을 모두 정규화한다.
- runtime profile의 runner 정보를 canonical `modelKind`로 변환한다.
- Pod Ready와 inference healthy를 분리한다.
- plan에는 precondition, blast radius, cooldown, rollback, post-check를 포함한다.
- 기존 배포를 adopt할 때 관찰자/소유자 모드를 구분한다.
- 모든 변경은 decision reason과 before/after state를 audit에 남긴다.

## 다른 Plane과의 경계

| Plane | 소유 책임 | MOC 역할 |
|---|---|---|
| LiteLLM | API compatibility, alias, auth/policy | model-info 동기화와 drift 조정 |
| Router | 요청별 backend 선택 | route drain/ejection policy 전달 |
| vLLM runtime | 실제 inference 실행 | replica/topology lifecycle 조정 |
| Kubernetes | resource scheduling과 workload lifecycle | 안전한 desired-state 변경 |
| Observability | metric/log/event 수집 | canonical signal과 verification에 사용 |

MOC는 위 구성요소를 대체하지 않는다. 사람이 하던 `관찰 → 판단 → drain → 이동 → 재배포 → 검증`을 재현 가능하고 감사 가능한 reconcile plan으로 바꾸는 계층이다.
