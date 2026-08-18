# 2026 Q3 Target Architecture

이 문서는 Q3 과제들이 각각 어디에 위치하고 최종적으로 어떻게 연결되는지 보여주는 상위 아키텍처 스케치다.

## 현재 운영 흐름

```text
Client / AI Service
  → Kong Gateway
  → Nginx Ingress
  → vLLM Proxy (FastAPI)
  → LiteLLM Proxy
  → model service / router
  → vLLM engine

Observability
  → Prometheus / Grafana / Loki / Alloy / DCGM
```

현재 구조는 이미 다중 모델 운영이 가능하지만, Q3부터는 단일 vLLM instance를 전제로 한 구조를 넘어 **P/D Cell, Multi-node, KV locality, Stateful Conversation**을 수용해야 한다.

---

## Q3 통합 목표 아키텍처

```mermaid
flowchart TB
    C[Clients / Agents / AI Clients]
    GW[Kong / Nginx Gateway]
    VP[vLLM Proxy / API Facade]
    LL[LiteLLM\nAPI Compatibility / Auth / Policy]

    CS[Conversation State Service\nconversation / response lineage]
    CR[Intelligent Routing Layer\nmodel / session / cache / load / health]

    subgraph INF[Inference Runtime Plane]
      SN[Single-node vLLM]
      PD[Node-local P/D Serving Group]
      MN[Multi-node vLLM\nTP / PP / DP / EP / TEP]
      MPD[Multi-node P/D Serving Group]
    end

    subgraph CACHE[KV Cache Plane]
      L0[GPU Local KV / APC]
      L1[Host / Local Offload]
      KT[P to D KV Transfer]
      L2[Shared / Remote KV]
    end

    subgraph RUNTIME[GPU / Network Runtime]
      K8S[Kubernetes / k3s]
      NWO[NVIDIA Network Operator]
      RDMA[IB / RDMA Resource]
      NCCL[NCCL / NVLink / NVSwitch / GDRDMA]
    end

    subgraph OBS[Observability & Performance Intelligence]
      PROM[Prometheus / Grafana / DCGM]
      PERF[TTFT / ITL / Queue / KV / MFU / HBM / NCCL]
    end

    MOC[MOC Operations Control Plane\nObserve → Decide → Plan → Guard → Apply → Verify]

    C --> GW --> VP --> LL
    LL --> CR
    LL <--> CS
    CS -. session identity .-> CR

    CR --> SN
    CR --> PD
    CR --> MN
    CR --> MPD

    SN <--> L0
    PD <--> KT
    MN <--> L0
    MPD <--> KT
    L0 --> L1 --> L2

    K8S --> INF
    NWO --> RDMA --> NCCL --> MN
    NCCL --> MPD

    INF --> PROM --> PERF
    CACHE --> PROM
    CR --> PROM
    CS --> PROM

    MOC -. reconcile .-> K8S
    MOC -. policy .-> CR
    MOC -. lifecycle .-> INF
    MOC -. capacity / tier policy .-> CACHE
    MOC -. operational policy .-> CS
    PERF -. state / signals .-> MOC
```

---

## 레이어별 책임

### 1. API / Compatibility — LiteLLM

주요 개념:
- OpenAI / Anthropic compatible API
- model alias / auth / policy
- streaming delta
- reasoning / tool-call translation

역할:
- 클라이언트 API 차이를 inference engine 밖에서 흡수한다.
- GPU topology나 P/D 내부 구조는 가능한 한 숨긴다.

### 2. Routing Plane

주요 개념:
- load-aware routing
- prefix / KV locality
- session affinity
- backend health / ejection
- P/D orchestration

역할:
- 요청마다 어떤 serving endpoint 또는 P/D group을 사용할지 결정한다.
- MOC처럼 deployment 자체를 생성/삭제하는 control plane과 분리한다.

### 3. Inference Runtime Plane

주요 개념:
- TP / PP / DP / EP / TEP
- Prefill / Decode disaggregation
- MoE / Expert Parallel
- chunked prefill
- speculative decoding / MTP
- distributed executor / native multiprocess / LWS

실행 형태:
- Single-node integrated vLLM
- Node-local P/D Cell
- 망A Multi-node vLLM
- 향후 Multi-node P/D Serving Group

### 4. KV Cache Plane

서로 다른 문제를 한 기능으로 취급하지 않는다.

- **GPU Local KV / APC** — 한 engine 내부 prefix reuse
- **KV Offloading** — GPU KV capacity를 CPU/외부 tier로 확장
- **P→D KV Transfer** — Prefill 결과를 Decode engine으로 전달
- **Shared KV** — 다른 instance에서도 prefix KV를 재사용
- **Cache-aware Routing** — KV가 존재할 가능성이 높은 backend로 요청을 보냄

주요 구현 후보:
- vLLM native connector
- LMCache
- NIXL
- Mooncake / MooncakeStore

### 5. Stateful Serving Plane

주요 개념:
- conversation_id / response_id
- previous_response_id
- session identity
- tool-call / result lineage
- durable semantic state

원칙:
- Conversation Store는 **대화 의미 상태**를 저장한다.
- KV Cache는 **재계산을 줄이기 위한 transient compute state**다.
- 둘은 session identity로 연결하되 lifecycle과 storage contract는 분리한다.

### 6. GPU / Network Runtime

망A:
- H200 8GPU × 7
- IB / GPUDirect RDMA 검증
- NVIDIA Network Operator / RDMA resource
- NVLink / NVSwitch / NCCL

망B:
- H200 8GPU × 5
- IB 없음
- Node-local P/D가 우선적인 topology

### 7. Observability / Performance Intelligence

최소 관측 축:
- TTFT / ITL / E2E latency
- running / waiting / queue depth
- KV usage / hit / transfer
- GPU utilization / MFU / HBM BW
- NCCL / RDMA 상태
- backend routing decision / ejection
- P/D phase별 metrics

목표는 dashboard 자체가 아니라 **배포 topology와 운영 의사결정에 사용할 canonical signal**을 만드는 것이다.

### 8. MOC Operations Control Plane

MOC는 위 모든 기능을 대체하는 플랫폼이 아니다.

MOC가 담당할 영역:
- actual state 수집 및 정규화
- policy evaluation
- scale / drain / node reclaim 계획
- serving group lifecycle
- cache / routing / conversation 운영 정책 연계
- safety guard / cooldown / protected workload
- 변경 후 verify 및 audit

즉 요청 처리 data path 밖에서 **플랫폼 전체를 안전하게 수렴시키는 cross-plane controller**로 둔다.

---

## 중요한 아키텍처 경계

1. `LiteLLM != Router != MOC`
2. `Conversation State != KV Cache`
3. `P/D KV Transfer != Shared KV Cache`
4. `Pod Ready != Inference Healthy`
5. `Model Profile != Runtime Topology`
6. `Multi-node transport capability != Model-specific deployment recipe`

이 경계를 유지해야 신규 모델이나 upstream 변경이 생겨도 전체 구조를 다시 뜯지 않는다.
