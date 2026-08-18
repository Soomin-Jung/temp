# vLLM Stack 0.1.8 Node-local P/D Cell 구현 계획

작성일: 2026-08-18

## 1. 결정 요약

현재 운영 기준선인 `Soomin-Jung/vllm-production-stack-custom`의 `agent/production-0.1.8-baseline-final` 브랜치를 유지한 상태에서, **0.1.12 이관보다 먼저 Node-local P/D Cell을 0.1.8 커스텀 차트에 추가한다.**

단, 0.1.8의 기존 `disaggregated_prefill` Router 구현을 재사용하지 않는다. Cell 내부 P/D orchestration에는 `disaggregated_prefill_orchestrated`를 지원하는 **검증된 최신 LMRouter 이미지(0.1.12 계열)** 를 별도 container로 사용한다.

핵심 결정은 다음과 같다.

1. `servingEngineSpec.modelSpec[]`는 계속 **모델/서비스의 최상위 선언 단위**로 사용한다.
2. Prefill과 Decode를 서로 다른 `modelSpec` 항목으로 만들지 않는다.
3. 하나의 모델 블록에 `deploymentMode: pdCell`과 `pdCell:` 하위 구성을 둔다.
4. P/D Cell 하나는 **Kubernetes Pod 하나**로 구현한다.
5. Cell Pod 안에 `P/D Router + N개의 Prefill Engine + M개의 Decode Engine`을 배치한다.
6. `modelSpec.replicaCount`는 P/D 모드에서 **Engine 개수가 아니라 Cell 개수**를 의미한다.
7. Pod가 Cell 전체 GPU를 요청하게 하여 Kubernetes scheduler가 Cell 전체를 한 노드에 자동 배치하게 한다.
8. 기존 Global LMRouter 0.1.8은 변경하지 않고 Cell Pod의 `:8000` P/D Router만 Serving Endpoint로 discovery한다.
9. LiteLLM은 모델별 Cell Service 하나만 바라본다.
10. 각 vLLM Engine의 `/metrics`는 Prometheus가 개별 scrape할 수 있도록 모든 Engine port를 Pod spec에 선언한다.

---

## 2. 왜 Prefill/Decode를 별도 modelSpec으로 만들지 않는가

다음 형태는 사용하지 않는다.

```yaml
modelSpec:
  - name: qwen36-prefill
    ...
  - name: qwen36-decode
    ...
```

이 구조는 Kubernetes 리소스 생성에는 쉽지만 플랫폼 의미가 틀어진다.

Prefill과 Decode는 서로 다른 사용자 모델이 아니라 **동일한 Served Model의 두 실행 단계(phase)** 이다. 둘을 독립 modelSpec으로 만들면 다음 문제가 생긴다.

- 동일 모델의 배포 identity가 두 개로 분리된다.
- Service와 `/v1/models` discovery 경계가 불필요하게 복잡해진다.
- P/D 비율을 변경할 때 하나의 배포 정의가 아니라 여러 modelSpec을 동시에 수정해야 한다.
- Cell replica와 P/D engine replica의 의미가 섞인다.
- metrics, readiness, 장애 도메인도 두 모델처럼 관리하게 된다.
- 장기적으로 `disaggregatedServingSpec`/PDSG 구조로 이관할 때 다시 합쳐야 한다.

따라서 `modelSpec`은 **"어떤 모델 endpoint를 배포할 것인가"**, `deploymentMode`는 **"그 endpoint를 어떤 실행 topology로 구현할 것인가"** 를 표현하도록 역할을 분리한다.

```text
modelSpec
  └─ qwen36-pd-a                 # 하나의 모델 배포 identity
       ├─ deploymentMode: pdCell
       └─ pdCell
            ├─ router
            ├─ prefill phase
            └─ decode phase
```

이 방식이 기존 integrated 배포와 장기 P/D 추상화를 동시에 보존한다.

---

## 3. 제안 values API

초기 구현안은 아래와 같이 잡는다.

```yaml
servingEngineSpec:
  modelSpec:
    # 기존 모델은 아무 변경 없음
    - name: existing-model
      replicaCount: 1
      repository: registry/vllm
      tag: existing
      requestGPU: 4
      vllmConfig:
        extraArgs:
          - --config
          - /profiles/existing.yaml

    # 신규 Node-local P/D Cell 모델
    - name: qwen36-pd-a
      deploymentMode: pdCell

      # P/D 모드에서 replicaCount == Cell replica 수
      replicaCount: 2

      # P/D vLLM Engine 공통 이미지/환경
      repository: registry/vllm
      tag: v0.26.0-mooncake
      env: []

      pdCell:
        router:
          repository: registry/lmstack-router
          tag: validated-0.1.12
          port: 8000

        prefill:
          count: 2
          requestGPU: 2
          portBase: 8101
          mooncakeBootstrapPortBase: 8998
          profile: /profiles/qwen36/pd-prefill.yaml

        decode:
          count: 2
          requestGPU: 2
          portBase: 8201
          profile: /profiles/qwen36/pd-decode.yaml

        service:
          port: 8000
```

### 의미

- `name`: Kubernetes 배포 identity. 같은 모델 가중치를 `qwen36-pd-a`, `qwen36-pd-b`처럼 동시에 실험 가능하다.
- `served_model_name`: 기존 운영 방식대로 profile에서 관리할 수 있다. A/B를 API에서 구분하려면 profile의 served model name도 다르게 지정한다.
- `replicaCount`: Cell 수.
- `prefill.count`, `decode.count`: Cell 내부 P/D Engine 개수.
- `requestGPU`: 각 Engine container가 요청할 GPU 수. TP2라면 일반적으로 2.
- `profile`: 역할별 vLLM profile. 기존 profile 중심 운영 방식을 유지한다.
- Router는 P/D Cell 내부 orchestration 전용이며 Global Router와 별개다.

초기 구현에서는 API를 과도하게 일반화하지 않는다. 향후 필요하면 role별 repository/tag/env override를 추가하되, **공통 설정은 modelSpec에 최대한 유지한다.**

---

## 4. 렌더링 결과

예: `P2:D2`, 각 Engine TP2, `replicaCount: 2`.

```text
Deployment/qwen36-pd-a
replicas: 2

Cell Pod #1 (Node A)
├─ pd-router      :8000   CPU only
├─ prefill-0      :8101   GPU 2
├─ prefill-1      :8102   GPU 2
├─ decode-0       :8201   GPU 2
└─ decode-1       :8202   GPU 2
                   total GPU = 8

Cell Pod #2 (Node B)
├─ pd-router      :8000
├─ prefill-0      :8101
├─ prefill-1      :8102
├─ decode-0       :8201
└─ decode-1       :8202
                   total GPU = 8
```

Pod는 Kubernetes scheduling의 원자 단위이므로 Cell 내부 container가 여러 노드로 분리되지 않는다. H200 8-GPU 노드에서 Cell이 GPU 8개를 요청하면 자연스럽게 한 Cell이 한 노드를 점유한다.

Node 이름을 values에 직접 적지 않는다. 기존 `schedulerName`, toleration, node affinity/nodeSelector 계층만 재사용한다.

---

## 5. Cell 내부 Router

Cell Router는 0.1.8 Global Router가 아니라 **0.1.12 계열 LMRouter image**를 사용한다.

Cell 내부에서는 Kubernetes discovery보다 static discovery가 더 단순하고 안전하다.

P2:D2 예:

```text
Prefill backends
  http://127.0.0.1:8101
  http://127.0.0.1:8102

Decode backends
  http://127.0.0.1:8201
  http://127.0.0.1:8202
```

템플릿이 `count`와 `portBase`를 기준으로 아래 Router args를 자동 생성한다.

```text
--service-discovery static
--routing-logic disaggregated_prefill_orchestrated
--static-backends <P endpoints>,<D endpoints>
--static-models <served model repeated>
--static-model-labels <prefill labels>,<decode labels>
--prefill-model-labels <prefill label>
--decode-model-labels <decode label>
```

이렇게 하면 Cell A Router가 Cell B의 Engine을 선택할 방법 자체가 없어져 **cross-node KV transfer를 구조적으로 차단**한다.

---

## 6. Profile 운영 방식

현재 커스텀 baseline의 profile 중심 실행 방식을 그대로 계승한다.

```text
/profiles/qwen36/
├─ pd-prefill.yaml
└─ pd-decode.yaml
```

각 Engine container는 공통 이미지와 global env/volume을 사용하고, 역할별 profile만 다르게 전달한다.

```text
prefill-N:
  vllm serve --host 0.0.0.0 --port <generated> --config /profiles/qwen36/pd-prefill.yaml

decode-N:
  vllm serve --host 0.0.0.0 --port <generated> --config /profiles/qwen36/pd-decode.yaml
```

동일 profile을 여러 Prefill container가 재사용하므로 HTTP port와 Mooncake bootstrap port처럼 **container마다 달라야 하는 값은 Helm template/env에서 생성**한다.

Mooncake producer/consumer 역할처럼 역할 자체에 고정된 vLLM 옵션은 P/D profile에 두는 것을 우선한다. Helm은 topology와 동적 identity/port에 집중한다.

---

## 7. 기존 0.1.8 Chart에서 수정할 파일

### 신규

```text
helm/templates/deployment-pd-cell.yaml
helm/templates/service-pd-cell.yaml
```

필요하면 validation helper를 `_helpers.tpl`에 추가한다.

### 최소 수정

#### `deployment-vllm-multi.yaml`

기존 non-Ray 조건에 `deploymentMode != pdCell` guard만 추가한다.

```text
기존 integrated model → 기존 template 그대로
pdCell model           → deployment-pd-cell.yaml
raySpec model          → 기존 Ray path
```

기존 integrated model의 rendered manifest가 바뀌지 않는 것을 회귀 테스트의 최우선 조건으로 둔다.

#### `service-vllm.yaml`

`pdCell` model은 기존 vLLM Service renderer에서 제외하고 `service-pd-cell.yaml`이 Router `:8000`을 target하도록 한다.

#### `values.schema.json`

`deploymentMode`, `pdCell.router`, `pdCell.prefill`, `pdCell.decode`, `pdCell.service` validation을 추가한다.

### 가능한 한 건드리지 않는 파일

- 기존 Global `deployment-router.yaml`
- 기존 Ray template
- 기존 일반 model resource/probe logic
- 기존 global env/volume merge contract

단기 기능 때문에 운영 integrated 모델 template을 대규모 수정하지 않는다.

---

## 8. 기존 Global LMRouter / LiteLLM 연계

Cell Pod의 외부 Serving Endpoint는 `pd-router:8000` 하나다.

```text
Global LMRouter 0.1.8
       │
       └─ K8s discovery → Cell Pod IP:8000 → /v1/models

LiteLLM
       │
       └─ model-specific Service → Cell Pod:8000
```

Global Router가 사용하는 기존 engine discovery label을 Cell Pod에도 부여한다. Prefill/Decode는 별도 Pod가 아니므로 Global Router에 노출되지 않는다.

따라서 현재 vLLM Proxy의:

```text
LiteLLM /v1/models
       ∩
Global LMRouter /v1/models
       ↓
사용자 노출 모델
```

구조를 유지할 수 있다.

LiteLLM에는 Cell replica를 개별 등록하지 않는다. `qwen36-pd-a-engine-service` 같은 모델별 Service 하나만 등록하며 Service가 Cell replica Pod들로 분산한다.

---

## 9. Prometheus /metrics

모든 vLLM Engine container의 port를 Pod spec에 명시한다.

```text
pd-router :8000
prefill-0 :8101
prefill-1 :8102
decode-0  :8201
decode-1  :8202
```

Prometheus `role: pod` discovery가 각 container port를 개별 target으로 발견하도록 한다.

기존 scrape rule의 `port == 8000` 조건은 P/D Engine을 제외하므로 P/D 전용 scrape job을 추가한다.

최종 target label은 최소한 다음을 제공한다.

```text
pd_deployment
pd_cell
pd_role        # prefill | decode
container
node
instance       # pod/container 또는 PodIP:port
```

Cell에서는 여러 vLLM process가 같은 Pod IP를 공유하므로 기존처럼 port를 제거한 Pod IP만 `instance`로 사용하면 안 된다.

---

## 10. 장애 복구 v1 정책

초기 0.1.8 구현은 **Strict Cell readiness**를 사용한다.

```text
Router Ready
AND all Prefill Ready
AND all Decode Ready
        ↓
Cell Ready
```

하나의 Engine이 crash하면:

```text
Engine container restart
→ Cell Pod NotReady
→ Service에서 해당 Cell 제외
→ 다른 Cell replica가 신규 요청 처리
→ Engine 모델 재로딩 및 readiness 성공
→ Cell 자동 재가입
```

이 정책은 degraded serving보다 가용 GPU 효율은 낮지만 구현이 단순하고 failure semantics가 명확하다.

P/D Engine 일부 장애에도 살아 있는 Engine으로 계속 처리하는 degraded Cell은 0.1.12 이관 전 필수 범위로 보지 않는다. 필요하면 후속 단계에서 LMRouter static backend health check + aggregate Cell readiness를 별도 구현한다.

운영 수준에서는 `replicaCount >= 2`를 권장한다.

---

## 11. Helm validation

`deployment-pd-cell.yaml` 렌더 전에 최소한 아래를 fail-fast 한다.

```text
prefill.count >= 1
decode.count >= 1
prefill.requestGPU >= 1
decode.requestGPU >= 1
router.port와 P/D port range 충돌 금지
Prefill/Decode port range 상호 충돌 금지
Mooncake bootstrap port 중복 금지
profile 값 필수
Router image/tag 필수
```

GPU 계산값도 렌더 로그/annotation으로 확인 가능하게 한다.

```text
cellGpu = prefill.count * prefill.requestGPU
        + decode.count  * decode.requestGPU
```

예:

```text
P2:D2 / TP2 → 2*2 + 2*2 = 8 GPU
P3:D1 / TP2 → 3*2 + 1*2 = 8 GPU
P2:D1 / TP2 → 6 GPU
```

`cellGpu < node GPU capacity`인 구성도 허용하되, 이 경우 "1 Cell = 1 Node 독점"은 자동 보장되지 않는다는 점을 명확히 한다. Node-locality 자체는 Pod 구조로 항상 보장된다.

---

## 12. 개발 순서

### Phase 0 — Baseline 보호

1. `agent/production-0.1.8-baseline-final`의 integrated model render 결과를 golden manifest로 저장한다.
2. P/D 변경 전/후 기존 model manifest가 동일한지 비교할 테스트를 만든다.
3. 기존 Global Router manifest가 변경되지 않는지 확인한다.

**Gate:** P/D 기능 추가가 기존 model rollout을 유발하지 않는 구조임을 `helm template` diff로 증명.

### Phase 1 — P1:D1 renderer

1. `deploymentMode: pdCell` schema 추가.
2. `deployment-pd-cell.yaml` 생성.
3. Router + P1 + D1 multi-container Pod render.
4. profile/global env/global volumes 적용.
5. static backend args 자동 생성.

**Gate:** `helm lint`, `helm template`, GPU/port validation 통과.

### Phase 2 — 실제 P1:D1 data path

1. vLLM 0.26 + Mooncake image 고정.
2. LMRouter 0.1.12 계열 image 고정.
3. Prefill → KV transfer → Decode 요청 성공.
4. streaming / non-streaming / 긴 context 검증.

**Gate:** 실제 OpenAI API 요청 E2E 성공.

### Phase 3 — Variable P:D

1. `count` loop renderer 적용.
2. P2:D2, P3:D1 렌더/구동.
3. 동일 모델을 다른 `name`으로 동시에 배포.

**Gate:** values 변경만으로 P:D 비율과 Cell replica를 변경 가능.

### Phase 4 — Cell replica / scheduling

1. `replicaCount: 1 → 2 → N` 검증.
2. Cell 전체가 한 Node에 배치되는지 확인.
3. H200×8에서 8-GPU Cell이 노드별로 하나씩 분산되는지 확인.
4. GPU 부족 시 Pending semantics 확인.

**Gate:** Node 이름을 직접 지정하지 않고 Cell scale-out 성공.

### Phase 5 — Service / discovery

1. P/D Service는 Router 8000만 노출.
2. LiteLLM upstream을 Service 하나로 연결.
3. Global LMRouter 0.1.8이 Cell Router `/v1/models`를 discovery하는지 확인.
4. 기존 vLLM Proxy model intersection 검증.

**Gate:** 기존 사용자 model availability 흐름과 통합.

### Phase 6 — Metrics

1. 모든 P/D Engine `/metrics` scrape.
2. `pd_role`, `pd_cell`, `pd_deployment` label 확인.
3. 동일 Pod IP의 각 Engine 시계열이 충돌하지 않는지 확인.

**Gate:** P/D Engine별 KV/queue/TTFT/ITL 관찰 가능.

### Phase 7 — Failure

1. Prefill container kill.
2. Decode container kill.
3. Router kill.
4. Node failure 또는 Cell Pod delete.
5. Cell Service endpoint 제거/복귀 시간 측정.

**Gate:** 다른 Cell은 계속 서비스하고, 재기동된 Cell이 수동 작업 없이 복귀.

---

## 13. 테스트 Matrix

최소 조합:

| Test | P:D | Cell replica | 목적 |
|---|---:|---:|---|
| T1 | 1:1 | 1 | data path 최초 성공 |
| T2 | 2:2 | 1 | multi-engine static routing |
| T3 | 3:1 | 1 | long-context Prefill 중심 구성 |
| T4 | 2:2 | 2 | 자동 Node scheduling / Cell HA |
| T5 | 3:1 | 2+ | 운영 후보 topology |

API 검증:

- `/v1/models`
- `/v1/chat/completions` non-streaming
- `/v1/chat/completions` streaming
- 긴 prompt
- cancellation
- timeout
- P/D engine restart

관측 항목:

- TTFT p50/p95/p99
- ITL p50/p95/p99
- P queue / D queue
- P/D GPU utilization
- Decode KV usage
- Mooncake KV transfer latency/error
- Engine별 `/metrics`

---

## 14. 0.1.12 장기 이관과의 관계

이 구현은 최종 P/D abstraction이 아니다. **Node-local P/D를 빨리 확보하기 위한 0.1.8 mode renderer**다.

그러나 다음 contract는 장기 구조에 그대로 가져간다.

```text
Model identity
Deployment mode
P/D phase definition
Role profile
Cell/Group replica
KV connector config
Serving Endpoint = P/D Router
metrics identity
readiness/failure domain
```

장기 0.1.12+에서는 대략 다음과 같이 일반화한다.

```text
modelSpec
  └─ deploymentMode: disaggregated
       └─ servingGroup
            ├─ topology: nodeLocal | fabric
            ├─ prefill pool
            ├─ decode pool
            ├─ engineRuntime: deployment | nativeMP/LWS
            └─ router implementation
```

따라서 단기 `pdCell`은 장기 `disaggregated serving`의 **nodeLocal topology subset**으로 취급한다.

---

## 15. 최종 판단

이번 단계에서는 **별도 top-level `pdCellSpec`보다 기존 `modelSpec` 안의 `deploymentMode: pdCell`이 더 적절하다.**

이유는 단순하다.

> 사용자가 배포하는 것은 Prefill 모델과 Decode 모델 두 개가 아니라, `qwen3.6-27b`라는 하나의 Serving Endpoint다.

P와 D는 그 endpoint를 구현하는 내부 topology다.

따라서 모델 선언은 하나여야 하고, Helm renderer가 내부적으로 Router/P/D containers를 만들어야 한다.

이 구조는 현재 0.1.8 integrated model을 건드리는 범위를 최소화하면서도, 이후 0.1.12의 범용 Disaggregated Serving 구조로 가장 자연스럽게 흡수할 수 있다.
