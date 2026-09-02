# vLLM Stack 0.1.8 Node-local P/D Cell 구현 계획

작성일: 2026-08-18  
최종 업데이트: 2026-09-03 KST

## 0. 현재 상태

이 문서는 최초 설계안이 아니라 실제 구현 PR과 런타임 검증 결과를 반영한 현재 기준선이다.

| 항목 | 상태 | 근거 / 다음 Gate |
|---|---|---|
| custom 0.1.8 baseline | 완료 | PR #1이 `main`에 merge됨 |
| P/D additive renderer | 구현 완료, Draft 유지 | PR #2: `pdCellSpec.models[]`, Deployment/Service/test/docs 추가 |
| 초기 P1:D1 runtime | 성공 | PR #2 이전 커밋 `36e45f0c277d4206ce233c8057a383e387c616c1`에서 배포, LiteLLM 연결, Anthropic 경로 확인 |
| PR #4 template audit | 테스트 중 | values 상속, router type, alias, 비대칭 GPU, model-local KV config 보완 |
| Qwen3.6-27B 운영 검증 | 진행 중 | P/D 역할별 profile 최적화와 topology 확대 검증 |
| Mooncake same-node NVLink | 0.3.10 overlay 확보 / 0.3.12 재검증 | PR #5 source-build skeleton을 vLLM 0.28 + Mooncake 0.3.12.post1에 적용하고 actual `nvlink_intra` path 인증 |
| Prometheus 다중 port scrape | 후순위 | runtime/data path 안정화 후 namespace/scrape rule 검증 |

초기 P1:D1 성공은 현재 PR #4 HEAD 전체가 검증됐다는 뜻이 아니다. PR #4는 같은 runtime에서 다시 P1:D1을 통과한 뒤 P2:D1, P2:D2, P1:D3 순서로 확장한다.

---

## 1. 현재 설계 결정

1. 단기 0.1.8에서는 기존 `servingEngineSpec.modelSpec[]`을 수정하지 않는다.
2. P/D 모델은 top-level additive extension인 `pdCellSpec.models[]`에 선언한다.
3. Prefill과 Decode를 서로 다른 사용자 모델로 만들지 않고 한 model block의 실행 phase로 둔다.
4. Cell 하나는 `Router + Prefill×N + Decode×M`을 담은 Kubernetes Pod 하나다.
5. `replicaCount`는 engine 수가 아니라 Cell Pod 수다. `0`도 허용해 topology 정의를 보존한 채 비활성화할 수 있다.
6. Kubernetes scheduler가 Cell 전체 GPU request를 한 Node에 원자적으로 배치한다.
7. 모델별 Service는 Cell Router port만 노출한다.
8. KV connector와 config는 `pdCellSpec.models[].kvTransfer`가 소유한다.
9. Router image별 CLI 차이를 `router.type: lmstack | vllm | custom`으로 분리한다.
10. 장기 0.1.12+에서는 같은 의미 계약을 `deploymentMode: disaggregated` 계열로 옮긴다.

### 왜 `pdCellSpec`인가

장기 모델링만 보면 `modelSpec.deploymentMode: pdCell`이 자연스럽다. 그러나 custom 0.1.8의 `modelSpec`은 integrated Deployment, Service, RayCluster renderer가 이미 직접 소비한다.

단기 기능을 기존 배열에 끼우면 현재 운영 모델의 manifest까지 바뀐다. 따라서 0.1.8에서는 기존 renderer를 건드리지 않는 additive root를 사용하고, P/D의 의미 계약만 장기 구조로 이관한다.

```text
0.1.8 short term
pdCellSpec.models[]
        ↓ semantic migration
0.1.12+
modelSpec
  deploymentMode: disaggregated
  disaggregatedServing.topology: nodeLocal | fabric
```

---

## 2. 현재 values API

아래 예시는 망B H200 한 Node의 8 GPU를 P1:D1, engine당 TP4로 사용하는 형태다. image tag는 source-built Mooncake artifact가 확정된 뒤 digest 또는 검증 tag로 고정한다.

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
      tag: vllm-validated-mooncake-0.3.10-post2
      replicaCount: 1

      kvTransfer:
        connector: MooncakeConnector
        config:
          kv_buffer_device: cuda
          kv_load_failure_policy: fail
          kv_connector_extra_config:
            mooncake_protocol: nvlink_intra
            num_workers: 16

      prefill:
        count: 1
        requestGPU: 4
        profile: /profiles/qwen36/pd-prefill-tp4.yaml
        env:
          - name: MC_INTRANODE_NVLINK
            value: "1"

      decode:
        count: 1
        requestGPU: 4
        profile: /profiles/qwen36/pd-decode-tp4.yaml
        env:
          - name: MC_INTRANODE_NVLINK
            value: "1"
```

### 핵심 field

| Field | 의미 |
|---|---|
| `models[].name` | Kubernetes resource / topology identity |
| `servedModelNames[]` | 첫 항목은 primary model ID, 나머지는 alias |
| `replicaCount` | Cell Pod 개수 |
| `prefill.count`, `decode.count` | Cell 내부 phase별 engine container 수 |
| `requestGPU` | container당 GPU 예약량. profile의 TP/PP/DP와 일치해야 함 |
| `profile` | 역할별 vLLM 설정 |
| `models[].kvTransfer` | 모델별 connector, failure policy, transport config |
| `router.type` | LMStack Router, vLLM Router, custom CLI 계약 선택 |

`count`와 TP를 혼동하지 않는다. 예를 들어 P2:D2에서 각 engine이 TP2면 Cell GPU request는 `2×2 + 2×2 = 8`이다.

---

## 3. 생성 리소스와 routing 경계

```text
Deployment/<release>-<name>-pd-cell
  replicas = models[].replicaCount

Cell Pod
  ├─ pd-router  :8000
  ├─ prefill-0  :8101
  ├─ ...
  ├─ decode-0   :8201
  └─ ...

Service/<release>-<name>-engine-service
  └─ targetPort: pd-router
```

Pod는 Kubernetes scheduling의 원자 단위이므로 Cell 내부 container가 서로 다른 Node로 갈 수 없다. 다만 Cell GPU 합계가 Node capacity보다 작으면 다른 workload와 Node를 공유할 수 있으므로, Node 독점이 필요하면 scheduler/affinity 정책을 별도로 적용한다.

### Router 2계층

```text
Global Router
  └─ Cell Service / Cell Router
       ├─ Prefill pool
       └─ Decode pool
```

- Global Router는 모델 endpoint 선택과 discovery를 담당한다.
- Cell Router는 한 Cell 안에서 Prefill→Decode orchestration을 담당한다.
- Cell 내부 backend는 localhost static endpoint이므로 다른 Cell의 engine으로 KV를 보낼 수 없다.
- LiteLLM은 개별 P/D engine이 아니라 모델별 Cell Service를 바라본다.

### Router 구현별 계약

| `router.type` | 핵심 계약 |
|---|---|
| `lmstack` | static discovery + `disaggregated_prefill_orchestrated` + model label/alias |
| `vllm` | `--vllm-pd-disaggregation` + 반복형 `--prefill`/`--decode` + connector/policy |
| `custom` | `command` 선택, `args` 필수; custom image용 escape hatch |

LMStack Router와 vLLM Router는 CLI가 호환되지 않는다. repository만 교체하고 args를 재사용하면 안 된다.

---

## 4. 모델 identity와 alias

`servedModelNames`는 primary와 alias를 하나의 목록으로 관리한다.

```yaml
servedModelNames:
  - Qwen3.6-27B-PD
  - standard
```

Helm은 P/D 양쪽에 동일한 `--served-model-name` 목록을 주입하고, LMStack Router에는 alias→primary 매핑을 생성한다.

주의:

- topology 비교 시 같은 served name을 여러 Cell Service가 노출하면 Global Router가 하나의 backend pool로 섞을 수 있다.
- P1:D1과 P2:D1을 분리 측정할 때는 생성된 Service를 직접 호출하거나 topology별 임시 alias를 쓴다.
- profile의 served model name도 values와 같은 순서로 유지한다.

---

## 5. KV transfer ownership과 failure policy

KV config는 모델, vLLM/Mooncake/NIXL version, attention layout, dtype, TP topology에 종속된다. 따라서 최상위 공통 상속을 제거하고 각 `models[]`가 직접 소유한다.

```yaml
models:
  - name: qwen36-pd-p1d1
    kvTransfer:
      connector: MooncakeConnector
      config:
        kv_buffer_device: cuda
        kv_load_failure_policy: fail
        kv_connector_extra_config:
          mooncake_protocol: nvlink_intra
```

운영 기본은 P/D 모두 `kv_load_failure_policy: fail`이다.

- 실질적인 load failure 처리는 Decode에서 발생한다.
- `recompute`는 transfer 실패를 감추고 Decode engine에 긴 Prefill 연산을 유입시킨다.
- 50K~200K 장기 context 비중이 높은 환경에서는 Decode tail latency와 진행 중 요청을 함께 악화시킬 수 있다.
- connector 검증 단계에서도 `fail`을 사용해야 전송 실패가 recompute 성공으로 위장되지 않는다.

---

## 6. Mooncake transport: `nvlink`와 `nvlink_intra`

두 이름은 같은 transport의 alias가 아니다.

| Mooncake protocol | 범위 | Build flag | Runtime 선택 |
|---|---|---|---|
| `nvlink` | Multi-Node NVLink(MNNVL) | `USE_MNNVL=ON` | `mooncake_protocol: nvlink`; HCA가 있으면 `MC_FORCE_MNNVL=1` |
| `nvlink_intra` | 한 Node 내부 NVIDIA NVLink/NVSwitch | `USE_INTRA_NVLINK=ON` | `mooncake_protocol: nvlink_intra` + `MC_INTRANODE_NVLINK=1` |
| `rdma` | IB/RoCE/GDRDMA | CUDA/RDMA support | `mooncake_protocol: rdma`, HCA/device 검증 |
| `tcp` | 일반 network | `USE_TCP=ON`; GPU buffer에는 `USE_CUDA=ON` | `mooncake_protocol: tcp` |

망B H200 Node-local P/D에서 필요한 경로는 `nvlink_intra`다. H200에 NVLink/NVSwitch가 있다는 이유만으로 MNNVL용 `nvlink`를 선택하지 않는다.

### Runtime selector 주의

Mooncake 0.3.10.post2 source는 `MC_INTRANODE_NVLINK`의 값이 아니라 **환경변수 존재 여부**를 검사한다. 끄려면 `MC_INTRANODE_NVLINK=0`을 넣는 것이 아니라 변수를 제거해야 한다.

또 vLLM 로그의 “using X as its protocol”은 전달한 `mooncake_protocol`을 출력한 값이다. Mooncake가 실제 설치한 transport는 Mooncake 로그에서 다음과 같이 별도로 확인한다.

```text
Using Intra-Node NVLink transport (MC_INTRANODE_NVLINK set)
Using cross-node NVLink transport (MC_FORCE_MNNVL or no HCA detected)
Using RDMA transport ...
```

요청 protocol과 실제 transport가 일치하지 않으면 startup 성공을 data path 성공으로 간주하지 않는다.

### image build 상태와 다음 migration

초기 `0.3.10.post2` official artifact에는 `nvlink_intra`가 없어 source build가 필요했고, 이 build path는 PR #5로 확보했다.

vLLM 0.28 migration에서는 Mooncake `0.3.12.post1`을 새 source pin으로 사용한다. 해당 official x86 release build에도 `USE_INTRA_NVLINK=ON`이 없으므로 아래 compile capability는 계속 필요하다.

```text
USE_CUDA=ON
USE_MNNVL=ON
USE_INTRA_NVLINK=ON
```

0.3.10 빌드/반입 계약은 [Mooncake 0.3.10-post2 폐쇄망 source build 계획](2026-08-24-mooncake-0.3.10-post2-offline-build.md)에 보존한다. 현재 0.28 migration은 [vLLM 0.28.0 Migration & KV Connector Compatibility](../2026-09-03-vllm-0.28-migration.md)를 우선한다.

---

## 7. P/D 역할별 profile

현재 운영 우선 모델은 Qwen3.6-27B다. Prefill과 Decode profile을 동일 옵션 복사본으로 두지 않고 역할별로 최적화한다.

Prefill 중심:

- long-context chunked prefill
- TTFT와 batch token budget
- compute throughput
- transfer-ready KV layout

Decode 중심:

- ITL / decode token throughput
- 동시 sequence와 KV + recurrent/state capacity
- transfer load failure 격리
- 진행 중 Decode 지연 보호
- `max_num_batched_tokens` / `max_num_seqs` 역할별 튜닝
- speculative decoding 사용 시 K와 scheduler budget joint sweep
- backend capability에 따라 `FULL_DECODE_ONLY` 등 CUDA Graph mode 검증

공통 불변 조건:

- model/revision
- vLLM과 connector build
- dtype / KV cache dtype
- attention backend와 KV layout
- served model name
- connector가 허용하는 TP/PP 조합

비대칭 GPU 예: Prefill TP4×1 + Decode TP2×2. Chart는 resource/render를 지원하지만 실제 지원 여부는 model architecture와 connector compatibility가 결정한다.

---

## 8. 현재 검증 순서

### Gate A — PR #4 P1:D1 재검증

1. Qwen3.6-27B P/D profile 확정
2. Cell Router, P, D startup/readiness
3. Router→Prefill→KV transfer→Decode
4. LiteLLM OpenAI/Anthropic 경로
5. streaming/non-streaming, 장문, cancellation

### Gate B — topology 확장

순서:

```text
P1:D1
  → P2:D1
  → P2:D2
  → P1:D3
```

각 단계에서 확인:

- Pod 총 GPU request
- engine별 port / internal port / side-channel 충돌
- static backend 및 model label
- 요청 분배
- connector handshake
- 실제 KV transfer success/error
- TTFT / ITL / throughput

### Gate C — lifecycle

- `replicaCount: 0 → 1 → 0 → 1`
- Node 이름 없이 scheduler 배치
- GPU 부족 시 Pending 의미
- Prefill, Decode, Router restart
- Cell endpoint 제거와 자동 복귀

### Gate D — observability

- P/D engine별 `/metrics`
- 동일 Pod IP에서 `PodIP:port` 또는 pod/container identity 유지
- `pd_role`, deployment, cell, node label
- Router metrics
- namespace/scrape rule은 runtime data path 안정화 후 적용

---

## 9. 장애 정책

초기 운영 정책은 strict Cell readiness다.

```text
Router Ready
AND all Prefill Ready
AND all Decode Ready
        ↓
Cell Ready
```

engine 하나가 죽으면 Cell을 신규 요청 route에서 제외하고 container 재기동과 model reload가 끝난 뒤 재가입시킨다. 부분 engine만으로 계속 서비스하는 degraded Cell은 단기 필수 범위가 아니다.

이 정책은 개별 container liveness만으로 자동 완성되지 않는다. 최종 acceptance에는 aggregate readiness 또는 Router backend health가 실제 Service endpoint 제거로 연결되는지 확인해야 한다.

---

## 10. 종료 조건

0.1.8 단기 트랙은 아래가 모두 충족되어야 merge 가능하다.

- 기존 integrated/Ray manifest regression 없음
- PR #4 HEAD에서 Qwen3.6-27B P1:D1 성공
- 최소 한 개 variable topology 성공
- LiteLLM / Global Router / model discovery 성공
- streaming / long-context / cancellation 확인
- engine별 metrics 수집
- Cell failure/ejection/rejoin 확인
- connector/image/tag/digest와 실제 transport가 재현 가능하게 고정

장기 0.1.12+로 넘길 범위:

- independent P/D pool scaling
- fabric P/D와 multi-node P/D engine
- degraded serving
- Native Multiprocess / LWS
- topology-aware routing

핵심 원칙:

> 모델 identity는 하나이고 Prefill/Decode는 내부 실행 phase다. 단기 `pdCellSpec`은 기존 0.1.8 renderer를 보호하기 위한 API 격리이며, 장기 의미 모델은 하나의 disaggregated serving group으로 수렴한다.


## 2026-09-03 Update Note

이 문서는 0.1.8 Node-local Cell의 Helm/API 계약을 보존한다. 현재 runtime/version 판단은 다음 문서를 우선한다.

- [vLLM 0.28.0 Migration & KV Connector Compatibility](../2026-09-03-vllm-0.28-migration.md)
- [Scheduler Budget / Speculative Decoding / CUDA Graph](../../study/inference-serving-optimization/01-scheduler-budget-spec-decode-cudagraph.md)
