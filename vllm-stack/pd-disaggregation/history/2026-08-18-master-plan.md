# vLLM Stack P/D Disaggregation 구현 Master Plan

작성일: 2026-08-18  
최종 업데이트: 2026-09-03 KST

## 현재 진행 상태

- PR #1: custom 0.1.8 baseline이 `main`에 merge됨
- PR #2: top-level `pdCellSpec` additive renderer 구현, Draft 유지
- Runtime: PR #2 이전 커밋 `36e45f0c277d4206ce233c8057a383e387c616c1`에서 P1:D1, LiteLLM, Anthropic 경로 성공
- PR #4: values/container/router 계약 audit 완료 후 Qwen3.6-27B runtime 확대 테스트 중
- Mooncake: `0.3.10.post2` source-built `nvlink_intra` overlay는 PR #5로 확보했다. 현재는 actual transport/runtime Gate와 vLLM 0.28용 Mooncake `0.3.12.post1` source-build compatibility가 다음 검증 대상
- Runtime migration: vLLM `0.27.1`은 비교 기준선, `0.28.0-cu129`은 차기 validation candidate
- Repository strategy: custom Production Stack은 upstream fork + thin overlay로 유지

초기 성공은 PR #4 HEAD의 완료 판정이 아니다. P1:D1 재검증 후 P2:D1 → P2:D2 → P1:D3, Cell replica, failure, metrics 순으로 Gate를 닫는다.

## 목표

현재 운영 기준선인 custom vLLM Production Stack 0.1.8을 보존하면서, 0.1.12 전면 이관 전에 node-local P/D Cell을 별도 확장 경로로 구현한다.

이번 단계의 핵심은 단순 P/D PoC가 아니라 다음 계약을 실제 Kubernetes/Helm 수준에서 검증하는 것이다.

- 한 모델 블록에서 Prefill/Decode topology를 함께 정의
- Cell replica 단위 scale-out
- Prefill/Decode 수량 조정
- 기존 profile 기반 vLLM 실행 방식 재사용
- Cell 내부 전용 P/D Router(LMStack/vLLM/custom image별 CLI 계약 분리)
- KV connector 기반 node-local transfer
- 기존 Global Router의 모델 discovery 구조 유지
- 모델별 단일 Service endpoint 유지
- 모든 P/D engine metrics 개별 수집 가능
- Cell 단위 장애 제거 및 자동 복귀

## 단기 구조 결정

0.1.8의 기존 `servingEngineSpec.modelSpec`은 단일 Deployment와 RayCluster renderer가 이미 직접 소비한다. 단기 P/D 기능을 여기에 강하게 끼워 넣으면 기존 운영 manifest까지 함께 변경해야 한다.

따라서 단기 0.1.8에서는 additive extension을 사용한다.

```yaml
pdCellSpec:
  enabled: true
  models:
    - name: example-pd
      servedModelNames: [example-model, example-alias]
      replicaCount: 2
      kvTransfer:
        connector: MooncakeConnector
      prefill:
        count: 2
      decode:
        count: 2
```

중요한 점은 Prefill과 Decode를 별도 modelSpec으로 만들지 않는다는 것이다. P/D는 동일 모델의 실행 phase이며 한 모델 block 내부 topology로 취급한다.

장기 0.1.12+에서는 이 semantic block을 `modelSpec.deploymentMode=disaggregated` 계열로 흡수한다.

## Runtime 구조

```text
Deployment
  replicas = Cell count

Cell Pod
  ├─ P/D Router
  ├─ Prefill Engine 0..N
  └─ Decode Engine 0..M
```

하나의 Pod가 Cell 전체 GPU resource를 요청하기 때문에 Kubernetes scheduler가 Cell을 한 Node에 원자적으로 배치한다. 단기 node-local 모드에서는 Node 이름을 직접 관리하지 않고 Cell replica 수로 확장한다.

## Router 계층

```text
Global Router
   │
   └─ Cell Router endpoint
         ├─ Prefill pool
         └─ Decode pool
```

Global Router는 Cell 내부 topology를 알 필요가 없다. Cell Router는 image에 맞는 P/D 계약을 사용하며, Cell 내부 엔진은 localhost/static membership으로 격리한다.

- `router.type=lmstack`: `disaggregated_prefill_orchestrated` + static backend/model/alias
- `router.type=vllm`: `--vllm-pd-disaggregation` + 반복형 Prefill/Decode endpoint
- `router.type=custom`: 명시적 command/args

LMStack Router와 vLLM Router의 CLI는 호환되지 않는다.

## KV Transfer와 transport

KV connector는 모델·profile·topology 호환성에 종속되므로 `pdCellSpec.models[].kvTransfer`가 직접 소유한다.

운영 기본 failure policy는 P/D 모두 `fail`이다. `recompute`는 transfer failure를 숨기고 Decode에 Prefill을 유입시키므로 장기-context production 기본값으로 사용하지 않는다.

Mooncake transport 의미:

| Protocol | 범위 | Build flag |
|---|---|---|
| `nvlink` | Multi-Node NVLink(MNNVL) | `USE_MNNVL=ON` |
| `nvlink_intra` | 동일 Node NVLink/NVSwitch | `USE_INTRA_NVLINK=ON` |
| `rdma` | IB/RoCE/GDRDMA | CUDA/RDMA support |
| `tcp` | 일반 network fallback | GPU buffer에는 `USE_CUDA=ON` |

Network B H200 Node-local P/D의 목표 Mooncake path는 `nvlink_intra`다. `0.3.10.post2` source-build overlay는 이미 확보했으며, vLLM 0.28 validation에서는 Mooncake `0.3.12.post1`을 동일한 air-gap/build contract로 재검증한다. official x86 wheel에는 `USE_INTRA_NVLINK=ON`이 없으므로 source build 자체는 여전히 필요하다.

## Profile 원칙

기존 운영 방식처럼 모델/엔진 동작 옵션은 profile에 둔다.

Helm이 관리:
- topology
- engine count
- resource
- port/runtime identity
- KV role
- scheduling
- routing

Profile이 관리:
- 모델 경로
- max length
- cache dtype
- batching
- 기타 vLLM engine option

## Observability

P/D Cell에서는 Pod 하나에 여러 vLLM process가 존재한다. 따라서 Prometheus target identity는 Pod가 아니라 engine/container 단위여야 한다.

필수 구분:
- deployment
- cell
- phase(prefill/decode)
- engine/container
- node

기존 port 하나만 수집하는 scrape rule과 분리하여 P/D engine용 scrape job을 추가한다.

## 단기 장애 정책

첫 버전은 strict Cell readiness를 사용한다.

```text
Router Ready
AND 모든 Prefill Ready
AND 모든 Decode Ready
→ Cell Ready
```

engine 하나가 재시작 중이면 해당 Cell 전체를 신규 요청 대상에서 제거한다. 다른 Cell replica가 서비스를 유지하고, engine이 다시 Ready가 되면 Cell이 자동 복귀한다.

engine-level degraded serving은 장기 PDSG 범위로 남긴다.

## 구현 파일

실제 구현 브랜치에서 신규 리소스를 additive하게 추가한다.

- P/D Cell Deployment template
- P/D Cell Service template
- example values
- Helm unit test
- 한글 구현/연결 구조 문서

기존 integrated/Ray renderer는 단기 기능 때문에 가능한 한 수정하지 않는다.

## 개발 Gate

1. Helm lint/template — 완료
2. PR #2 이전 커밋 P1:D1 단일 Cell — 완료
3. PR #4 HEAD P1:D1 재검증 — 진행 중
4. P2:D1 → P2:D2 → P1:D3 topology 변경
5. Cell replica 0↔1 및 scale-out
6. `/v1/models` / Service / 상위 Router 연동
7. 모든 engine metrics 수집
8. P/D/Router container 장애 및 자동 복귀
9. 기존 integrated deployment regression
10. integrated vs P/D 성능 비교

## 단기 종료선

0.1.8에서는 아래까지만 완성한다.

- node-local Cell
- configurable P:D
- Cell replica scaling
- profile reuse
- image별 CLI가 검증된 Cell P/D Router
- KV transfer
- discovery/service integration
- metrics
- strict failure recovery

다음은 current upstream fork의 장기 트랙으로 이동한다. 특정 0.1.12 snapshot 자체가 목표가 아니라 upstream primitive를 계속 흡수하는 것이 목표다.

- independent P/D pool scaling
- degraded Cell serving
- fabric P/D
- multi-node P/D engine
- Native MP/LWS
- topology-aware routing

## 장기 매핑

```text
0.1.8
pdCellSpec.models[]
      ↓ semantic migration
0.1.12+
modelSpec
  deploymentMode: disaggregated
  disaggregatedServing:
    topology: nodeLocal | fabric
```

단기 작업에서 재사용해야 하는 것은 모델 identity, P/D topology, profile contract, KV policy, Router contract, metrics label, failure semantics이다.

상세 현재 기준:

- [Node-local P/D Cell 구현 계획](../node-local-pd-cell-vllm-stack-0.1.8.md)
- [Mooncake 0.3.10-post2 폐쇄망 Source Build 계획](2026-08-24-mooncake-0.3.10-post2-offline-build.md)

## 원칙

> 0.1.8은 기존 운영 경로를 흔들지 않는 additive P/D Cell로 구현한다.
>
> Prefill/Decode는 별도 모델이 아니라 한 모델 내부의 실행 topology다.
>
> 단기 구현에서 얻은 계약은 0.1.12 범용 Disaggregated Serving으로 그대로 이동할 수 있어야 한다.


## 2026-09-03 Update Note

이 문서의 0.1.8/0.1.12 매핑은 당시 설계 경계를 설명하는 historical semantic map이다. 현재 migration 판단과 connector version은 다음 문서를 우선한다.

- [vLLM 0.28.0 Migration & KV Connector Compatibility](../../migrations/vllm-0.28.md)
- [P/D Disaggregation Index](../README.md)
- [Inference Serving Optimization](../../../study/llm-serving-optimization/README.md)
