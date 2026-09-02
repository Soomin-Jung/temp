# vLLM Stack

vLLM Production Stack의 운영 기준선, version migration, 배포 topology, P/D Disaggregation, endpoint 검증 계약을 한곳에서 관리한다.

## 문서 지도

| 영역 | 문서 | 상태 |
|---|---|---|
| Historical execution plan | [2026-08-18 execution plan](history/2026-08-18-stack-execution-plan.md) | 당시 설계/진행 snapshot; current 기준 아님 |
| P/D Disaggregation | [P/D 문서 인덱스](pd-disaggregation/README.md) | PR #2 P1:D1 성공, PR #4 검증 중 |
| vLLM 0.28 migration | [0.28.0 Migration & KV Connector Compatibility](migrations/vllm-0.28.md) | 차기 validation candidate |
| Historical Mooncake build | [0.3.10-post2 최초 폐쇄망 Source Build](pd-disaggregation/history/2026-08-24-mooncake-0.3.10-post2-offline-build.md) | source-build skeleton 기록; current 0.28 path는 migration 문서 우선 |
| API routing | [Chat / Messages / Responses Routing Contract](api-routing-contract.md) | 구현 기준 |
| Stateful Serving | [Responses / Codex State Architecture](stateful-conversation-architecture.md) | Agentic API candidate validation |
| 배포 검증 | [Model Serving Validation Contract](model-serving-validation.md) | 구현 요구사항·golden test |

## 구조 원칙

1. 의미상 모델 block은 사용자에게 노출되는 모델/endpoint identity를 나타낸다. 단기 0.1.8 P/D의 실제 root는 `pdCellSpec.models[]`이다.
2. integrated, P/D, multi-node는 같은 모델을 구현하는 runtime topology다.
3. 단기 0.1.8 변경은 additive extension으로 두고 기존 integrated renderer의 회귀를 막는다.
4. custom repository는 특정 tag의 장기 복사본이 아니라 **upstream Production Stack fork + 최소 overlay**로 유지하고, upstream sync/rebase를 운영 workflow로 둔다.
5. version migration은 파일 복사가 아니라 model identity, topology, routing, metrics, failure semantics의 semantic migration으로 수행한다.
6. vLLM runtime baseline과 Production Stack version은 독립 axis로 관리한다. 현재 vLLM 0.27.x는 비교 기준선, 0.28.0-cu129는 차기 validation candidate다.
7. 배포 완료는 Pod Ready가 아니라 API contract, inference health, streaming/tool/reasoning 결과까지 검증된 상태를 의미한다.
8. Mooncake `nvlink`와 `nvlink_intra`를 구분하고, requested protocol이 아니라 실제 installed transport 로그로 data path를 판정한다.
9. `/v1/chat/completions`, `/v1/messages`, `/v1/responses`는 URL alias가 아니라 semantics별 routing contract로 관리한다.
