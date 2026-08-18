# vLLM Stack

vLLM Production Stack의 운영 기준선, version migration, 배포 topology, P/D Disaggregation, endpoint 검증 계약을 한곳에서 관리한다.

## 문서 지도

| 영역 | 문서 | 상태 |
|---|---|---|
| 전체 진행 순서 | [2026-08-18 진행 계획](2026-08-18-진행계획.md) | 기준 계획 |
| P/D Disaggregation | [P/D 문서 인덱스](pd-disaggregation/README.md) | 단기 Node-local Cell 우선 |
| 배포 검증 | [Model Serving Validation Contract](model-serving-validation.md) | 구현 요구사항·golden test |

## 구조 원칙

1. `modelSpec`은 사용자에게 노출되는 모델/endpoint identity를 나타낸다.
2. integrated, P/D, multi-node는 같은 모델을 구현하는 runtime topology다.
3. 단기 0.1.8 변경은 additive extension으로 두고 기존 integrated renderer의 회귀를 막는다.
4. 0.1.12+ 이관은 파일 복사가 아니라 model identity, topology, routing, metrics, failure semantics의 semantic migration으로 수행한다.
5. 배포 완료는 Pod Ready가 아니라 API contract, inference health, streaming/tool/reasoning 결과까지 검증된 상태를 의미한다.
