# P/D Disaggregation

Prefill과 Decode를 서로 다른 실행 자원으로 분리하되, 사용자에게는 하나의 served model endpoint로 유지하는 배포 구조를 정리한다.

## 현재 결정

- 단기 기준선: custom vLLM Production Stack 0.1.8
- 우선 topology: 한 Pod 안의 Node-local P/D Cell
- Cell 구성: 전용 orchestrated Router + Prefill pool + Decode pool
- scale 단위: 초기에는 Cell replica
- 장애 정책: engine 일부가 실패하면 Cell 전체를 route에서 제외하고 완전 복구 후 재가입
- 관측 단위: Pod가 아니라 router/prefill/decode container와 port
- 장기 방향: 0.1.12+의 범용 disaggregated serving과 fabric/multi-node P/D로 semantic migration

## 읽는 순서

1. [Master Plan](2026-08-18-vllm-stack-pd-disaggregation-master-plan.md)
2. [0.1.8 Node-local P/D Cell 상세 계획](2026-08-18-node-local-pd-cell-0.1.8-plan.md)
3. [vLLM Stack 전체 진행 계획](../2026-08-18-진행계획.md)

## 핵심 경계

- Prefill/Decode는 별도 모델이 아니라 같은 모델의 실행 phase다.
- Global Router와 Cell Router의 책임을 분리한다.
- P→D KV transfer와 여러 replica가 공유하는 remote KV cache는 다른 문제다.
- Node-local 성공을 fabric P/D 성공으로 간주하지 않는다.
- engine/container별 metrics와 inference-aware readiness가 없으면 운영 완료로 보지 않는다.
