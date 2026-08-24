# P/D Disaggregation

업데이트: 2026-08-24 KST

Prefill과 Decode를 서로 다른 실행 자원으로 분리하되, 사용자에게는 하나의 served model endpoint로 유지하는 배포 구조를 정리한다.

## 현재 결정

- 단기 기준선: custom vLLM Production Stack 0.1.8
- 우선 topology: 한 Pod 안의 Node-local P/D Cell
- Cell 구성: 전용 orchestrated Router + Prefill pool + Decode pool
- scale 단위: 초기에는 Cell replica
- 장애 정책: engine 일부가 실패하면 Cell 전체를 route에서 제외하고 완전 복구 후 재가입
- 관측 단위: Pod가 아니라 router/prefill/decode container와 port
- long-context production의 KV load failure policy: Prefill/Decode 모두 `fail` 우선
- 장기 방향: 0.1.12+의 범용 disaggregated serving과 fabric/multi-node P/D로 semantic migration

## 구현 / PR 상태

### PR #1 — 0.1.8 baseline

- merge 완료
- 현재 `main`의 short-term baseline

### PR #2 — Node-local P/D Cell

- Open Draft
- 기본 Helm lint/template/unit/pre-commit 검증 완료
- 실제 환경에서 P/D Cell runtime smoke 확인
- 추가 검증: 장기 context, streaming/cancellation, P2:D2/P3:D1, router discovery, LiteLLM path, metrics, restart/recovery, integrated 대비 성능
- GPU runtime matrix가 충분히 끝나기 전에는 merge하지 않음

### PR #4 — values/template/router contract audit

- PR #2 위에 stacked된 Open Draft
- 현재 실제 환경 테스트 중
- 주요 변경:
  - `kvTransfer` ownership을 model 단위로 고정
  - `servedModelNames` primary + alias 지원
  - LMStack / vLLM / custom Router contract 분리
  - env/envFrom/volumeMount/extraVolumes 상속 보완
  - 비대칭 Prefill/Decode GPU topology 문서화
  - full values reference 추가

## 현재 blocker — Mooncake NVLink transport

Network B는 InfiniBand가 없으므로 node-local P/D의 목표 KV 경로는 host/TCP 우회가 아니라 NVLink/NVSwitch 기반 GPU-local path다.

현재 기준:

- target: `mooncake-transfer-engine==0.3.10-post2`
- CUDA runtime ABI 문제는 공식 wheel 재설치로 해소
- 그러나 기본 wheel에는 요구하는 `nvlink_intra` transport가 포함되지 않는 빌드 조합이 있음
- 따라서 vLLM image 내부에서 Mooncake source build가 필요

다음 작업은 P/D Helm PR과 분리한 image-build PR로 진행한다.

1. Mooncake 0.3.10-post2 source 고정
2. `nvlink` / `nvlink_intra` 사용 가능한 build 옵션 고정
3. 폐쇄망에서 필요한 Git/Go/native dependency 사전 반입 목록 작성
4. 재현 가능한 Dockerfile / Kaniko build context 구성
5. build 후 transport enumeration 및 실제 P→D KV path 검증

Mooncake transport가 확정되기 전까지 node-local P/D의 최종 성능 평가는 보류한다.

## 읽는 순서

1. [Current Context](../../CURRENT_CONTEXT.md)
2. [Master Plan](2026-08-18-vllm-stack-pd-disaggregation-master-plan.md)
3. [0.1.8 Node-local P/D Cell 상세 계획](2026-08-18-node-local-pd-cell-0.1.8-plan.md)
4. [vLLM Stack 전체 진행 계획](../2026-08-18-진행계획.md)

## 핵심 경계

- Prefill/Decode는 별도 모델이 아니라 같은 모델의 실행 phase다.
- Global Router와 Cell Router의 책임을 분리한다.
- P→D KV transfer와 여러 replica가 공유하는 remote KV cache는 다른 문제다.
- Node-local 성공을 fabric P/D 성공으로 간주하지 않는다.
- engine/container별 metrics와 inference-aware readiness가 없으면 운영 완료로 보지 않는다.
