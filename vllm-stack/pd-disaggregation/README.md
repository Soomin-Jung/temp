# P/D Disaggregation

업데이트: 2026-08-25 KST

Prefill과 Decode를 서로 다른 실행 자원으로 분리하되, 사용자에게는 하나의 served model endpoint로 유지하는 배포 구조를 정리한다.

## 현재 결정

- 단기 기준선: custom vLLM Production Stack 0.1.8
- 구현 API: 기존 renderer와 분리된 top-level `pdCellSpec.models[]`
- 우선 topology: 한 Pod 안의 Node-local P/D Cell
- Cell 구성: 전용 P/D Router + Prefill pool + Decode pool
- scale 단위: 초기에는 Cell replica
- 장애 정책: engine 일부가 실패하면 Cell 전체를 route에서 제외하고 완전 복구 후 재가입
- 관측 단위: Pod가 아니라 router/prefill/decode container와 port
- long-context production의 KV load failure policy: P/D 모두 `fail` 우선
- 장기 방향: 0.1.12+의 범용 disaggregated serving과 fabric/multi-node P/D로 semantic migration

## 2026-08-25 상태

- PR #1: custom 0.1.8 baseline merge 완료
- PR #2: P/D Cell additive extension 구현, Draft
- P1:D1: PR #2 이전 커밋에서 배포/LiteLLM/Anthropic 경로 성공
- PR #4: values, router, alias, 비대칭 GPU, KV config 계약 보완 후 runtime matrix 검증 중
- PR #5: Mooncake `0.3.10.post2`를 폐쇄망 source build하고 `nvlink`/`nvlink_intra` transport를 포함하는 custom vLLM image build 경로 merge 완료
- Mooncake image build 성공과 실제 P/D KV transfer transport 인증은 별도 Gate로 관리
- 최신 upstream 분석 기준: vLLM `v0.27.1`; NIXL runtime pin `1.3.1`; Mooncake 실무 재현 기준 `0.3.10.post2`
- PR #4 GPU runtime matrix가 충분히 끝나기 전에는 P/D PR을 merge하지 않음

## 읽는 순서

1. [Platform Engineering State](../../docs/context/platform-state.md)
2. [Master Plan](2026-08-18-vllm-stack-pd-disaggregation-master-plan.md)
3. [0.1.8 Node-local P/D Cell 상세 계획](2026-08-18-node-local-pd-cell-0.1.8-plan.md)
4. **[KV Transfer Backends — vLLM/Mooncake/NIXL 전체 지도](kv-transfer-backends/README.md)**
5. [vLLM KV Transfer Runtime Path](kv-transfer-backends/vllm-kv-transfer-path.md)
6. [Mooncake vs NIXL 선택/비교](kv-transfer-backends/backend-selection.md)
7. [Mooncake Transfer Engine Deep Dive](kv-transfer-backends/mooncake-transfer-engine/README.md)
8. [NIXL Deep Dive](kv-transfer-backends/nixl/README.md)
9. [Mooncake 0.3.10-post2 최초 폐쇄망 Build 기록](2026-08-24-mooncake-0.3.10-post2-offline-build.md) — historical incident/build note
10. [vLLM Stack 전체 진행 계획](../2026-08-18-진행계획.md)

## KV Transfer 문서 구조

```text
kv-transfer-backends/
├── README.md
├── vllm-kv-transfer-path.md
├── backend-selection.md
├── mooncake-transfer-engine/
│   ├── README.md
│   ├── transport-and-runtime-paths.md
│   ├── source-build-airgap.md
│   └── vllm-connector-flow-debugging.md
└── nixl/
    ├── README.md
    ├── vllm-integration.md
    ├── backends-data-path.md
    ├── source-build-airgap.md
    └── debugging-validation.md
```

### 문서 분리 원칙

- `kv-transfer-backends/`: vLLM 공통 connector lifecycle과 backend 선택 문제
- `mooncake-transfer-engine/`: Mooncake **Transfer Engine 자체**와 `rdma` / `nvlink` / `nvlink_intra` / `tcp` 실제 구현
- `nixl/`: NIXL **Agent / Memory Section / descriptor / plugin architecture 자체**를 먼저 설명하고 그 위에 vLLM Pull/Push integration을 연결
- Mooncake와 NIXL의 이름이 겹치는 protocol/backend를 동일 계층으로 해석하지 않음

## 핵심 경계

- Prefill/Decode는 별도 모델이 아니라 같은 모델의 실행 phase다.
- Global Router와 Cell Router의 책임을 분리한다.
- P→D KV transfer와 여러 replica가 공유하는 remote KV cache는 다른 문제다.
- Control plane(metadata/bootstrap/handshake)과 Data plane(KV bytes)을 분리해서 관측한다.
- Node-local 성공을 fabric P/D 성공으로 간주하지 않는다.
- Mooncake `nvlink`(MNNVL/Fabric Memory 계열)와 `nvlink_intra`(동일 Node CUDA IPC/P2P)를 혼용하지 않는다.
- `NixlConnector`는 network protocol 이름이 아니며 v0.27.1에서는 `NixlPullConnector`의 호환 alias다.
- NIXL 기본 `UCX` backend와 UCX가 실제 선택하는 CUDA IPC/RDMA/TCP lane을 구분한다.
- NCCL tuning과 KV transfer backend tuning은 별도다.
- image build/import 성공과 실제 GPU-direct KV transfer 성공을 분리한다.
- engine/container별 metrics와 inference-aware readiness가 없으면 운영 완료로 보지 않는다.

## 네트워크별 우선 검증

### Network B — node-local P/D

1. Mooncake `nvlink_intra` actual data path 인증
2. NIXL `NixlPushConnector + UCX` same-node CUDA IPC/P2P 평가
3. NIXL Pull과 동일 workload 비교
4. NVLink/NVSwitch/PCIe 실제 physical counter까지 확인

### Network A — cross-node RDMA

1. NIXL `UCX` VRAM-to-VRAM RDMA baseline
2. Mooncake `rdma`
3. 필요 시 NIXL `LIBFABRIC`
4. heterogeneous TP / multi-node P/D 확장

이 순서는 제품 우열을 확정한 것이 아니라 현재 platform topology와 upstream integration maturity를 기준으로 한 검증 우선순위다.
