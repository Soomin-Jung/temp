# 2026 Q3 — LLM E2E Platform 확장

기준일: 2026-08-18

## 분기 목표

Q3의 목표는 개별 모델을 추가 배포하는 수준을 넘어, 현재 운영 중인 LLM Serving Stack을 **분산 추론 / P-D 분리 / 신규 모델 / KV Cache / Stateful Serving / 운영 자동화**까지 확장 가능한 공통 플랫폼으로 정리하는 것이다.

MOC는 전체 과제의 중심이 아니라 여러 플랫폼 capability를 운영적으로 연결하는 **Operations Control Plane** 중 하나로 본다.

## 현재 인프라 기준

### HPC망

- H200 8GPU 노드 × 7 — 총 56 GPU
  - NVLink/NVSwitch
  - InfiniBand 구성
  - vLLM NCCL 로그에서 GPUDirect RDMA / IB channel 확인
  - NVIDIA Network Operator 기반 RDMA device resource 사용
- H100 4GPU 노드 × 6 — 총 24 GPU
  - NVBridge 미장착
- L40S 8GPU 노드 × 2 — 총 16 GPU

### DSCloud망

- H200 8GPU 노드 × 5 — 총 40 GPU
- InfiniBand 없음

총 20 GPU 노드 / 136 GPU

---

## Q3 우선순위

### P0 — 고객 일정에 직접 연결된 과제

1. **P/D Disaggregation 통합 배포**
   - custom vLLM Production Stack 0.1.8 운영 기준선을 보존하면서 Node-local P/D Cell을 우선 구현
   - 하나의 modelSpec 안에서 Prefill / Decode topology 정의
   - orchestrated LMRouter + KV transfer + Cell replica + Prometheus + failure recovery 통합
   - 상세: [`pd-disaggregation/`](../../../pd-disaggregation/)

2. **Kimi-K3 HPC Multi-node 운영 배포**
   - H200 + IB/GDRDMA 환경에서 멀티노드 배포를 운영 수준으로 정립
   - 단순 기동 성공이 아니라 topology, lifecycle, health, failure recovery, benchmark까지 포함
   - Ray는 현재/검증 경로로 활용하되 장기적으로 vLLM native multiprocess + LWS 계열을 검토
   - 이후 Multi-node P/D topology까지 확장 검토

3. **최신 모델 Enablement / 장애 분석**
   - Kimi-K3
   - DeepSeek-V4 Flash 0731 / Pro 0813 계열
   - Qwen3.8-27B 계열
   - 신규 모델별 architecture / kernel / speculative decoding / parser / topology compatibility를 운영 profile로 축적

### P1 — P0를 지속 가능하게 만드는 플랫폼 기반

4. **vLLM Production Stack Modernization**
   - 0.1.8 custom baseline 유지 및 regression 검증
   - Node-local P/D를 단기 additive extension으로 구현
   - 이후 0.1.12+로 semantic rebase
   - custom fork 확대보다 upstream primitive + organization overlay 구조를 지향

5. **LiteLLM / Router / Observability Contract 정리**
   - OpenAI / Anthropic API compatibility
   - streaming / reasoning / tool-call regression test
   - Global Router와 P/D Cell Router의 책임 분리
   - engine-level metrics와 canonical health model 정립

### P2 — 다음 단계 플랫폼 확장

6. **KV Cache / Cache-aware Routing**
   - GPU local APC, host offload, P→D KV transfer, shared remote KV를 서로 다른 capability로 분리
   - LMCache / NIXL / Mooncake / vLLM native connector 호환성 정리
   - cache locality + load + health를 함께 보는 routing 구조 검토

7. **Stateful Conversation Serving**
   - `/v1/responses` 계열의 durable conversation state를 inference engine과 분리
   - conversation / response lineage와 transient KV state를 구분
   - session identity를 conversation store, router, cache plane이 공유할 수 있는 구조 검토

8. **MOC Operations Control Plane**
   - 다른 과제보다 구현 우선순위는 뒤지만 전체 운영 구조의 장기 제어 계층으로 유지
   - Core Framework는 state collection / policy / planner / safety / adapter / reconcile 구조를 기준으로 함
   - 각 serving capability가 안정된 뒤 scale, drain, node reclaim, topology lifecycle, cache policy 등을 controller로 흡수

---

## 6개 Program Workstream

분기 관리는 세부 기능 수십 개 대신 아래 6개 축으로 유지한다.

| Program | 포함 범위 | Q3 상태 |
|---|---|---|
| Serving Foundation | Production Stack, LiteLLM, LMRouter, Observability | 진행 중 |
| Distributed Inference | P/D, Multi-node, GDRDMA, Native MP/LWS | 최우선 진행 |
| Model Enablement & Performance | 신규 모델, topology, SD, tuning, metric 분석 | 상시 진행 |
| Cache Plane | KV offload, P/D transfer, shared KV, cache-aware routing | 설계/검증 트랙 |
| Stateful Serving | Responses state, conversation store, session routing | 아키텍처 과제 |
| Operations Control | MOC Core + 운영 비즈니스 로직 | 설계 유지 / 구현 후순위 |

자세한 상태와 Gate는 [`workstreams.md`](workstreams.md), 전체 구조는 [`architecture.md`](architecture.md)를 참고한다.

---

## Q3 핵심 성공 기준

- [ ] Node-local P/D Cell이 기존 integrated deployment를 깨지 않고 운영 Chart에 통합된다.
- [ ] P/D Cell의 LiteLLM / Global Router / metrics / failure recovery 경로가 E2E로 검증된다.
- [ ] Kimi-K3가 HPC H200 멀티노드 환경에서 재현 가능한 운영 profile로 정립된다.
- [ ] Multi-node 장애와 engine process failure를 Pod Ready 이상의 health 기준으로 탐지/복구할 수 있다.
- [ ] vLLM Production Stack 0.1.8 custom baseline과 0.1.12+ 이관 경계가 명확해진다.
- [ ] 신규 모델 배포 결과가 일회성 명령이 아니라 Model/Runtime Profile로 축적된다.
- [ ] KV Cache / Stateful Conversation / MOC가 차기 구현 시 현재 Serving Stack을 다시 뒤엎지 않도록 interface boundary를 확정한다.

## 상세 문서 연결

- [vLLM Stack 진행 계획](../../../vllm-stack/2026-08-18-%EC%A7%84%ED%96%89%EA%B3%84%ED%9A%8D.md)
- [P/D Disaggregation Master Plan](../../../pd-disaggregation/2026-08-18-vllm-stack-pd-disaggregation-master-plan.md)
- [Node-local P/D Cell 상세 계획](../../../pd-disaggregation/2026-08-18-node-local-pd-cell-0.1.8-plan.md)
- [DeepSeek-V4 모델/장애 조사](../../../models/deepseek-v4/)
- [Study Notes](../../../study/)
