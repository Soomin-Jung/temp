# 2026 Q3 — LLM E2E Platform 확장

기준일: 2026-08-18  
최근 검토: 2026-09-03 KST

## 분기 목표

Q3의 목표는 개별 모델을 추가 배포하는 수준을 넘어, 현재 운영 중인 LLM Serving Stack을 **분산 추론 / P-D 분리 / 신규 모델 / KV Cache / Stateful Serving / 운영 자동화**까지 확장 가능한 공통 플랫폼으로 정리하는 것이다.

MOC는 전체 과제의 중심이 아니라 여러 플랫폼 capability를 운영적으로 연결하는 **Operations Control Plane** 중 하나로 본다.

## 현재 인프라 기준

### 망A

- H200 중심 GPU pool
- 노드 내부 NVLink/NVSwitch topology
- InfiniBand 구성
- vLLM NCCL 로그에서 GPUDirect RDMA / IB channel 확인
- NVIDIA Network Operator 기반 RDMA device resource 사용
- H100/L40S를 포함한 heterogeneous GPU pool은 topology capability를 별도 profile로 관리

### 망B

- H200 중심 GPU pool
- InfiniBand 없음
- 우선 검증 topology는 Node-local P/D

실제 node/GPU 수량과 hostname/IP는 공개 저장소에 기록하지 않는다.

---

## Q3 우선순위

### P0 — 현재 분기 우선 추진 과제

1. **P/D Disaggregation 통합 배포**
   - custom vLLM Production Stack 0.1.8 운영 기준선을 보존하면서 Node-local P/D Cell을 우선 구현
   - 단기 0.1.8에서는 기존 renderer와 분리한 `pdCellSpec.models[]` 한 block에서 Prefill / Decode topology 정의
   - PR #2 이전 커밋에서 P1:D1 + LiteLLM + Anthropic 경로 성공, PR #4 확대 테스트 진행
   - Qwen3.6-27B P/D profile과 P1:D1 → P2:D1 → P2:D2 → P1:D3 검증
   - Cell Router + KV transfer + Cell replica + Prometheus + failure recovery 통합
   - Network B same-node Mooncake `nvlink_intra` source-build overlay는 0.3.10.post2 기준으로 확보했고, vLLM 0.28 migration에서는 Mooncake 0.3.12.post1 계열로 재검증
   - 상세: [`vllm-stack/pd-disaggregation/`](../../../vllm-stack/pd-disaggregation/)

2. **Kimi-K3 망A Multi-node 운영 배포**
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

4. **vLLM / Production Stack Modernization**
   - 0.1.8 custom baseline PR #1 merge 완료
   - Node-local P/D를 단기 additive extension으로 구현
   - vLLM `0.27.1-cu129`을 비교 기준선으로 유지하면서 `0.28.0-cu129`을 차기 validation candidate로 승격
   - vLLM 0.28의 Model Runner V2 E/P/D, Kimi-K3/DeepSeek-V4 최적화, tiered KV, scheduler default 변화 검증
   - custom repository는 **upstream Production Stack fork + 최소 overlay**로 운영
   - 특정 chart version에 patch를 계속 누적하기보다 upstream primitive를 지속 sync하고 organization-specific profile/policy만 유지

5. **API Routing / LiteLLM / Router / Observability Contract 정리**
   - `/v1/chat/completions`, `/v1/completions` → LiteLLM hosted_vllm lane
   - `/v1/messages` → LiteLLM Anthropic lane
   - `/v1/responses` → Agentic API → LMStack Router lane
   - pinned LMStack Router 0.1.9의 Responses route와 Messages 부재를 version-specific contract로 관리
   - Global Router와 P/D Cell Router의 책임 분리
   - engine-level metrics와 canonical health model 정립
   - direct path와 full proxy path의 streaming / reasoning / tool-call / typed item fidelity를 golden test로 비교

### P2 — 다음 단계 플랫폼 확장

6. **KV Cache / Cache-aware Routing**
   - GPU local APC, host offload, P→D KV transfer, shared remote KV를 서로 다른 capability로 분리
   - vLLM 0.28 기준 LMCache `>=0.3.9`, NIXL `1.3.2`, Mooncake `>=0.3.12` compatibility 재검증
   - Mooncake `nvlink`(MNNVL)와 `nvlink_intra`(same-node)를 구분하고 compile feature를 compatibility matrix에 포함
   - Mooncake 0.3.12.post1 official x86 wheel에도 `USE_INTRA_NVLINK=ON`이 없으므로 source-build requirement 유지
   - cache locality + load + health를 함께 보는 routing 구조 검토

7. **Stateful / Agentic Responses Serving**
   - `/v1/responses` durable state를 inference engine과 분리
   - Agentic API를 model-agnostic State Facade + PostgreSQL + tool orchestration POC 후보로 검증
   - Agentic API는 하나의 logical `LLM_API_BASE` 뒤 multi-model LMStack Router를 사용
   - conversation / response lineage와 transient KV state를 구분
   - tenant authorization / retention / multi-replica recovery를 production Gate로 유지

8. **MOC Operations Control Plane**
   - 다른 과제보다 구현 우선순위는 뒤지만 전체 운영 구조의 장기 제어 계층으로 유지
   - Core Framework는 구현되어 있으나 운영 business capability가 완성된 것은 아님
   - Production Stack adoption, vLLM adapter, endpoint/LiteLLM sync, dynamic placement/scale, GPU reclaim/repacking을 명시적 backlog로 관리
   - 각 serving capability가 안정된 뒤 scale, drain, node reclaim, topology lifecycle, cache policy 등을 controller로 흡수
   - 상세: [`moc/README.md`](../../../moc/README.md)

---

## 6개 Program Workstream

분기 관리는 세부 기능 수십 개 대신 아래 6개 축으로 유지한다.

| Program | 포함 범위 | Q3 상태 |
|---|---|---|
| Serving Foundation | vLLM 0.28 migration, Production Stack fork, API routing, LiteLLM, LMRouter, Observability | 진행 중 |
| Distributed Inference | P/D, Multi-node, GDRDMA, Native MP/LWS | 최우선 진행 |
| Model Enablement & Performance | 신규 모델, topology, SD, tuning, metric 분석 | 상시 진행 |
| Cache Plane | KV offload, P/D transfer, shared KV, cache-aware routing | Mooncake transport 검증 진행 |
| Stateful Serving | Agentic API, Responses state, PostgreSQL, session routing | POC / architecture validation |
| Operations Control | MOC Core + 운영 비즈니스 로직 | Core 가용 / Business logic backlog |

자세한 상태와 Gate는 [`workstreams.md`](workstreams.md), 전체 구조는 [`architecture.md`](architecture.md)를 참고한다.

---

## Q3 핵심 성공 기준

- [ ] Node-local P/D Cell이 기존 integrated deployment를 깨지 않고 운영 Chart에 통합된다.
- [x] PR #2 이전 커밋에서 P1:D1 배포와 LiteLLM/Anthropic 기본 data path가 확인된다.
- [ ] PR #4 HEAD와 source-built Mooncake image에서 Qwen3.6-27B P1:D1이 재현된다.
- [ ] P/D Cell의 LiteLLM / Global Router / metrics / failure recovery 경로가 E2E로 검증된다.
- [ ] Kimi-K3가 망A H200 멀티노드 환경에서 재현 가능한 운영 profile로 정립된다.
- [ ] Multi-node 장애와 engine process failure를 Pod Ready 이상의 health 기준으로 탐지/복구할 수 있다.
- [ ] vLLM 0.28.0-cu129이 주요 모델/P-D/connector Gate를 통과하고 0.27.x 대비 승격 여부가 결정된다.
- [ ] Production Stack custom repository가 upstream fork + thin overlay workflow로 정착한다.
- [ ] 신규 모델 배포 결과가 일회성 명령이 아니라 Model/Runtime Profile로 축적된다.
- [ ] KV Cache / Stateful Conversation / MOC가 차기 구현 시 현재 Serving Stack을 다시 뒤엎지 않도록 interface boundary를 확정한다.

## 상세 문서 연결

- [vLLM Stack 진행 계획](../../../vllm-stack/2026-08-18-%EC%A7%84%ED%96%89%EA%B3%84%ED%9A%8D.md)
- [P/D Disaggregation Master Plan](../../../vllm-stack/pd-disaggregation/2026-08-18-vllm-stack-pd-disaggregation-master-plan.md)
- [Node-local P/D Cell 상세 계획](../../../vllm-stack/pd-disaggregation/2026-08-18-node-local-pd-cell-0.1.8-plan.md)
- [Mooncake 0.3.10-post2 폐쇄망 Source Build](../../../vllm-stack/pd-disaggregation/2026-08-24-mooncake-0.3.10-post2-offline-build.md)
- [Model Serving Validation Contract](../../../vllm-stack/model-serving-validation.md)
- [vLLM 0.28.0 Migration](../../../vllm-stack/2026-09-03-vllm-0.28-migration.md)
- [API Routing Contract](../../../vllm-stack/api-routing-contract.md)
- [Stateful Conversation / Agentic API](../../../vllm-stack/stateful-conversation-architecture.md)
- [Inference Serving Optimization](../../../study/inference-serving-optimization/README.md)
- [MOC Capability Backlog](../../../moc/README.md)
- [DeepSeek-V4 모델/장애 조사](../../../models/deepseek-v4/)
- [Kimi-K3 모델 중심 학습 경로](../../../models/kimi-k3/)
- [Study Notes](../../../study/)
