# 작업 메모 / 계획 저장소

## 디렉터리 기준

- `roadmap/` : 연도/분기별 주요 과제, 목표 아키텍처, Workstream 현황을 보는 상위 인덱스
- `vllm-stack/` : vLLM Production Stack 커스텀, 버전 이관, P/D Disaggregation, 배포·검증 구조
- `multi-node/` : 멀티노드 vLLM, Ray 대체, native multiprocess
- `kv-cache/` : KV Cache, connector, offloading, cache-aware routing
- `moc/` : MOC 설계/구현 관련 중요 결정
- `study/` : 공부하면서 남길 가치가 있는 핵심 정리
- `models/` : 모델별 architecture, checkpoint, 배포 호환성, 장애 조사
- `weekly-report/` : vLLM 생태계, 오픈웨이트 모델, LLM serving·최적화·논문 주간 보고서
- `docs/` : 저장소 작성·렌더링·검증 규칙
- `misc/` : 임시 메모

## Roadmap

- [2026 Q3 — LLM E2E Platform 확장](roadmap/2026/Q3/README.md)
  - [Q3 Target Architecture](roadmap/2026/Q3/architecture.md)
  - [Q3 Workstreams](roadmap/2026/Q3/workstreams.md)

## 주요 문서

### Git / GitHub Collaboration & Automation

- [Git & GitHub: 현업 협업과 자동화 학습 지도](study/git-github/README.md)
  - [Git object와 commit DAG](study/git-github/01-git-object-model-and-dag.md)
  - [Branch, merge, rebase](study/git-github/03-branches-merge-and-rebase.md)
  - [Pull Request, review, merge](study/git-github/05-pull-requests-review-and-merge.md)
  - [동기화, conflict, stacked PR](study/git-github/06-sync-conflicts-and-stacked-prs.md)
  - [되돌리기, 복구, 디버깅](study/git-github/07-undo-recovery-and-debugging.md)
  - [GitHub Actions와 CI](study/git-github/09-github-actions-and-ci.md)
  - [Delivery, release, 공급망 보안](study/git-github/10-delivery-release-and-security.md)
  - [Platform Engineering 실전](study/git-github/11-platform-engineering-playbook.md)
  - [Hands-on lab](study/git-github/12-hands-on-labs.md)
  - [초대형 Open Source 기여 생애주기](study/git-github/13-large-open-source-contribution-lifecycle.md)
  - [vLLM 기여 절차와 실제 PR 해부](study/git-github/14-vllm-contribution-and-pr-anatomy.md)
  - [장기 PR의 upstream 동기화](study/git-github/15-long-running-pr-upstream-sync.md)
  - [명령·사고 대응 runbook](study/git-github/90-command-and-incident-runbook.md)

### Attention Architecture / LLM Sequence Modeling

- [Attention Architecture Study Guide](study/attention/README.md)
  - [수학적 기반과 Transformer Block](study/attention/00-foundations.md)
  - [Full Softmax Attention — MHA, MQA, GQA와 KV Cache](study/attention/01-full-attention.md)
  - [MLA와 Low-Rank Attention](study/attention/02-mla-low-rank-attention.md)
  - [Linear/Recurrent Attention — DeltaNet, GDN, KDA](study/attention/03-linear-recurrent-attention.md)
  - [Local, Sparse, Compressed, Hybrid Attention](study/attention/04-local-sparse-compressed-hybrid.md)
  - [Kimi-K3 — KDA, Gated MLA, Attention Residuals](study/attention/10-kimi-k3.md)
  - [Qwen — GQA에서 Gated DeltaNet Hybrid까지](study/attention/11-qwen.md)
  - [DeepSeek — MLA에서 DSA, CSA/HCA까지](study/attention/12-deepseek.md)
  - [다른 주요 LLM의 Attention 설계 비교](study/attention/13-other-major-models.md)
  - [Serving Engineer 관점 — Cache, State, Kernel, Prefix Cache, 분산 추론](study/attention/90-serving-engineer-view.md)
  - [논문 읽기 순서, 수식 Cheat Sheet, 용어집](study/attention/99-papers-and-glossary.md)

### vLLM Production Stack / P-D Disaggregation

- [vLLM Stack 문서 인덱스](vllm-stack/README.md)
- [vLLM Stack 진행 계획](vllm-stack/2026-08-18-진행계획.md)
- [vLLM 0.28.0 Migration & KV Connector Compatibility](vllm-stack/2026-09-03-vllm-0.28-migration.md)
- [Chat / Messages / Responses API Routing Contract](vllm-stack/api-routing-contract.md)
- [Stateful Conversation / Agentic API Architecture](vllm-stack/stateful-conversation-architecture.md)
- [P/D Disaggregation 인덱스](vllm-stack/pd-disaggregation/README.md)
  - [P/D Disaggregation Master Plan](vllm-stack/pd-disaggregation/2026-08-18-vllm-stack-pd-disaggregation-master-plan.md)
  - [Node-local P/D Cell 0.1.8 현재 구현 계획](vllm-stack/pd-disaggregation/2026-08-18-node-local-pd-cell-0.1.8-plan.md)
  - [Mooncake 0.3.10-post2 폐쇄망 Source Build](vllm-stack/pd-disaggregation/2026-08-24-mooncake-0.3.10-post2-offline-build.md)
- [Model Serving Validation Contract](vllm-stack/model-serving-validation.md)

### MOC Operations Control Plane

- [MOC 현재 구현 경계와 Business Capability Backlog](moc/README.md)

### GPU Architecture / CUDA

- [GPU Architecture: 구조에서 LLM Serving 병목까지](study/gpu-architecture/README.md)
- [Study 인덱스](study/README.md)

### Speculative Decoding / Serving Optimization

- [Speculative Decoding Study 인덱스](study/speculative-decoding/README.md)
  - [전통 SD, MTP, DFlash, DSpark의 구조적 차이](study/speculative-decoding/2026-08-18-speculative-decoding-mtp-dspark.md)
  - [vLLM SD Token Budget Deep Dive — MBT, scheduled tokens, method별 draft slots](study/speculative-decoding/2026-09-03-vllm-speculative-decoding-token-budget-deep-dive.md)
- [Inference Serving Optimization Mental Model](study/inference-serving-optimization/README.md)
- [Scheduler Budget, Speculative Decoding & CUDA Graph](study/inference-serving-optimization/01-scheduler-budget-spec-decode-cudagraph.md)

### DeepSeek-V4

- [DeepSeek-V4 Architecture / Training / Serving Deep Dive](models/deepseek-v4/README.md)
- [개선판의 DSpark checkpoint와 vLLM 구현 차이](models/deepseek-v4/2026-08-18-dspark-checkpoint-and-vllm-implementation.md)
- [vLLM 0.27.x DeepGEMM SM90 CUDA IMA 분석](models/deepseek-v4/2026-08-18-vllm-0.27-deepgemm-sm90-cuda-ima.md)

### GLM-5.2

- [GLM-5.2 Architecture / Agentic Training / Systems Deep Dive](models/glm-5.2/README.md)

### Kimi K3 / Kimi Architecture Lineage

- [Kimi K3 — Model Architecture & Lineage Study Guide](models/kimi-k3/README.md)
  - [Kimi 계보와 K3 학습 지도](models/kimi-k3/00-lineage-and-study-map.md)
  - [Kimi K2 — K3의 Large-Scale Foundation](models/kimi-k3/01-kimi-k2-foundation.md)
  - [Kimi Linear — KDA Hybrid에서 K3로 가는 다리](models/kimi-k3/02-kimi-linear-bridge.md)
  - [Sparse MoE → LatentMoE → Stable LatentMoE](models/kimi-k3/03-moe-latentmoe-stable-latentmoe.md)
  - [Attention Residuals — Depth Mixing](models/kimi-k3/04-attention-residuals-depth-mixing.md)
  - [Native Multimodal — Kimi-VL, K2.5, MoonViT-V2](models/kimi-k3/05-native-multimodal-lineage.md)
  - [Optimization / Scaling / Pretraining — Muon 계보](models/kimi-k3/06-optimization-scaling-and-pretraining.md)
  - [Post-Training / Agentic RL / MTP / Speculative Decoding](models/kimi-k3/07-posttraining-agentic-rl-and-speculative.md)
  - [K3 Architecture Reconstruction — 2.8T 구조 직접 계산](models/kimi-k3/08-k3-architecture-reconstruction.md)
  - [K3 Systems & Serving — EP, CP, Hybrid Cache, P/D](models/kimi-k3/09-k3-systems-and-serving.md)
  - [B300 8-GPU Single-Node Feasibility](models/kimi-k3/2026-08-24-b300-single-node-feasibility.md)
  - [논문 읽기 순서, 핵심 수식, 용어집](models/kimi-k3/99-papers-and-glossary.md)

### LLM Serving Weekly Report

- [주간 보고서 인덱스와 검증 정책](weekly-report/README.md)
- [주간 보고서 템플릿](weekly-report/_TEMPLATE.md)

### 문서 품질

- [Markdown 수식·링크·렌더링 규칙](docs/markdown-rendering.md)

## 운영 원칙

1. 중요한 논의가 끝나면 해당 주제 디렉터리에 요약 문서를 남깁니다.
2. 연도/분기별 우선순위와 의존성은 `roadmap/`에서 관리하고 상세 설계는 주제 디렉터리에 둡니다. 현재 구현 상태 인덱스는 `docs/context/platform-state.md`에서 관리하되 루트 README의 전면 진입점으로 노출하지 않습니다.
3. 계획은 현재 상태 / 다음 단계 / 검증 기준 / 보류사항을 분리해 기록합니다.
4. 실제 구현 저장소와 연결되는 경우 관련 저장소, PR, 커밋 번호를 적습니다.
5. 완료된 계획도 삭제하지 않고 결과를 덧붙여 이력으로 남깁니다.
6. 공개 저장소인 동안에는 민감한 내부정보를 기록하지 않습니다.
7. display math는 `$$` 대신 GitHub 공식 fenced `math` 문법을 사용하고, 변경 후 `node tools/validate_markdown.mjs .`로 전체 문서를 검사합니다.
