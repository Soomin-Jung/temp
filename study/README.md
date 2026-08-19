# Study

공부하면서 다시 찾아볼 가치가 있는 개념, 구조, 비교, 검증 절차를 모은다. 제품명과 결론만 기록하지 않고 `왜 그런가 → 내부에서 어떻게 동작하는가 → 어떤 지표로 검증하는가` 순서로 정리한다.

## 학습 모듈

### Git & GitHub

- [Git & GitHub: 현업 협업과 자동화 학습 지도](git-github/README.md)
  - [Git·GitHub 전체 구조](git-github/00-big-picture-and-mental-model.md)
  - [Git object와 commit DAG](git-github/01-git-object-model-and-dag.md)
  - [로컬 작업과 index](git-github/02-local-workflow-and-index.md)
  - [Branch, merge, rebase](git-github/03-branches-merge-and-rebase.md)
  - [Remote, fork, 분산 협업](git-github/04-remotes-forks-and-collaboration.md)
  - [Pull Request, review, merge](git-github/05-pull-requests-review-and-merge.md)
  - [동기화, conflict, stacked PR](git-github/06-sync-conflicts-and-stacked-prs.md)
  - [되돌리기, 복구, 디버깅](git-github/07-undo-recovery-and-debugging.md)
  - [팀 workflow와 repository governance](git-github/08-team-workflow-and-governance.md)
  - [GitHub Actions와 CI](git-github/09-github-actions-and-ci.md)
  - [Delivery, release, 공급망 보안](git-github/10-delivery-release-and-security.md)
  - [Platform Engineering 실전](git-github/11-platform-engineering-playbook.md)
  - [Hands-on lab](git-github/12-hands-on-labs.md)
  - [명령·사고 대응 runbook](git-github/90-command-and-incident-runbook.md)
  - [용어집과 공식 자료](git-github/99-glossary-and-references.md)

### Attention Architecture

- [Attention Architecture Study Guide](attention/README.md)
  - [수학적 기반과 Transformer Block](attention/00-foundations.md)
  - [Full Softmax Attention — MHA, MQA, GQA와 KV Cache](attention/01-full-attention.md)
  - [MLA와 Low-Rank Attention](attention/02-mla-low-rank-attention.md)
  - [Linear/Recurrent Attention — DeltaNet, GDN, KDA](attention/03-linear-recurrent-attention.md)
  - [Local, Sparse, Compressed, Hybrid Attention](attention/04-local-sparse-compressed-hybrid.md)
  - [Kimi-K3 — KDA, Gated MLA, Attention Residuals](attention/10-kimi-k3.md)
  - [모델 중심 계보 — Kimi Linear에서 Kimi-K3까지](../models/kimi-k3/README.md)
  - [Qwen — GQA에서 Gated DeltaNet Hybrid까지](attention/11-qwen.md)
  - [DeepSeek — MLA에서 DSA, CSA/HCA까지](attention/12-deepseek.md)
  - [다른 주요 LLM의 Attention 설계 비교](attention/13-other-major-models.md)
  - [Serving Engineer 관점 — Cache, State, Kernel, Prefix Cache, 분산 추론](attention/90-serving-engineer-view.md)
  - [논문 읽기 순서, 수식 Cheat Sheet, 용어집](attention/99-papers-and-glossary.md)

### GPU Architecture

- [GPU Architecture: 구조에서 LLM Serving 병목까지](gpu-architecture/README.md)
  - [GPU를 보는 다섯 개의 축](gpu-architecture/00-mental-model-and-terms.md)
  - [보드에서 SM까지](gpu-architecture/01-physical-anatomy-and-hierarchy.md)
  - [CUDA 실행 모델과 SM](gpu-architecture/02-cuda-execution-model-and-sm.md)
  - [메모리와 데이터 이동](gpu-architecture/03-memory-and-data-movement.md)
  - [NVIDIA 세대와 주요 GPU 비교](gpu-architecture/04-nvidia-architecture-generations.md)
  - [정밀도와 Tensor Core](gpu-architecture/05-numerical-precision-and-tensor-cores.md)
  - [GPU 메트릭과 관측 도구](gpu-architecture/06-metrics-and-observability.md)
  - [LLM Serving 병목 진단](gpu-architecture/07-llm-serving-bottleneck-diagnosis.md)
  - [실습과 검증 절차](gpu-architecture/08-hands-on-labs.md)
  - [GPU 통신을 보는 지도](gpu-architecture/09-gpu-communication-mental-model.md)
  - [PCIe·NVLink·NVSwitch](gpu-architecture/10-pcie-nvlink-nvswitch.md)
  - [RDMA와 GPUDirect RDMA](gpu-architecture/11-rdma-and-gpudirect-rdma.md)
  - [InfiniBand·RoCE와 GPU network 하드웨어](gpu-architecture/12-infiniband-roce-and-hardware.md)
  - [NCCL·집단통신·관측 실습](gpu-architecture/13-nccl-collectives-observability-labs.md)
  - [GPU Architecture 용어집](gpu-architecture/glossary.md)

GPU 자체는 0→8, multi-GPU와 cluster 통신은 9→13 순서로 읽는다. 운영 이슈를 진단할 때는 메트릭(6) → 병목 진단(7) → NCCL 관측(13) → IB/RoCE(12) → RDMA/GDRDMA(11) → local topology(10) 순서로 역추적한다.

### Speculative Decoding

- [전통 SD, MTP, DFlash, DSpark의 구조적 차이](speculative-decoding/2026-08-18-speculative-decoding-mtp-dspark.md)

## 작성 원칙

1. architecture, chip, SKU, form factor, system처럼 범위가 다른 용어를 분리한다.
2. 기능 지원과 workload의 실제 사용을 구분한다.
3. 용량, 활동률, 대역폭, 지연을 같은 지표로 취급하지 않는다.
4. 공식 원문과 확인일을 남기고, 공개되지 않은 구조는 추측하지 않는다.
5. 실험은 가설, 고정 조건, 변경 변수, 반증 조건을 함께 기록한다.
6. 외부에 공개할 수 없는 호스트명, 주소, 수량, 조직·프로젝트명은 기록하지 않는다.
7. display math는 `$$` 대신 fenced `math` 문법을 사용하고 [Markdown 렌더링 규칙](../docs/markdown-rendering.md)에 따라 전체 검증한다.
