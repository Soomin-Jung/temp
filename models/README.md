# Models

모델별 architecture, checkpoint, framework/runtime compatibility, 배포 topology, 장애 분석을 모은다. 공통 이론은 `study/`에 두고, 여기서는 특정 모델의 실제 config와 serving contract로 다시 연결한다.

## 인덱스

- [DeepSeek-V4](deepseek-v4/README.md)
- [Kimi-K3](kimi-k3/README.md) — architecture lineage / systems deep dive
- [GLM-5.2](glm-5.2/README.md) — MLA + DSA + IndexShare + MoE + MTP + agentic RL / EDA-agent deep dive

## 기록 형식

1. Model Profile — architecture, config, dtype, context, parser/capability
2. Architecture Reconstruction — attention/MoE/residual/speculative path를 config와 source code로 복원
3. Training & Agentic Capability — pretraining, post-training, RL, tool/agent infrastructure
4. Runtime Profile — vLLM/image/kernel, topology, flags
5. Validation Result — API, accuracy sanity, performance, failure/recovery
6. Known Issues — model/framework/kernel/topology 중 어느 층의 문제인지 분리

공통 attention/MoE/GPU 이론은 `study/`에 두고, 특정 모델 문서에서는 해당 이론이 실제 config/kernel/serving topology에 어떻게 나타나는지 연결한다.
