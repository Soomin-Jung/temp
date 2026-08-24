# Models

모델별 architecture, checkpoint, framework/runtime compatibility, 배포 topology, 장애 분석을 모은다. 공통 이론은 `study/`에 두고, 여기서는 특정 모델의 실제 config와 serving contract로 다시 연결한다.

## 인덱스

- [DeepSeek-V4](deepseek-v4/README.md)
- [Kimi-K3](kimi-k3/README.md)
- [GLM-5.2](glm-5.2/README.md) — historical deployment decision / parked track

## 기록 형식

1. Model Profile — architecture, config, dtype, context, parser/capability
2. Runtime Profile — vLLM/image/kernel, topology, flags
3. Validation Result — API, accuracy sanity, performance, failure/recovery
4. Known Issues — model/framework/kernel/topology 중 어느 층의 문제인지 분리
