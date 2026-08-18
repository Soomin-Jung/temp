# DeepSeek-V4

DeepSeek-V4 계열 checkpoint와 vLLM runtime/kernel 호환성 조사 문서다.

- [DSpark checkpoint와 vLLM 구현 차이](2026-08-18-dspark-checkpoint-and-vllm-implementation.md)
- [vLLM 0.27.x DeepGEMM SM90 CUDA IMA 분석](2026-08-18-vllm-0.27-deepgemm-sm90-cuda-ima.md)

배포 판단 시 모델 architecture, checkpoint metadata, vLLM implementation, DeepGEMM/CUDA kernel, GPU compute capability를 서로 다른 층으로 분리한다.
