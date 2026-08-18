# 작업 메모 / 계획 저장소


## 디렉터리 기준

- `vllm-stack/` : vLLM Production Stack 커스텀, 버전 이관, 배포 구조
- `pd-disaggregation/` : Prefill/Decode 분리 배포, Cell/Fabric 구조, 장애 복구
- `multi-node/` : 멀티노드 vLLM, Ray 대체, native multiprocess
- `kv-cache/` : KV Cache, connector, offloading, cache-aware routing
- `moc/` : MOC 설계/구현 관련 중요 결정
- `study/` : 공부하면서 남길 가치가 있는 핵심 정리
- `models/` : 모델별 architecture, checkpoint, 배포 호환성, 장애 조사
- `misc/` : 임시 메모

## 주요 문서

### Speculative Decoding

- [전통 SD, MTP, DFlash, DSpark의 구조적 차이](study/speculative-decoding/2026-08-18-speculative-decoding-mtp-dspark.md)

### DeepSeek-V4

- [개선판의 DSpark checkpoint와 vLLM 구현 차이](models/deepseek-v4/2026-08-18-dspark-checkpoint-and-vllm-implementation.md)
- [vLLM 0.27.x DeepGEMM SM90 CUDA IMA 분석](models/deepseek-v4/2026-08-18-vllm-0.27-deepgemm-sm90-cuda-ima.md)

## 운영 원칙

1. 중요한 논의가 끝나면 해당 주제 디렉터리에 요약 문서를 남깁니다.
2. 계획은 현재 상태 / 다음 단계 / 검증 기준 / 보류사항을 분리해 기록합니다.
3. 실제 구현 저장소와 연결되는 경우 관련 저장소, PR, 커밋 번호를 적습니다.
4. 완료된 계획도 삭제하지 않고 결과를 덧붙여 이력으로 남깁니다.
5. 공개 저장소인 동안에는 민감한 내부정보를 기록하지 않습니다.
