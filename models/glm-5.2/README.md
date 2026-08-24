# GLM-5.2 Deployment Context

업데이트: 2026-08-24 KST  
상태: Historical deployment decision / parked track

## 결정 요약

- GGUF 기반 vLLM 배포안은 폐기했다.
- 이후 검토 방향은 **GLM-5.2-FP8**을 Network A의 H200 multi-node + InfiniBand 환경에서 배포하는 것이다.
- 배포 기반은 당시 custom vLLM Production Stack 0.1.8이었다.
- multi-node orchestration 후보로 Ray cluster를 검토했다.

## 현재 해석

이 문서는 당시 결정의 보존용이다. 현재 플랫폼의 multi-node 기본 방향은 Ray 장기 의존보다 vLLM native multiprocess/backend와 최신 Production Stack migration을 우선 검토하므로, GLM-5.2를 다시 활성화할 때는 최신 vLLM 지원 상태와 multi-node backend를 재검증한다.

## 삭제해도 되는 과거 탐색

- GGUF 배포 가능성 검토
- GGUF vs FP8 선택 논의
- H200 multi-node/IB 배포 초안

위 내용은 이 문서의 historical context로 대체한다.
