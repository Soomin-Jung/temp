# Platform Evolution and Operations History

업데이트: 2026-08-25 KST  
상태: Historical context / archived decisions

이 문서는 현재 아키텍처의 배경이 된 과거 운영 경험 중 다시 참고할 가치가 있는 것만 압축 보존한다. 현재 설계의 source of truth는 `docs/context/platform-state.md`와 영역별 canonical 문서다.

## 1. Docker / NFS xattr 제약과 Singularity 우회

과거 일부 GPU 서버 환경에서 Docker data root가 NFSv3 계열 스토리지에 위치했고, extended attribute 제약 때문에 container image pull/unpack이 실패하는 문제가 있었다.

당시 실용적인 우회는:

1. xattr 제약이 없는 Docker 환경에서 이미지를 pull/export
2. 해당 이미지를 Singularity/Apptainer 계열 artifact로 변환
3. GPU 서버에 반입해 실행

핵심 교훈은 container runtime 오류를 image 자체 문제로 단정하지 않고 **storage filesystem capability(xattr/overlayfs)와 runtime unpack 요구사항을 분리해 확인**하는 것이다.

현재 Kubernetes 기반 serving의 기본 배포 방식은 이 historical path와 다르므로, 위 방식은 fallback/legacy reference로만 본다.

## 2. 단일 Docker 서비스에서 Kubernetes 기반 플랫폼으로의 전환

초기에는 OpenWebUI, n8n, Dify 같은 애플리케이션을 폐쇄망 Docker 환경에서 직접 배포·운영하는 비중이 컸다.

이후 서비스 수와 운영 복잡도가 증가하면서 k3s/Kubernetes 및 vLLM Production Stack 기반으로 전환했고, 다음 요소가 중요해졌다.

- 선언형 deployment
- replica/health/restart 관리
- ingress/service 기반 endpoint 관리
- GPU workload scheduling
- observability와 rollout validation
- application layer와 inference runtime 분리

현재 플랫폼 설계는 이 Kubernetes 기반 운영 모델을 전제로 한다.

## 3. LMCache / KV Offload 실험

LMCache 계열 CPU/KV offload를 실제 inference 환경에서 검증한 경험이 있으며, prefix reuse가 존재하는 workload에서 TTFT 개선 가능성을 확인했다.

현재는 단순히 "LMCache가 빠르다"는 결론보다 다음을 분리해서 평가한다.

- P→D KV transfer
- shared/remote KV cache
- prefix cache hit rate
- CPU/GPU/network data movement 비용
- cache-aware routing

따라서 당시 결과는 cache/offload가 실제로 유효할 수 있다는 historical evidence로만 두고, 현재 시스템에서는 workload별 stage-gate로 재검증한다.

## 4. LiteLLM / API Compatibility Layer

과거부터 inference runtime 앞에 proxy/compatibility layer를 두고 OpenAI/Anthropic 계열 API 차이를 흡수하는 구조를 운영·검토해왔다.

주요 교훈:

- model alias와 실제 served model identity를 분리
- reasoning field, tool call, streaming semantics를 단순 text proxy로 다루지 않음
- API compatibility와 conversation-state durability는 별도 문제
- LiteLLM 같은 proxy layer와 vLLM runtime의 책임을 분리

현재 구체적인 stateful conversation 계약은 `vllm-stack/stateful-conversation-architecture.md`, 배포 검증 계약은 `vllm-stack/model-serving-validation.md`를 따른다.

## 5. Historical context 사용 원칙

이 문서의 내용은 과거 운영 맥락을 이해하기 위한 것이다.

우선순위는 다음과 같다.

1. 실제 code / PR current state
2. `docs/context/platform-state.md`
3. 영역별 canonical 문서
4. 최신 runtime validation
5. 이 historical document

오래된 탐색 기록은 current source of truth로 사용하지 않는다.
