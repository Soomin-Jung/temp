# 작업 메모 / 계획 저장소

이 저장소는 ChatGPT와 논의한 **중요 계획, 기술 검토 결과, 작업 순서, 결정사항**을 빠르게 기록해 두는 용도입니다.

> 현재 이 저장소는 **공개(public)** 상태입니다. 회사 내부 IP, 호스트명, 실제 운영 경로, 계정/토큰/비밀값, 상세한 사내 구성정보는 기록하지 않습니다.

## 디렉터리 기준

- `vllm-stack/` : vLLM Production Stack 커스텀, 버전 이관, 배포 구조
- `pd-disaggregation/` : Prefill/Decode 분리 배포, Cell/Fabric 구조, 장애 복구
- `multi-node/` : 멀티노드 vLLM, Ray 대체, native multiprocess
- `kv-cache/` : KV Cache, connector, offloading, cache-aware routing
- `moc/` : MOC 설계/구현 관련 중요 결정
- `study/` : 공부하면서 남길 가치가 있는 핵심 정리
- `misc/` : 임시 메모

## 운영 원칙

1. 중요한 논의가 끝나면 해당 주제 디렉터리에 요약 문서를 남깁니다.
2. 계획은 현재 상태 / 다음 단계 / 검증 기준 / 보류사항을 분리해 기록합니다.
3. 실제 구현 저장소와 연결되는 경우 관련 저장소, PR, 커밋 번호를 적습니다.
4. 완료된 계획도 삭제하지 않고 결과를 덧붙여 이력으로 남깁니다.
5. 공개 저장소인 동안에는 민감한 내부정보를 기록하지 않습니다.
