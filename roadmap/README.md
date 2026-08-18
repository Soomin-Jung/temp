# Platform Roadmap

연도/분기별 주요 과제와 플랫폼 방향을 관리하는 상위 인덱스입니다.

상세 기술 조사와 구현 메모는 기존 주제별 디렉터리에 남기고, 이 경로에서는 **분기 목표 / 현재 상태 / 의존성 / 종료 기준**만 유지합니다.

## 구조

```text
roadmap/
└─ YYYY/
   └─ QN/
      ├─ README.md        # 분기 총괄
      ├─ architecture.md  # 해당 분기 목표 아키텍처
      └─ workstreams.md   # 주요 Workstream 상태와 Gate
```

## 원칙

1. 로드맵은 상세 설계문서를 복제하지 않는다.
2. 과제는 제품명이 아니라 플랫폼 capability 기준으로 묶는다.
3. 각 분기 문서에는 현재 상태, 다음 Gate, 선행 의존성을 남긴다.
4. 트러블슈팅 상세는 `models/`, `vllm-stack/`, `pd-disaggregation/`, `study/` 등 원본 주제 경로를 링크한다.
5. 완료된 과제는 삭제하지 않고 결과와 다음 분기 이관 여부를 기록한다.

## 연도별

- [2026 Q3 — LLM E2E Platform 확장](2026/Q3/README.md)
