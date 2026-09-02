# Repository Structure and Document Lifecycle

업데이트: 2026-09-03 KST

이 문서는 `temp` 저장소의 정보 구조와 문서 수명주기를 정의한다. 이 저장소는 더 이상 단순 작업 메모가 아니라 LLM serving engineering 지식을 축적하는 knowledge base로 취급한다.

## 1. 최상위 디렉터리의 책임

| 경로 | 책임 | 넣지 않는 것 |
|---|---|---|
| `study/` | 모델을 가로지르는 개념, mental model, 원리, 실습 | 특정 checkpoint만의 운영 기록 |
| `models/` | 모델 family별 architecture, training, serving 특성 | 범용 vLLM 운영 원리의 반복 설명 |
| `vllm-stack/` | serving platform의 구현 계약, migration, runtime integration | 순수 모델 이론 |
| `roadmap/` | 분기 목표, 상태, 의존성, 종료 기준 | 상세 기술 deep dive |
| `docs/context/` | 현재 플랫폼 상태와 과거 의사결정 맥락 | 개별 기술 튜토리얼 |
| `docs/conventions/` | 저장소 작성·검증 규칙 | 운영 사고 기록 |
| `docs/operations/` | 플랫폼 공통 운영 reference | 모델별 실험 노트 |
| `docs/tooling/` | 개발·운영 도구 사용 및 troubleshooting | serving architecture |
| `moc/` | MOC control-plane 설계와 capability | data-path 구현 세부 |
| `weekly-report/` | 주간 생태계 변화 추적 | 장기 canonical 설명 |
| `tools/` | 저장소 검증/자동화 코드 | 문서 본문 |

## 2. Canonical 문서와 기록성 문서를 분리한다

### Evergreen / canonical

현재도 반복해서 참조하는 지식은 의미 기반의 안정적인 파일명을 사용한다.

예:

- `scheduler-token-budget.md`
- `api-routing-contract.md`
- `node-local-pd-cell-vllm-stack-0.1.8.md`
- `migrations/vllm-0.28.md`

날짜는 파일명의 identity가 아니라 문서 본문의 `업데이트` 또는 Git history로 관리한다.

### Time-bound record

특정 시점의 사실을 보존하는 문서는 날짜를 유지한다.

예:

- incident
- runtime validation
- benchmark snapshot
- historical plan
- 특정 version에서 발생한 compatibility failure

이런 문서는 `history/`, `records/`, `notes/` 아래에 둔다.

## 3. 날짜를 파일명에 넣는 기준

날짜를 넣는다:

- 그 날짜에 발생한 사건 자체가 문서의 정체성일 때
- 당시 판단을 나중에 재현해야 할 때
- 결과가 최신 상태로 자동 승격되어서는 안 될 때

날짜를 넣지 않는다:

- 계속 갱신될 architecture 설명
- study chapter
- current operational contract
- migration guide의 대상 version이 이미 파일명에 포함될 때

## 4. 문서를 쪼개는 기준

문서 길이만으로 나누지 않는다. 다음 중 하나가 성립할 때 분리한다.

1. **독립적인 검색 의도**가 있다.
2. **업데이트 주기**가 다르다.
3. **source of truth**가 다르다.
4. 한 부분만 version-specific이고 나머지는 evergreen이다.
5. 장애 기록과 정상 architecture 설명이 섞여 current state를 오해하게 만든다.

반대로 하나의 개념을 단계적으로 이해해야 하는 study chapter는 길더라도 흐름을 유지한다.

## 5. 중복 설명의 허용 범위

같은 개념을 여러 위치에서 완전히 재서술하지 않는다.

- `study/`: 일반 원리와 cross-model 비교
- `models/`: 해당 모델에 원리를 적용한 구체적 shape, config, runtime 의미
- `vllm-stack/`: 실제 serving runtime/API/connector 계약
- `roadmap/`: 현재 추진 여부와 의존성만

필요한 배경은 짧게 요약하고 canonical 문서로 링크한다.

## 6. 상태 정보의 소유권

현재 플랫폼 상태의 canonical source는 `docs/context/platform-state.md`다.

- roadmap은 목표와 program status를 요약한다.
- 각 영역 README는 navigation과 영역별 결정 경계를 제공한다.
- incident/history 문서는 현재 상태를 선언하지 않는다.

동일한 현재 상태를 여러 문서에 복사하면 시간이 지나며 서로 어긋나므로 피한다.

## 7. README의 역할

각 디렉터리 README는 다음만 담당한다.

1. 이 디렉터리의 책임
2. 읽는 순서
3. canonical 문서와 기록성 문서의 구분
4. 인접 영역으로 가는 링크

본문 deep dive를 README에 다시 복제하지 않는다.

## 8. 번호 체계

학습/모델 chapter는 다음 패턴을 권장한다.

- `00-`: mental model / lineage / foundation
- `01-89`: 주제별 chapter
- `90-`: runbook 또는 특수 관점이 정말 chapter일 때만 사용
- `99-`: glossary / references / source map

과거 기록을 번호 `90`으로 가장하지 않고 `notes/` 또는 `history/`로 분리한다.

## 9. 파일명 규칙

- 기본은 lowercase kebab-case
- 일반 문서의 한글 파일명은 새로 만들지 않는다.
- product/version identifier는 의미 보존을 위해 허용한다.
- `plan`, `master-plan`, `final`, `new`, `latest` 같은 상대적 이름은 가급적 피한다.
- current contract는 대상과 범위를 파일명에 명시한다.
- historical plan은 `history/YYYY-MM-DD-...`로 보존한다.

## 10. 변경 전 체크리스트

새 문서를 만들기 전에 확인한다.

- 기존 canonical 문서의 section으로 충분하지 않은가?
- `study`, `models`, `vllm-stack` 중 누가 이 지식의 owner인가?
- 이 문서는 6개월 뒤에도 같은 이름이 자연스러운가?
- 날짜가 정말 identity인가?
- current state와 historical evidence가 섞이지 않았는가?
- README와 상위 index에서 발견 가능한가?
