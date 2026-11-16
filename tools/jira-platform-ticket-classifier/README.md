# Jira 플랫폼 운영 티켓 분류기

Jira Data Center 프로젝트의 티켓을 수집한 뒤 **OpenAI 호환 내부 LLM**으로 분류하여 Excel에서 검토 가능한 파이프(`|`) 구분 파일로 내보내는 독립 실행 스크립트입니다.

실제 조직명·내부 주소·프로젝트 키·토큰·티켓 원문·분류 결과는 이 저장소에 포함하지 않습니다. **이 저장소는 공개 저장소입니다.** 실제 데이터는 접근이 승인된 내부 환경에서만 취급하세요.

## 실행 환경

- Python 3.10+
- `pip install requests openai`
- Jira Data Center REST API 접근 권한
- OpenAI 호환 `/v1/chat/completions` 내부 모델 엔드포인트

### 1. 환경변수

Linux / macOS:

```bash
export JIRA_BASE_URL='https://jira.example.internal'
export JIRA_TOKEN='REPLACE_WITH_TOKEN'
export LLM_BASE_URL='https://llm.example.internal/v1'
export LLM_API_KEY='REPLACE_WITH_TOKEN'
export LLM_MODEL='pro'
```

- Jira Data Center PAT이면 기본 `Authorization: Bearer ...`를 사용합니다.
- Jira가 Basic 인증을 사용하면 `export JIRA_USERNAME='my-user'`를 추가하고 `--jira-auth basic` 옵션을 사용하세요. 이때 `JIRA_TOKEN`에는 해당 환경이 요구하는 암호/토큰을 넣습니다.
- 사내 API Gateway가 `X-API-Key` 헤더를 요구하면 `--jira-auth x-api-key`를 사용하세요.
- `LLM_BASE_URL`은 **OpenAI SDK의 base_url**입니다. 보통 URL이 `/v1`로 끝나야 합니다. 코드에서 `/chat/completions`를 자동으로 덧붙입니다.
- Jira(`requests.Session.verify=False`)와 내부 LLM(`openai.DefaultHttpxClient(verify=False)`) 양쪽 모두 **SSL/TLS 서버 인증서 검증이 비활성화**되어 있습니다. 사설 CA를 설정할 수 없는 내부 시험 환경을 전제로 하며, 서버 인증이 되지 않아 중간자 공격에 취약할 수 있습니다.
- 원본 토큰을 소스 파일에 하드코딩하거나 결과 디렉터리를 Git에 커밋하지 마세요.

### 2. 실행

처음 20건으로 분류 동작 및 프롬프트 확인:

```bash
python classify_jira.py --project DEMO --limit 20 --workers 10 --output ./local-results
```

전체 처리(캐시가 있으면 재사용):

```bash
python classify_jira.py --project DEMO --workers 10 --output ./local-results
```

진행 메시지 예시(터미널에서 실시간 갱신):

```text
LLM 워커 상태 | 활동 3/3
W01 [   ███      ] LLM 요청    SAMPLE-101      시도 1          경과 00:00:23 | 처리 5 | 오류 0
W02 [      ███   ] 429 대기    SAMPLE-102      6초 남음        경과 00:00:11 | 처리 4 | 오류 0
W03 [███         ] 응답 검증   SAMPLE-103      JSON·원문 대조   경과 00:00:03 | 처리 7 | 오류 0
전체 [███████████░░░░░░░░░░░░░░░░░░] 310/800 (38.8%) | 신규 성공 16 | 실패 0 | 남음 490 | ETA 00:12:30
```

- **각 워커**: 현재 처리 중인 티켓 키, 모델 요청/응답 검증/429 대기/재시도/오류 상태, 티켓 경과 시간, 누적 처리 건수.
- **워커 막대는 활동 표시(움직이는 indeterminate bar)**입니다. 한 번의 비스트리밍 LLM 요청 내부 진행률은 알 수 없으므로 가짜 0~100%를 표시하지 않습니다.
- **하단 전체 막대는 실제 시도 완료 건수**를 기준으로 계산합니다. 실패한 호출도 시도 완료로 집계하되 성공/실패 건수는 따로 보여줍니다. 기존 체크포인트 완료 건수도 포함합니다.
- ETA는 최근 완료 속도로 추정합니다. 최초 처리 중이거나 표본이 부족하면 `--:--`로 표시됩니다.
- TTY에서는 한 화면을 갱신하며, 로그 파일 리다이렉션 또는 터미널 높이가 부족한 환경에서는 일정 간격으로 워커별 상태와 전체 진행률을 줄 단위로 출력합니다.

표시 모드 변경:

```bash
# 자동(TTY 실시간 / 비TTY 일반 로그), 기본값
python classify_jira.py --project DEMO --progress auto

# ANSI 실시간 대시보드 강제
python classify_jira.py --project DEMO --progress live

# SSH 로그·파이프 출력: 10초마다 워커별 상태와 전체 진행률 출력
python classify_jira.py --project DEMO --progress plain --progress-log-interval 10
```

Jira 수집 단계는 기존처럼 한 줄 진행률을 사용하고, **LLM 분류 단계에서만 워커별 대시보드가 나타납니다.**

**주의:** `--limit 20`은 Jira 데이터를 **전부 수집한 뒤** 앞의 20건만 LLM으로 분류합니다. 전체 수집량 확인, 댓글 처리, API 제한 테스트가 목적일 때 유용합니다.

### 3. 추론 강도와 분류 일관성

기본 모델은 환경변수 `LLM_MODEL`(예: `pro`)을 사용하며, **추론 강도 `max`를 기본 요청**합니다.

```bash
# 권장: 본격 처리 전 1건으로 내부 서버가 max를 지원하는지 점검
python classify_jira.py --project DEMO --limit 1 --workers 1 --output ./local-results

# 기본값은 max. 내부 API가 이 필드를 거절할 때만 명시적으로 끄기
python classify_jira.py --project DEMO --reasoning-effort off --output ./local-results

# 출력 토큰 한도가 추론/답변에 부족한 경우
python classify_jira.py --project DEMO --max-output-tokens 8192 --output ./local-results
```

- OpenAI 호환 `/chat/completions`의 요청 본문에 `reasoning_effort: "max"`를 넣습니다. **내부 서버의 모델·API Gateway가 `max`를 실제 해석/적용하는지는 별도 확인이 필요**합니다. 단순히 HTTP 200이 나왔다고 실제 추론 강도가 바뀌었다고 단정할 수 없습니다. 400이면 최초 호출에서 중지하고 `--reasoning-effort off` 등을 안내합니다.
- `max`에서 **시간 지연과 생성 토큰 수 증가**가 발생할 수 있어 출력 상한 기본값을 4,096으로 조정했습니다. `LLM_TIMEOUT` 기본값은 240초입니다. 일부 서버는 `max_tokens`에 추론 토큰을 포함해 JSON 답변이 잘릴 수 있습니다.
- 매 티켓의 분류 결과를 다음 요청에 넣는 **최근 3건 동적 주입은 사용하지 않습니다**. 병렬 워커의 종료 순서가 달라지고 오분류가 누적될 위험이 있기 때문입니다.
- 대신 선택적으로 **사람이 확인한 고정 예시 3건**(포함/제외/검토 각 1건)을 모든 요청 앞에 똑같이 넣을 수 있습니다.

```bash
# 저장소의 예제는 실제 데이터가 아닌 합성 사례입니다.
python classify_jira.py --project DEMO --limit 20 --workers 4 \
  --examples-file ./few_shot_examples.sample.json --output ./local-results

# 실제 운영 시: 사람이 검증한 3건을 내부 파일로 작성. Git에 올리지 마세요.
python classify_jira.py --project DEMO --workers 10 \
  --examples-file ./few_shot_examples.local.json --output ./local-results
```

예시 JSON은 `[{"ticket": {...}, "result": {...}}, ...]` 배열입니다. `ticket`은 분류기에 전달되는 정규화된 Jira 필드(`key`, `title`, `description`, `comments` 등), `result`는 분류 JSON 전체 스키마여야 합니다. 스크립트는 예시 개수(정확히 3개), INCLUDE/EXCLUDE/REVIEW 균형, INCLUDE의 원문 인용 일치 여부를 검증합니다.

**권장:** 이미 오판이 많았던 경계 사례를 세 개만 골라 사람이 정답을 확정하세요. 예를 들어 (1) 사용자 오류였지만 실제 공통 API 서버를 수정해 해결한 **포함**, (2) 개발자 PC·클라이언트 설정만 안내한 **제외**, (3) 플랫폼 장애 의심만 있고 조치 기록이 없는 **검토**입니다. 합성 예시는 형식 시연용이며 정확도 향상을 보장하지 않습니다.

설정별 재현성을 위해 체크포인트에는 분류 스키마, 모델명, 추론 강도, 고정 예시의 SHA-256 축약 지문을 저장합니다. 하나라도 바뀌면 Jira 원본 캐시는 재사용하되 **LLM 분류는 새로 수행**하며, 예전 결과는 남습니다. 같은 설정으로 재실행하면 처리 완료 건을 건너뜁니다.

추론 강도나 예시 도입 전후를 판단하려면 **사람이 정답을 확정한 동일한 약 60~80건**(INCLUDE/EXCLUDE/REVIEW 균형 + 어려운 경계 사례)에 대해 비교하세요. 특히 INCLUDE의 오포함(기술 조치가 없는데 포함), 중요 기술 사례의 누락, 동일 티켓 재실행 시 판정 일치율을 보세요. 전체 포함 건수가 늘었다고 정확도가 높아진 것은 아닙니다.

### 4. 출력 파일 (증빙 중심 v3)

기존 분류 파일은 보존하고 새 결과에는 `_evidence-v3` 접미사가 붙습니다.

| 파일 | 내용 |
| --- | --- |
| `all_tickets_evidence-v3.psv` | 전체 티켓(모델 분석 실패 시 검토필요 표시) |
| `selected_tickets_evidence-v3.psv` | 직접적인 플랫폼 대응 성과 후보 |
| `high_priority_tickets_evidence-v3.psv` | 원문 증빙·조치·효과가 비교적 명확한 1차 심층검토 후보 |
| `excluded_tickets_evidence-v3.psv` | 일반 사용자 문의 등 제외 |
| `review_tickets_evidence-v3.psv` | 실제 수행 여부 또는 근거 확인이 필요한 건 |
| `summary_evidence-v3.json` | 분류 건수 및 지원 주장별 건수 |
| `jira_issues.jsonl` | Jira 원본 검색 캐시(민감 데이터 가능) |
| `classifications.jsonl` | 이전 버전과 신규 버전을 함께 담는 체크포인트 |
| `errors.jsonl` | LLM 실패 건. 다음 실행 시 다시 시도 |

Excel에서 **데이터 → 텍스트/CSV에서** 열고 구분자를 `|`로 선택합니다. 인코딩은 UTF-8 BOM입니다. 3줄 요약은 별도 세 열이며, 줄바꿈과 원문의 `|`는 Excel 단일 행에 맞도록 정리합니다.

CSV 열:

```text
티켓번호|생성일|티켓제목|요약1|요약2|요약3|관련성|주요유형|세부항목|지원주장|실제기술조치|확인된운영결과|직접운영권연계|원문근거|근거출처|시간증빙원문|보고우선순위|추가확인사항|판단근거|신뢰도
```

### 5. 분류 및 경영진 보고용 주장

먼저 기술적으로 **INCLUDE / EXCLUDE / REVIEW**를 판단한 다음, INCLUDE에 해당하는 경우에만 아래 중 실제 활동으로 뒷받침되는 주장을 0~3개 연결합니다.

| 지원 주장 | 입증할 실제 업무 |
| --- | --- |
| 장애 원인 분석·복구 | 플랫폼 장애의 기술 분석, 완화, 복구 |
| 모델·Serving 변경 대응 | 모델 교체, 서빙 파라미터·배포 구조 변경 |
| 다계층 E2E 문제 해결 | Gateway, Proxy, Router, Engine 등 복수 계층의 연계 문제 분석 |
| 특수 워크로드 수용 | 특수 업무를 위해 공통 API·Serving을 변경 |
| 신규 아키텍처·플랫폼 구축 | PDD 등 아키텍처나 공통 플랫폼의 실제 구축·개선 |

구분은 **업무 실적**에 대한 것이지, AX와의 비교 평가나 추가 GPU의 필수성을 증명하는 지표가 아닙니다. 별도 검증 없이 "더 빠르다", "장애가 없다"는 주장을 만들지 않습니다.

다음 필드를 특히 중요하게 취급합니다.

- **실제기술조치**: 티켓에서 확인된 플랫폼 팀의 작업. 없으면 "확인되지 않음".
- **확인된운영결과**: 조치 후 검증된 결과. 단순 티켓 Close만으로 복구되었다고 판단하지 않음.
- **원문근거 / 근거출처**: 설명이나 댓글에 실제 존재하는 문구를 그대로 발췌하고, 스크립트가 원문과 일치하는지 검증.
- **시간증빙원문**: 시작·복구·운영 반영 시간에 대한 명시적 문구가 있을 때만 기록. 비어 있으면 처리 시간 미확인.
- **직접운영권연계**: 실제 배포·설정·복구 권한 행사 확인 여부. 독립 GPU 추가 필요성과 동일하지 않음.
- **보고우선순위**: 상/중/하 1차 검토 순위. **상**이라고 해서 자동으로 최종 보고서에 채택하면 안 됨.
- **추가확인사항**: TAT, 사용자 영향, 실제 배포 시각 등 사람의 확인이 필요한 항목.

일반 Client/IDE/CLI 설정 문의는 제외하되, 클라이언트 증상을 해결하기 위해 서버 측 플랫폼을 실제 분석·수정했다면 포함합니다. 신뢰도 0.65 미만의 포함은 검토필요로 내려갑니다.

**검증 순서 추천:** INCLUDE 30건, EXCLUDE 20건, REVIEW 20건으로 표본 정확도 검토 → 전체 INCLUDE 요약 탐색 → 우선검토 후보 약 20~40건 원문·배포/모니터링 로그 대조 → 경영진용 대표 사례 3~5건 선별. 중요 사례는 시각과 효과를 별도로 검증해야 합니다.


### 6. 실행 제어

```bash
# 모델 분류 정책을 바꾸었을 때 전체 다시 분류 (Jira 캐시는 재사용)
python classify_jira.py --project DEMO --output ./local-results --reclassify

# Jira에서 최신 이슈와 댓글 다시 수집
python classify_jira.py --project DEMO --output ./local-results --refresh-jira

# Jira 추가 댓글 호출을 생략하고 빠르게 시범 실행
python classify_jira.py --project DEMO --output ./local-results --limit 20 --no-full-comments

# 보수적으로 동시 호출 수 조절
python classify_jira.py --project DEMO --output ./local-results --workers 4
```

Jira는 페이지 단위로 순차 수집하며, 모델 추론만 기본 10개 **스레드**로 병렬 호출합니다. API 호출은 I/O 중심이므로 멀티프로세스보다 스레드 풀이 적합합니다.

**429 처리:** Jira와 LLM 모두 HTTP 429 수신 시 최소 6초를 기다린 뒤 최대 100회 재시도합니다. LLM 워커는 동일한 쿨다운 상태를 공유해 재시도 폭주를 완화합니다. 일반적인 5xx/네트워크 오류는 짧게 제한하여 재시도합니다. 전체 사용량 제한을 위반하지 않는다는 보장은 없으므로 필요 시 `--workers`를 낮추세요.

**재시작:** Jira 수집 결과는 원자적으로 저장하고, LLM 분류는 티켓 단위로 체크포인트에 즉시 추가합니다. 중단 후 **같은 모델·추론 강도·예시**로 다시 실행하면 완료된 **v3** 분류는 건너뜁니다. 기존 v1/v2 분류는 보존하지만 새 결과에는 사용하지 않으므로 **처음 v3를 실행하면 전체를 다시 모델로 분석합니다.** `--reclassify`는 기존 체크포인트를 타임스탬프가 포함된 백업 파일로 옮긴 다음 재분류합니다. 단, 신규/수정 티켓을 반영하려면 `--refresh-jira` 후 관련 분류를 `--reclassify`해야 합니다. 단순 새로고침만으로는 이미 분류된 티켓이 자동 재분류되지 않습니다.

## 주의사항

1. 분류 결과는 기술 실적 후보의 **1차 선별 데이터**입니다. 조직의 상대적 업무 능력, 복구 시간, 실제 배포 완료를 이 CSV만으로 증명할 수는 없습니다.
2. 티켓이 처리된 실제 시간은 Jira 상태 전이·댓글·배포·모니터링 기록으로 별도 교차검증해야 합니다.
3. 데이터에는 시스템 경로, 개인정보, 내부 URL, 비밀정보가 포함될 수 있습니다. 간단한 토큰 패턴 마스킹은 완전한 비식별화 기능이 아닙니다.
4. 필요하다면 데이터 접근 권한이 있는 승인된 내부 머신에서만 실행하고 로컬 출력 파일을 저장소로 푸시하지 마세요.
