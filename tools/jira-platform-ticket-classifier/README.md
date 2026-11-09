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

진행 메시지 예시:

```text
Jira 검색 | 완료 150/800 | 남음 650 | ETA 00:00:43
LLM 분류 | 완료 389/800 | 남음 411 | ETA 00:07:28
완료 | 전체=800 | 포함=... | 제외=... | 검토=... | 실패=0
```

**주의:** `--limit 20`은 Jira 데이터를 **전부 수집한 뒤** 앞의 20건만 LLM으로 분류합니다. 전체 수집량 확인, 댓글 처리, API 제한 테스트가 목적일 때 유용합니다.

### 3. 출력 파일

스크립트는 실행 디렉터리가 아닌 `--output` 지정 폴더에 파일을 생성합니다.

| 파일 | 내용 |
| --- | --- |
| `all_tickets.psv` | 모든 티켓(미분류·오류는 검토필요로 표기) |
| `selected_tickets.psv` | 인프라·모델 Serving 실적 관련 INCLUDE 티켓 |
| `excluded_tickets.psv` | 일반 사용자·클라이언트 문의 등 EXCLUDE |
| `review_tickets.psv` | 서버 조치가 불확실하거나 근거가 부족한 REVIEW |
| `summary.json` | 분류별 집계 건수 |
| `jira_issues.jsonl` | Jira 원본 검색 결과와 필요한 추가 댓글(민감 데이터 포함 가능) |
| `classifications.jsonl` | LLM 분류 체크포인트(재실행 시 자동 재사용) |
| `errors.jsonl` | 실패한 티켓 키와 오류 요약. 다음 실행 시 자동 재시도 |

Excel: **데이터 → 텍스트/CSV에서** 파일을 불러와 구분 기호를 `|`로 지정하고 문자 인코딩을 UTF-8로 설정합니다. BOM을 기록하므로 일반적인 Excel 한글 인식도 지원합니다.

CSV 열:

```text
티켓번호|생성일|티켓제목|요약1|요약2|요약3|관련성|주요유형|세부항목|서버측조치|판단근거|신뢰도
```

3줄 요약은 Excel에서 필터링·편집하기 편하도록 **서로 다른 세 열**로 저장합니다. 제목과 요약에 들어 있는 `|` 문자는 전각 슬래시로 치환하고, 행 내부 개행은 공백으로 변환합니다.

### 4. 핵심 분류 정책

- **INCLUDE**: 실제 Serving/GPU/Kubernetes/라우팅/모델/인증 플랫폼 문제를 분석·수정·운영한 증거가 확인된 경우
- **EXCLUDE**: 단순 설치 안내·클라이언트 설정·API 키 사용법·일반 사용자 문의 등 플랫폼 조치가 없는 경우
- **REVIEW**: 보고된 현상은 플랫폼 문제일 수 있지만 구체적인 기술 분석·변경 이력이 없거나 정보가 부족한 경우

이슈가 Claude/Codex/Dify 등 클라이언트에서 발생했다는 이유만으로 제외하지 않습니다. **실제 서버/Serving 대응 여부**가 핵심입니다. 신뢰도 0.65 미만의 INCLUDE는 REVIEW로 낮춥니다.

분류는 설명 본문뿐 아니라 **댓글의 실제 처리 경과**도 참고합니다. 분량을 제한하기 위해 초반 댓글 2개와 최근 댓글을 우선 투입하고, 각 본문은 앞·뒤를 보존하며 잘라냅니다. 긴 첨부파일의 내용은 자동으로 다운로드하지 않습니다. 높은 정확도가 필요한 사례는 담당자 검토를 권장합니다.

### 5. 실행 제어

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

**재시작:** Jira 수집 결과는 원자적으로 저장하고, LLM 분류는 티켓 단위로 체크포인트에 즉시 추가합니다. 중단 후 같은 명령을 다시 실행하면 완료된 분류는 건너뜁니다. 단, 신규/수정 티켓을 반영하려면 `--refresh-jira` 후 관련 분류를 `--reclassify`해야 합니다. 단순 새로고침만으로는 이미 분류된 티켓이 자동 재분류되지 않습니다.

## 주의사항

1. 분류 결과는 기술 실적 후보의 **1차 선별 데이터**입니다. 조직의 상대적 업무 능력, 복구 시간, 실제 배포 완료를 이 CSV만으로 증명할 수는 없습니다.
2. 티켓이 처리된 실제 시간은 Jira 상태 전이·댓글·배포·모니터링 기록으로 별도 교차검증해야 합니다.
3. 데이터에는 시스템 경로, 개인정보, 내부 URL, 비밀정보가 포함될 수 있습니다. 간단한 토큰 패턴 마스킹은 완전한 비식별화 기능이 아닙니다.
4. 필요하다면 데이터 접근 권한이 있는 승인된 내부 머신에서만 실행하고 로컬 출력 파일을 저장소로 푸시하지 마세요.
