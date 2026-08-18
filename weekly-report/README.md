# LLM Serving Weekly Report

vLLM 생태계와 오픈웨이트 모델, LLM 배포·추론·최적화 기술을 매주 검증해 남기는 운영형 주간 보고서입니다.

## 발행 기준

- 발행 시점: 매주 월요일 새벽(Asia/Seoul)
- 집계 구간: 직전 월요일 00:00부터 일요일 23:59:59까지(Asia/Seoul)
- 파일명: `YYYY/YYYY-MM-DD.md` — 파일 날짜는 발행일 기준
- 기본 골격: [`_TEMPLATE.md`](_TEMPLATE.md)
- 시간 구간 밖의 자료는 맥락 설명에 꼭 필요한 경우에만 쓰고 `배경`으로 표시합니다.

## 수집 범위

1. **vLLM core**
   - release, tag, changelog, 주요 PR·issue, breaking change, 보안·안정성, 성능 회귀/개선
2. **vLLM 연계 생태계**
   - vLLM Production Stack, llm-d, KServe, Ray Serve, LMCache, Mooncake, NIXL 등 실제 통합·의존 관계가 있는 프로젝트
3. **오픈웨이트 모델**
   - 신규 checkpoint, model card, license, `config.json`, tokenizer/chat template, vLLM 지원 상태와 배포상 주의점
4. **배포·추론·최적화**
   - CUDA/Triton kernel, FlashInfer/FlashAttention/DeepGEMM, quantization, KV cache, speculative decoding, TP/PP/DP/EP/CP, P/D disaggregation, routing/scheduling, observability
5. **놓치면 안 될 논문·기술**
   - arXiv/학회 논문, 공식 코드, 재현 결과 중 serving architecture나 운영 판단을 바꿀 만한 내용

프로젝트 목록은 고정 watchlist가 아니라 실제 영향도에 따라 확장합니다. 단순 홍보성 발표는 제외합니다.

## 근거 우선순위

| 등급 | 사용할 수 있는 근거 | 표기 원칙 |
|---|---|---|
| A — 공식 확인 | 공식 release/tag/changelog, merged PR, 문서, model card/config, 논문·공식 코드 | 사실로 기술하고 버전·날짜·직접 링크를 남김 |
| B — 교차 확인 | 공식 근거와 서로 독립적인 추가 근거가 일치 | 해석과 사실을 분리하고 두 근거를 함께 남김 |
| C — 실험 보고 | 공식 자료가 없고 재현 가능한 개인/커뮤니티 실험만 존재 | `커뮤니티 실험`으로 명시하고 HW/SW/명령·워크로드·한계를 기록 |
| 제외 | 출처 없는 재전파, 캡처만 있는 주장, 루머, 삭제되어 확인 불가한 게시물 | 본문 사실로 채택하지 않음 |

- 모델 가중치 공개와 소스코드 공개는 구분하며, 근거 없이 `오픈소스 모델`이라 부르지 않습니다.
- 출시일, 게시일, merge일, benchmark 실행일을 섞지 않습니다.
- benchmark는 동일 조건이 아니면 순위를 만들지 않고 방향성만 설명합니다.
- 고영향 주장에는 가능한 한 두 개 이상의 독립된 근거를 붙입니다. 단일 근거뿐이면 그 사실을 표시합니다.
- 추론이나 전망은 `해석` 또는 `추정`으로 분리합니다.

## 보고서 품질 게이트

발행 전 아래 항목을 모두 확인합니다.

- [ ] 집계 구간과 실제 사건 발생일이 맞는가
- [ ] 모델·프로젝트·옵션 이름을 공식 표기와 대조했는가
- [ ] 버전, tag, commit/PR 번호, GPU 세대와 precision을 혼동하지 않았는가
- [ ] 링크가 검색 결과가 아니라 해당 주장을 직접 뒷받침하는 원문인가
- [ ] benchmark의 하드웨어, 동시성, 입력/출력 길이, quantization, batch 조건을 기록했는가
- [ ] vLLM에서 `지원됨`, `실험적`, `미지원`, `외부 fork 필요`를 구분했는가
- [ ] 같은 사건을 release·블로그·SNS 항목으로 중복 집계하지 않았는가
- [ ] 커뮤니티 실험을 공식 성능 수치처럼 표현하지 않았는가
- [ ] 맞춤법, 표 안 숫자, 단위, 링크, Markdown 렌더링을 다시 확인했는가
- [ ] 민감한 내부 환경·조직·주소·계정 정보가 없는가
- [ ] 지난 보고서의 오류나 변경된 사실이 있으면 `정정` 절에 남겼는가

하나라도 충족하지 못한 핵심 주장은 삭제하거나 `확인 필요`로 낮춥니다.

## Git 운영 원칙

- 예약 실행은 임시 branch와 PR을 만들지 않습니다.
- 자료 수집과 검증을 모두 끝낸 뒤에만 `main`의 날짜별 보고서와 이 인덱스를 갱신합니다.
- 가능하면 한 번의 원자적 commit으로 반영합니다.
- 실행이 실패하면 불완전한 초안, 빈 보고서, 작업 branch를 남기지 않습니다.
- 같은 발행일 파일이 이미 있으면 새 이름으로 복제하지 않고 검증된 내용만 갱신합니다.
- 사람이 검토해야 할 불확실성이 크면 GitHub에는 쓰지 않고 실행 결과에서 보류 사유를 알립니다.

## 보고서 인덱스

아직 발행된 보고서가 없습니다. 첫 예약 실행부터 최신순으로 추가합니다.
