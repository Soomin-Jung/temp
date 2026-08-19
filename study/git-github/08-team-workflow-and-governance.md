# 08. 팀 workflow와 repository governance

## 1. 협업 구조는 branch 이름보다 통합 정책이다

팀 workflow를 설계할 때 먼저 정할 질문:

- source of truth branch는 무엇인가?
- 누가 직접 push할 수 있는가?
- 어떤 review와 check가 merge 조건인가?
- release와 deployment는 branch, tag, artifact 중 무엇을 기준으로 하는가?
- 긴급 변경은 어떤 예외 절차를 거치는가?
- 여러 지원 version을 동시에 유지하는가?

이 답 없이 `develop`, `release/*`, `hotfix/*` branch만 만들면 절차가 목적을 대신한다.

## 2. 대표 workflow

### 2.1 GitHub Flow

- `main`에서 짧은 branch 생성
- commit/push 후 PR
- review와 check
- merge 후 배포

지속 배포가 가능하고 장기 지원 version이 많지 않은 팀의 좋은 기본값이다.

### 2.2 Trunk-based development

- 매우 짧은 branch 또는 직접 trunk 통합
- 작은 변경을 자주 merge
- 미완성 기능은 feature flag로 숨김
- 강력하고 빠른 CI 필요

branch를 적게 쓰는 것이 핵심이 아니라, 통합 batch를 작게 만들고 trunk를 항상 건강하게 유지하는 것이 핵심이다.

### 2.3 Git Flow

`main`, `develop`, feature, release, hotfix branch를 장기간 운용한다. 여러 release line과 명시적 stabilization 기간이 있는 제품에는 맞을 수 있지만, 지속 배포 팀에서는 merge-back과 branch drift 비용이 크다.

### 2.4 Release branch

현재 주 개발은 main에서 계속하면서 지원 중인 version별 critical fix를 backport한다.

```text
main
release/1.8
release/1.9
```

각 branch의 지원 기간, 허용 변경, cherry-pick 순서, security patch 정책을 문서화한다.

## 3. 작은 platform 팀의 권장 baseline

- default branch: `main`
- 직접 push 금지
- 짧은 feature/fix/docs branch
- PR 최소 1인 review, 중요 영역은 CODEOWNER review
- required CI: formatting, static validation, unit test, render/schema test
- 새 push 시 stale approval dismiss
- conversation resolution 요구
- squash merge 기본
- remote feature branch 자동 삭제
- release는 annotated tag와 immutable artifact digest로 추적

인원이 한 명이어도 main protection과 CI는 의미가 있다. 실수와 자동화 agent의 잘못된 push를 막는 두 번째 경계가 되기 때문이다.

## 4. Branch protection과 ruleset

보호 규칙으로 다음을 강제할 수 있다.

- PR 없이 직접 변경 금지
- required approval 수
- CODEOWNER approval
- required status check
- 최신 base 상태 요구
- resolved conversation
- signed commit
- linear history
- force push와 branch deletion 제한
- merge queue

Ruleset은 branch/tag/push 규칙을 더 일관되게 적용하고 여러 규칙의 적용 상태를 확인하는 데 유용하다. branch protection과 ruleset이 함께 적용되면 일반적으로 더 제한적인 요구가 결과에 반영될 수 있으므로 중복 정책을 주기적으로 감사한다.

## 5. CODEOWNERS

예:

```text
*                         @platform-team
/charts/                  @serving-team
/.github/workflows/       @platform-security @devex-team
/security/                @platform-security
```

CODEOWNERS는 파일 경로별 review 요청과 ownership을 표현한다. 실제 merge 강제는 branch/ruleset에서 “code owner review required”를 활성화해야 한다.

주의:

- workflow, deployment manifest, permission policy는 application code만큼 민감하다.
- ownership이 한 사람에게만 몰리면 병목과 bus factor 문제가 생긴다.
- 마지막 matching pattern이 적용되는 규칙과 pattern 제약을 공식 문서 기준으로 검증한다.

[예제 CODEOWNERS](examples/CODEOWNERS)는 학습용이며 실제 적용 시 `.github/CODEOWNERS` 등 GitHub가 인식하는 위치에 둔다.

## 6. Required check 설계

PR마다 모든 expensive integration test를 무조건 돌리는 것이 최선은 아니다. feedback 속도와 신뢰 수준을 계층화한다.

| 층 | 시점 | 예시 |
|---|---|---|
| Local | commit 전 | formatter, fast unit test |
| PR fast gate | 수 분 | lint, schema, unit, Helm render |
| PR integration | 조건부/병렬 | ephemeral cluster, API contract |
| Merge queue | merge group | 최신 main 조합 회귀 |
| Post-merge | main | package/image build, wider test |
| Pre-deploy | environment | policy, signature, smoke, approval |

required check가 flaky하면 사람들은 재실행을 정상 절차로 받아들이고 gate 신뢰가 무너진다. flaky test는 별도 quarantine만 하지 말고 owner와 수정 기한을 둔다.

## 7. PR template와 review SLA

[PR template 예제](examples/pull_request_template.md)는 다음을 강제하지 않고 빠뜨리기 쉬운 정보를 드러낸다.

- 변경 이유와 비변경 범위
- test 증거
- migration과 rollback
- 보안·성능·운영 영향
- reviewer가 집중할 지점

큰 조직에서는 review 요청만 보내고 owner가 무기한 기다리지 않도록 응답 시간, escalation, 휴가 시 대체 owner도 합의한다.

## 8. Commit/PR 서명과 audit

commit signature는 작성자가 가진 signing credential로 commit object를 서명했음을 검증한다. 다음을 모두 대신하지는 않는다.

- 코드가 정확하다는 review
- pusher가 authorization을 받았다는 branch policy
- workflow와 artifact가 안전하다는 provenance
- signing key 자체가 탈취되지 않았다는 보장

서명, branch rule, review, CI, artifact attestation은 서로 다른 위협을 막는다.

## 9. Bot과 automation account

- 최소 repository 권한만 부여한다.
- 사람 PAT를 공유하지 않는다.
- GitHub App 또는 OIDC 기반 short-lived credential을 선호한다.
- bot이 만든 PR도 동일한 required review/check를 거치게 한다.
- 자동 merge 범위와 label trigger를 명시한다.
- generated change에 generator version과 source를 남긴다.

## 10. 정책 변경도 코드처럼 관리한다

branch rule, workflow permission, runner group, environment protection은 장애와 보안에 직접 영향을 준다.

- 변경 이유와 영향 기록
- 최소 두 사람 review
- repository/organization audit log 확인
- rollback 경로
- 정기 policy drift 점검

UI에서 클릭한 설정도 IaC나 API export로 재현 가능하게 만드는 것이 이상적이다.
