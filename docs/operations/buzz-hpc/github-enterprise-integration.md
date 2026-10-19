# Buzz / GitHub Enterprise Server 통합 검토

기준: 2026-10-19, block/buzz commit 3debf18f1448334aa7bc202d328d93bb9651e168.
상태: Upstream 소스 정적 분석 완료, 실제 GHES 연동 테스트 전.

## 결론

사내 설치형 GitHub Enterprise Server(GHES)의 Clone URL을 Buzz Project의 외부 Git 주소로 기록할 수는 있다. 하지만 **Buzz Desktop이 사내 GHES Private Repository를 직접 Clone/인증/탐색하는 네이티브 통합은 현재 소스에서 지원되지 않는다.** 공식 Helm values를 바꿔도 해결되지 않는 Desktop 구현 제약이다.

| 기능 | 확인 결과 |
| --- | --- |
| GHES URL을 NIP-34 Repo Announcement의 clone Tag로 기록 | 가능 |
| 외부 호스트 링크 표시 | 가능 |
| Buzz Desktop에서 GHES Private Repo Clone | hostname/인증 구현상 기본 미지원 |
| Buzz UI로 GHES PR/Issue 자동 동기화 | 전용 GHES Bridge 미확인 |
| 별도 Agent 환경에서 GHES Git 및 API 사용 | 독립 Git/CLI 인증 경로로 가능 |
| Buzz 내장 Git 서버 사용 | 가능하지만 GHES와는 별도 Repository |

## 관련 코드

- [projectRepositoryCreation.ts](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/desktop/src/features/projects/projectRepositoryCreation.ts): 입력 Clone URL을 Git 저장소 Announcement에 기록.
- [projectCloneUrl.ts](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/desktop/src/features/projects/lib/projectCloneUrl.ts): 외부 Clone URL 우선.
- [project_git_exec.rs](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/desktop/src-tauri/src/commands/project_git_exec.rs): GitHub URL 검증은 HTTPS github.com/owner/repo 형식만 허용. 그 외는 Buzz Relay의 /git/owner/repo 형태만 허용.
- [project_git_workflow.rs](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/desktop/src-tauri/src/commands/project_git_workflow.rs): 실제 Clone 전에 URL 검증 수행.
- [projectGitError.ts](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/desktop/src/features/projects/lib/projectGitError.ts): 공개 GitHub는 인증정보 없이 Clone한다는 설명.
- [ProjectWorkspaceTabs.tsx](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/desktop/src/features/projects/ui/ProjectWorkspaceTabs.tsx): 외부 코드는 Buzz에 미러되어 있지 않다는 UI.

따라서 **외부 저장소 링크 등록 가능과 네이티브 GHES 작업 가능을 구분**해야 한다. 특히 호스트 허용만 수정해서는 Private Repository 자격증명이 제공되지 않는다.

## 권장안 A — 기존 GHES를 유지하고 Agent Runtime에서 Git 연동

Buzz는 Collaboration/Agent 대화 채널로 사용하고, GHES를 기존 Git/PR/Issue의 System of Record로 유지한다. 실제 Git 클라이언트가 **사용자 PC에서 동작하는지, HPC의 Agent Pod에서 동작하는지**에 따라 각각 GHES DNS/443 접근 권한이 필요하다. Buzz Relay Pod에만 Egress를 허용해도 해결되지 않는다.

```text
Buzz Client ↔ Buzz Relay (Nostr/WS + 외부 Repo 메타데이터)
                   |
                   └─ Agent Runtime (독립 Desktop Process 또는 Pod)
                         └─ git / gh / GHES 전용 MCP·REST Adapter
                               └─ Internal GitHub Enterprise Server
```

GHES에 접근 가능한 Agent 실행 환경 또는 개발자 PC에서 먼저 확인한다.

```bash
export GH_HOST="ghe.example.internal" # 샘플 hostname
gh auth login --hostname "$GH_HOST"   # 개발자 PC 대화형 검증
gh api --hostname "$GH_HOST" repos/ORG/REPO --jq .full_name
git ls-remote "https://$GH_HOST/ORG/REPO.git" HEAD
```

운영 Agent에는 대화형 로그인 대신 전용 자격증명(예: GitHub App Installation Token, 제한된 PAT)을 **해당 Agent만 접근할 수 있는 Secret**으로 공급한다. Git 인증정보를 Buzz Relay의 Kubernetes Secret이나 Git URL/로그에 넣지 않는다. 사내 TLS CA 신뢰도 Client와 Agent 런타임 각각 검증한다.

GHES REST API 기본 주소는 `https://HOST/api/v3`이다. `gh auth setup-git --hostname HOST`로 해당 Host용 Git Credential Helper를 구성할 수 있다. 지원하는 MCP 서버가 있다면 GHES Host와 인증 구현을 해당 서버 소스 기준으로 따로 검증한다. **현재 Buzz Helm이 GHES 또는 MCP 서버를 자동 설치하지는 않는다.**

- [GHES REST API](https://docs.github.com/en/enterprise-server@3.19/rest/using-the-rest-api/getting-started-with-the-rest-api)
- [GHES Git HTTPS 인증](https://docs.github.com/en/enterprise-server@3.20/get-started/git-basics/about-remote-repositories)
- [GitHub CLI Enterprise auth](https://cli.github.com/manual/gh_auth_login)
- [GH CLI Git Credential Helper](https://cli.github.com/manual/gh_auth_setup-git)

## 대안 B — Buzz Desktop 네이티브 GHES 통합 개발

아래 항목은 **이미 제공되는 옵션이 아니라 추가 구현 대상**이다.

1. 하드코딩된 `github.com` 검증을 회사의 승인된 GHES Host allowlist로 확장. 허용 Scheme/Host/Port/Path와 Redirect 정책 및 임의 외부 URL 접근 위험까지 검사.
2. Buzz의 Git 실행 설정은 `GIT_CONFIG_GLOBAL=/dev/null`, 글로벌 Credential Helper 초기화 및 Nostr Git Helper를 사용한다. 이 로직과 별개로 **GHES 호스트 전용 인증 경로**를 설계해야 한다. Nostr 사용자 키를 GHES용 Git 프로세스에 넘기지 않는 보안 분리가 중요하다.
3. Desktop Clone/Fetch/Pull/Push, Git File/Commit/Branch UI 및 인증 오류 상태를 변경·검증.
4. GHES 자체 PR/Issue와 Buzz Nostr PR/Issue는 별개이다. 양방향 동기화 요구 시 GHES REST API/Webhook Adapter, 사용자 권한 매핑, 중복 이벤트 방지 로직 필요.
5. Desktop은 Tauri 네이티브 앱이다. Relay Helm 배포만 변경해도 Desktop 소스 수정이 사용자에게 적용되지 않는다. **폐쇄망 Desktop 바이너리 배포·업데이트**도 구축해야 한다.

## 검증 계획

| 검증 | 예상 또는 합격 기준 |
| --- | --- |
| 프로젝트에 GHES URL 등록 | Announcement의 clone 태그·외부 Git Host 표시 |
| 기본 Buzz Desktop에서 GHES Clone | 현 소스 기준 URL 검증 실패 예상; 설치된 실제 Desktop 버전에서 재확인 |
| 사용자 PC 또는 Agent Pod에서 GHES git ls-remote | 내부 DNS/사내 CA/인증 성공 |
| 같은 실행 환경에서 GHES REST Repo API | 사용자 또는 App Token 범위 내 성공 |
| Agent가 PR 생성 | GHES 권한 정책과 별도 Git/gh 도구 기반 검증 |
| Buzz PR과 GHES PR의 자동 동기화 | 기본 제공으로 간주하지 않음. 별도 Adapter 없으면 미지원 |
| Public github.com 네트워크 Egress 차단 | GHES를 사용하는 필수 실행 경로에 외부 통신이 남지 않아야 함 |

기본 권장: **Relay는 공식 Helm으로 배포하고 GHES는 Agent 도구부터 연동한다.** 네이티브 Desktop Git 통합이 필수인지 확인한 뒤 별도 소스 커스터마이징 과제로 분리한다.
