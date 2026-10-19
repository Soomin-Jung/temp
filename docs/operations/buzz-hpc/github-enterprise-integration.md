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
