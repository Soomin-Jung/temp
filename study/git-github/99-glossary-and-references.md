# 99. 용어집과 공식 자료

공식 자료 확인일: 2026-08-19

## 1. Git 용어

| 용어 | 의미 |
|---|---|
| Working tree | 현재 checkout하여 편집 중인 파일 집합 |
| Index / staging area | 다음 commit snapshot을 준비하는 영역 |
| Repository | object database와 refs를 포함한 Git 저장 단위 |
| Blob | 파일 내용을 담는 object |
| Tree | 파일 이름·mode와 blob/subtree 관계를 담는 directory snapshot object |
| Commit | tree, parent, author/committer, message를 담는 object |
| DAG | parent 방향으로 연결되며 cycle이 없는 commit graph |
| Ref | commit/tag 등 object ID에 붙인 이름 |
| Branch | 일반적으로 commit을 가리키며 새 commit 때 이동하는 ref |
| `HEAD` | 현재 branch 또는 detached commit을 가리키는 ref |
| Tag | 특정 object에 붙인 release/bookmark ref; annotated tag는 별도 object 포함 |
| Remote | 다른 repository URL과 fetch/push 설정에 붙인 이름 |
| Remote-tracking ref | 마지막 fetch 때 본 원격 branch의 local 관측값 |
| Upstream branch | local branch가 기본 pull/push 비교 대상으로 tracking하는 branch |
| Merge base | 두 commit의 공통 조상 중 merge 기준이 되는 commit |
| Fast-forward | 새 commit 없이 branch ref만 descendant로 이동하는 통합 |
| Merge commit | 둘 이상의 parent를 가진 통합 commit |
| Rebase | commit patch를 새 base 위에 재적용해 새 commit을 만드는 operation |
| Cherry-pick | 선택한 commit의 변경을 현재 branch에 새 commit으로 적용 |
| Revert | 기존 commit 효과를 반대로 적용하는 새 commit 생성 |
| Reset | branch ref와 선택적으로 index/working tree를 target에 맞춤 |
| Restore | working tree 또는 index의 파일 내용을 source에서 복원 |
| Reflog | local ref와 `HEAD`의 이동 기록 |
| Refspec | remote ref와 local ref의 mapping 규칙 |
| Detached HEAD | `HEAD`가 branch가 아닌 commit을 직접 가리키는 상태 |
| Conflict | Git이 양쪽 변경 의도를 자동 결합할 수 없는 상태 |
| Rerere | 이전 conflict resolution을 기록하고 같은 conflict에 재사용하는 기능 |
| Worktree | 한 repository object database에 연결된 추가 working directory |

## 2. GitHub 용어

| 용어 | 의미 |
|---|---|
| Pull Request | head 변경을 base에 합치기 위한 제안·검토 객체 |
| Draft PR | 아직 merge-ready가 아님을 표시한 PR |
| Review | approve, comment, request changes 상태를 가진 검토 |
| CODEOWNERS | path별 owner와 자동 review 요청 규칙을 표현하는 파일 |
| Check run | 특정 commit SHA에 대한 자동 검사 결과 |
| Required check | 성공해야 merge를 허용하도록 rule에 연결한 check |
| Branch protection | 특정 branch의 push/merge 조건과 금지를 강제하는 규칙 |
| Ruleset | branch, tag, push 등에 적용하는 GitHub repository/organization rule 집합 |
| Merge queue | 최신 base와 queue 선행 변경을 포함한 merge group을 검증·직렬 통합하는 기능 |
| Stacked PR | 의존 관계가 있는 작은 PR을 chain으로 구성한 구조 |
| Workflow | event에 반응해 job을 실행하는 Actions YAML |
| Job | 하나의 runner에서 실행되는 step 묶음 |
| Step | shell command 또는 action 호출 단위 |
| Runner | Actions job을 실행하는 machine/VM/container |
| Artifact | workflow run이 생성해 보존·전달하는 결과 파일 |
| Cache | 후속 run 속도를 위한 재사용 데이터 |
| Environment | deployment 승인, secret, branch 제한, history를 연결하는 GitHub 객체 |
| `GITHUB_TOKEN` | workflow run에 제공되는 repository-scoped token |
| OIDC | 외부 provider가 workflow identity를 검증해 short-lived credential을 발급하는 federation 방식 |
| Release | tag에 release note와 asset을 연결한 GitHub 객체 |
| Attestation | artifact에 관한 provenance 등 statement를 검증 가능하게 연결한 증명 |

## 3. Git 공식 학습 순서

### 기초와 내부 구조

- [Pro Git: What is Git?](https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F)
- [Pro Git: Recording Changes](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository)
- [Pro Git: Git Objects](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects)
- [Pro Git: Git References](https://git-scm.com/book/en/v2/Git-Internals-Git-References)
- [Pro Git: The Refspec](https://git-scm.com/book/en/v2/Git-Internals-The-Refspec)

### Branch와 통합

- [Pro Git: Branches in a Nutshell](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)
- [Pro Git: Basic Branching and Merging](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)
- [Pro Git: Remote Branches](https://git-scm.com/book/en/v2/Git-Branching-Remote-Branches)
- [Pro Git: Rebasing](https://git-scm.com/book/en/v2/Git-Branching-Rebasing)
- [Pro Git: Distributed Workflows](https://git-scm.com/book/en/v2/Distributed-Git-Distributed-Workflows)

### 복구와 조사

- [Git reset, restore and revert overview](https://git-scm.com/docs/git#_reset_restore_and_revert)
- [Pro Git: Reset Demystified](https://git-scm.com/book/en/v2/Git-Tools-Reset-Demystified)
- [Pro Git: Advanced Merging](https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging)
- [git-reflog](https://git-scm.com/docs/git-reflog)
- [git-rerere](https://git-scm.com/docs/git-rerere)
- [git-bisect](https://git-scm.com/docs/git-bisect)
- [git-worktree](https://git-scm.com/docs/git-worktree)

## 4. GitHub 공식 학습 순서

### 협업

- [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow)
- [About pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-pull-requests)
- [Pull request merges](https://docs.github.com/en/pull-requests/reference/pull-request-merges)
- [Keeping a PR branch in sync](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/keeping-your-pull-request-in-sync-with-the-base-branch)
- [Stacked pull requests](https://docs.github.com/en/pull-requests/reference/stacked-pull-requests)
- [About CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)

### Repository policy

- [About protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [About rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)
- [Available rules for rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
- [Managing a merge queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue)

### Actions와 보안

- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [Workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
- [Use `GITHUB_TOKEN` for authentication](https://docs.github.com/en/actions/reference/authentication-in-a-workflow)
- [Secure use reference](https://docs.github.com/en/actions/reference/security/secure-use)
- [Self-hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners)
- [OIDC reference](https://docs.github.com/en/actions/reference/openid-connect-reference)
- [Control workflow concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)

### Release와 공급망

- [Managing releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
- [Supply chain security](https://docs.github.com/en/code-security/concepts/supply-chain-security/supply-chain-security)
- [Artifact attestations](https://docs.github.com/en/actions/concepts/security/artifact-attestations)
- [Immutable releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases)

## 5. 자주 하는 오해

| 오해 | 정확한 설명 |
|---|---|
| commit은 이전 파일과의 diff다 | 논리적으로 tree snapshot과 parent를 가리키는 object다. 저장 최적화에서 delta compression을 쓸 수 있다. |
| branch는 코드 복사본이다 | commit을 가리키는 가벼운 ref다. |
| `origin/main`은 GitHub의 실시간 main이다 | 마지막 fetch로 갱신된 local remote-tracking ref다. |
| `git add`는 GitHub upload다 | index에 다음 snapshot을 준비한다. |
| PR이 merge되면 내 local main도 갱신된다 | local에서는 fetch와 fast-forward가 별도로 필요하다. |
| rebase는 commit을 이동한다 | 새 parent 위에 새 commit을 만든다. |
| force-with-lease면 언제나 안전하다 | 다른 사람 update 유실을 일부 막지만 shared history rewrite 자체의 위험은 남는다. |
| CI가 초록이면 production이 안전하다 | 정의한 검사 범위와 실행 환경에서 성공했다는 뜻이다. |
| secret을 Git history에서 지우면 끝이다 | 노출된 credential은 먼저 revoke/rotate해야 한다. |
| Git tag와 GitHub Release는 같다 | tag는 Git ref, Release는 tag에 metadata/asset을 연결한 GitHub 객체다. |

## 6. 더 공부할 주제

- partial clone, sparse checkout와 대형 monorepo
- commit-graph, packfile, bitmap과 Git 성능
- submodule, subtree, vendoring trade-off
- signed commit/tag와 SSH/GPG/Sigstore 계열 서명
- Git LFS와 large artifact policy
- organization ruleset, runner group, reusable workflow governance
- merge queue와 monorepo path-aware CI
- SLSA provenance와 artifact admission policy
