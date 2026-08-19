# 90. 명령 선택과 Git 사고 대응 runbook

## 1. 모든 작업의 시작

```bash
git status -sb
git branch --show-current
git remote -v
git log --graph --decorate --oneline --all -20
```

## 2. 상태 확인

```bash
git diff                    # working tree vs index
git diff --cached           # index vs HEAD
git diff HEAD               # working tree vs HEAD
git branch -vv              # tracking 관계
git rev-parse HEAD
git merge-base origin/main HEAD
```

## 3. 일상 feature workflow

```bash
git switch main
git fetch origin
git merge --ff-only origin/main
git switch -c feature/<name>

# edit/test
git add -p
git diff --cached
git commit -m "<type>: <change>"
git push -u origin feature/<name>
```

PR merge 후:

```bash
git switch main
git fetch origin
git merge --ff-only origin/main
git branch -d feature/<name>
git fetch --prune origin
```

## 4. 최신 main 반영

같은 repository에서 작업하는 개인 branch:

```bash
git fetch origin
git switch feature/<name>
git rebase origin/main
git push --force-with-lease
```

공동 branch:

```bash
git fetch origin
git switch feature/<name>
git merge origin/main
git push
```

외부 Open Source fork의 개인 PR branch:

```bash
git fetch upstream --prune
git fetch origin --prune
git switch feature/<name>
git branch backup/<name>-before-rebase HEAD
git rebase upstream/main

# test와 diff 확인 후
git fetch origin --prune
git log --oneline HEAD..origin/feature/<name>
git push --force-with-lease origin HEAD:feature/<name>
```

## 5. A에서 파생한 B, A squash merge 완료

```bash
git fetch origin
git switch branch-b
git branch backup/branch-b-before-onto
git rebase --onto origin/main <last-a-commit> branch-b
git range-diff <last-a-commit>..backup/branch-b-before-onto origin/main..branch-b
git push --force-with-lease origin branch-b
```

경계를 모르겠다면 중단하고 graph에서 A/B 고유 commit을 식별한다.

## 6. 장기 Open Source PR 갱신

```bash
# read-only 관찰
git fetch upstream --prune
git fetch origin --prune
git rev-list --left-right --count upstream/main...HEAD
git log --left-right --cherry-pick --oneline upstream/main...HEAD

# rewrite 전 복구점
git branch backup/<topic>-before-rebase HEAD

# 개인 branch만
git rebase upstream/main

# patch 유실·중복과 remote-only commit 확인
git diff --stat upstream/main...HEAD
git diff --check upstream/main...HEAD
git fetch origin --prune
git log --oneline HEAD..origin/<topic>
```

push 후에는 PR의 base/head, current head SHA, Files changed, outdated thread, approval, 최신 SHA의 CI를 확인한다. stacked PR의 하위 branch가 squash merge되었다면 단순 rebase 전에 `--onto` 경계를 확인한다. 전체 절차는 [장기 PR upstream 동기화](15-long-running-pr-upstream-sync.md)를 따른다.

## 7. Conflict

```bash
git status
git diff --name-only --diff-filter=U
git ls-files -u
rg '^(<<<<<<<|=======|>>>>>>>)' .
```

해결 후:

```bash
git add <resolved-files>
git merge --continue       # merge
git rebase --continue      # rebase
git cherry-pick --continue # cherry-pick
```

취소:

```bash
git merge --abort
git rebase --abort
git cherry-pick --abort
```

## 8. 되돌리기 결정표

```text
아직 commit 전인가?
├─ working tree 편집 취소 -> git restore <file>
└─ stage만 취소          -> git restore --staged <file>

commit했지만 아직 나만 쓰는가?
├─ message/내용 보정     -> git commit --amend
├─ commit 재구성         -> git rebase -i
└─ commit만 풀기         -> git reset --soft/mixed

이미 공유되었는가?
└─ 효과를 취소           -> git revert <sha>
```

## 9. 잃어버린 commit

```bash
git reflog -50
git show <candidate-sha>
git branch rescue/<name> <candidate-sha>
```

복구 branch를 만든 뒤에만 추가 정리 작업을 한다.

## 10. 잘못 push함

### 일반 main/shared branch

```bash
git revert <bad-sha>
git push origin <branch>
```

### secret 포함

1. credential 즉시 revoke/rotate
2. 접근 log와 사용 흔적 확인
3. repository owner/security 담당자와 history rewrite 범위 결정
4. fork/cache/artifact까지 노출 범위 확인
5. 보호 rule과 secret scan 보완

단순 commit 삭제는 이미 복제된 secret을 무효화하지 못한다.

## 11. CI 실패

```text
workflow가 시작되지 않음
  -> event, branch/path filter, YAML 위치, Actions enable 상태

job이 queued
  -> runner label, capacity, concurrency, approval

job이 skipped
  -> if 조건, needs 실패, event context

step 실패
  -> 첫 실패 command, exit code, tool version, permission

동일 SHA에서 flaky
  -> seed/time/network/shared state, retry 횟수와 실패율 기록
```

## 12. 실행 전 별도 승인이 필요한 명령

```bash
git reset --hard
git clean -fd
git push --force
git branch -D
git tag -f
```

개인 실습 repository 외에는 대상, 영향, 복구점, 공유 여부를 먼저 확인한다. 가능하면 recoverable alternative를 사용한다.

## 13. 도움 요청 시 함께 전달할 정보

secret을 제외하고 다음을 제공한다.

```bash
git status -sb
git branch -vv
git log --graph --decorate --oneline --all -30
git reflog -20
git remote -v
```

그리고 다음을 문장으로 적는다.

- 하려던 일
- 마지막으로 성공한 command
- 실제 실행한 command와 error
- 이미 push했는지
- branch를 다른 사람이 사용하는지
- 보존해야 할 local 변경이 있는지
