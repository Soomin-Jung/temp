# 07. 되돌리기, 복구와 Git 기반 디버깅

## 1. “되돌린다”는 말부터 분해한다

되돌리기 명령이 많은 이유는 대상이 다르기 때문이다.

- working tree의 편집만 버릴 것인가?
- index에서 stage만 취소할 것인가?
- local branch ref를 과거로 옮길 것인가?
- 이미 공유된 변경을 새 commit으로 상쇄할 것인가?
- 다른 branch의 특정 변경만 가져올 것인가?
- 이름을 잃은 commit을 다시 branch로 붙잡을 것인가?

명령 전에 보존해야 할 데이터를 먼저 정한다.

## 2. 안전도 순서

대체로 다음 순서로 생각한다.

1. 조회: `status`, `diff`, `log`, `show`, `reflog`
2. 새 이력으로 수정: 새 commit, `revert`
3. local 상태 재배치: `restore`, mixed reset, rebase
4. 파괴적 삭제: hard reset, clean, 강제 push

파괴적 명령이 빠르다는 이유로 첫 선택이 되어서는 안 된다.

## 3. 상황별 명령 선택

| 상황 | 권장 명령 | 공개 이력 영향 |
|---|---|---|
| unstaged 파일 편집 취소 | `git restore <file>` | 없음 |
| staged 상태만 취소 | `git restore --staged <file>` | 없음 |
| 파일을 특정 commit 버전으로 복원 | `git restore --source=<commit> <file>` | 없음, 이후 commit 필요 |
| 마지막 local commit message 수정 | `git commit --amend` | commit SHA 변경 |
| 여러 local commit 정리 | `git rebase -i` | commit SHA 변경 |
| 이미 push된 나쁜 commit 취소 | `git revert <sha>` | 새 취소 commit 추가 |
| local branch ref를 과거로 이동 | `git reset` | 선택한 mode에 따라 index/worktree 변경 |
| 잃어버린 commit 찾기 | `git reflog` | 조회만 |
| 다른 branch의 특정 commit 적용 | `git cherry-pick <sha>` | 새 commit 추가 |

## 4. `restore`

```bash
# working tree를 index 상태로 복원
git restore path/to/file

# stage 취소, working tree 수정은 유지
git restore --staged path/to/file

# 특정 commit의 파일 내용 가져오기
git restore --source=HEAD~2 path/to/file
```

첫 번째 명령은 아직 commit되지 않은 편집을 잃을 수 있다. 실행 전 `git diff -- path/to/file`을 확인한다.

## 5. `reset`의 세 mode

`reset`은 주로 current branch ref, index, working tree 세 층을 어디까지 맞출지 결정한다.

| mode | Branch ref | Index | Working tree |
|---|---|---|---|
| `--soft` | 이동 | 유지 | 유지 |
| `--mixed` | 이동 | target에 맞춤 | 유지 |
| `--hard` | 이동 | target에 맞춤 | target에 맞춤 |

예:

```bash
# 마지막 local commit만 풀고 변경은 staged로 유지
git reset --soft HEAD~1

# 마지막 local commit을 풀고 변경은 unstaged로 유지
git reset HEAD~1
```

`reset --hard`는 tracked working tree 변경을 버린다. 정확한 repository, branch, target SHA, 백업 ref를 확인하기 전에는 사용하지 않는다.

```bash
git branch backup/before-reset
git reset --hard <verified-sha>
```

공유 branch를 reset하고 force push하면 다른 사람의 history도 바뀐다. production main의 장애 변경은 보통 reset이 아니라 revert로 취소한다.

## 6. `revert`

```bash
git revert <bad-commit>
```

기존 commit을 삭제하지 않고 그 효과를 반대로 적용하는 새 commit을 만든다. audit 가능한 공개 이력에 적합하다.

merge commit을 revert할 때는 mainline parent를 지정해야 한다.

```bash
git revert -m 1 <merge-commit>
```

`-m 1`은 첫 번째 parent를 mainline으로 간주하고 다른 parent에서 들어온 효과를 취소한다. parent를 잘못 선택하면 의도하지 않은 변경을 제거할 수 있으므로 `git show --no-patch --pretty=raw <merge>`와 graph를 먼저 확인한다.

merge revert 후 같은 branch를 다시 merge할 때 Git이 이미 merge된 ancestry로 판단할 수 있다. 단순히 revert를 revert할지, 새 fix commit을 만들지 graph와 release 계획을 함께 결정한다.

## 7. `reflog`: ref 이동 기록

```bash
git reflog
git reflog show feature/my-work
```

reflog는 local ref와 `HEAD`가 과거에 가리켰던 commit을 기록한다. 다음 사고에서 유용하다.

- branch를 실수로 삭제
- rebase 전 commit SHA를 잃음
- 잘못 reset
- detached HEAD에서 만든 commit을 놓침

복구:

```bash
git branch rescue/lost-work <reflog-sha>
```

reflog는 local이며 영구 보관 계약이 아니다. 만료와 garbage collection 전에 복구한다. GitHub remote가 local reflog를 대신 보관해 주는 것도 아니다.

## 8. `ORIG_HEAD`와 operation abort

merge, rebase, reset 같은 일부 operation은 이전 위치를 `ORIG_HEAD`에 남긴다.

```bash
git show ORIG_HEAD
```

진행 중 operation은 해당 abort 명령을 우선 사용한다.

```bash
git merge --abort
git rebase --abort
git cherry-pick --abort
git revert --abort
```

중간 state에서 임의 reset을 하면 operation metadata까지 복잡해질 수 있다.

## 9. `bisect`: 장애 도입 commit 찾기

좋았던 commit과 나쁜 commit 사이를 이진 탐색한다.

```bash
git bisect start
git bisect bad HEAD
git bisect good <known-good-sha>

# Git이 checkout한 각 commit에서 판정
git bisect good
git bisect bad

git bisect reset
```

자동화:

```bash
git bisect run ./scripts/reproduce.sh
```

exit code 0은 good, 1~127은 bad, 125는 test 불가로 skip한다. 각 commit이 build/test 가능한 atomic 상태일수록 bisect가 강력해진다.

## 10. `blame`과 `log -S/-G`

```bash
git blame -L 40,80 path/to/file
git log -S 'max_num_batched_tokens' --oneline --all
git log -G 'selector:.*vllm' -p --all
```

- blame은 현재 line이 마지막으로 바뀐 commit을 찾는다. 책임자를 비난하는 도구가 아니라 context로 가는 index다.
- `-S`는 특정 문자열의 등장 횟수가 바뀐 commit을 찾는다.
- `-G`는 diff가 regex와 맞는 commit을 찾는다.

찾은 commit에서 PR, issue, incident 기록을 따라가야 변경 이유를 이해할 수 있다.

## 11. 실수 직후 공통 runbook

```bash
git status -sb
git branch --show-current
git log --graph --decorate --oneline --all -30
git reflog -30
```

그다음:

1. 추가 commit, reset, clean, force push를 멈춘다.
2. 복구 후보 SHA에 backup branch를 만든다.
3. working tree 변경이 있으면 patch 또는 임시 branch로 보존한다.
4. local 문제인지 remote 공개 이력 문제인지 분리한다.
5. 복구 후 diff와 test를 실행한다.

> 이 장의 본질: 안전한 복구는 과거를 무조건 지우는 것이 아니라, **어느 상태 공간과 ref를 바꿀지 제한하고 복구 가능한 object를 먼저 붙잡는 일**이다.
