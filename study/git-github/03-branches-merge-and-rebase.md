# 03. Branch, merge와 rebase

## 1. Branch를 쓰는 이유

branch는 변경을 격리하는 폴더가 아니라 commit graph의 별도 tip이다. 목적은 다음과 같다.

- 불완전한 변경이 `main`의 안정성을 깨뜨리지 않게 한다.
- 하나의 변경 제안을 독립적으로 review하고 test한다.
- 여러 작업의 수명과 승인 시점을 분리한다.
- release/hotfix처럼 서로 다른 통합 정책을 적용한다.

branch가 오래 유지될수록 `main`과의 차이, conflict 가능성, 통합 비용이 커진다. 따라서 feature branch는 작고 짧게 유지하는 것이 기본이다.

## 2. 생성과 이동

```bash
git switch main
git fetch origin
git merge --ff-only origin/main
git switch -c feature/cache-aware-routing
```

- `switch -c`는 branch 생성과 이동을 함께 한다.
- 새 branch는 현재 commit을 가리키며 시작한다.
- 최신 main 기반이 필요하면 branch를 만들기 전에 fetch와 fast-forward를 확인한다.

`git checkout`은 branch 이동, 파일 복원 등 여러 역할을 겸한다. 최근 Git에서는 의도를 분리한 `git switch`와 `git restore`가 초심자에게 더 명확하다.

## 3. Fast-forward

base branch가 갈라진 뒤 별도 commit을 만들지 않았다면 ref만 앞으로 옮길 수 있다.

```text
C0---C1 main
       \
        F1---F2 feature
```

main을 F2로 fast-forward하면 새 merge commit이 필요 없다.

```bash
git switch main
git merge --ff-only feature
```

`--ff-only`는 branch가 갈라졌다면 자동 merge commit을 만들지 않고 실패한다. “단순 최신화만 허용”하는 안전 guard로 유용하다.

## 4. 3-way merge

양쪽 branch가 모두 전진했다면 Git은 다음 세 snapshot을 비교한다.

1. 공통 조상, merge base
2. 현재 branch tip, ours
3. 합칠 branch tip, theirs

겹치지 않는 변경은 자동 결합하고, 같은 영역에 호환되지 않는 변경이 있으면 conflict를 남긴다. 성공하면 두 parent를 가진 merge commit이 생길 수 있다.

```bash
git switch main
git merge --no-ff feature/cache-aware-routing
```

merge commit은 “이 두 개발선이 여기서 통합되었다”는 topology를 보존한다. 하지만 모든 작은 PR마다 merge commit을 남기면 history가 복잡해질 수 있다.

## 5. Rebase의 실제 의미

rebase는 commit을 물리적으로 이동시키지 않는다. 기존 branch의 고유 commit을 골라 새 base 위에 patch를 순서대로 적용하여 **새 commit**을 만든다.

변경 전:

```text
          F1---F2 feature
         /
C0---C1---M1 main
```

변경 후:

```text
C0---C1---M1 main
              \
               F1'---F2' feature
```

`F1'`, `F2'`는 내용이 비슷해도 parent와 committer metadata가 달라 새로운 SHA를 가진다.

```bash
git fetch origin
git switch feature/cache-aware-routing
git rebase origin/main
```

conflict가 나면 한 commit씩 해결한다.

```bash
git add <resolved-files>
git rebase --continue

# 현재 rebase 전체 취소
git rebase --abort

# 정말 불필요한 현재 commit만 건너뜀
git rebase --skip
```

`--skip`은 변경을 잃을 수 있으므로 “이미 upstream에 동일 patch가 있고 현재 commit이 완전히 불필요하다”는 것을 확인한 경우에만 사용한다.

## 6. Merge와 rebase 선택

| 기준 | Merge | Rebase |
|---|---|---|
| 기존 commit ID | 보존 | 새 commit으로 변경 |
| branch topology | 보존 | 선형화 |
| conflict 해결 | 통합 시 한 번 | 재생되는 commit마다 발생 가능 |
| 공유 branch | 안전한 기본값 | 협업자와 합의 없으면 위험 |
| 개인 feature branch 정리 | 가능 | 적합 |
| audit에서 실제 통합선 보존 | 유리 | 단순화됨 |

원칙:

- 다른 사람이 기반으로 사용 중인 shared branch는 함부로 rebase하지 않는다.
- 개인 feature branch는 PR merge 전 rebase할 수 있다.
- 최종 merge 방법은 repository 정책과 release/revert 요구를 따른다.

## 7. Force push는 왜 필요한가

rebase 후 remote branch tip과 local branch tip은 서로 다른 history가 된다. 일반 push는 non-fast-forward update를 거부한다.

```bash
git push --force-with-lease origin feature/cache-aware-routing
```

`--force-with-lease`는 remote ref가 내가 마지막으로 본 상태와 같을 때만 덮어쓴다. 그사이 다른 사람이 push했다면 실패해 변경 유실을 막는다. 무조건 덮는 `--force`보다 안전하지만, shared branch를 rebase해도 된다는 허가가 되는 것은 아니다.

## 8. Interactive rebase

push 전 개인 commit을 review 가능한 이야기로 정리할 때 사용한다.

```bash
git rebase -i origin/main
```

주요 action:

- `pick`: 유지
- `reword`: message 수정
- `edit`: commit 내용 수정
- `squash`: 이전 commit과 합치고 message 편집
- `fixup`: 이전 commit과 합치고 현재 message 폐기
- `drop`: 제거

CI fix commit을 무조건 숨길 필요는 없다. review 중 어떤 피드백을 반영했는지 commit별로 보는 팀도 있다. 최종 history 정책과 PR review 편의 사이에서 정한다.

## 9. Cherry-pick

```bash
git switch release/1.2
git cherry-pick <commit-sha>
```

특정 commit의 patch를 현재 branch에 적용해 새 commit을 만든다. hotfix backport에 유용하지만 같은 논리 변경이 서로 다른 SHA로 여러 branch에 존재하게 된다. 장기간 남용하면 어느 branch가 정본인지 추적하기 어려워진다.

## 10. Merge 전략과 PR merge 방법은 구분한다

GitHub PR에서 일반적으로 선택하는 방법:

- **Merge commit**: head의 commit을 보존하고 base에 merge commit 추가
- **Squash merge**: PR 변경 전체를 base의 새 commit 하나로 기록
- **Rebase merge**: head commit들을 base 위에 새 SHA로 순서대로 추가

local에서 feature branch를 rebase했는지와 GitHub에서 PR을 어떤 방식으로 merge하는지는 별도 결정이다.

## 11. 실무 기본 추천

작은 platform 팀에서 별도 요구가 없다면 다음이 단순하다.

- `main`은 항상 배포 가능한 상태
- 짧은 feature branch + PR
- PR 하나가 하나의 rollback 단위가 되도록 구성
- 최종 base history는 squash merge 중심
- 여러 commit 자체가 중요한 migration/release 작업은 merge commit 또는 rebase merge를 예외 적용
- feature branch 동기화는 개인 branch면 rebase, 공동 branch면 merge

모든 repository에 하나의 전략을 강제할 필요는 없다. source library, deployment config, release maintenance repository는 history 요구가 다를 수 있다.
