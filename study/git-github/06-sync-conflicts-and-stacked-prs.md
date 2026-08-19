# 06. Branch 동기화, conflict와 stacked PR

## 1. 가장 흔한 상황

```text
main:      C0
branch A:  C0---A1---A2
branch B:             \---B1---B2
```

B는 A의 변경이 필요해서 A tip에서 만들었다. A와 B의 PR을 각각 열었고, 그사이 A가 main에 먼저 merge되었다. 이제 B를 최신 main 기준으로 정리해야 한다.

먼저 알아야 할 점:

- conflict가 없고 repository가 최신 base를 강제하지 않으면 B PR이 그대로 merge 가능할 수 있다.
- 하지만 최신 main 조합으로 CI를 확인하고 PR diff를 깨끗하게 유지하려면 B를 동기화하는 것이 일반적이다.
- A가 어떤 방식으로 merge되었는지에 따라 graph와 정리 방법이 달라진다.

## 2. 일반 branch의 최신 main 반영

### 방법 A: main을 feature에 merge

```bash
git fetch origin
git switch branch-b
git merge origin/main
```

장점:

- 기존 commit ID를 보존한다.
- 공동 작업 branch에서도 안전한 기본값이다.
- 일반 push로 갱신할 수 있다.

```bash
git push origin branch-b
```

단점:

- 동기화할 때마다 merge commit이 생길 수 있다.
- linear history 정책과 맞지 않을 수 있다.

### 방법 B: feature를 최신 main 위로 rebase

```bash
git fetch origin
git switch branch-b
git rebase origin/main
git push --force-with-lease origin branch-b
```

장점:

- PR의 고유 변경이 최신 main 뒤에 선형으로 정리된다.
- base 조합에서 각 commit을 다시 검증한다.

단점:

- commit SHA가 바뀐다.
- 공동 branch에서는 다른 작업자의 history를 깨뜨릴 수 있다.

## 3. A가 merge commit으로 합쳐진 경우

A의 original commit이 main history의 ancestor로 남으면 B와 main의 merge base가 보통 A tip까지 전진할 수 있다. 이때 B에서 `git rebase origin/main`하면 B 고유 commit만 새 main 위에 재생되는 형태가 된다.

```text
main: C0---A1---A2---M
                  \
B old:             B1---B2

B new: C0---A1---A2---M---B1'---B2'
```

## 4. A가 squash merge된 경우

squash merge는 A1+A2의 전체 diff를 main의 새 commit S 하나로 기록한다.

```text
main: C0---S
       \
A/B:    A1---A2---B1---B2
```

main에는 A1, A2 object가 ancestor로 존재하지 않을 수 있다. 단순 `git rebase origin/main`은 Git의 patch equivalence 탐지에 따라 A commit을 skip할 수도 있지만, commit 분할과 squash 결과가 다르면 A 변경을 다시 적용하려다 conflict나 중복 diff가 생길 수 있다.

이때 의도를 가장 명확히 표현하는 명령은 `--onto`다.

```bash
git fetch origin
git switch branch-b

# A2 이후의 B 고유 commit만 origin/main 위에 재생
git rebase --onto origin/main A2 branch-b

git push --force-with-lease origin branch-b
```

`A2`는 “B에 포함되어 있지만 버릴 A 계층의 마지막 commit”이다. 결과:

```text
main: C0---S
           \
B new:      B1'---B2'
```

SHA나 경계를 확신할 수 없다면 새 branch를 main에서 만들고 B 고유 commit만 cherry-pick하는 방법이 더 관찰 가능하다.

```bash
git switch -c branch-b-clean origin/main
git cherry-pick B1 B2
```

## 5. Stacked PR의 정식 구조

Stacked PR은 큰 변경을 의존 관계가 있는 작은 PR chain으로 나눈다.

```text
main <- PR A <- PR B <- PR C
```

- PR A: 기반 refactor
- PR B: A를 이용한 기능
- PR C: B를 이용한 운영·문서 변경

각 PR의 base를 바로 아래 branch로 설정하면 reviewer는 해당 계층의 diff만 본다. 아래 PR이 merge되면 위 PR의 base를 main 또는 새 하위 branch로 retarget/rebase한다.

2026년 GitHub의 stacked PR 기능과 `gh stack` 명령은 stack 생성, cascading rebase, retarget과 merge를 지원한다. 조직에서 사용할 수 있다면 수동 `--onto` 실수를 줄일 수 있다. 다만 underlying graph를 이해하지 못한 채 도구에만 의존하면 conflict나 squash 결과를 해석하기 어렵다.

## 6. Conflict marker를 읽는 법

일반 merge conflict:

```text
 <<<<<<< HEAD
현재 branch 쪽 내용
 =======
합치려는 branch 쪽 내용
 >>>>>>> origin/main
```

rebase 중에는 “ours/theirs” 직관이 뒤집혀 느껴질 수 있다. 현재 화면의 label만 믿지 말고 다음을 사용한다.

```bash
git status
git diff
git diff --base <file>
git diff --ours <file>
git diff --theirs <file>
```

conflict 해결은 marker를 지우는 작업이 아니다. 양쪽 변경의 **의도**를 결합하여 최종 파일을 만든 뒤 test해야 한다.

## 7. Conflict 해결 절차

1. 현재 작업이 merge인지 rebase인지 확인한다.
2. conflict 파일과 공통 조상을 확인한다.
3. 양쪽 PR의 목적과 invariant를 읽는다.
4. 최종 결과를 직접 편집한다.
5. marker 잔존 여부와 syntax를 검사한다.
6. 파일을 stage하고 operation을 계속한다.
7. 전체 test와 diff를 다시 확인한다.

```bash
git status
git diff --name-only --diff-filter=U
rg '^(<<<<<<<|=======|>>>>>>>)' .

git add <resolved-files>
git merge --continue       # merge 중
# 또는
git rebase --continue      # rebase 중
```

진행 방향이 잘못되었으면 억지로 끝내지 않는다.

```bash
git merge --abort
git rebase --abort
git cherry-pick --abort
```

## 8. Generated file conflict

Helm rendered manifest, lockfile, generated schema 같은 파일은 양쪽 text를 수동 결합하기보다 source와 generator 기준을 먼저 정한다.

1. source conflict 해결
2. tool version 고정 확인
3. generated output 재생성
4. diff와 deterministic 여부 검증

generated file만 “ours”나 “theirs”로 선택하면 source와 결과가 불일치할 수 있다.

## 9. 동기화 후 PR에서 확인할 것

- Files changed에 A의 이미 merge된 변경이 다시 나타나지 않는가?
- B 고유 commit이 모두 남았는가?
- base branch가 `main` 또는 의도한 stack 하위 branch로 설정되었는가?
- approval이 dismiss되었는가?
- CI가 rebase/merge 후 새 head SHA를 검사했는가?
- force push 사이에 다른 사람이 올린 commit이 유실되지 않았는가?

## 10. 상황별 결정표

| 상황 | 기본 선택 |
|---|---|
| 개인 feature branch, 최신 main 필요 | `rebase origin/main` |
| 공동 feature branch | `merge origin/main` |
| A에서 파생한 B, A가 regular merge | B를 최신 main에 rebase |
| A에서 파생한 B, A가 squash merge | `rebase --onto` 또는 clean branch + cherry-pick |
| conflict가 너무 복잡하고 경계 불명확 | abort 후 graph와 commit 범위 재확인 |
| GitHub stacked PR 사용 가능 | stack 도구 사용, graph도 함께 확인 |

> 이 장의 본질: branch 동기화는 “최신 코드를 복사하는 일”이 아니라, **내 변경의 부모를 최신 통합 지점으로 다시 연결하고 그 조합을 재검증하는 일**이다.
