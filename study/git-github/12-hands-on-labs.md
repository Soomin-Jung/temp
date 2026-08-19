# 12. 단계별 hands-on lab

## 실습 원칙

- 실제 업무 repository가 아닌 새 임시 directory에서 시작한다.
- 각 단계에서 `git status`와 graph를 먼저 예측한 뒤 실행한다.
- command 결과가 예상과 다르면 다음 command를 계속 넣지 말고 object/ref 관점에서 설명한다.
- destructive operation 전 backup branch를 만든다.

## Lab 1. Working tree, index, HEAD 세 버전 만들기

```bash
mkdir git-lab
cd git-lab
git init -b main
git config user.name "Git Lab"
git config user.email "git-lab@example.com"

printf 'version 1\n' > app.txt
git add app.txt
git commit -m "add app"

printf 'version 2 staged\n' > app.txt
git add app.txt
printf 'version 3 working tree\n' > app.txt
```

관찰:

```bash
git status
git show HEAD:app.txt
git show :app.txt
sed -n '1,20p' app.txt
git diff
git diff --cached
```

목표: 같은 경로의 세 version을 설명한다.

## Lab 2. Object 직접 보기

```bash
git rev-parse HEAD
git cat-file -p HEAD
git cat-file -p 'HEAD^{tree}'
git ls-tree -r HEAD
```

질문:

- commit object 안에 파일 내용이 직접 들어 있는가?
- author와 committer는 무엇인가?
- app.txt의 이름은 blob과 tree 중 어디에 있는가?

## Lab 3. Branch와 fast-forward

```bash
git switch -c feature/readme
printf '# Git Lab\n' > README.md
git add README.md
git commit -m "add readme"

git switch main
git log --graph --decorate --oneline --all
git merge --ff-only feature/readme
git log --graph --decorate --oneline --all
```

목표: 새 merge commit 없이 main ref만 이동했음을 확인한다.

## Lab 4. 3-way merge와 conflict

```bash
printf 'mode: default\n' > config.yaml
git add config.yaml
git commit -m "add config"

git switch -c feature/mode
printf 'mode: feature\n' > config.yaml
git commit -am "set feature mode"

git switch main
printf 'mode: production\n' > config.yaml
git commit -am "set production mode"

git merge feature/mode
```

관찰:

```bash
git status
git diff --name-only --diff-filter=U
git ls-files -u
```

단순히 한쪽을 고르지 말고 최종 contract를 정한 뒤 수정한다.

```bash
printf 'mode: production\nfeature_enabled: true\n' > config.yaml
git add config.yaml
git commit
git log --graph --decorate --oneline --all
```

## Lab 5. Merge와 rebase graph 비교

새 repository에서 같은 초기 graph를 두 번 만들고 한쪽은 merge, 다른 쪽은 rebase한다.

```bash
git log --graph --decorate --oneline --all
git merge-base main feature/example
```

확인:

- merge commit parent 수
- rebase 전후 feature commit SHA
- `main..feature`와 `main...feature` log/diff 차이

## Lab 6. Staged hunk 분리

한 파일의 서로 다른 함수 두 개를 수정한다.

```bash
git add -p
git diff --cached
git commit -m "fix request validation"
git add -p
git commit -m "refactor log formatting"
```

목표: 파일 단위가 아니라 변경 이유 단위로 commit한다.

## Lab 7. Squash merge 이후 파생 branch 정리

graph:

```text
main -> A branch(A1,A2) -> B branch(B1,B2)
```

로컬에서 A 변경을 squash한 commit S를 main에 만든 뒤 다음을 비교한다.

```bash
git rebase main branch-b
```

실습을 reset한 후:

```bash
git rebase --onto main A2 branch-b
```

확인:

- 어떤 commit이 replay되었는가?
- B diff에 A 변경이 중복되는가?
- `git range-diff`로 rebase 전후 B patch가 같은가?

```bash
git range-diff <old-base>..<old-b-tip> main..branch-b
```

## Lab 8. Reflog 복구

```bash
git switch -c feature/lost
printf 'important\n' > important.txt
git add important.txt
git commit -m "important work"
git rev-parse HEAD

git switch main
git branch -D feature/lost
git reflog
git branch rescue/lost-work <found-sha>
git show rescue/lost-work:important.txt
```

목표: branch ref 삭제와 object 즉시 삭제가 다름을 확인한다.

## Lab 9. Revert와 reset 비교

공개되었다고 가정한 bad commit을 각각 처리한다.

```bash
git branch experiment/revert
git branch experiment/reset

git switch experiment/revert
git revert <bad-sha>

git switch experiment/reset
git reset --hard <bad-sha>^
```

graph와 diff를 비교하고, 다른 개발자가 bad commit 위에서 작업 중이라면 어떤 방법이 안전한지 설명한다.

## Lab 10. Bisect 자동화

여러 commit 중 하나가 output을 깨뜨리도록 만든 뒤 다음 script를 준비한다.

```bash
#!/usr/bin/env bash
set -euo pipefail
test "$(./app.sh)" = "expected"
```

```bash
git bisect start
git bisect bad HEAD
git bisect good <known-good>
git bisect run ./test-output.sh
git bisect reset
```

목표: commit이 독립적으로 build/test 가능해야 하는 이유를 체감한다.

## Lab 11. Pull Request review simulation

작은 연습 repository에서 다음을 수행한다.

1. main protection 설정
2. feature branch push
3. draft PR 생성
4. PR template 작성
5. reviewer 지정
6. CI 실패를 의도적으로 만들고 log에서 첫 실패 step 확인
7. fix push 후 approval과 check 상태 관찰
8. squash merge
9. local branch prune

PR 본문에는 실행한 command와 미검증 영역을 쓴다.

## Lab 12. Actions CI 확장

이 저장소의 `docs-ci.yml`을 출발점으로 다음을 한 번에 하나씩 추가한다.

1. timeout
2. concurrency
3. matrix
4. artifact upload
5. reusable workflow
6. environment approval이 있는 수동 delivery job

각 변경마다 다음을 기록한다.

- 어떤 failure를 막는가?
- token permission이 늘어나는가?
- fork PR에서 실행해도 안전한가?
- required check 이름이 바뀌는가?
- runner 비용과 feedback time은 어떻게 달라지는가?

## 최종 과제

Helm chart repository를 가정하여 다음 deliverable을 만든다.

- branch/PR/merge 정책 1쪽
- CODEOWNERS
- PR template
- lint/schema/render/policy CI
- main ruleset 요구사항
- release tag와 artifact promotion 흐름
- upstream update와 emergency hotfix runbook
- 일부러 만든 conflict와 reflog recovery 시연

성공 기준은 command를 모두 실행하는 것이 아니라 각 graph 변화, 권한 경계, 검증 보장 범위를 설명하는 것이다.
