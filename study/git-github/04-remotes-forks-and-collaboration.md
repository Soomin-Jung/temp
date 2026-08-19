# 04. Remote, fork와 분산 협업

## 1. Remote의 정체

remote는 다른 repository의 URL과 fetch/push 설정에 붙인 이름이다. `origin`은 관례적인 기본 이름일 뿐 특별한 서버 타입이 아니다.

```bash
git remote -v
git remote get-url origin
git remote show origin
```

한 local repository에 여러 remote를 둘 수 있다.

```bash
git remote add upstream https://github.com/upstream/project.git
git remote add company ssh://git.example.com/platform/project.git
```

일반적인 fork workflow:

- `origin`: 내가 push할 fork
- `upstream`: 원본 project

## 2. Remote-tracking branch

`origin/main`은 GitHub의 branch에 직접 연결된 실시간 pointer가 아니다. 마지막 `fetch` 때 관측한 원격 main 상태를 local에 기록한 remote-tracking ref다.

```text
main         local branch
origin/main  마지막 fetch 시점의 remote main
```

GitHub에서 main이 전진해도 fetch 전까지 local `origin/main`은 움직이지 않는다.

```bash
git fetch origin
```

fetch는 object와 remote-tracking ref를 갱신하지만 현재 working tree나 local branch를 자동 통합하지 않는다. 그래서 상태를 관찰하기 좋은 안전한 첫 단계다.

## 3. Fetch, pull, push

### Fetch

```bash
git fetch --prune origin
```

- 원격 object와 ref 정보를 가져온다.
- 현재 branch에 merge/rebase하지 않는다.
- `--prune`은 원격에서 삭제된 branch의 stale tracking ref를 정리한다.

### Pull

`git pull`은 대체로 `fetch + integrate`다. integrate는 설정과 option에 따라 merge 또는 rebase가 된다.

```bash
git pull --ff-only
git pull --rebase
```

초심자에게 pull이 혼란스러운 이유는 두 동작이 합쳐져 있기 때문이다. 중요한 branch에서는 `fetch` 후 graph를 보고 명시적으로 merge/rebase하는 습관이 안전하다.

### Push

```bash
git push -u origin feature/endpoint-routing
```

- local object를 remote로 전송한다.
- remote branch ref를 갱신한다.
- `-u`는 local branch와 remote branch의 upstream tracking 관계를 설정한다.

push는 PR을 자동으로 merge하지 않고, GitHub에 branch가 생겼다고 review가 완료되는 것도 아니다.

## 4. Ahead와 behind

```bash
git status -sb
git rev-list --left-right --count origin/main...main
```

예를 들어 `2 3`이면 한쪽에만 있는 commit 수가 각각 표시된다. 정확한 방향은 명령 operand 순서와 함께 해석한다.

```bash
git log --oneline origin/main..main
git log --oneline main..origin/main
```

서로 모두 commit이 있다면 branch가 diverged한 상태다. 이때 단순 fast-forward는 불가능하며 merge 또는 rebase 정책을 선택해야 한다.

## 5. Refspec의 최소 이해

일반 clone의 fetch refspec 예:

```text
+refs/heads/*:refs/remotes/origin/*
```

원격의 모든 branch ref를 local `refs/remotes/origin/*`에 대응시킨다는 뜻이다. `+`는 fetch 과정에서 remote-tracking ref의 non-fast-forward 갱신도 허용한다.

PR ref, 특정 branch만 가져오기, mirror repository를 다룰 때 refspec 이해가 필요하다. 일상 업무에서는 remote-tracking branch가 remote branch의 local 관측본이라는 점만 정확히 잡아도 충분하다.

## 6. Fork workflow

권한이 없는 upstream project에 기여하거나 조직 경계를 분리할 때 fork를 사용한다.

```bash
git clone <my-fork-url>
cd project
git remote add upstream <upstream-url>
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
git switch -c feature/my-change
```

PR의 base repository는 upstream이고 head repository는 fork가 된다.

보안상 fork PR은 신뢰하지 않은 코드다. secret과 내부 self-hosted runner가 fork의 workflow code에 노출되지 않도록 Actions event와 runner policy를 별도로 설계해야 한다.

## 7. Upstream fork를 회사에서 유지하는 경우

vLLM Production Stack 같은 upstream project를 회사 요구에 맞게 custom할 때는 단순 feature branch보다 장기 동기화 전략이 필요하다.

권장 기록:

- upstream remote와 기준 tag/commit
- 회사 patch를 기능별 commit 또는 patch series로 분리
- upstream update 시 적용 순서와 conflict resolution 근거
- upstream에 기여 가능한 변경과 회사 전용 변경의 경계
- generated chart, CRD, lockfile의 재생성 방법

`vendor/upstream-version.md` 같은 파일에 기준 SHA를 적는 것만으로는 부족하다. 실제 Git remote, tag, merge-base와 CI 재현 결과를 함께 관리한다.

## 8. Authentication과 author identity

다음은 서로 다르다.

- commit author name/email: 누가 변경을 작성했다고 기록했는가?
- GitHub authentication: 누가 push할 권한으로 서버에 접속했는가?
- commit signature: commit/tag가 특정 signing key로 서명되었는가?
- branch protection: 그 push 또는 merge가 policy를 만족했는가?

name/email을 다른 값으로 설정한다고 그 사람의 GitHub 권한이 생기는 것은 아니다. 반대로 shared machine에서 credential을 잘못 사용하면 commit author와 실제 pusher가 다를 수 있다.

## 9. 협업 전 확인 명령

```bash
git remote -v
git fetch --all --prune
git branch -vv
git log --graph --decorate --oneline --all -20
git merge-base origin/main HEAD
```

이 다섯 결과를 보면 “어디에 push하는가”, “무엇을 기반으로 했는가”, “누가 먼저 전진했는가”를 대부분 설명할 수 있다.
