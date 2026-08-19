# vLLM contribution 실전 checklist

- 확인일: 2026-08-19
- 사용법: PR 하나당 복사하여 실행한 command, SHA, 결과를 실제 값으로 채운다.
- 우선순위: [current contributing guide](https://docs.vllm.ai/en/latest/contributing/)와 [current `AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md)가 이 checklist보다 우선한다.

## 1. 작업 선택

- [ ] current main에서 issue를 재현했다.
- [ ] issue discussion과 linked PR을 읽었다.
- [ ] issue number로 open PR을 검색했다.
- [ ] component·symptom keyword로 open PR을 검색했다.
- [ ] 같은 해결이 최근 main commit에 들어오지 않았는지 확인했다.
- [ ] security issue라면 공개 PR을 만들지 않고 `SECURITY.md` 경로를 따른다.
- [ ] major architecture 변경이면 code 전에 RFC를 열었다.

```bash
gh issue view <issue-number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open \
  --search '<issue-number> in:body'
gh pr list --repo vllm-project/vllm --state open \
  --search '<component> <symptom>'
```

기록:

```text
Issue:
Duplicate search date:
Nearest related PRs:
Why this work is still unique:
```

## 2. 규칙과 owner

- [ ] `CONTRIBUTING.md`와 `docs/contributing/README.md`를 읽었다.
- [ ] root와 변경 directory에 적용되는 `AGENTS.md`/domain guide를 읽었다.
- [ ] `DCO`, PR template, `CODEOWNERS`, relevant CI config를 읽었다.
- [ ] 변경 path의 기존 test와 최근 merged PR을 확인했다.

```bash
find .. -name AGENTS.md -print
git log --oneline -- path/to/change
rg 'target_symbol' tests vllm
```

기록:

```text
Applicable guides:
Likely owners/reviewers:
Existing tests/fixtures:
```

## 3. Fork와 base

- [ ] `origin`은 내 fork, `upstream`은 `vllm-project/vllm`인지 확인했다.
- [ ] local main을 `upstream/main`으로 fast-forward했다.
- [ ] main이 아니라 새 feature branch에서 작업한다.

```bash
git remote -v
git fetch upstream --prune
git switch main
git merge --ff-only upstream/main
git switch -c <type>/<topic>
```

```text
Base SHA:
Branch:
```

## 4. 개발 환경

- [ ] Python 3.12와 project-approved `uv`/`.venv` 환경을 사용한다.
- [ ] Python-only, CUDA/C++, Rust, docs 중 필요한 층만 구성했다.
- [ ] bare system `python3`와 bare `pip install`을 사용하지 않는다.
- [ ] pre-commit hook을 설치했다.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install
```

```text
OS:
Python:
PyTorch:
CUDA/ROCm/XPU:
GPU/CPU:
Model/checkpoint:
```

## 5. 문제와 설계

- [ ] base에서 실패하는 reproducer 또는 성능 baseline을 저장했다.
- [ ] root cause를 call path와 invariant로 설명할 수 있다.
- [ ] 변경 범위와 non-goal을 적었다.
- [ ] user/API/numerical/performance/security risk를 검토했다.
- [ ] 가장 싼 test 층에서 regression을 잡도록 설계했다.

```text
Symptom:
Root cause:
Invariant to preserve:
Change:
Non-goals:
Risk:
Baseline command/result:
```

## 6. 구현과 commit

- [ ] existing pattern, fixture, helper를 우선 재사용했다.
- [ ] unrelated cleanup을 섞지 않았다.
- [ ] generated file은 source를 고친 뒤 공식 command로 재생성했다.
- [ ] staged diff를 직접 읽었다.
- [ ] 모든 commit에 DCO sign-off가 있다.
- [ ] AI assistance를 project policy에 맞게 attribution했다.

```bash
git add -p
git diff --cached --check
git diff --cached
git commit -s -m '<message>'
git log --format=full -5
```

## 7. Local validation

### 공통

- [ ] staged file에 pre-commit을 실행했다.
- [ ] targeted regression test를 실행했다.
- [ ] 인접 component suite를 실행했다.
- [ ] pass/fail/skip 수와 미실행 범위를 기록했다.

```bash
pre-commit run
.venv/bin/python -m pytest tests/path/to/test_file.py -v
git diff --check upstream/main...HEAD
```

### 변경 유형별

- [ ] bugfix: unmodified base에서 regression test가 실패한다.
- [ ] model/output: reference parity와 model eval을 실행했다.
- [ ] kernel: schema/meta-function/opcheck/correctness를 확인했다.
- [ ] performance: hardware·dtype·shape·warm-up을 고정한 before/after가 있다.
- [ ] Rust/build: binary, package, wheel/Docker path별 결과를 확인했다.
- [ ] docs: MkDocs preview/build와 example command를 확인했다.

```text
Command:
Result:
Base or branch SHA:
Environment:
Not run and why:
```

## 8. PR 생성

- [ ] current guide의 title prefix를 썼다.
- [ ] 관련 issue/RFC와 duplicate check를 연결했다.
- [ ] Purpose에 symptom, root cause, scope를 적었다.
- [ ] Test Plan과 Test Result를 분리했다.
- [ ] exact command, environment, before/after, skip을 적었다.
- [ ] documentation 필요 여부를 설명했다.
- [ ] AI assistance를 명시했다.
- [ ] 미완료 검증이 있으면 Draft로 연다.
- [ ] maintainers may edit 권한 정책을 확인했다.

```text
PR URL:
PR base:
PR head:
Head SHA:
Draft/ready:
```

## 9. Review 대응

- [ ] 모든 comment를 이해하거나 clarification을 요청했다.
- [ ] 반영한 comment에는 commit과 test 결과를 연결했다.
- [ ] 기술적으로 동의하지 않으면 invariant와 측정 근거를 제시했다.
- [ ] `action-required`를 처리한 뒤 reviewer에게 re-review를 요청했다.
- [ ] reviewer 배정 후 7일 이상 update가 없을 때만 예의 있게 ping한다.

```text
Changed:
Commit:
Re-run:
Result:
Remaining question:
```

## 10. 기다리는 동안 upstream 동기화

- [ ] main 변화와 dependency PR 상태를 fetch로 확인했다.
- [ ] unrelated main commit 때문에 매일 rebase하지 않는다.
- [ ] relevant refactor, dependency merge, conflict, re-review/final CI를 sync trigger로 삼는다.
- [ ] rewrite 전에 origin/upstream fetch와 backup branch를 만들었다.
- [ ] 개인 branch만 rebase하고, 공동 branch는 합의 없이 rewrite하지 않는다.
- [ ] stacked PR의 dependency가 squash merge되었으면 `--onto` 경계를 확인한다.
- [ ] rebase 후 `range-diff`, 전체 PR diff, test를 다시 확인했다.
- [ ] push 전 remote-only commit이 없는지 확인했다.

```bash
git fetch upstream --prune
git fetch origin --prune
git branch backup/<topic>-before-rebase HEAD
git rebase upstream/main

git diff --stat upstream/main...HEAD
git fetch origin --prune
git log --oneline HEAD..origin/<topic>
git push --force-with-lease origin HEAD:<topic>
```

```text
New base SHA:
Dropped because upstreamed:
Conflict resolution:
New head SHA:
Tests re-run:
```

## 11. CI

- [ ] local cheap tests를 먼저 통과시켰다.
- [ ] approval 또는 `ready` 등 current CI authorization 조건을 확인했다.
- [ ] 필요한 경우에만 `/ci run` 또는 hardware-specific CI를 요청했다.
- [ ] CI build가 current head SHA를 가리킨다.
- [ ] failure를 code, expectation, flaky/infra로 분류했다.
- [ ] fix push 후 새 head에서 필요한 CI를 다시 실행했다.

```text
CI command/comment:
Build URL:
Build SHA:
Failed job/root cause:
Fix commit:
Final result:
```

## 12. Merge 직전

- [ ] PR base/head와 Files changed가 의도한 범위다.
- [ ] 이미 upstream된 diff나 generated artifact가 섞이지 않았다.
- [ ] unresolved/outdated thread와 approval dismissal을 확인했다.
- [ ] current head의 DCO, pre-commit, required check가 성공했다.
- [ ] PR 본문은 과거 revision이 아니라 current diff를 설명한다.
- [ ] user-facing change의 docs/release note 요구를 확인했다.

## 13. Merge 후

- [ ] official main에서 merge commit과 follow-up CI를 확인했다.
- [ ] local main을 fast-forward했다.
- [ ] 사용이 끝난 local/remote feature branch를 정리했다.
- [ ] 후속 issue, known limitation, benchmark 결과를 연결했다.
- [ ] user-visible change라면 포함 release를 추적한다.

```bash
git fetch upstream --prune
git switch main
git merge --ff-only upstream/main
git branch -d <topic>
git fetch origin --prune
```

> 완료 기준: merge button이 눌린 것이 아니라 **current main에서 변경과 검증 결과를 다시 추적할 수 있는 상태**다.
