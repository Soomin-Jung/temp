# Claude Code File-Deletion Guard in User-Controlled Environments

업데이트: 2026-08-24 KST  
상태: Design / operational safety note

## 문제 정의

공용 서버에서 `/etc` 같은 system-wide 정책을 수정할 권한이 없고, 사용자는 각자 홈 디렉터리에서 Claude Code를 실행한다.

단순한 settings rule 예:

```json
{
  "deny": ["Bash(rm *)"]
}
```

만으로는 충분하지 않다.

이유:

- `rm` 외의 삭제 명령/스크립트가 존재할 수 있음
- Bash 외 built-in tool 또는 추가 tool/MCP가 filesystem mutation을 수행할 수 있음
- shell wrapper, Python, Perl, find -delete 등 우회 경로가 존재
- 사용자가 설치한 도구의 mutation semantics까지 문자열 deny rule로 모두 열거하기 어려움

따라서 목표는 특정 command string 차단이 아니라 **filesystem destructive action을 중앙 guard에서 판정**하는 것이다.

## 현실적인 배포 경계

system-wide `/etc` 정책을 사용할 수 없으므로 사용자 환경에서 강제 가능한 경계를 사용한다.

```text
source claude.env
  -> Claude Code runtime/env 초기화
  -> user-level settings / hook config 주입
  -> PreToolUse guard
  -> allow / deny
```

`claude.env`를 공통 진입점으로 사용한다면 Node/Claude 관련 환경변수뿐 아니라 guard script 위치와 settings/hook 경로를 함께 고정할 수 있다.

단, 사용자가 `claude.env`를 우회해 Claude Code를 직접 실행할 수 있는 환경이라면 이는 OS-level mandatory access control이 아니다. **사용자 수준 정책 강제**와 **운영체제 수준 강제**를 구분해야 한다.

## PreToolUse Hook의 역할

PreToolUse hook은 tool이 실제 실행되기 전에 tool name과 input을 검사해 실행 허용/거부를 결정하는 interception point로 사용한다.

장점:

- `Bash(rm *)` 같은 단일 문자열 rule보다 넓은 문맥을 볼 수 있음
- tool별 JSON input을 파싱해 path / operation을 검사 가능
- 공통 정책 로직을 한 스크립트에 모을 수 있음
- logging/audit 가능

핵심은 **tool 이름만 보고 차단하지 않고 tool의 mutation intent와 target path를 함께 검사하는 것**이다.

## 정책 모델

### 1. Protected path

삭제/파괴적 overwrite를 금지할 경로를 정의한다.

예:

```text
$HOME/work
$HOME/project
공용 작업 디렉터리
중요 artifact / model / source tree
```

실제 내부 경로는 공개 문서에 기록하지 않는다.

### 2. Destructive operation taxonomy

최소 다음 범주를 검사한다.

- delete / unlink / remove
- recursive directory removal
- move that overwrites or evacuates protected content
- truncate / destructive overwrite
- filesystem format/reset 계열
- tool-specific delete mutation

### 3. Bash / shell

shell command는 AST 수준 분석이 가장 이상적이지만, hook에서 모든 shell semantics를 완벽히 해석하기는 어렵다.

따라서 실용적인 정책은:

1. 명백한 destructive command 차단
2. protected path가 포함된 destructive expression 차단
3. shell interpreter를 통한 우회(`python -c`, `find -delete` 등)는 high-risk로 분류
4. 불명확한 destructive intent는 deny 또는 explicit approval 대상으로 처리

## Bash 외 tool

Claude Code가 Bash 하나만 사용하는 것으로 가정하면 안 된다.

Guard는 다음 두 층으로 구성하는 것이 안전하다.

### Known tools

알고 있는 built-in / MCP tool에 대해서는 input schema별 adapter를 둔다.

```text
Tool event
  -> normalize
      { operation, target_paths, destructive, source_tool }
  -> policy engine
  -> allow / deny
```

예를 들어 filesystem tool이 `{path, operation}`을 주면 shell parsing을 하지 않고 직접 delete semantics를 판단한다.

### Unknown / user-added tools

사용자가 추가한 tool은 semantics를 사전에 알 수 없으므로 fail-open으로 두면 "모든 삭제 방지" 목표와 충돌한다.

권장 원칙:

- mutation capability가 불명확한 unknown tool은 기본적으로 deny/approval
- read-only로 명확히 allowlist된 tool만 자동 허용
- 새 tool을 추가할 때 adapter 또는 permission classification을 등록

즉 **denylist보다 capability allowlist에 가깝게 운영**한다.

## 방어 계층

```text
Layer 1: Claude settings permissions
Layer 2: PreToolUse policy hook
Layer 3: tool-specific adapter / capability allowlist
Layer 4: filesystem Unix permission / ownership
Layer 5: container/sandbox/immutable workspace (가능한 환경)
```

PreToolUse 하나가 OS permission을 대체하지 않는다.

반대로 `/etc` 권한이 없는 환경에서도 Layer 1~3은 사용자 홈/공통 `claude.env` 배포 체계로 현실적으로 적용할 수 있다.

## 구현 시 요구사항

Guard script는 다음 특성을 갖는다.

- deterministic
- deny reason 출력
- path canonicalization (`..`, symlink, relative path 고려)
- protected path prefix boundary 정확히 처리
- command/input 원문을 과도하게 로그하지 않음
- allow/deny audit 기록
- hook 자체 실패 시 정책(fail-closed 또는 approval)을 명시
- 버전 고정 및 checksum/배포 버전 표시

## 우회 방지 시 주의점

문자열 검색만으로는 충분하지 않다.

예:

```text
rm file
find ... -delete
python -c 'os.remove(...)'
perl -e 'unlink ...'
mv source /tmp/...  # 목적에 따라 사실상 제거
```

따라서 "모든 가능한 삭제 문법을 정규식으로 완벽히 잡는다"가 목표가 되어서는 안 된다.

정확한 목표는:

1. known structured tool은 schema 기반으로 판정
2. shell은 high-risk destructive behavior를 보수적으로 차단
3. unknown mutation tool은 기본 허용하지 않음
4. 중요한 데이터는 Unix permission/sandbox로 최종 방어

## 현재 결론

공용 서버에서 system-wide `/etc` 제어가 불가능한 조건에서는:

**공통 `claude.env` + user-level settings + PreToolUse guard + known-tool adapter + unknown mutation tool deny/approval** 조합이 현실적인 Claude Code 레벨의 정책 경계다.

단, 사용자가 이 실행 경로 자체를 우회할 수 있다면 이를 절대적인 보안 경계로 표현하지 않는다. truly mandatory한 삭제 금지는 filesystem permission, ACL, container/sandbox 같은 OS/runtime enforcement가 필요하다.
