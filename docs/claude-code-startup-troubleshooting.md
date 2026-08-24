# Claude Code Startup Hang Troubleshooting

업데이트: 2026-08-24 KST  
상태: Troubleshooting reference

## 문제 유형

Claude Code가 특정 사용자 환경에서 시작 직후 멈추고, 로그가 mTLS / Git remote detection 부근에서 더 진행되지 않는 형태의 startup hang을 진단한다.

과거 사례 기준 버전: Claude Code 2.1.153.

## 우선 의심 영역

사용자별로만 재현될 경우 binary 자체보다 다음 user-scoped state를 우선 의심한다.

- `additionalDirectories`처럼 접근 불가능하거나 느린 경로
- MCP server / hook / plugin 초기화
- user settings/config/cache corruption
- proxy / certificate / mTLS 관련 환경변수
- HOME 하위 파일 권한
- shell profile에서 주입되는 환경변수/명령
- Git remote 또는 credential helper 접근 지연

## 진단 순서

1. **clean environment 비교**
   - 임시 HOME 또는 최소 user config로 실행해 user-scoped state와 binary/runtime 문제를 분리한다.
2. **settings 최소화**
   - MCP, hook, plugin, additional directory를 한 번에 하나씩 제외한다.
3. **환경변수 비교**
   - 정상 사용자와 문제 사용자의 proxy, cert, Node, Claude 관련 env 차이를 본다.
4. **filesystem 접근 확인**
   - HOME/config/cache 및 참조 디렉터리의 ownership/permission/응답 지연을 확인한다.
5. **system-call tracing**
   - Linux에서는 `strace` 등으로 마지막 blocking syscall/path를 확인한다.
6. **Git/credential 경로 분리**
   - repository 밖에서 실행하거나 Git remote/credential helper를 임시 배제해 Git 초기화 구간을 분리한다.

## 해석 원칙

- 로그의 마지막 줄이 원인이라고 단정하지 않는다. 그 직후 호출된 filesystem/network/plugin 작업에서 blocking됐을 수 있다.
- 동일 binary가 다른 사용자에게 정상이라면 global install보다 HOME/config/env 차이를 먼저 본다.
- 설정 파일을 무작정 삭제하기보다 clean-home 비교 후 원인 범위를 좁혀 재현 가능한 최소 조건을 만든다.

## 삭제 방지 정책과의 관계

이 문서는 startup hang 진단용이다. 파일 삭제 제한/PreToolUse 정책은 [Claude Code File-Deletion Guard](claude-code-delete-guard.md)에서 별도로 관리한다.
