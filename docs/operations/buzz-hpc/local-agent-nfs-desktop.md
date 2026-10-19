# Buzz HPC — Linux X 세션 + NFS 공용 Desktop + Local Agent 운영안

검토일: 2026-10-19 KST. **설계 및 공식 소스 정적 검토; 사내 Linux X 세션/NFS/사내 CA/GHES 실기동 검증 전.**  
검토 범위: `block/buzz` 2026-10-19 공개 main의 Desktop/Agent 소스, 공식 AppImage 문서. 기존 [Relay 설치 가이드](README.md)는 별도의 고정 커밋(`3debf18f...`) 기준이므로 **Desktop 릴리스와 Relay API 호환성은 선택한 두 버전으로 별도 검증한다.**

## 1. 결론 / 배포 책임 경계

**기본값은 Local Agent**: `BackendKind::Local`이며 `emptyWhereToRunDraft.runOn="local"`. `buzz-backend-kubernetes`를 사용자에게 제공하거나 K8s RBAC을 부여할 필요가 없다.

```text
HPC Kubernetes                       Linux X Session Host (multiuser)
┌──────────────────────────┐          ┌───────────────────────────────────────┐
│ buzz-relay (Helm)        │◄── WSS ──┤ Buzz Desktop [user A, NFS binary]     │
│  ├─ Postgres / Redis     │          │   └─ buzz-acp → Coding Agent → tools  │
│  └─ S3 object storage    │          │      └─ git / shell / user workspace  │
└──────────────────────────┘          │ Buzz Desktop [user B, same binary]    │
                                      │   └─ buzz-acp → Coding Agent → tools  │
┌──────────────────────────┐          └──────────────────┬────────────────────┘
│ Internal vLLM/LiteLLM    │◄──── inference HTTP(S) ─────┘
└──────────────────────────┘

NFS (/nfs/apps/buzz): 공용 읽기 전용 실행 배포물.  각 사용자 HOME: 사설 상태·개인키·스킬·작업공간.
```

- **여기서 Local이란 접속 단말이 아니라 Desktop 프로세스를 실제 실행하는 Linux X 서버**이다. SSH X11 forwarding/X2Go/VNC 등에서는 호스트 CPU·메모리·파일시스템과 Git egress 정책이 적용된다.
- Relay는 Nostr 이벤트/인증/협업 저장소이지 LLM 추론이나 코드 편집 프로세스가 아니다. GPU 추론은 기존 vLLM이 담당한다.
- Desktop이 띄운 `buzz-acp`와 Coding Agent가 개발 작업 및 GHES 접근을 수행한다. 기본 OSS `buzz-agent` 외 Goose/Codex/Claude Code는 각 런타임의 설치·호환성·자격증명이 별도로 필요할 수 있다.
- 각 사용자는 **Kubernetes 권한 없음**. NFS에 단일 공용 설치, 실행 프로세스와 데이터는 사용자 UID별 격리.
- Desktop 종료 시 관리하던 Local Agent는 정리된다. 장시간 무인 작업/PC·X 서버 장애 지속 실행을 요구하면 별도의 Remote Agent 설계가 필요하다.

검증 근거:
- [Local backend 기본값](https://github.com/block/buzz/blob/main/desktop/src-tauri/src/managed_agents/types.rs), [Run on 기본값](https://github.com/block/buzz/blob/main/desktop/src/features/agents/ui/whereToRunIntent.ts)
- [Local Agent 종료 처리](https://github.com/block/buzz/blob/main/desktop/src-tauri/src/shutdown.rs), [Relay 아키텍처](https://github.com/block/buzz/blob/main/ARCHITECTURE.md)
- [원격 Agent는 선택 경로](https://github.com/block/buzz/blob/main/docs/remote-agents.md)

## 2. NFS 공용 디렉터리 배치

```text
/nfs/apps/buzz/                         # 관리자 쓰기 / 일반 사용자 읽기·실행
├── env.sh                              # nfs-client/env.sh를 사내 주소로 수정
├── bin/
│   └── buzz-desktop                    # nfs-client/buzz-desktop
├── versions/
│   ├── desktop-v<version-A>/
│   │   ├── Buzz.AppImage               # 검증된 배포본, 업데이트 시 덮어쓰기 금지
│   │   └── SHA256SUMS
│   └── desktop-v<version-B>/
│       └── Buzz.AppImage
└── current -> versions/desktop-v<검증된 버전>/
```

**NFS mount가 `noexec`이면 AppImage를 이 경로에서 직접 실행할 수 없다.** 실행 허용 NFS export 또는 각 X 서버의 관리자 관리 로컬 실행 디렉터리가 필요하다. `noexec`를 우회하려고 임의 `LD_LIBRARY_PATH`, `--no-sandbox` 등을 강제하지 않는다.

공유 배포물을 여러 사용자가 실행해도 사용자별 홈의 `~/.buzz`, XDG 데이터 및 Linux Keyring은 서로 공유하지 않아야 한다. `/nfs/apps/buzz`에 사용자별 설정/인증키를 저장하지 않는다. 관리자 외 일반 사용자에게는 버전/심볼릭 링크 수정 권한을 주지 않는다.

### 관리자 설치 예시 (반입한 AppImage를 미리 검증한 후)

```bash
ROOT=/nfs/apps/buzz
VER=desktop-v<검증된버전>
ASSET=/path/to/approved/Buzz_x86_64.AppImage

# sha256sum -c <승인된 checksum 파일> 등 외부 검증 후 반입.
install -d -m 0755 "$ROOT/versions/$VER" "$ROOT/bin"
install -m 0755 "$ASSET" "$ROOT/versions/$VER/Buzz.AppImage"
sha256sum "$ROOT/versions/$VER/Buzz.AppImage" \
  > "$ROOT/versions/$VER/SHA256SUMS"

# repo의 nfs-client 템플릿을 운영 경로로 복사하며 주소/경로를 수정한다.
install -m 0644 nfs-client/env.sh "$ROOT/env.sh"
install -m 0644 nfs-client/buzz-desktop "$ROOT/bin/buzz-desktop"
# env.sh의 BUZZ_NFS_ROOT와 BUZZ_RELAY_URL / HTTP를 실제 값으로 변경.

# 버전 디렉터리를 절대 덮어쓰지 않음. 같은 NFS 파일시스템에서 원자적 교체.
ln -s "versions/$VER" "$ROOT/.current.next"
mv -Tf "$ROOT/.current.next" "$ROOT/current"
```

참고: `mv -T`는 GNU coreutils 문법. 공용 AppImage가 이미 실행 중일 때 파일을 교체하지 않는다. 신규 버전을 별도 디렉터리에 두고 `current` 심볼릭 링크만 교체한다. 업데이트 정책은 사용자 재시작과 승격/롤백으로 관리한다. **Buzz 내장 updater가 읽기 전용 NFS 설치를 안전하게 갱신한다고 가정하지 않는다.**

## 3. 사용자 실행: source → buzz

`[nfs-client/env.sh](nfs-client/env.sh)` 예시는 `BUZZ_NFS_ROOT`, `BUZZ_RELAY_URL`, `BUZZ_RELAY_HTTP`, X11용 `GDK_BACKEND`를 설정하고 **대화형 Bash 함수 `buzz`를 정의**한다. `[nfs-client/buzz-desktop](nfs-client/buzz-desktop)`는 실제 AppImage를 실행한다.

```bash
source /nfs/apps/buzz/env.sh
buzz
# 그래픽 장애 시 선택:
buzz --safe-rendering
```

**왜 `buzz`를 실제 PATH 실행 파일로 만들지 않는가?** Buzz Agent가 사용하는 JSON CLI의 실제 바이너리 이름 또한 `buzz`이기 때문이다. `/nfs/apps/buzz/bin/buzz` Desktop 래퍼를 PATH에 넣으면 Agent가 `buzz messages ...`를 실행할 때 GUI 래퍼와 충돌할 위험이 있다. **대화형 셸 함수만 `buzz`로 노출**하고 프로세스에는 내보내지 않는다. 프로세스 실행 경로는 `buzz-desktop`으로 유지한다.

Launcher는 `BUZZ_PRIVATE_KEY` 또는 `BUZZ_SHARE_IDENTITY`가 설정되어 있으면 거부한다. 공용 `env.sh`에는 **어떠한 사용자 비밀, 토큰, Nostr nsec도 넣지 않는다**. 키는 각 사용자 Desktop이 관리한다. `BUZZ_RELAY_URL`는 Desktop에서 런타임 환경변수로 읽지만, 앱 UI에서 변경 후 저장한 workspace relay override가 우선할 수 있다.

## 4. Desktop Linux 호환성 — 실제 X 서버에서 필수 사전 검증

공식 릴리스는 Linux `.AppImage`와 `.deb`를 제공한다. NFS 단일파일 공용 배포에는 AppImage가 운영상 간편하지만 **배포 파일이라고 해서 모든 사내 Linux에서 실행되는 것은 아니다**.

1. **CPU 아키텍처**: X 서버 `uname -m`과 AppImage 아키텍처 일치.
2. **glibc**: `getconf GNU_LIBC_VERSION`. 공식 [Linux rendering troubleshooting](https://github.com/block/buzz/blob/main/docs/linux-rendering-troubleshooting.md)에 따르면 AppImage v0.5.2 이상은 **glibc 2.39**를 최소 요구한다고 명시되어 있음. RHEL 8/9 등 오래된 HPC 배포판에서는 실행 실패 가능성이 높음. 버전별 실측이 필수.
3. **GLib / GTK / WebKitGTK / GStreamer / 그래픽**: 공식 [RELEASING.md](https://github.com/block/buzz/blob/main/RELEASING.md)의 GLib·호스트 그래픽 라이브러리 조건과 실제 바이너리의 링킹 결과를 확인한다. `.deb`도 오래된 OS에서 무조건 동작한다고 가정하지 않는다.
4. **AppImage + FUSE**: 실행 호스트의 FUSE 사용 정책과 `libfuse2` 또는 `libfuse2t64` 필요 여부. FUSE가 막혀 있다면 관리자가 AppImage를 **한 번만** `--appimage-extract`하여 별도 AppDir를 만들 수 있지만, `AppRun`과 동봉 sidecar가 해당 호스트에서 정상 동작하는지 재검증 필요. **각 사용자 로그인 때 `--appimage-extract-and-run` 하는 형태는 비추천**(반복 I/O/시작 지연).
5. **GUI 세션**: `DISPLAY`, `XAUTHORITY`, 사용자 세션 D-Bus (`DBUS_SESSION_BUS_ADDRESS`) 및 `XDG_RUNTIME_DIR` 확인. 기존 X 세션 환경을 임의로 재설정하지 않는다. GDK_BACKEND=x11은 X11 운영 가정이다.
6. **NFS**: 실행 허용 (`noexec` 아님), ACL, 장시간 읽기 및 서버 장애 처리. `/tmp`가 `noexec`이면 AppImage의 FUSE 마운트/보조 바이너리 실행도 호스트별 확인.

```bash
uname -m
getconf GNU_LIBC_VERSION
findmnt -T /nfs/apps/buzz -o TARGET,FSTYPE,OPTIONS
printf 'DISPLAY=%s\nDBUS_SESSION_BUS_ADDRESS=%s\nXDG_RUNTIME_DIR=%s\n' \
  "${DISPLAY:-}" "${DBUS_SESSION_BUS_ADDRESS:-}" "${XDG_RUNTIME_DIR:-}"
source /nfs/apps/buzz/env.sh
buzz
```

**호환성 차단 시**: 패키지 자체를 실제 HPC 배포판에서 빌드해 검증하거나, 앱 실행이 가능한 X 세션 호스트를 별도로 제공한다. glibc 강제 교체는 금지.

공식 참고: [AppImage FUSE](https://docs.appimage.org/user-guide/troubleshooting/fuse.html), [Buzz Linux Rendering](https://github.com/block/buzz/blob/main/docs/linux-rendering-troubleshooting.md).

## 5. 사용자별 상태, 비밀키, Agent CWD

| 데이터 | 권장 경로 및 보안 |
|---|---|
| 공용 Desktop 실행 파일 | `/nfs/apps/buzz/versions/<ver>/Buzz.AppImage`, admin만 쓰기 |
| Agent Nest / Skills | `$HOME/.buzz/` (`~/.buzz/.agents/skills` 포함), 사용자 전용 |
| Desktop 앱 상태 | `${XDG_DATA_HOME:-$HOME/.local/share}/xyz.block.buzz.app` 등 앱 번들 ID 기반 경로 |
| Nostr identity, agent keys | Linux Secret Service(Keyring) 우선; 실패 시 앱의 사용자 전용 `0600` 파일 대체 경로 |
| XDG_RUNTIME_DIR / DBus | 로그인 세션에서 제공, 공용 NFS 경로에 재할당 금지 |
| Git checkout/빌드 | 사용자 전용 `$HOME` 또는 승인된 로컬 scratch; 실사용 repo/skill 지속성이 필요하면 별도 PVC/NFS HOME 정책 |

NFS HOME을 사용한다면 UID 분리, 홈 디렉터리 ACL, 비밀파일 권한, 키 백업 및 NFS 장애를 점검한다. Secret Service가 없는 X 서버에서는 개인키가 사용자 파일로 저장되므로 **HOME이 네트워크 공유 저장소라는 사실을 보안 검토에 반영**한다. 여러 X 호스트에서 동일한 HOME을 동시에 열면 앱의 단일 인스턴스/키링·설정 저장을 교차 호스트까지 조정해주는지 보장되지 않으므로 동시 실행을 초기에는 제한한다.

근거: [Buzz SECURITY.md](https://github.com/block/buzz/blob/main/SECURITY.md), [Desktop Nest](https://github.com/block/buzz/blob/main/desktop/src-tauri/src/managed_agents/nest.rs).

## 6. 내부 LLM, GHES 설정 책임

Local Agent는 **X 서버 프로세스**이므로 X 서버에서 사내 vLLM DNS/443, GHES/443, 사내 CA, Git/빌드 도구 접근을 검증한다. Relay Pod의 NetworkPolicy와 무관하다.

**기본 `buzz-agent`를 사용할 때만** 다음 변수가 해당된다. 사용자마다 모델/권한을 달리 쓸 수 있으므로 기본값과 자격증명을 분리한다.

```bash
export BUZZ_AGENT_PROVIDER=openai
export OPENAI_COMPAT_BASE_URL=https://llm.hpc.example.internal/v1
export OPENAI_COMPAT_MODEL=<실제 served-model-name>
export OPENAI_COMPAT_API=chat
# OPENAI_COMPAT_API_KEY는 사용자별 Secret/보안 설정에서 공급한다.
# 실 서버가 무인증이라도 buzz-agent는 해당 변수가 필요하므로 정책 검토 후 처리.
```

다른 런타임(Goose/Codex/Claude Code)은 **동일 변수들을 자동 해석하는 것이 아님**. 런타임별 설정을 별도로 확인한다. 기본 Buzz CLI와 Desktop GUI는 다른 프로그램이므로 `buzz` 실행 이름 충돌을 피한다.

GHES가 사내 최초 개발 코드의 source of truth라면:
- 사용자의 `git` / `gh`에서 GHES hostname, TLS trust, 개인별 로그인/토큰 범위를 검증한다.
- **Buzz 프로젝트 내장 GHES private clone/PR UI가 바로 활성화되는 것은 아니다.** 자세한 갭: [GHES 통합](github-enterprise-integration.md).
- 사용자별 작업/AI 호출/보안감사를 별도 정책으로 보완한다.

[공식 buzz-agent Provider 환경 변수](https://github.com/block/buzz/blob/main/crates/buzz-agent/README.md)

## 7. PoC 합격 기준

- [ ] 2개 사용자 UID가 **동일한 NFS AppImage**로 로그인하고 각자 서로 다른 Nostr identity로 메시지 송수신
- [ ] 사용자 A/B의 `~/.buzz`, 앱 데이터/Secret Service가 분리되고 서로 읽기 불가
- [ ] `source env.sh; buzz`로 X11 UI 실행, **`buzz` CLI 이름 충돌 없이** Local Agent가 메시지 수신·답변
- [ ] X 서버에서 로컬 Agent 프로세스 존재, `kubectl` 권한 없이 동작
- [ ] X 서버→내부 LLM 호출 성공, Agent 작업 위치/Repo Clone 및 테스트 확인
- [ ] Desktop 종료 시 Local Agent 종료; Relay 메시지/과거 내역 영속성 유지
- [ ] GHES git clone/push 및 필요한 경우 gh api를 **사용자별 권한**으로 성공 (Buzz 네이티브 GHES UI와 구분)
- [ ] 내부 CA, GLib/glibc, FUSE, NFS `exec`, 리소스 상한, 10+ 동시 UI/Agent 기동 검증
- [ ] 사용자에게 공용 앱/쉘 환경을 변경할 권한 없음, 버전 승격/롤백 가능
- [ ] 내부 updater/런타임의 외부 접속 시도 및 읽기 전용 설치 동작 확인

**배포 판정**: Linux X 서버의 OS/glibc 버전 확인이 가장 먼저다. AppImage가 해당 환경에서 작동하지 않으면 이 NFS 실행 방식 자체를 사전 재검토한다.
