# Buzz Linux X session: NFS + Singularity/Apptainer

**검토일:** 2026-10-20 · 소스/스크립트 정적 검토 및 가짜 Apptainer 실행 테스트 완료. **SIF 빌드·실제 HPC X11 실행은 미검증.**

## Codex/Claude ACP 및 UTF-8 업데이트 (2026-10-20)

- 최신 `Buzz.def`는 Node.js 24, `@agentclientprotocol/claude-agent-acp`, `@agentclientprotocol/codex-acp`, `@openai/codex`, `en_US.UTF-8` locale을 포함.
- **[DTGPT 환경변수·Codex config.toml·Claude settings.json 및 API Key 오류/로그 진단](dtgpt-acp-configuration.md)** 참조.
- 사용자가 `BUZZ_DESKTOP_LOG_TO_FILE=1 buzz`로 실행하면 Launcher와 GUI stdout/stderr를 `~/.local/state/buzz/`에 저장할 수 있음.
- Codex는 Responses API, Claude는 Anthropic Messages API가 필요하며, DTGPT의 단순 OpenAI Chat Completions 호환성만으로 모두 가능하지 않음.

## 변경 이유

모든 사용자 X 세션 호스트에 `libffi.so.8`이 있지 않고, 호스트 패키지를 관리할 수 없다. 따라서 공식 `.AppImage`를 호스트에서 실행하지 않는다. 대신 **공식 Buzz `.deb`를 Ubuntu 24.04 + libffi8 + WebKitGTK 환경에 설치한 뒤 읽기 전용 `Buzz.sif`로 배포**한다.

AppImage를 SIF 내부에서 다시 실행하는 것보다, native `.deb`를 설치하면 AppImage FUSE와 AppRun의 번들 라이브러리 충돌을 피할 수 있다.

## 1. 빌드 서버에서 SIF 생성

버전과 checksum을 승인받은 [Buzz Release](https://github.com/block/buzz/releases)의 `Buzz_<VERSION>_amd64.deb`를 확보한다. 임의 latest 다운로드 대신 정확한 자산과 SHA-256을 기록한다.

```bash
# Buzz.def가 있는 디렉터리에서 실행
cp /path/to/Buzz_<VERSION>_amd64.deb ./Buzz.deb
sha256sum ./Buzz.deb                  # 승인된 SHA256과 대조

apptainer build Buzz.sif Buzz.def
# 또는 singularity build Buzz.sif Buzz.def

apptainer exec --cleanenv Buzz.sif /bin/bash -c 'ldconfig -p | grep libffi.so.8'
apptainer exec --cleanenv Buzz.sif /bin/bash -c 'command -v buzz-desktop && ! ldd /usr/bin/buzz-desktop | grep "not found"'\napptainer exec --cleanenv Buzz.sif /bin/bash -c 'locale; node --version; codex --version; codex-acp --version; claude-agent-acp --version'
sha256sum Buzz.sif
```

빌드는 관리자/격리된 staging 환경에서 수행하며 베이스 OCI 이미지 및 Ubuntu apt 저장소 접근이 필요하다. 폐쇄망 빌드라면 승인된 registry/apt mirror를 쓴다. **Def의 Ubuntu tag와 apt 버전은 기본적으로 floating**하므로 재현성이 엄격하면 OCI digest 및 apt snapshot을 추가로 고정한다. SIF digest는 승인 기록으로 남긴다.

사내 HTTPS Relay/GHES/vLLM이 비공개 CA를 사용한다면 `%files`에 CA PEM을 `/usr/local/share/ca-certificates/company.crt`로 복사하고 `%post`에서 `update-ca-certificates`를 실행하도록 빌드 정의에 추가해야 한다.

## 2. NFS 배치

```text
/nfs/apps/buzz/
├── env.sh                              # nfs-client/env.sh 사용
├── bin/buzz-desktop                    # nfs-client/buzz-desktop 래퍼
├── versions/desktop-<VER>-ubuntu24/
│   └── Buzz.sif                        # 승인된 read-only SIF
└── current -> versions/desktop-<VER>-ubuntu24
```

```bash
ROOT=/nfs/apps/buzz
VER=desktop-<VER>-ubuntu24
install -d -m 0755 "$ROOT/versions/$VER" "$ROOT/bin"
install -m 0644 Buzz.sif "$ROOT/versions/$VER/Buzz.sif"
install -m 0755 ../nfs-client/buzz-desktop "$ROOT/bin/buzz-desktop"
# env.sh는 기존 배포 유지. BUZZ_NFS_ROOT, BUZZ_RELAY_URL을 실제 운영값으로 수정
ln -s "versions/$VER" "$ROOT/.current.next"
mv -Tf "$ROOT/.current.next" "$ROOT/current"
```

SIF는 일반 사용자가 수정할 수 없는 경로에 둔다. 실행 중 SIF 파일 덮어쓰기를 금지하고 디렉터리와 `current` 링크를 교체한다. NFS `noexec` 환경에서 SIF가 동작하는지는 Singularity 마운트 정책에 따라 별도 검증한다.

## 3. 사용자는 기존 명령 유지

```bash
source /nfs/apps/buzz/env.sh
buzz
# GUI 문제가 있으면 Buzz 공식 지원 옵션
buzz --safe-rendering
```

`buzz`는 사용자 Bash 함수이다. 실제 래퍼는 호스트의 `apptainer` 또는 `singularity`를 찾아 `run --cleanenv /nfs/apps/buzz/current/Buzz.sif`를 수행한다.

- `--cleanenv`: HPC 모듈의 `LD_LIBRARY_PATH`, `PYTHONPATH` 등이 컨테이너 라이브러리를 오염시키는 현상을 줄인다.
- X11: `DISPLAY`, `XAUTHORITY`, `/tmp/.X11-unix` 전달.
- D-Bus 및 사용자 Keyring: `XDG_RUNTIME_DIR`, `DBUS_SESSION_BUS_ADDRESS` 전달 및 사용자 런타임 디렉터리 bind.
- HOME와 CWD는 일반적인 Singularity/Apptainer 기본 바인드를 사용. 외부 Git 작업 디렉터리는 별도 bind 필요할 수 있다.
- 내부 LLM: `BUZZ_AGENT_PROVIDER`, `OPENAI_COMPAT_*` 등 허용된 환경 변수만 전달. **사용자별 자격증명은 공용 env.sh/SIF에 넣지 않는다.**
- Local Agent의 Shell/Git/테스트도 이 SIF 안에서 돌아간다. 필요 도구(추가 언어 Runtime, gh, 컴파일러, 사용자 Skills)는 별도 이미지 확장 대상.
- `buzz`라는 이름은 Buzz agent용 JSON CLI에도 쓰이므로 **공용 PATH 실행 파일 이름으로 사용하지 말고** 사용자 함수만 `buzz`라고 둔다.
- **호스트 의존성이 0이 되는 것은 아니다.** 호스트 X11·세션 D-Bus·Singularity 실행 권한 및 bind 정책·커널·네트워크는 여전히 필요하다. Secret Service가 없으면 Buzz 개인키가 사용자 전용 파일로 떨어질 수 있어 NFS HOME 권한을 검증한다.

## 4. 수용 테스트

1. SIF 안에서 `ldd /usr/bin/buzz-desktop`에 `not found` 없음, `libffi.so.8` 존재.
2. 두 사용자 UID가 동일한 NFS SIF를 사용해 서로 다른 Nostr 신원으로 로그인.
3. Pod/RBAC 없이 `buzz-acp → buzz-agent` Local 작업 실행.
4. 사내 vLLM/Relay/GHES 접근과 TLS CA 검증, Git clone/push·MCP 실행 테스트.
5. 다른 X 세션 호스트에서 DISPLAY / D-Bus / NFS mount 및 Singularity 실행 검증.
6. `current` 버전 교체·롤백 및 사용자별 상태/Keyring 분리 확인.

## 근거

- [Buzz 공식 릴리스 패키징](https://github.com/block/buzz/blob/main/RELEASING.md)
- [Buzz Linux 렌더링 가이드](https://github.com/block/buzz/blob/main/docs/linux-rendering-troubleshooting.md)
- [Buzz Secret Storage](https://github.com/block/buzz/blob/main/SECURITY.md)
- [Apptainer Definition](https://apptainer.org/docs/user/latest/definition_files.html)
- [Apptainer Bind](https://apptainer.org/docs/user/latest/bind_paths_and_mounts.html)
