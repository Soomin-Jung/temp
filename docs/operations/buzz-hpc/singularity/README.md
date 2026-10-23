# Buzz Desktop — Singularity/Apptainer NFS 배포

**기준: 2026-10-23, 정적 검토 완료 / 사용자 HPC에서 실제 SIF 빌드·GUI 검증 전.**

## 핵심 문서

- **[사내 Relay WSS UnknownIssuer 해결 (Root CA + Rust TLS 재빌드)](private-ca-relay-tls.md)**
- [Buzz.def — Ubuntu 24.04, Node 24, ACP, UTF-8/IBus, CA](Buzz.def)
- [최소 Rust TLS 패치](buzz-tls-native-roots.patch)
- [DTGPT/ACP/커뮤니티/로그/한국어](dtgpt-acp-setup.md)
- [NFS 런처](../nfs-client/buzz-desktop) / [셸 환경](../nfs-client/env.sh)

## 1. 빌드 입력

`Buzz.def`가 있는 디렉터리에 다음 파일을 준비한다.

| 파일 | 목적 |
| --- | --- |
| `Buzz.deb` | 공식 또는 **사내 CA를 신뢰하도록 수정·재빌드한** Buzz Desktop .deb |
| `node-v24.18.0-linux-x64.tar.xz` | 공식 Node 24.18.0 x64 tarball, SHA256 검증 |
| `company-root-ca.crt` | 승인된 **PEM Root CA 공개 인증서**, 개인키 포함 금지 |

**주의:** 현 upstream의 WebSocket은 WebPKI 공개 루트 기능을 사용하므로 CA를 SIF에 넣는 것만으로 `native_relay_client: UnknownIssuer`가 해결된다고 가정하지 않는다. 반드시 [WebSocket native-root patch](private-ca-relay-tls.md)를 검토하고 해당 코드를 적용해 재빌드한 Buzz .deb를 사용한다.

공식 `.deb` 버전을 확인하고 **소스가 동일한 버전**인지 검증한다.

```bash
dpkg-deb -f /path/to/Buzz.deb Version
sha256sum /path/to/Buzz.deb
openssl x509 -in company-root-ca.crt -noout -subject -fingerprint -sha256
```

## 2. SIF 생성

관리 가능한 빌드 호스트에서 (일반 사용자의 HPC X11 호스트가 아님):

```bash
# 인증서/Node tarball/패치 적용된 Buzz.deb 검증 후,
# 이 파일들이 Buzz.def와 같은 작업 디렉터리에 있는 상태:
singularity build Buzz.sif Buzz.def

# 또는
apptainer build Buzz.sif Buzz.def

apptainer exec --cleanenv Buzz.sif bash -c '
  test -x /usr/bin/buzz-desktop
  node --version
  command -v codex-acp && command -v claude-agent-acp
  grep -q "BEGIN CERTIFICATE" /etc/ssl/certs/ca-certificates.crt
  openssl verify -CAfile /etc/ssl/certs/ca-certificates.crt \
    /usr/local/share/ca-certificates/company-root-ca.crt
'
```

`%post`에서 apt, npm registry에 접근하므로 인터넷 차단 시 사내 승인 미러가 필요하다. 버전 고정하더라도 npm의 간접 의존성과 apt 패키지 버전은 저장소 스냅샷을 별도로 고정하지 않으면 변할 수 있다.

## 3. NFS 배포 및 사용자 실행

```text
/nfs/apps/buzz/
├── env.sh
├── bin/buzz-desktop
├── versions/
│   └── <verified-release>/
│       └── Buzz.sif
└── current -> versions/<verified-release>
```

SIF는 관리자만 쓰기 가능한 NFS에 배포하고 사용자들은 읽기 전용으로 실행한다. 앱 프로세스가 살아 있는 동안 SIF 파일 자체를 덮어쓰지 않는다. 새 버전 경로를 만든 후 심볼릭 링크를 교체한다.

```bash
source /nfs/apps/buzz/env.sh
buzz

# 터미널 로그를 바로 확인하고 싶다면
BUZZ_LOG_FOREGROUND=1 buzz
```

기본 실행 래퍼는 `apptainer run --cleanenv` 또는 `singularity run --cleanenv`를 사용하고 X11/D-Bus를 바인딩한다. `--cleanenv`로 인해 사용자별 LLM credential은 런처에서 명시 전달한 환경변수만 사용할 수 있다. 내부 인증서 경로는 호스트의 `SSL_CERT_FILE`이 아니라 SIF의 `/etc/ssl/certs/ca-certificates.crt`를 사용하도록 설정했다.

## 4. 운영 점검

- WSS: 사내 Root CA + Ingress intermediate chain + Hostname SAN + **Rust native root 재빌드** 확인
- `node/npm`: 빌드 시 설치, 사용자 세션 `npx` 다운로드 불필요
- Local Agent: 사용자의 HOME/.buzz 및 Git worktree bind, 사용자별 권한·비밀 보호
- X11/IME: `DISPLAY`, `XAUTHORITY`, D-Bus, `ko_KR.UTF-8`, `ibus-hangul`
- 내부 LLM: Codex Responses vs Claude Messages vs buzz-agent Chat Completions 분리
- 로그: `${XDG_STATE_HOME:-$HOME/.local/state}/buzz/desktop.log`
- GHES·DTGPT가 동일 사내 CA를 사용하면 Git/Curl/Node 등의 TLS도 확인
