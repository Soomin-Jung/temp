# Buzz Desktop → 사내 Relay wss://: UnknownIssuer 복구

**2026-10-23 정적 검토 / HPC 실제 TLS 재연결 미검증.**

## 증상과 근본 원인

```text
buzz-desktop: native_relay_client: connect failed:
  WebSocket error: IO Error:
  invalid peer certificate: UnknownIssuer
```

Buzz는 `native_relay_client.rs` → `buzz-ws-client/src/connection.rs` → `tokio_tungstenite::connect_async` 경로를 사용한다. 공식 Desktop의 `desktop/src-tauri/Cargo.toml` 및 workspace `Cargo.toml`은 `rustls-tls-webpki-roots`만 활성화한다. 이는 Mozilla/WebPKI Root만 로드할 수 있는 설정으로, **사내 CA를 Ubuntu의 /etc/ssl/certs에 등록하는 것만으로 네이티브 WebSocket이 인식한다는 보장이 없다**.

`curl https://buzz.internal.com`은 성공해도 Buzz에서 `UnknownIssuer`가 지속될 수 있다. 단순히 `SSL_CERT_FILE`이나 `NODE_EXTRA_CA_CERTS`를 추가하는 것으로 수정된다고 단정하지 말 것.

근거:
- [Desktop WebSocket dependency](https://github.com/block/buzz/blob/main/desktop/src-tauri/Cargo.toml)
- [Workspace WebSocket dependency](https://github.com/block/buzz/blob/main/Cargo.toml)
- [NativeRelayClient](https://github.com/block/buzz/blob/main/desktop/src-tauri/src/native_relay_client.rs)
- [buzz-ws-client connection](https://github.com/block/buzz/blob/main/crates/buzz-ws-client/src/connection.rs)
- [tokio-tungstenite native trust store](https://docs.rs/tokio-tungstenite/0.29.0/tokio_tungstenite/)
- [rustls-native-certs SSL_CERT_FILE](https://github.com/rustls/rustls-native-certs)

## A. 사내 CA를 Singularity 이미지에 넣기 (GUI/HTTP/Node/CLI 공통)

**준비물:** 보안팀에서 발급한 공개 Root CA 인증서. PEM 형식(내용이 `-----BEGIN CERTIFICATE-----` ... `-----END CERTIFICATE-----`), 파일명은 `company-root-ca.crt`. Private Key, `.key`, `.p12` **절대 포함하지 않는다**. Relay 인증서가 Intermediate로 서명됐다면 Ingress는 서버 인증서 + intermediate chain을 올바르게 송신해야 한다.

현재 [Buzz.def](Buzz.def)에서는 다음이 반영되어 있다.

```text
%files
    Buzz.deb /tmp/Buzz.deb
    node-v24.18.0-linux-x64.tar.xz /tmp/node.tar.xz
    company-root-ca.crt /usr/local/share/ca-certificates/company-root-ca.crt

%post
    # apt install ca-certificates, openssl 등 완료 후:
    openssl x509 -in /usr/local/share/ca-certificates/company-root-ca.crt -noout -subject
    update-ca-certificates

%environment
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
    export GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt
    export NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/company-root-ca.crt
```

예시 검증:

```bash
openssl x509 -in company-root-ca.crt -noout -subject -issuer -fingerprint -sha256

# 현재 내부 Cert를 먼저 외부 바인드해서 검증 (SIF에 신뢰번들이 아직 없으면
# PEM만 파일로 마운트하고 curl --cacert로 연결 체크)
singularity exec --bind "$PWD/company-root-ca.crt:/tmp/company-root-ca.crt:ro" \
  Buzz.sif curl --fail --silent --show-error \
  --cacert /tmp/company-root-ca.crt https://buzz.internal.com/

# Root CA + 고정된 WebSocket trust 수정된 Desktop deb를 빌드한 후 SIF 재빌드
singularity build Buzz.sif Buzz.def

singularity exec Buzz.sif sh -c '
  openssl verify -CAfile /etc/ssl/certs/ca-certificates.crt \
     /usr/local/share/ca-certificates/company-root-ca.crt
  echo "SSL_CERT_FILE=$SSL_CERT_FILE"
  curl -I --fail https://buzz.internal.com/
'
```

**참고:** TLS handshake 성공 여부는 HTTPS 응답 상태와 별개. 최상위 `/`가 HTTP 404여도 TLS 인증서 검증은 성공할 수 있다. `curl -v`의 TLS 검증 결과로 구분한다.

## B. Rust WebSocket RootStore 수정(중요: Desktop 재빌드 필요)

> **바로 따라할 실행 가이드**: [동일 릴리스 checkout → Cargo patch → sidecar build → Tauri .deb → SIF 재빌드](source-build-native-ca.md). [자동화 빌드 스크립트](build-patched-buzz-deb.sh)도 제공한다.


[최소 변경 패치](buzz-tls-native-roots.patch)를 기존 Buzz Desktop과 **동일한 git tag / source revision**에 적용한다. 패치 내용은 두 `Cargo.toml`에 `rustls-tls-native-roots` feature만 추가한다.

```toml
tokio-tungstenite = { version = "0.29",
  features = ["rustls-tls-webpki-roots", "rustls-tls-native-roots"] }
```

`tokio-tungstenite`는 enabled feature가 둘 다 있으면 WebPKI 공개 루트와 Linux native CA bundle을 함께 trust root로 읽는다. `rustls-native-certs`는 `SSL_CERT_FILE` 또는 Linux 시스템 CA 파일을 참조한다.

**중요:** 기존에 배포된 공식 `Buzz.deb` 바이너리는 **패치된 소스가 아니다**. SIF에 CA를 추가해도 이 공식 바이너리의 `tokio-tungstenite` 컴파일 옵션은 바뀌지 않는다. **패치된 소스로 Buzz Desktop + 필요 sidecar를 다시 빌드하여 새로운 `.deb`로 SIF에 넣어야 한다.** 교체할 `.deb`와 패치 대상 릴리스의 버전이 정확히 일치해야 한다.

```bash
# 예: 운영 데스크톱 .deb와 동일 버전의 upstream 태그를 checkout 한 상태
git checkout <matching-release-tag>
git apply --check /path/to/buzz-tls-native-roots.patch
git apply /path/to/buzz-tls-native-roots.patch

# GUI + ACP CLI/sidecar 빌드, x86_64-unknown-linux-gnu 예시.
# 도구: Rust, Cargo, pnpm, Node, WebKitGTK/GTK 개발 패키지 등.
cargo build --release --target x86_64-unknown-linux-gnu \
  -p buzz-acp -p buzz-agent -p buzz-backend-kubernetes \
  -p buzz-dev-mcp -p git-credential-nostr -p buzz-cli
./scripts/bundle-sidecars.sh x86_64-unknown-linux-gnu
pnpm install --frozen-lockfile
cd desktop
pnpm tauri build --target x86_64-unknown-linux-gnu --bundles deb
# 결과물 위치와 이름은 빌드 로그에서 확인 후 Buzz.deb로 staging
```

정확한 빌드 의존성·sidecar 제작은 선택한 릴리스의 `Justfile`, `RELEASING.md`와 `desktop/src-tauri/tauri.conf.json`에서 확인한다. 현재 Tauri 패키징 버전과 차트/Relay와의 호환성도 검증한다. Patch는 2026-10-23 upstream 메인 소스의 `tokio-tungstenite=0.29`를 기준으로 하므로 과거 release에서는 `git apply`가 실패할 수 있다. 이 경우 구 버전의 실제 manifest에 같은 feature를 수동 적용한다.

## C. 배포 전 CA 서버 체인 검증

```bash
# 인증서를 제공받은 최초 호스트 또는 빌드 서버에서:
openssl s_client -connect buzz.internal.com:443 \
  -servername buzz.internal.com \
  -CAfile ./company-root-ca.crt \
  -verify_return_error </dev/null
```

`Verify return code: 0 (ok)`가 필요하다. `unable to get local issuer certificate`면 사내 CA 파일이 다르거나 Relay Ingress가 Intermediate를 누락했을 수 있다. `hostname mismatch`면 SAN/DNS가 잘못된 것이며 CA 추가로 해결되지 않는다. 서비스가 443이 아닌 다른 포트라면 변경.

```bash
# SIF 교체 후:
source /nfs/apps/buzz/env.sh
BUZZ_LOG_FOREGROUND=1 buzz
```

우선 로그의 `UnknownIssuer`가 사라졌는지 확인. 이후 멤버십 관련 `auth-required`, `restricted`, NIP-42 인증 실패 등이 나타나면 **TLS 단계는 넘어갔고** 별도 인증 이슈로 진단한다.

## D. 신뢰·배포 정책

- CA PEM은 공개 Root 인증서만 배포하며 해시와 인증서 유효기간을 기록. 인증서 사설키는 절대 배포하지 않는다.
- 공용 NFS에 버전별 불변 `Buzz.sif`를 배포하고 사용자 실행 중 파일 덮어쓰기 금지.
- 신뢰 설정을 Rust/Node/Git/HTTP 별로 구분: `SSL_CERT_FILE`이 모든 TLS 라이브러리에 자동 적용되는 것은 아니다.
- `ws://`, `--insecure`, `curl -k`, `danger_accept_invalid_certs` 방식으로 우회하지 않는다.
