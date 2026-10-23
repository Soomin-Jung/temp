# Buzz Desktop self-hosted WSS CA — patched .deb 재빌드 실전 절차

검토 2026-10-23. **정적 소스/CI 조사 완료.** 실제 Ubuntu 24.04에서 컴파일, 사내 CA/WSS 수용 시험은 운영 환경에서 실시해야 한다.

## 1. 재빌드가 필요한 이유

- [Buzz PR #7029](https://github.com/block/buzz/pull/7029) / [#3455](https://github.com/block/buzz/pull/3455)는 운영 사고와 정확히 같은 \`UnknownIssuer\` 원인.
- Buzz Desktop \`Cargo.toml\`과 workspace \`Cargo.toml\`의 \`tokio-tungstenite\`는 \`rustls-tls-webpki-roots\`로 컴파일되어, \`SSL_CERT_FILE\` 또는 OS CA에 등록한 사내 CA를 WSS에서는 읽지 않는다. **바이너리의 컴파일 feature를 바꾸려면 .deb를 재빌드해야 한다.**
- 현재 \`temp\`에는 [최소 패치](buzz-tls-native-roots.patch)가 존재한다. 두 manifest에 \`rustls-tls-native-roots\`를 **추가**하고 기존 공개 root 지원은 남긴다.
- \`Buzz.def\`는 Root CA를 system CA bundle에 통합하고 \`SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt\`를 지정하도록 준비되어 있다. 패치된 .deb와 사내 CA를 넣고 SIF 재생성한다.
- **중요**: \`buzz-acp\`, \`buzz-agent\`, \`buzz-dev-mcp\`, \`buzz\` 등의 Tauri \`externalBin\`은 실제 최신 릴리스 버전에서 다시 빌드해서 넣어야 하므로, Tauri 전에 \`cargo build -p ...\` + \`bundle-sidecars.sh\` 실행이 필수. 스텁만 포함하지 않는다.

## 2. Ubuntu 24.04 x86_64 빌드 머신 준비

SIF 빌드 가능한 **관리 빌드 머신** 또는 Ubuntu 24.04 빌드 컨테이너에서 수행한다. 일반 X11 로그인 노드에서 \`apt install\`하지 않는다. upstream Linux release CI도 Ubuntu 24.04를 사용한다.

\`\`\`bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl git file wget pkg-config \
  libasound2-dev libayatana-appindicator3-dev libgtk-3-dev \
  librsvg2-dev libssl-dev libwebkit2gtk-4.1-dev libxdo-dev \
  patchelf xdg-utils
\`\`\`

원본 Buzz 저장소는 [Hermit](https://cashapp.github.io/hermit/)으로 **Rust 1.95.0**, Node, pnpm 11.4.0을 고정한다. \`source ./bin/activate-hermit\`로 초기화하며 첫 실행 시 지정 툴체인을 다운로드해야 한다. 기업 인증서로 빌드망 egress가 TLS inspection 되는 경우 **빌드 환경의 CA도 먼저 등록**한다.

## 3. 현재 배포된 .deb와 **정확히 같은 태그** checkout

**실행 중인 앱과 소스의 정확한 버전 일치가 필수.** \`main\`을 무작정 빌드하면 Relay/API/Sidecar의 버전 차이와 신규 기능 차이까지 섞인다.

\`\`\`bash
# SIF에 입력했던 원본 공식 .deb에서 읽는다. 예: 0.5.28
dpkg-deb -f /path/to/original/Buzz.deb Version

VERSION=0.5.28     # 위 버전과 동일하게 바꾸기
git clone https://github.com/block/buzz.git
cd buzz
git fetch --tags
git checkout "desktop-v${VERSION}"
git rev-parse HEAD
\`\`\`

해당 \`desktop-v<version>\` 태그가 없으면 releases의 tag/release commit SHA로 이동한다. \`git checkout main\`으로 대체하지 않는다.

## 4. 최소 TLS 패치 적용 및 .deb 패키징

방법 A: 통합 스크립트 [build-patched-buzz-deb.sh](build-patched-buzz-deb.sh) 사용. \`temp\` 저장소를 별도 checkout하거나 raw 파일을 받아서 사용한다.

\`\`\`bash
# block/buzz root에서
curl -fsSLo /tmp/buzz-tls-native-roots.patch \
  https://raw.githubusercontent.com/Soomin-Jung/temp/main/docs/operations/buzz-hpc/singularity/buzz-tls-native-roots.patch
curl -fsSLo /tmp/build-patched-buzz-deb.sh \
  https://raw.githubusercontent.com/Soomin-Jung/temp/main/docs/operations/buzz-hpc/singularity/build-patched-buzz-deb.sh

# 두 파일 적용 대상으로 원본 tag가 일치해야 함 (스크립트는 불일치 시 중단)
bash /tmp/build-patched-buzz-deb.sh \
  /tmp/buzz-tls-native-roots.patch \
  /path/to/singularity-build-dir
\`\`\`

방법 B: 핵심 명령을 수동으로 실행한다.

\`\`\`bash
# block/buzz root
git apply --check /tmp/buzz-tls-native-roots.patch
git apply /tmp/buzz-tls-native-roots.patch

source ./bin/activate-hermit
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# Cargo sidecar 6개 먼저. 실전 Linux release CI와 같은 순서
cargo build --release \
  -p buzz-acp -p buzz-agent -p buzz-backend-kubernetes \
  -p buzz-dev-mcp -p git-credential-nostr -p buzz-cli
./scripts/bundle-sidecars.sh

pnpm install --frozen-lockfile
(
  cd desktop
  pnpm tauri build --ci --bundles deb
)

ls -lh desktop/src-tauri/target/release/bundle/deb/*.deb
cp desktop/src-tauri/target/release/bundle/deb/*.deb \
   /path/to/singularity-build-dir/Buzz.deb
\`\`\`

수동 작업에서 \`pnpm tauri build\`는 \`desktop\` 디렉터리에서 실행한다. 또한 \`--features mesh-llm\`은 현재 Local ACP와 WSS 패치를 적용하기 위한 필수 feature가 아니므로 이 스크립트에서 제외한다. 만약 기존 배포판에서 \`mesh-llm\` 기능 자체를 사용했다면, 동일한 기능 집합을 유지하도록 \`--features mesh-llm\`을 추가한다.

빌드된 \`.deb\`가 지정된 원본 태그 버전인지 검증한다.

\`\`\`bash
dpkg-deb -f /path/to/singularity-build-dir/Buzz.deb Version
sha256sum /path/to/singularity-build-dir/Buzz.deb
\`\`\`

## 5. 기존 Buzz.def로 SIF 재빌드

\`/path/to/singularity-build-dir/\`에 다음이 존재해야 한다.

\`\`\`text
Buzz.def
Buzz.deb                  # 이번에 패치된 .deb
company-root-ca.crt       # 승인된 PEM 사내 Root CA, 사설키 금지
node-v24.18.0-linux-x64.tar.xz
\`\`\`

\`\`\`bash
cd /path/to/singularity-build-dir
apptainer build Buzz-with-internal-ca.sif Buzz.def
# 또는 singularity build Buzz-with-internal-ca.sif Buzz.def

apptainer exec --cleanenv Buzz-with-internal-ca.sif sh -c '
  openssl verify -CAfile /etc/ssl/certs/ca-certificates.crt \
    /usr/local/share/ca-certificates/company-root-ca.crt
  echo "SSL_CERT_FILE=$SSL_CERT_FILE"
'
\`\`\`

기존 SIF가 작업 중이라면 덮어쓰지 말고 별도 버전 디렉터리에 업로드해 \`/nfs/apps/buzz/current\` 링크를 변경한다.

## 6. TLS와 NIP-42 테스트

\`\`\`bash
# 1) 인증서/Ingress 체인 자체가 유효한지 먼저 (호스트에서)
openssl s_client -connect buzz.internal.com:443 \
  -servername buzz.internal.com \
  -CAfile company-root-ca.crt \
  -verify_return_error </dev/null

# 2) SIF 내부 HTTPS 인증서 체인 확인
apptainer exec --cleanenv Buzz-with-internal-ca.sif \
  curl -vI https://buzz.internal.com/

# 3) 새 SIF 실행 후 Desktop 'native_relay_client: UnknownIssuer' 재발 여부
source /nfs/apps/buzz/env.sh
BUZZ_LOG_FOREGROUND=1 buzz
\`\`\`

**주의:** \`curl\`로 HTTPS 인증서를 통과한 사실이 수정된 Rust WSS의 성공을 증명하지는 않는다. GUI의 WSS 연결 로그가 성공하고 NIP-42 인증/메시지 수신까지 검증해야 한다. TLS가 성공한 다음 \`restricted\`, \`auth-required\` 등은 Relay membership 문제로 별도 분리한다.

## 근거

- [upstream PR #7029](https://github.com/block/buzz/pull/7029) (실제 feature 추가·인증서 검증)
- [upstream PR #3455](https://github.com/block/buzz/pull/3455)
- [upstream 공식 Linux release workflow](https://github.com/block/buzz/blob/main/.github/workflows/release.yml)
- [upstream scripts/bundle-sidecars.sh](https://github.com/block/buzz/blob/main/scripts/bundle-sidecars.sh)
- [private CA / Relay 기존 트러블슈팅](private-ca-relay-tls.md)
