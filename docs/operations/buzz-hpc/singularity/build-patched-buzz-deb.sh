#!/usr/bin/env bash
# Run INSIDE the block/buzz source checkout, at the exact Desktop release tag.
# Usage: bash /path/to/build-patched-buzz-deb.sh PATCH_FILE SIF_BUILD_CONTEXT
# Prereq: Ubuntu 24.04 Tauri build libraries, pinned Rust 1.95, Node/pnpm.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/path/buzz-tls-native-roots.patch /absolute/path/sif-build-context" >&2
  exit 2
fi
patch="$(realpath "$1")"
out="$(realpath -m "$2")"
[[ -f "$patch" ]] || { echo "Patch missing: $patch" >&2; exit 2; }

[[ -f desktop/src-tauri/Cargo.toml && -f Cargo.toml && -x scripts/bundle-sidecars.sh ]] || {
  echo "Run this script from the root of an exact-tag block/buzz source checkout." >&2
  exit 2
}
[[ "$(uname -m)" == x86_64 ]] || {
  echo "This script builds native x86_64 Ubuntu .deb only." >&2
  exit 2
}

src_version="$(node -p "require('./desktop/package.json').version")"
echo "Buzz source version: $src_version"
echo "Buzz source commit: $(git rev-parse HEAD)"

# Match the SIF's Ubuntu 24.04 base, and the exact source release tag
# (checkout is the caller's responsibility; do not automatically jump to main).
if git apply --check "$patch"; then
  git apply "$patch"
  echo "Applied native+webpki TLS patch."
elif git apply --reverse --check "$patch"; then
  echo "Native+webpki TLS patch is already applied."
else
  echo "Patch does not match this release; inspect its two Cargo.toml lines." >&2
  exit 1
fi

grep -Fq '"rustls-tls-native-roots"' Cargo.toml
grep -Fq '"rustls-tls-native-roots"' desktop/src-tauri/Cargo.toml

# Use the repo's pinned Hermit toolchain when available (Rust 1.95 / pnpm 11).
if [[ -f ./bin/activate-hermit ]]; then
  # shellcheck disable=SC1091
  source ./bin/activate-hermit
fi

echo "Toolchain versions:"
rustc --version
cargo --version
node --version
pnpm --version

export CMAKE_POLICY_VERSION_MINIMUM=3.5
# Build the REAL sidecars first: Tauri externalBin requires them; never ship stubs.
cargo build --release \
  -p buzz-acp -p buzz-agent -p buzz-backend-kubernetes \
  -p buzz-dev-mcp -p git-credential-nostr -p buzz-cli
./scripts/bundle-sidecars.sh

# Root workspace lockfile with frozen, reproducible JS dependencies.
pnpm install --frozen-lockfile
(
  cd desktop
  # No upstream signing keys required for an internal, unsigned .deb.
  # Not enabling optional mesh-llm: Local Agent + WSS + ACP do not require it.
  pnpm tauri build --ci --bundles deb
)

shopt -s nullglob
debs=(desktop/src-tauri/target/release/bundle/deb/*.deb)
if (( ${#debs[@]} != 1 )); then
  printf 'Expected exactly one .deb; found %s in desktop/src-tauri/target/release/bundle/deb\n' "${#debs[@]}" >&2
  exit 1
fi
built_version="$(dpkg-deb -f "${debs[0]}" Version)"
if [[ "$built_version" != "$src_version"* ]]; then
  echo "Version mismatch: source=$src_version deb=$built_version" >&2
  exit 1
fi

mkdir -p "$out"
install -m 0644 "${debs[0]}" "$out/Buzz.deb"
sha256sum "$out/Buzz.deb" > "$out/Buzz.deb.sha256"
echo "Patched Buzz Debian package: $out/Buzz.deb"
cat "$out/Buzz.deb.sha256"
echo "NEXT: stage Node tarball + approved company-root-ca.crt alongside Buzz.def"
echo "      then: apptainer build Buzz.sif Buzz.def"
