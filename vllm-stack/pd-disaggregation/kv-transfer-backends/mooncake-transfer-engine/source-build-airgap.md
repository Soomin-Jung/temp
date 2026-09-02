# Mooncake Transfer Engine Source Build in an Air-gapped vLLM Image

기준 구현 reference: `Soomin-Jung/vllm-production-stack-custom` PR #5  
검증 이력: Mooncake `0.3.10.post2` source-build overlay  
차기 validation: vLLM `v0.28.0-cu129` + Mooncake `0.3.12.post1`

이 문서는 PR #5를 단순 복사하지 않고 **최신 vLLM base image에도 재사용 가능한 폐쇄망 build pattern**으로 일반화한다.

---

## 1. 왜 source build가 필요한가

Mooncake에서 package version만 같다고 feature set이 같다고 볼 수 없다.

Mooncake transport implementation은 compile-time option에 영향을 받는다.

```text
USE_MNNVL=ON
 -> NvlinkTransport compiled
 -> protocol "nvlink" 가능

USE_INTRA_NVLINK=ON
 -> IntraNodeNvlinkTransport compiled
 -> protocol "nvlink_intra" 가능
```

공식 wheel에 해당 옵션이 빠져 있으면:

```text
pip install 성공
import 성공
TransferEngine 생성 성공
```

까지 가더라도 runtime에서:

```text
Unsupported transport nvlink_intra, please rebuild Mooncake
```

처럼 실패할 수 있다.

따라서 source build의 목표는 **버전을 바꾸는 것**이 아니라 **필요한 native feature를 명시적으로 포함한 reproducible artifact를 만드는 것**이다.

---

## 2. vLLM base image가 바뀔 때 먼저 확인할 것

vLLM release마다 default CUDA/Python/Torch 조합과 별도 CUDA variant가 바뀔 수 있다. v0.28.0은 default CUDA 13 image와 `v0.28.0-cu129` image를 함께 제공하므로, connector migration과 driver/CUDA migration을 같은 변경으로 묶지 않는다.

**Dockerfile을 쓰기 전에 실제 base image 안에서 확인한다.**

```bash
python3 --version
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY

nvcc --version || true
ls -l /usr/local/cuda || true
cat /etc/os-release
ldd --version | head -1
```

기록할 것:

```text
base image repository/tag
base image digest
Python
PyTorch
CUDA runtime
CUDA toolkit availability
Ubuntu/glibc
NVIDIA driver requirement
```

---

## 3. upstream vLLM KV connector dependency 정책

```text
vLLM 0.27.1
  nixl == 1.3.1
  mooncake-transfer-engine >= 0.3.8

vLLM 0.28.0
  lmcache >= 0.3.9
  nixl == 1.3.2
  mooncake-transfer-engine >= 0.3.12
```

Dockerfile에서는 `INSTALL_KV_CONNECTORS=false`가 기본이고, true일 때 KV connector dependencies를 설치한다.

즉 derived image 전략은 두 가지다.

### Strategy A — upstream base에 KV deps 없음

```text
FROM vllm/vllm-openai:v0.27.1...
 -> custom Mooncake wheel만 설치
 -> 필요한 Python deps도 closure에 포함
```

### Strategy B — upstream KV deps 포함 image

```text
FROM prebuilt image with connector deps
 -> 기존 official Mooncake uninstall
 -> custom source-built Mooncake wheel force install
```

PR #5는 Strategy B와 유사하게 runtime에서 기존 Mooncake package를 제거하고 custom wheel을 설치한다.

### 0.3.12.post1에서도 source build가 필요한가

Network B의 same-node `nvlink_intra` 기준으로는 **그렇다**. 0.3.12.post1 official x86 release workflow는 `USE_CUDA=ON`을 사용하지만 `USE_INTRA_NVLINK=ON`을 포함하지 않는다. 따라서 0.3.10의 build pattern은 재사용하되 source/submodule lock, build requirements, CMake target, wheel ABI를 새 version에 맞춰 다시 검증한다.

---

# Part A. 폐쇄망 source closure

## 4. GitHub source archive만 받아오면 안 되는 이유

Mooncake는 submodule/source dependency가 있다.

일반 GitHub source archive는 git submodule working tree를 채워주지 않는다.

PR #5의 `0.3.10.post2` closure:

```text
Mooncake source
├── extern/pybind11/       # pinned source populated
└── extern/yalantinglibs/  # pinned source populated
```

이를 외부 반입 단계에서 완성해 하나의 tarball로 만든다.

---

## 5. Source lock

반입 artifact는 최소 다음을 pin한다.

```text
Mooncake release/tag
Mooncake commit SHA
pybind11 commit SHA
yalantinglibs commit SHA
SHA256 of final tarball
```

PR #5 baseline:

```text
Mooncake: 0.3.10.post2
Mooncake commit: e1d6d6f6f49fbbd77b7ee6e5d0c77349f341b3e3
pybind11: 58c382a8e3d7081364d2f5c62e7f429f0412743b
yalantinglibs: 73dea196d23ad8fcd4914c6ef1238f390b9a1c48
```

새 버전으로 올릴 때는 숫자를 재사용하지 않고 새 source lock을 만든다.

---

## 6. 권장 vendor layout

```text
docker/mooncake/
├── SOURCE_LOCK.env
├── requirements-build.txt
├── vendor/
│   ├── mooncake-offline_<version>.tar.gz
│   └── SHA256SUMS
└── scripts/
    ├── verify-source-lock.sh
    └── verify-install.py
```

Git 자체가 없는 build environment에서도 source identity를 검사할 수 있게 한다.

---

# Part B. 사내 mirror / CA 계층

## 7. Base configuration은 builder와 runtime 공통 부모로

권장:

```dockerfile
FROM ${VLLM_BASE_IMAGE} AS configured-base
USER root

COPY certs/ /opt/certs/
COPY pip.conf /etc/pip.conf
COPY sources.list /etc/apt/sources.list

RUN set -eux; \
    rm -fv /etc/apt/sources.list.d/*.list; \
    find /opt/certs -type f -name '*.crt' \
      -exec cp -v {} /usr/local/share/ca-certificates/ \;; \
    update-ca-certificates; \
    rm -rf /opt/certs
```

이 구조의 이유:

- builder도 내부 APT/pip/CA 필요
- final runtime도 사내 CA 필요
- external repository list와 internal mirror가 섞이는 것 방지
- 민감한 실제 URL/credential은 public repo에 넣지 않음

---

# Part C. CUDA build ABI

## 8. runtime image에 compiler가 없을 수 있다

`vllm-openai` final image는 runtime/JIT에 필요한 일부 CUDA components는 있지만, Mooncake native source build에 필요한 모든 devel package가 있다고 가정하지 않는다.

Builder stage에서 확인:

```text
nvcc
cuda_runtime.h
CUDA driver stub for link
Python.h
C/C++ compiler
CMake/Ninja
verbs headers
```

없으면 **base image의 CUDA version과 정확히 맞는** devel package를 사내 APT mirror에서 설치한다.

PR #5의 CUDA 12.9 baseline:

```text
cuda-nvcc-12-9
cuda-cudart-dev-12-9
cuda-driver-dev-12-9
```

CUDA 13 image에서는 이 이름을 그대로 복사하지 말고 해당 image/repository의 package version을 확인한다.

---

## 9. `libcuda.so` stub 문제

container build 시 host NVIDIA kernel driver는 runtime처럼 mount되지 않는다.

native link가 CUDA driver symbol을 요구하면 toolkit stub을 사용해야 한다.

PR #5 해결:

```dockerfile
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
```

### `LIBRARY_PATH`와 `LD_LIBRARY_PATH`를 구분

```text
build/link time:
  LIBRARY_PATH -> stub libcuda.so

runtime:
  NVIDIA Container Runtime -> host real libcuda.so.1
```

**stub directory를 runtime `LD_LIBRARY_PATH` 앞쪽에 넣지 않는다.**

그렇게 하면 GPU Pod에서 실제 driver library 대신 stub을 load할 위험이 있다.

---

# Part D. Build dependency 최소화

## 10. Transfer Engine만 필요한 경우 Store를 끈다

P/D direct KV transfer만 목표라면 Mooncake Store stack은 불필요하다.

PR #5 profile:

```cmake
-DWITH_TE=ON
-DWITH_STORE=OFF
-DWITH_STORE_RUST=OFF
-DWITH_STORE_GO=OFF
-DWITH_P2P_STORE=OFF
-DWITH_RUST_EXAMPLE=OFF
-DWITH_EP=OFF
```

장점:

- Rust/Go/etcd dependency closure 감소
- 폐쇄망 반입량 감소
- builder 복잡도 감소
- runtime image attack/debug surface 감소

---

# Part E. NVIDIA transport build profiles

## 11. Profile 1 — same-node P/D Cell

목적: Network B node-local P/D.

```cmake
-DUSE_CUDA=ON
-DUSE_INTRA_NVLINK=ON
-DUSE_MNNVL=OFF
-DUSE_TCP=ON
```

runtime:

```bash
MC_INTRANODE_NVLINK=1
```

### 왜 MNNVL을 굳이 끄는 profile도 고려하나

Mooncake 0.3.10.post2 auto-selection에서 MNNVL/Intra-NVLink compile branch는 TCP fallback semantics가 다르다.

node-local production artifact라면 **사용하지 않는 cross-node transport를 제거하여 선택 ambiguity를 줄이는 것**도 좋은 전략이다.

단, build compatibility와 upstream CMake dependency를 실제로 검증한 뒤 적용한다.

---

## 12. Profile 2 — RDMA + TCP fallback

목적: Network A.

```cmake
-DUSE_CUDA=ON
-DUSE_MNNVL=OFF
-DUSE_INTRA_NVLINK=OFF
-DUSE_TCP=ON
```

이 branch에서는 HCA discovery에 따라 RDMA/TCP 선택이 비교적 명확하다.

---

## 13. Profile 3 — MNNVL

목적: 실제 MNNVL fabric 보유 환경.

```cmake
-DUSE_CUDA=ON
-DUSE_MNNVL=ON
-DUSE_INTRA_NVLINK=<필요 여부>
```

runtime:

```bash
MC_FORCE_MNNVL=1
```

추가로 Fabric Memory capability를 검증한다.

---

## 14. Profile 4 — validation superset (PR #5 유형)

```cmake
-DUSE_CUDA=ON
-DUSE_MNNVL=ON
-DUSE_INTRA_NVLINK=ON
-DUSE_TCP=ON
```

이 image는 여러 실험에 편하지만 **runtime auto-selection branch를 정확히 관리해야 한다.**

PR #5는 이 계열이다.

---

# Part F. PR #5에서 검증된 CMake baseline

## 15. 핵심 configure

```cmake
-DCMAKE_BUILD_TYPE=Release
-DBUILD_EXAMPLES=ON
-DBUILD_SHARED_LIBS=OFF
-DBUILD_UNIT_TESTS=OFF
-DWITH_TE=ON
-DWITH_STORE=OFF
-DWITH_STORE_RUST=OFF
-DWITH_STORE_GO=OFF
-DWITH_P2P_STORE=OFF
-DWITH_RUST_EXAMPLE=OFF
-DWITH_EP=OFF
-DUSE_CUDA=ON
-DUSE_MNNVL=ON
-DUSE_INTRA_NVLINK=ON
-DUSE_TCP=ON
-DUSE_HTTP=ON
-DUSE_ETCD=OFF
-DUSE_REDIS=OFF
-DUSE_TENT=OFF
-DWITH_METRICS=ON
```

`yalantinglibs`는 Mooncake configure 전에 local CMake package로 먼저 build/install한다.

---

## 16. CMakeCache를 artifact evidence로 보존

Builder에서 적어도:

```bash
grep -E \
'^(USE_CUDA|USE_MNNVL|USE_INTRA_NVLINK|USE_TCP|WITH_STORE|WITH_STORE_RUST|USE_ETCD):' \
  build/CMakeCache.txt
```

결과를 runtime image에 `/opt/mooncake-build-info/` 같은 경로로 남긴다.

운영자가 image에서 직접:

```text
"이 wheel에 nvlink_intra가 실제 compile됐나?"
```

를 확인할 수 있어야 한다.

---

# Part G. Python wheel build

## 17. CUDA major variant

Mooncake build script는 CUDA major에 따라 wheel variant가 달라질 수 있다.

PR #5에는:

```text
MOONCAKE_CU13_BUILD=0
CU13_BUILD=${MOONCAKE_CU13_BUILD}
```

가 있다.

vLLM v0.27.1 default Docker base를 CUDA 13으로 쓸 경우 이 값을 자동으로 1이라고 추측해서 넣지 말고 **Mooncake 해당 release build script의 semantics를 확인**한 뒤 맞춘다.

권장 Docker build logic:

```text
read torch.version.cuda / nvcc
 -> determine CUDA major
 -> assert supported Mooncake wheel variant
 -> fail closed if unknown
```

---

# Part H. Multi-stage runtime image

## 18. builder 전체를 final image에 남기지 않는다

권장:

```text
configured-base
  |
  +--> mooncake-builder
  |      - compilers
  |      - CUDA devel
  |      - CMake/Ninja
  |      - source tree
  |      - wheel output
  |
  `--> runtime
         - upstream vLLM runtime
         - custom Mooncake wheel only
         - required runtime .so
         - build manifest
```

이렇게 해야 image 다이어트와 supply-chain 추적을 동시에 잡는다.

---

## 19. Runtime install

기존 package가 있다면 명시적으로 제거한다.

```bash
python3 -m pip uninstall -y \
  mooncake-transfer-engine \
  mooncake-transfer-engine-cuda13 || true
```

그 뒤:

```bash
python3 -m pip install \
  --no-index \
  --no-deps \
  --force-reinstall \
  /tmp/mooncake-dist/*.whl
```

`--no-deps`를 쓰려면 runtime dependency가 base image에 존재하는지 verify script에서 검사한다.

---

# Part I. Image validation

## 20. Gate 1 — package identity

```bash
python3 -m pip show mooncake-transfer-engine
python3 -c 'import mooncake; print(mooncake)'
```

package/module name은 release에 따라 확인한다.

---

## 21. Gate 2 — native linkage

```bash
ldd <Mooncake native .so>
```

확인:

- unresolved library 없음
- 잘못된 CUDA major library 없음
- stub `libcuda.so`가 runtime 우선순위에 들어오지 않음

---

## 22. Gate 3 — build feature manifest

image 내부:

```text
SOURCE_LOCK.env
CMAKE_FEATURES.txt
```

과 OCI label을 확인한다.

권장 label:

```text
ai.mooncake.version
ai.mooncake.source.commit
ai.mooncake.transport.rdma
ai.mooncake.transport.nvlink
ai.mooncake.transport.nvlink_intra
ai.mooncake.transport.tcp
```

---

## 23. Gate 4 — engine initialize

목표 transport별 Pod를 띄워 실제:

```text
Transfer Engine initialized
intended transport installed
memory registration success
```

를 확인한다.

---

## 24. Gate 5 — actual KV transfer

최종 인증은 반드시 P/D request로 한다.

```text
P prefill
 -> D requests remote KV
 -> P actual send
 -> D receive completion
 -> decode
```

그리고:

```text
D가 prompt를 local recompute하지 않음
```

을 확인한다.

---

# Part J. 최신 vLLM base로 PR #5 pattern 옮길 때 변경해야 할 것

## 25. 그대로 가져가도 되는 것

- `configured-base`에서 CA/pip/APT 구성
- source tarball + SHA/source lock
- multi-stage builder/runtime
- `LIBRARY_PATH` CUDA stub 처리
- Store를 제거한 Transfer Engine-only build
- CMake feature verification
- runtime package replacement
- install verification
- OCI build metadata

---

## 26. 그대로 가져가면 안 되는 것

- `VLLM_BASE_IMAGE=v0.26.0-cu129`
- `cuda-*-12-9` package names
- `MOONCAKE_CU13_BUILD=0`
- base image에 이미 존재한다고 가정한 Python/runtime dependencies
- 이전 PyTorch/CUDA ABI 기준 native wheel

새 base마다 재검증한다.

---

# Part K. NIXL과 함께 image를 만들 경우

vLLM v0.27.1은 NIXL 1.3.1을 exact pin한다.

하나의 P/D image에서 Mooncake와 NIXL을 모두 실험하려면:

```text
vLLM
 + custom Mooncake wheel
 + nixl==1.3.1 + matching native CUDA package/plugin
```

을 독립적으로 검증한다.

**Mooncake custom build 때문에 NIXL/UCX shared-library resolution이 깨지지 않는지 `ldd`, import, runtime smoke를 모두 수행**한다.

---

## 27. 최종 build checklist

```text
[ ] base image digest pinned
[ ] Python/PyTorch/CUDA recorded
[ ] Mooncake source commit pinned
[ ] all submodule source populated
[ ] tarball SHA verified
[ ] CA/APT/pip point only to approved internal sources
[ ] no live GitHub fetch in Docker build
[ ] correct CUDA devel packages
[ ] CUDA stub only in link path
[ ] required Mooncake transports compiled
[ ] CMakeCache feature manifest saved
[ ] Store/Rust/Go omitted unless intentionally required
[ ] wheel native linkage verified
[ ] runtime image does not carry unnecessary builder toolchain
[ ] actual transport initialization verified
[ ] actual P/D KV transfer verified
[ ] hardware path verified
```

---

## References

- PR #5: https://github.com/Soomin-Jung/vllm-production-stack-custom/pull/5
- PR #5 Dockerfile: https://github.com/Soomin-Jung/vllm-production-stack-custom/blob/main/docker/Dockerfile.vllm-mooncake
- vLLM v0.27.1 Dockerfile: https://github.com/vllm-project/vllm/blob/v0.27.1/docker/Dockerfile
- vLLM KV requirements: https://github.com/vllm-project/vllm/blob/v0.27.1/requirements/kv_connectors.txt
