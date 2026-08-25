# NIXL Source Build and Air-gapped Packaging for vLLM

기준:

- vLLM `v0.27.1`
- required NIXL `1.3.1`
- NVIDIA/CUDA 중심

NIXL은 Mooncake와 source-build 문제가 다르다. Mooncake는 특정 transport compile option이 핵심이라면, NIXL은 **NIXL core + Python/native CUDA package + backend plugin + UCX/runtime libraries**를 하나의 dependency closure로 관리해야 한다.

---

# 1. 먼저: 가능하면 upstream packaged artifact를 우선 평가

NIXL 공식 `v1.3.1` README는:

```bash
pip install nixl
```

로 Python API와 native libraries를 설치할 수 있으며 CUDA 12/13 backend를 제공하고 PyTorch CUDA version에 따라 runtime backend를 선택한다고 설명한다.

vLLM v0.27.1도:

```text
nixl == 1.3.1
```

을 exact pin한다.

따라서 사내 PyPI proxy가 upstream wheel closure를 온전히 mirror할 수 있다면 **직접 source-build하기 전에 공식 1.3.1 artifacts를 검증하는 것이 우선**이다.

source build는 다음 경우 필요하다.

- 폐쇄망 mirror에 필요한 native wheel 없음
- UCX build를 사내 driver/fabric에 맞춰 통제해야 함
- plugin set을 최소화해야 함
- architecture-specific issue/patch 필요
- wheel portability/ABI를 직접 관리해야 함

---

# 2. vLLM image에서 package 구조 확인

vLLM upstream Dockerfile는 KV connector dependency 설치 후 CUDA major에 맞는 native NIXL wheel을 force-reinstall하는 logic을 가진다.

개념:

```text
nixl meta package
+
nixl-cu12 or nixl-cu13 native package
+
backend plugins / bundled runtime libs
```

따라서 단순:

```bash
pip wheel nixl==1.3.1
```

한 파일만 반입하면 closure가 완성된다고 가정하지 않는다.

확인:

```bash
python3 -m pip show nixl
python3 -m pip freeze | grep -i nixl
python3 - <<'PY'
import torch
print(torch.version.cuda)
import nixl
print(nixl)
PY
```

---

# 3. source-build version lock

반드시 pin:

```text
vLLM version/digest
NIXL tag + commit SHA
UCX branch/tag + commit SHA
Python
PyTorch
CUDA major/minor
compiler
Ubuntu/glibc
selected NIXL plugins
```

NIXL version은 vLLM 요구와 맞춘다.

```text
NIXL_VERSION=1.3.1
```

---

# 4. vLLM helper script의 중요한 함정

vLLM v0.27.1에는:

```text
tools/install_nixl_from_source_ubuntu.py
```

가 있다.

좋은 reference지만 production에 그대로 쓰면 안 되는 지점이 두 가지 있다.

### 4.1 NIXL version autodetect

script:

```python
NIXL_VERSION = os.environ.get("NIXL_VERSION", get_latest_nixl_version())
```

즉 env를 안 넣으면 실행 시점 GitHub latest release를 가져올 수 있다.

그러면:

```text
vLLM 0.27.1 requirement = 1.3.1
helper runtime fetch = 1.4.x or future
```

처럼 drift가 발생한다.

폐쇄망/reproducible build에서는 반드시 **source tarball 자체를 pin**하고 live GitHub lookup을 제거한다.

---

## 4.2 UCX version discrepancy

vLLM v0.27.1 helper는 source에서:

```text
git checkout v1.19.x
```

를 수행한다.

반면 NIXL `v1.3.1` 공식 README source-build 문서는:

```text
NIXL was tested with UCX version 1.22.x
```

라고 하고 `v1.22.x` checkout example을 제공한다.

이 차이는 매우 중요하다.

### 처리 원칙

- vLLM helper의 UCX 1.19.x를 "공식 required version"이라고 해석하지 않는다.
- NIXL 1.3.1 자체 source-build baseline은 공식 문서의 1.22.x를 우선 reference로 둔다.
- 그러나 vLLM 0.27.1 integration compatibility는 별도 테스트한다.
- 사내 artifact에서는 **검증된 UCX exact commit**을 source lock에 기록한다.

즉 "latest UCX"도, "helper script가 우연히 적은 branch"도 production version policy가 아니다.

---

# Part A. 권장 source closure

## 5. 최소 UCX-based vLLM P/D closure

```text
nixl-offline/
├── NIXL-1.3.1/
├── UCX-<validated-version>/
├── source-lock.env
├── SHA256SUMS
└── python-build-wheels/
    ├── meson
    ├── ninja
    ├── pybind11
    ├── tomlkit
    ├── auditwheel (if bundling/repairing)
    └── ...
```

GitHub clone은 Docker build 중 수행하지 않는다.

---

## 6. vLLM P/D에 필요 없는 NIXL components

vLLM NixlConnector가 자체 ZMQ side channel로 remote agent metadata를 교환하므로 NIXL ETCD integration은 P/D 기본 경로에 필요하지 않다.

따라서 최소 build에서는:

```text
UCX plugin
Python bindings
CUDA-aware support
```

를 우선한다.

GDS/OBJ/Azure/HF3FS/Mooncake 등 다른 plugins는 사용 계획이 없으면 빼서 closure를 줄인다.

NIXL Meson option:

```text
enable_plugins
```

을 사용해 예를 들어 UCX만 명시적으로 build하는 전략을 검토한다.

실제 option spelling/case는 해당 pinned tag의 Meson help로 확인한다.

---

# Part B. UCX source build

## 7. official NIXL 1.3.1 baseline

공식 README example은 UCX 1.22.x에 대해 대략:

```bash
./autogen.sh
./contrib/configure-release-mt \
  --enable-shared \
  --disable-static \
  --disable-doxygen-doc \
  --enable-optimizations \
  --enable-cma \
  --enable-devel-headers \
  --with-cuda=<cuda install> \
  --with-verbs \
  --with-dm \
  --with-gdrcopy=<gdrcopy install>
```

를 사용한다.

### GPU Direct에서 중요한 옵션

- `--with-cuda`
- `--with-verbs`
- GDRCopy가 있으면 `--with-gdrcopy`

GDRCopy는 최대 성능에 유리하지만 NIXL/UCX 동작 자체의 절대 prerequisite로 일반화하지 않는다.

---

## 8. UCX installation prefix를 분리

예:

```text
/opt/ucx
```

에 설치하고 NIXL build에서:

```text
PKG_CONFIG_PATH=/opt/ucx/lib/pkgconfig
LD_LIBRARY_PATH=/opt/ucx/lib:/opt/ucx/lib/ucx
```

또는 NIXL `ucx_path` build option을 사용한다.

목표는 system UCX와 custom UCX가 섞이지 않게 하는 것이다.

---

# Part C. NIXL core/plugin build

## 9. compiler/tool requirements

NIXL 1.3.1 공식 문서:

```text
C++20
GCC >= 11 or Clang >= 14
Meson
Ninja
pybind11
```

base vLLM image에서 compiler가 존재한다고 가정하지 않는다.

builder stage에서만 설치한다.

---

## 10. Meson build profile

개념적으로:

```bash
meson setup build \
  --buildtype=release \
  -Ducx_path=/opt/ucx \
  -Denable_plugins=UCX

ninja -C build
ninja -C build install
```

실제 pinned NIXL tag에서:

```bash
meson configure build
```

결과를 artifact manifest로 보존한다.

---

# Part D. Python package build

## 11. NIXL 1.3.1 공식 packaging 구조

README는 source installation에서:

1. NIXL binaries install
2. CUDA-specific package build/install (`nixl-cu12` 또는 `nixl-cu13`)
3. `nixl` meta-package build/install

을 구분한다.

CUDA major를 잘못 맞추면 import는 일부 성공해도 native module loading에서 실패할 수 있다.

---

## 12. CUDA 12 vs CUDA 13

base image에서:

```python
import torch
print(torch.version.cuda)
```

를 source of truth 중 하나로 확인한다.

NIXL 공식 package는 PyTorch CUDA major를 기준으로 backend variant를 선택한다.

custom build도 같은 contract를 깨지 않도록 한다.

---

# Part E. vLLM image build pattern

## 13. 권장 multi-stage

```text
FROM vLLM image AS configured-base
  -> CA / pip.conf / sources.list

FROM configured-base AS nixl-builder
  -> UCX source build
  -> NIXL source build
  -> Python/native wheels
  -> build manifest

FROM configured-base AS runtime
  -> only validated UCX/NIXL runtime libs + wheels
  -> no compiler/source tree
```

Mooncake PR #5와 같은 패턴을 쓸 수 있지만 **NIXL에서는 UCX runtime `.so` closure까지 포함**해야 한다.

---

## 14. internal CA/repositories

Mooncake와 동일하게:

```dockerfile
COPY certs/ /opt/certs/
COPY pip.conf /etc/pip.conf
COPY sources.list /etc/apt/sources.list
```

을 configured-base에서 처리한다.

Docker build 중 외부:

- GitHub
- PyPI
- UCX source

를 직접 fetch하지 않는다.

---

# Part F. wheel self-containment

## 15. vLLM helper의 useful pattern

vLLM helper는:

- UCX를 별도 prefix에 build
- NIXL wheel build
- `auditwheel repair`
- UCX libraries를 self-contained wheel 쪽으로 묶는 형태

를 사용한다.

이 pattern 자체는 폐쇄망 artifact 배포에 유용하다.

다만 version autodetect/live clone 부분은 제거한다.

---

## 16. RPATH/loader 검증

runtime image에서:

```bash
ldd <nixl native .so>
ldd <libplugin_UCX.so>
```

를 실행한다.

확인:

- UCX libraries가 원하는 prefix/wheel에서 resolve
- system의 다른 UCX가 우연히 잡히지 않음
- CUDA 12/13 mismatch 없음
- verbs/CUDA libraries resolve

---

# Part G. NIXL-specific risk: UCX resource usage

## 17. UAR exhaustion

vLLM v0.27.1 NIXL worker는 UCX thread 수가 Mellanox UAR resource를 과도하게 소비해 다른 RDMA component(NVSHMEM/DeepEP 등)의 initialization을 깨는 상황을 명시적으로 고려한다.

기본:

```text
num_threads=4
```

를 사용한다.

한 GPU Pod에:

- NIXL/UCX
- NCCL
- NVSHMEM
- DeepEP

가 동시에 있으면 HCA resource pressure를 반드시 본다.

---

## 18. `UCX_RCACHE_MAX_UNRELEASED`

vLLM `nixl_utils.py`는 NIXL import 전에:

```text
UCX_RCACHE_MAX_UNRELEASED=1024
```

를 기본 설정해 rare UCX registration-cache leak을 회피하려 한다.

NIXL을 너무 일찍 import하면 이 env를 설정할 수 없다는 warning path가 있다.

custom image/startup wrapper가 import order를 바꾸는 경우 확인한다.

---

# Part H. runtime validation gates

## 19. Gate 1 — package identity

```bash
python3 -m pip freeze | grep -Ei 'nixl|ucx'
```

NIXL exact version 확인.

---

## 20. Gate 2 — NIXL agent creation

공식 smoke:

```bash
python3 -c "import nixl; agent = nixl.nixl_agent('agent1')"
```

UCX backend instantiated log가 보여야 한다.

---

## 21. Gate 3 — UCX capabilities

```bash
ucx_info -v
ucx_info -d
```

확인:

- CUDA support
- verbs/RDMA devices
- cuda_ipc/cuda copy capability
- intended HCA

---

## 22. Gate 4 — NIXL standalone GPU transfer

vLLM을 넣기 전에 NIXL example/nixlbench로:

```text
VRAM -> VRAM
same node
```

그리고 Network A라면:

```text
VRAM -> VRAM
cross node
```

을 검증한다.

이렇게 하면 vLLM connector bug와 NIXL/UCX 환경 문제를 분리할 수 있다.

---

## 23. Gate 5 — vLLM Pull/Push integration

```text
NixlConnector Pull
NixlPushConnector Push
```

각각:

- handshake
- memory registration
- actual READ/WRITE
- notification
- no prompt recompute

를 검증한다.

---

# Part I. build manifest

## 24. 반드시 image에 남길 것

```text
/opt/nixl-build-info/
├── NIXL_SOURCE_LOCK.env
├── UCX_SOURCE_LOCK.env
├── MESON_OPTIONS.txt
├── UCX_CONFIGURE.txt
└── WHEEL_SHA256SUMS
```

OCI labels 예:

```text
ai.nixl.version=1.3.1
ai.nixl.source.commit=<sha>
ai.nixl.backends=UCX
ai.ucx.version=<validated>
ai.ucx.source.commit=<sha>
ai.vllm.version=0.27.1
```

---

# Part J. 추천 production strategy

## 25. 1차 — official artifact path

사내 proxy에:

```text
nixl==1.3.1
matching nixl-cu12/13
```

를 mirror하고 vLLM upstream installation contract에 최대한 맞춘다.

## 26. 2차 — source-build path

official artifact가 환경 요구를 못 맞출 때만:

```text
NIXL 1.3.1
+ validated UCX exact commit
+ UCX-only plugin profile
```

으로 source build한다.

## 27. 1.4.0 upgrade

NIXL 최신 기능이 필요하면 vLLM 0.27.1의 exact pin을 override하지 않고 **별도 test image**에서:

- API compatibility
- vLLM unit/integration tests
- Pull/Push P/D
- hetero TP
- HMA

를 통과한 뒤 version policy를 변경한다.

---

# 28. 최종 체크리스트

```text
[ ] vLLM version/digest pinned
[ ] NIXL exactly 1.3.1 unless intentional experiment
[ ] UCX exact source version/commit pinned
[ ] no live latest-release lookup
[ ] no live GitHub clone in image build
[ ] CUDA major matched
[ ] C++20 compiler only in builder
[ ] UCX built with CUDA + verbs for NVIDIA RDMA use case
[ ] plugin set minimized
[ ] NIXL/UCX loader paths deterministic
[ ] NIXL standalone VRAM transfer passes
[ ] vLLM handshake compatibility passes
[ ] Pull/Push actual transfer passes
[ ] physical CUDA IPC/RDMA path verified
```

---

## References

- NIXL 1.3.1 README/build: https://github.com/ai-dynamo/nixl/blob/v1.3.1/README.md
- vLLM source-build helper: https://github.com/vllm-project/vllm/blob/v0.27.1/tools/install_nixl_from_source_ubuntu.py
- vLLM Dockerfile: https://github.com/vllm-project/vllm/blob/v0.27.1/docker/Dockerfile
- vLLM NIXL pin: https://github.com/vllm-project/vllm/blob/v0.27.1/requirements/kv_connectors.txt
