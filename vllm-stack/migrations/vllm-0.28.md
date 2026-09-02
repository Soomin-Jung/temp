# vLLM 0.28.0 Migration & KV-Connector Compatibility

업데이트: 2026-09-03 KST  
상태: migration decision / validation baseline candidate

이 문서는 기존 vLLM 0.26.x~0.27.x 운영 기준에서 **vLLM 0.28.0**으로 올라갈 때 무엇을 그대로 가져가고, 무엇을 다시 검증해야 하는지 정리한다. 단순 release-note 요약이 아니라 P/D disaggregation, Mooncake/NIXL, 모델별 runtime, CUDA Graph, scheduler까지 실제 운영 변경 경계를 정의한다.

## 1. 결론

vLLM 0.28.0은 현재 플랫폼에서 **새 validation baseline 후보**로 올릴 가치가 충분하다.

다만 다음처럼 다룬다.

```text
0.26.0 / 0.27.1
  = 기존 재현 기준선

0.28.0-cu129
  = 차기 validation candidate

0.28.0 default CUDA 13 image
  = 별도 CUDA/driver 승격 track
```

현재 CUDA 12.9 계열 운영 검증을 유지해야 하는 환경에서는 `v0.28.0-cu129` image를 우선 사용한다. v0.28.0의 default image가 CUDA 13.0으로 이동했다는 이유만으로 host driver/runtime 승격을 같은 변경에 묶지 않는다.

공식 release:
- https://github.com/vllm-project/vllm/releases/tag/v0.28.0
- CUDA 12.9 image: `vllm/vllm-openai:v0.28.0-cu129`

## 2. 운영상 중요한 0.28.0 변경

### 2.1 Scheduler default

v0.28.0은 기본 `max_num_batched_tokens`를 **8192 → 16384**로 올렸다.

이 변경은 "더 큰 값이 항상 좋다"는 뜻이 아니다.

`max_num_batched_tokens`는 한 scheduler iteration에서 처리할 수 있는 token budget이고 다음을 동시에 바꾼다.

- chunked prefill chunk 크기
- prefill/decode interference
- kernel shape와 arithmetic intensity
- CUDA Graph / compile shape
- temporary activation/workspace pressure
- speculative decoding을 사용할 때 실제 scheduler가 쓸 수 있는 순수 token budget

따라서 기존 profile에 값을 명시적으로 pin하고 있었다면 upstream default 변화가 직접 적용되지는 않지만, 0.28 전환 benchmark에서는 **기존 값과 16K를 모두 A/B**한다.

### 2.2 Model Runner V2와 P/D

0.28.0은 Model Runner V2에 E/P/D disaggregation을 포함한다.

현재 custom Production Stack의 P/D Cell과는 계층이 다르다.

- Production Stack P/D Cell: deployment / routing / lifecycle topology
- vLLM Model Runner P/D support: engine/runtime execution capability

따라서 "vLLM 0.28이 P/D를 지원하므로 custom P/D Cell이 불필요하다"로 해석하지 않는다. 오히려 장기 migration에서 custom topology를 upstream runtime primitive와 더 얇게 연결할 수 있는 근거로 본다.

### 2.3 모델 특화

0.28.0 release에서 특히 현재 관심 모델과 직접 연결되는 항목:

- Kimi-K3
  - Decode Context Parallel
  - fused FlashKDA decode/prefill
  - shared-expert sharding
  - adaptive speculative token budget
- DeepSeek V4
  - sparse MLA plain/MTP/DSpark E2E
  - CUDA Graph region 정리
  - sparse top-k metadata 최적화
- speculative decoding
  - DFlash2
  - DSpark confidence-scheduled verification
  - draft model async scheduling
- tiered KV offloading
  - disk tier
  - secondary tier plugin
  - tier metrics

따라서 0.28은 단순 bugfix release가 아니라 현재 platform workload와 직접 맞닿는 runtime feature release다.

## 3. KV connector dependency 변화

vLLM 0.28.0의 `requirements/kv_connectors.txt` 기준:

```text
lmcache >= 0.3.9
nixl == 1.3.2
mooncake-transfer-engine >= 0.3.12
```

공식 source:
https://github.com/vllm-project/vllm/blob/v0.28.0/requirements/kv_connectors.txt

기존 실무 기준과 비교:

| 영역 | 기존 기준 | vLLM 0.28 계열 |
|---|---|---|
| Mooncake | 0.3.10.post2 | >= 0.3.12, 현재 검토 artifact 0.3.12.post1 |
| NIXL | 1.3.1 계열 | 1.3.2 |
| LMCache | 이전 0.3.x | >= 0.3.9 |
| vLLM CUDA default | CUDA 12.9 계열 사용 | CUDA 13.0 default, cu129 image 별도 제공 |

이 표의 의미는 **connector package도 vLLM image ABI의 일부로 취급해야 한다**는 것이다.

## 4. Mooncake 0.3.12.post1과 `nvlink_intra`

### 4.1 가장 중요한 결론

Mooncake 버전이 0.3.12.post1로 올라가도 **x86_64 official wheel만으로 same-node `nvlink_intra` 요구사항이 해결됐다고 가정하면 안 된다.**

Mooncake `v0.3.12.post1` release workflow의 x86 build CMake args에는:

```text
USE_CUDA=ON
```

은 존재하지만,

```text
USE_INTRA_NVLINK=ON
USE_MNNVL=ON
```

은 포함되지 않는다.

ARM64 release job에는 `USE_MNNVL=ON`이 들어가지만 `USE_INTRA_NVLINK=ON`은 여전히 없다.

공식 release workflow:
https://github.com/kvcache-ai/Mooncake/blob/v0.3.12.post1/.github/workflows/release.yaml

Mooncake build option 자체도 `USE_INTRA_NVLINK`의 default가 OFF다.

따라서 Network B의 same-node NVLink/NVSwitch P/D는 계속 다음 계약을 가진다.

```text
Mooncake official wheel
    ≠ nvlink_intra guaranteed

same-node direct transport 필요
    -> source-built Mooncake
    -> USE_CUDA=ON
    -> USE_INTRA_NVLINK=ON
```

MNNVL 비교가 필요하면 `USE_MNNVL=ON`도 별도로 포함한다.

### 4.2 0.3.10 source-build PR을 그대로 재사용할 수 있는가

**아이디어와 build skeleton은 재사용할 수 있지만 source만 교체하면 끝나는 작업으로 보지 않는다.**

재사용 가능:
- air-gap source closure
- fixed source/submodule manifest
- CA / pip / apt proxy injection
- Kaniko-compatible multi-stage build
- no-GitHub/no-Go-fetch static check
- CMake feature flag 검증
- extension load 검증
- `nvlink_intra` runtime smoke 구조

반드시 재검토:
- Mooncake source/submodule lock
- Python build requirements
- CMake option/target 변화
- generated wheel package name
- CUDA 12 / CUDA 13 package variant
- base image Python/Torch/glibc/CUDA ABI
- transport selection logic
- vLLM Mooncake connector interface 변화
- hybrid cache / Mamba/GDN state handling

즉 0.28용 image는 **Dockerfile도 version-aware update 대상**이다.

## 5. Source build image 전략

장기적으로 `vllm-production-stack-custom`은 upstream Production Stack의 fork를 유지하고 필요한 overlay만 얹는 방식으로 간다.

Mooncake image도 같은 원칙을 적용한다.

```text
upstream vLLM image
    +
small deterministic Mooncake build overlay
    =
validated runtime image
```

지양:
- full vLLM image Dockerfile을 통째로 fork해 장기 유지
- package version 변경 때마다 unrelated build logic 복제
- runtime image와 chart 변경을 한 PR에 묶음

권장 책임 분리:

| 책임 | 위치 |
|---|---|
| vLLM base image version/digest | image build config |
| Mooncake source lock | Mooncake overlay |
| CA / package proxy | common build layer |
| P/D values / kvTransfer config | Production Stack fork |
| model scheduler profile | model/runtime profile |
| transport runtime verification | validation script/runbook |

## 6. 0.28 migration Gate

### G0 — artifact freeze

- `vllm-openai:v0.28.0-cu129` digest
- Python/Torch/CUDA versions
- Mooncake 0.3.12.post1 source commit
- NIXL 1.3.2
- image 내부 Transformers exact version 기록; v0.28.0 requirement floor는 `transformers >= 5.5.3`
- model revision/tokenizer/config checksum

### G1 — integrated baseline

P/D와 SD를 끄고 기존 모델을 먼저 검증한다.

- model load
- raw completion/chat
- streaming
- reasoning/tool parser
- long-context ladder
- eager/graph baseline

### G2 — P/D connector

- Prefill/Decode extension load
- same-node `nvlink_intra`
- requested protocol이 아니라 actual Mooncake transport log 확인
- KV transfer correctness
- cancellation/restart
- no silent TCP fallback

### G3 — scheduler/graph

- 기존 MBT
- 8192
- 16384
- workload-specific 후보

를 비교한다.

P와 D는 별도 sweep한다.

### G4 — speculative decoding

`num_speculative_tokens`와 scheduler budget을 독립 변수가 아니라 묶어서 본다.

특히 v0.28은 v0.26/v0.27의 `max_num_seqs` 기준 static worst-case drafting headroom과 달리, scheduler logical token budget과 physical input-slot budget을 분리해 실제 scheduled request별 drafting slot을 accounting한다. 이전 version의 `MBT - slots × max_num_seqs` 식을 0.28에 그대로 적용하지 않는다.

상세 source-level accounting은 [Speculative Decoding Token Budget Deep Dive](../../study/speculative-decoding/02-vllm-token-budget-and-slot-topology.md)를 따른다.

- K = 0 / 3 / 5 / 7 등
- concurrency ladder
- acceptance
- effective scheduled token budget
- TPOT/ITL
- TTFT
- GPU utilization
- CUDA Graph capture/replay

### G5 — soak

- long context
- concurrency
- prefix cache
- streaming/cancel
- restart/recovery
- malformed output / NaN / CUDA IMA / Xid = 0

## 7. Promotion 원칙

새 버전 승격 판단은 기능 개수로 하지 않는다.

```text
correctness
  -> lifecycle safety
  -> reproducible data path
  -> capacity
  -> latency/throughput
  -> operational simplicity
```

0.28이 더 빠르더라도 connector, parser, state, long-context correctness가 깨지면 production baseline으로 승격하지 않는다.

반대로 0.28의 upstream primitive가 기존 custom patch를 제거할 수 있다면, 단순 throughput보다 **custom surface 감소** 자체를 중요한 운영 가치로 본다.

## 8. 공식 근거

- vLLM 0.28.0 release  
  https://github.com/vllm-project/vllm/releases/tag/v0.28.0
- vLLM 0.28.0 KV connector requirements  
  https://github.com/vllm-project/vllm/blob/v0.28.0/requirements/kv_connectors.txt
- vLLM CUDA Graph design  
  https://github.com/vllm-project/vllm/blob/v0.28.0/docs/design/cuda_graphs.md
- vLLM 0.28.0 common requirements  
  https://github.com/vllm-project/vllm/blob/v0.28.0/requirements/common.txt
- Mooncake 0.3.12.post1 release  
  https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.12.post1
- Mooncake 0.3.12.post1 release workflow  
  https://github.com/kvcache-ai/Mooncake/blob/v0.3.12.post1/.github/workflows/release.yaml
- Mooncake build options  
  https://github.com/kvcache-ai/Mooncake/blob/v0.3.12.post1/mooncake-common/common.cmake
