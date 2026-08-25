# CURRENT CONTEXT — Canonical State

> Last updated: 2026-08-25 KST
>
> 이 문서는 과거 ChatGPT 대화방을 삭제한 뒤에도 작업을 이어갈 수 있도록 만든 **session-independent canonical context**다.
> 과거 채팅의 탐색 과정이나 당시 가설보다 실제 GitHub 상태와 각 영역의 canonical 문서를 우선한다.

## 0. 사용 원칙

1. 현재 상태는 **실제 GitHub code/PR current state → 이 문서 → 영역별 canonical docs → 최신 runtime test** 순으로 판단한다.
2. 과거 채팅과 이 문서가 충돌하면 과거 내용은 historical context로만 취급한다.
3. 공개 저장소에서는 실제 조직/망 이름 대신 **Network A / Network B**를 사용한다.
4. 개인 경력, 개인정보, 내부 식별자는 이 공개 저장소에 저장하지 않는다.
5. 새 결정이나 runtime 결과가 생기면 채팅을 보존하는 대신 관련 문서와 이 파일을 갱신한다.

---

# 1. LLM E2E Platform / Infrastructure

## Network A

- H200 8-GPU NVSwitch 노드군
- H100 4-GPU 노드군
- L40S 8-GPU 노드군
- H200 fabric 영역은 InfiniBand + GPUDirect RDMA 검증 완료
- multi-node vLLM/NCCL, Kimi-K3 등 fabric-aware serving 실험의 주 대상

## Network B

- H200 8-GPU NVSwitch 노드군
- InfiniBand 사용 불가
- 동일 노드 내부 GPU 간 통신은 NVLink/NVSwitch 경로 우선
- node-local P/D Cell의 우선 검증 환경

## Serving plane

```text
Client
  -> API Gateway / Ingress
  -> custom vLLM proxy / API policy layer
  -> LiteLLM
  -> vLLM Production Stack / Router / Service
  -> vLLM engine replicas
  -> GPU / KV transfer / cache layer
```

Observability는 Prometheus / Grafana / Loki / Alloy / DCGM 계열을 사용하며 inference runtime에서는 모델/엔진/라우터 단위 metrics를 분리해 본다.

폐쇄망이므로 외부 GitHub/Hugging Face/일반 패키지 registry 의존은 사전 반입 또는 Artifactory proxy를 전제로 한다. Docker image build는 Kaniko 흐름과 결합될 수 있다.

---

# 2. vLLM / Production Stack 기준선

## vLLM

- active validation 계열: **vLLM 0.27.x**
- 현재 P/D 실험의 실질 기준 image: **`v0.27.1-cu129` 계열**
- 현재 운영 GPU server driver 기준: **575.57.08**; CUDA 13 계열을 현재 그대로의 runtime 전제로 두지 않는다.
- 모델별로 0.26.0 ↔ 0.27.x regression을 stage-gate로 확인한다.
- `기능 지원`과 `production-safe`를 구분한다.

## vLLM Production Stack

### Short-term

- custom **0.1.8 baseline** 유지
- 기존 renderer를 최대한 건드리지 않고 additive extension으로 P/D를 검증

### Long-term

- **0.1.12+** upstream P/D orchestration/routing/observability primitive를 기준으로 semantic migration
- 0.1.8 custom patch를 무한 누적하지 않고 필요한 조직별 정책만 custom layer로 재적용

상세 문서:

- [vLLM Stack 진행계획](vllm-stack/2026-08-18-진행계획.md)
- [P/D Master Plan](vllm-stack/pd-disaggregation/2026-08-18-vllm-stack-pd-disaggregation-master-plan.md)
- [Node-local P/D Cell](vllm-stack/pd-disaggregation/2026-08-18-node-local-pd-cell-0.1.8-plan.md)
- [Mooncake 0.3.10-post2 폐쇄망 Source Build](vllm-stack/pd-disaggregation/2026-08-24-mooncake-0.3.10-post2-offline-build.md)
- [Stateful Conversation Architecture](vllm-stack/stateful-conversation-architecture.md)

---

# 3. P/D Disaggregation — 현재 상태

Repository: `Soomin-Jung/vllm-production-stack-custom`

## PR #1 — 0.1.8 Custom Baseline

- **Merged**
- main의 downstream baseline
- 실제 운영 manifest와 semantic diff가 없도록 복원/검증한 기준선
- schema/PDB 등 장기 개선은 baseline에 억지로 넣지 않고 이후 migration에서 재검토

## PR #2 — Node-local P/D Cell

상태: **Open Draft / runtime smoke 확인됨 / merge 전**

```text
PD Cell Pod
  ├─ Cell Router
  ├─ Prefill × N
  └─ Decode × M
```

핵심 계약:

- `pdCellSpec.models[]` additive root
- Prefill/Decode는 동일 served model의 phase
- Cell replica가 scheduler의 scale/failure 기본 단위
- Cell 전체 GPU request를 한 node에 원자적으로 배치
- Cell-local router가 localhost P/D backend 연결
- 외부 Service는 Cell Router만 노출
- P/D/Router container별 metrics port 노출
- Helm lint/template/test/pre-commit 완료
- P1:D1 + LiteLLM + Anthropic 경로 smoke 확인

남은 gate:

- long context
- cancellation
- restart/recovery
- P2:D2 / P3:D1
- global/router discovery
- LiteLLM path 재검증
- metrics scrape
- integrated vs P/D 성능 비교

## PR #4 — P/D values / template / router contract audit

상태: **Open Draft / PR #2 위에 stacked / 실제 환경 테스트 중**

검증 순서:

```text
Qwen3.6-27B P1:D1
  -> P2:D1
  -> P2:D2
  -> P1:D3
  -> replica 0↔1
```

주요 반영:

- `kvTransfer` ownership = `pdCellSpec.models[]`
- `servedModelNames: [primary, alias...]`
- LMStack Router / vLLM Router / custom router 계약 분리
- env/envFrom/volumeMount/extraVolumes 상속
- 비대칭 topology 지원
- 실제 parallelism과 `requestGPU` 일치 원칙
- `PROMETHEUS_MULTIPROC_DIR=/tmp`
- 최소 values + full reference values
- long-context production의 KV load failure policy는 **P/D 모두 fail** 우선

## 최신 runtime 관찰 — Qwen GDN/Mamba hybrid path

- vLLM 0.27.1에서 Mooncake connector의 hybrid cache interface mismatch가 있었음.
- local patch로 `cache_or_caches`를 single cache list 형태로 정규화한 뒤 Mooncake transfer init과 CUDA Graph capture까지 통과한 사례가 있음.
- Decode profile의 `max_num_batched_tokens=256`은 Mamba/GDN `block_size=1600` assertion과 충돌했으므로 profile tuning 시 hybrid-state block constraint를 함께 본다.
- 이 항목은 upstream/general guarantee가 아니라 **현재 runtime observation**이며 최종 image/patch contract가 확정되면 별도 문서/PR로 승격한다.

## Failure boundary

단기 P/D Cell에서는 부분 복구를 지나치게 복잡하게 만들지 않는다.

- P/D engine 일부 unhealthy/dead → Cell 전체 route 제외
- 완전 복구 후 재가입
- Pod Ready와 inference healthy를 동일시하지 않음

---

# 4. Mooncake / KV Transfer — 현재 상태

Network B의 node-local P/D는 host/TCP 우회가 아니라 **NVLink/NVSwitch GPU-local path**가 목표다.

## 목표

- `mooncake-transfer-engine == 0.3.10.post2`
- `nvlink`
- 특히 same-node용 `nvlink_intra`

Transport build mapping:

- `nvlink` → `USE_MNNVL=ON`
- `nvlink_intra` → `USE_INTRA_NVLINK=ON`

기본 배포 wheel 조합에는 원하는 `nvlink_intra` transport가 포함되지 않는 경우가 있어 단순 wheel 재설치만으로는 요구사항을 만족하지 못한다.

## PR #5 — Mooncake NVLink transport image build

Repository: `Soomin-Jung/vllm-production-stack-custom`  
상태: **Merged / 폐쇄망 Kaniko image build 성공 / GPU runtime은 PR #4에서 검증 중**

Merged commit:

```text
88185894123342e22a0a60c9c2e3f0f9ab115a18
```

구현:

- `docker/Dockerfile.vllm-mooncake` dedicated overlay
- 내부 검증 base 기본값 `vllm/vllm-openai:v0.26.0-cu129`
- corporate CA, `pip.conf`, `sources.list`를 builder/runtime에 공통 적용
- base의 `/etc/apt/sources.list.d/*.list` 제거
- Mooncake `v0.3.10.post2`와 pybind11/yalantinglibs 고정
- 수동 반입 source closure: `mooncake-offline_0.3.10.post2.tar.gz`
- Python/APT dependency는 사내 Artifactory proxy 사용; GitHub/Go online fetch 없음
- `USE_CUDA=ON`, `USE_MNNVL=ON`, `USE_INTRA_NVLINK=ON`
- Store/etcd/Rust/Go/EP/TENT 제외
- Mooncake CMake link용 `LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}`
- `SHA256SUMS`가 있으면 archive checksum 검증
- `SOURCE_MANIFEST.env`가 있으면 고정 commit까지 검증; 없으면 version과 populated submodule 필수 파일 검증
- wheel 생성, final image 설치, 전용 static validation까지 실환경에서 통과

실제 transport 선택 주의:

```text
NVLink compile flag가 하나라도 ON인 build:

MC_INTRANODE_NVLINK 존재
  -> nvlink_intra
else MC_FORCE_MNNVL 존재 또는 HCA 없음
  -> nvlink
else
  -> rdma
```

환경변수는 값이 아니라 **존재 여부**를 보므로 비활성화할 때 `"0"`으로 남기지 않고 항목을 제거한다. 이 compile branch에는 TCP 자동 fallback이 없다. `USE_TCP=ON`은 TCP 구현체를 compile할 뿐 자동 선택을 뜻하지 않는다. vLLM의 `mooncake_protocol` log는 requested protocol이며, 실제 설치 transport는 Mooncake log와 remote segment metadata로 확인한다.

PR #4에서 남은 runtime gate:

1. Prefill/Decode 양쪽 extension load
2. `nvlink_intra` 실제 P→D GPU KV transfer
3. `nvlink` 비교와 실제 선택 transport 확인
4. long-context / streaming / cancellation / concurrency
5. transfer latency / TTFT 비교
6. Cell restart/recovery

상세 compile/selection/전송 경로는 [Mooncake 0.3.10-post2 폐쇄망 Source Build](vllm-stack/pd-disaggregation/2026-08-24-mooncake-0.3.10-post2-offline-build.md)를 source of truth로 사용한다. Network B node-local P/D의 최종 성능 평가는 위 runtime gate 이후 확정한다.

---

# 5. KV Cache / Routing 원칙

서로 다른 문제를 분리한다.

1. **P→D KV transfer** — Prefill 결과를 Decode GPU로 전달
2. **remote/shared KV cache** — replica/session 사이 KV 재사용
3. **cache-aware routing** — cache hit 가능성이 높은 backend 선택

검토/사용 대상:

- Mooncake
- LMCache
- NIXL
- vLLM native/offload 기능
- LMRouter / cache-aware routing 계열

Topology 원칙:

- same-node NVLink/NVSwitch → GPU-local direct path 우선
- cross-node IB 가능 → GDRDMA/IB path 우선
- Ethernet/TCP fallback은 long-context P/D 기본 목표로 보지 않음

---

# 6. MOC — Model-Ops-Controller

Repository: `Soomin-Jung/model-ops-controller`

## Canonical identity

- **WADM 폐기**; active component/migration target 아님
- MOC = 독립적인 **LLM E2E Operations Control Plane**
- inference request data path를 직접 대체하지 않고 actual state를 관찰해 safe reconcile plan을 수행

## 현재 구현 상태

- **MOC Core Framework v0.1.0 released**
- Core runtime/API/CRD/controller/envtest/release pipeline까지 완료
- 남은 Core operational checkpoint: **real-cluster smoke validation**
- 이후 다음 구현 milestone: **Workload & Route Lifecycle**

Core execution model:

```text
Observe
  -> Normalize
  -> Decide
  -> Plan
  -> Validate / Safety Guard
  -> Execute
  -> Verify / Reconcile
  -> Audit / Status
```

Core v0.1은 아직 실제 vLLM workload 생성, route traffic, GPU placement/autoscaling, multi-node/P-D, KV/conversation state를 직접 수행하지 않는다.

## Business capability backlog

1. Production Stack discovery/adoption + vLLM runtime adapter
2. safe route membership / drain lifecycle
3. Kubernetes Service endpoint dynamic management
4. LiteLLM logical model/endpoint synchronization
5. workload-aware model placement
6. time/event based scaling
7. donor/victim GPU reclaim
8. eviction/repacking / TP8 admission
9. distributed serving
10. P/D, KV/cache routing, conversation policy domain modules

관련 설계는 MOC repo의 README / roadmap / backlog / RFC가 source of truth다.

---

# 7. Model-specific active tracks

## Kimi-K3

Canonical:

- [Kimi-K3 Study / Architecture](models/kimi-k3/README.md)
- [B300 8-GPU Single-Node Feasibility](models/kimi-k3/2026-08-24-b300-single-node-feasibility.md)

현재 production-validation 방향:

- vLLM 0.27.x 공식 지원 경로 검증
- **H200 ×16에서 안전성 검증 후 ×32로 확장**
- 초기 scope는 text / 짧은 context / seq1 / eager부터 시작
- Ray 의존 축소와 native multiprocess backend 검토
- KDA / Gated MLA / Stable LatentMoE / EP·TP·TEP / communication을 함께 검증
- KV cache뿐 아니라 KDA recurrent state를 별도 관점으로 관측

B300 단일노드:

- TP8 기반 low-concurrency feasibility candidate
- 현재 H200 staged production-validation track의 대체안은 아님
- CUDA 13 / R580+ 계열 stack이 전제이므로 현재 575.57.08 + cu129 환경과 분리해서 본다.

## DeepSeek V4

- Flash / Pro 계열 deployment + DSpark 검증
- 0.26.0에서 성공하던 구성과 0.27.x DeepGEMM/CUDA regression 분리 추적
- production promotion은 H100/H200 stage-gate 후 결정

## Qwen 3.x

- P/D 운영 검증 우선: Qwen3.6-27B
- Qwen3.8-27B + MTP deployment/streaming validation
- reasoning profile과 speculative/MTP path 검증
- GDN/Mamba hybrid cache는 일반 dense-attention 모델과 다른 runtime constraint를 가짐

## Gemma / 기타

- throughput 증가 시 repetition/looping 등 correctness regression을 TPS와 별도로 추적

---

# 8. Stateful Conversation

Canonical:

- [Stateful Conversation Architecture](vllm-stack/stateful-conversation-architecture.md)

핵심 결론:

```text
Durable Conversation Store
  + Conversation State Facade
  + optional Router Affinity
```

- process-local response/session state를 durability source of truth로 쓰지 않는다.
- `affinity != durability`.
- OpenAI Responses / Anthropic 호환은 message text뿐 아니라 response identity, reasoning item, tool call/result, streaming terminal state, retry/idempotency까지 구조화해 보존해야 한다.
- MOC는 conversation primary store가 아니라 rollout/drain/health policy와 integration할 수 있는 control plane이다.

---

# 9. Study canonical sources

Root:

- [Study README](study/README.md)
- [Attention](study/attention/README.md)
- [GPU Architecture](study/gpu-architecture/README.md)
- [Git & GitHub](study/git-github/README.md)
- [Speculative Decoding](study/speculative-decoding/2026-08-18-speculative-decoding-mtp-dspark.md)

학습 원칙:

- 결과 암기보다 **왜 필요한가 → 무엇을 보존/변환하는가 → 실제 LLM/system 어디에 쓰이는가**를 연결
- hardware / runtime / model architecture / serving behavior를 분리하지 않고 연결
- math rendering은 repository Markdown 규칙 준수

Backlog:

- Dense FFN → sparse MoE → modern MoE 계보
- vector/norm 기초와 이후 linear algebra 연결

---

# 10. 2026 Q3 Architecture / Workstreams

Canonical:

- [2026 Q3 Overview](roadmap/2026/Q3/README.md)
- [Q3 Architecture](roadmap/2026/Q3/architecture.md)
- [Q3 Workstreams](roadmap/2026/Q3/workstreams.md)

주요 축:

- API / compatibility
- Stateful conversation
- Intelligent routing
- Inference runtime
- KV/cache
- GPU / network
- Observability
- Operations control plane (MOC)

MOC는 전체 플랫폼의 일부 control plane이며 플랫폼 자체와 동일시하지 않는다.

---

# 11. Reporting / recurring knowledge

- `weekly-report/`: vLLM, dependent projects, open-weight models, serving/optimization, 주요 논문/기술 변화의 canonical archive
- 최신 뉴스보다 실제 stage-gate runtime test가 운영 판단에 우선
- 오래된 일회성 뉴스 ChatGPT 대화는 source of truth가 아님

Automation 자체는 ChatGPT task state에 존재하므로 대화방을 보존해야만 유지되는 정보로 보지 않는다. 단, 실제 활성/중지 여부는 automation state를 직접 확인한다.

---

# 12. Deprecated / historical decisions

되살리지 않는다.

- WADM을 active project로 취급
- Production Stack 0.1.8에 모든 장기 기능 누적
- node-local P/D 성공을 multi-node/fabric P/D 성공으로 간주
- TCP fallback을 Network B 목표 KV transport로 해석
- Pod Ready만으로 P/D Cell health 판단
- Prefill/Decode를 사용자 관점의 서로 다른 served model로 모델링
- affinity를 durable conversation persistence로 간주
- Kimi-K3의 단일 B300 feasibility를 현재 H200 production plan으로 대체

---

# 13. Chat deletion contract

대화방 삭제 후 우선순위:

```text
1. 실제 GitHub code / PR current state
2. CURRENT_CONTEXT.md
3. 각 영역의 canonical README / design docs
4. 최신 runtime test 결과
5. Memory / private context의 장기 preference
6. 과거 ChatGPT transcript
```

과거 채팅은 **결정의 원본 로그**일 뿐 source of truth가 아니다. 핵심이 GitHub/docs/memory/runtime evidence로 승격됐으면 삭제해도 된다.
