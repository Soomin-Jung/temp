# CURRENT CONTEXT — Canonical State

> Last updated: 2026-08-24 KST
>
> 이 문서는 과거 ChatGPT 대화방을 삭제한 뒤에도 작업을 이어갈 수 있도록 만든 **session-independent canonical context**다.
> 과거 채팅의 탐색 과정이나 당시 가설보다 이 문서와 각 영역의 상세 문서를 우선한다.

## 0. 사용 원칙

1. **현재 상태는 이 문서를 우선**한다.
2. 세부 기술 근거는 각 영역의 README/상세 문서를 따른다.
3. 과거 채팅에서 이 문서와 충돌하는 설계/상태가 나오면 과거 내용을 historical context로만 취급한다.
4. 공개 저장소에서는 실제 조직/망 이름을 쓰지 않고 **Network A / Network B**로 표기한다.
5. 개인 경력, 개인정보, 내부 식별자 등은 이 공개 저장소의 canonical context에 저장하지 않는다.
6. 완료된 탐색 과정은 채팅에 의존하지 않는다. 향후 새 결정이 생기면 해당 영역 문서와 이 파일을 갱신한다.

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
- 동일 노드 내부 GPU 간 통신은 NVLink/NVSwitch 경로를 최대한 활용해야 함
- 현재 node-local P/D Cell의 우선 검증 환경

## Serving plane

현재 시스템은 대략 다음 계층으로 본다.

```text
Client
  -> API Gateway / Ingress
  -> custom vLLM proxy / API policy layer
  -> LiteLLM
  -> vLLM Production Stack / Router / Service
  -> vLLM engine replicas
  -> GPU / KV transfer / cache layer
```

Observability는 Prometheus / Grafana / Loki / Alloy / DCGM 계열을 사용하며, inference runtime에서는 모델/엔진/라우터 단위 metrics를 분리해 보는 것을 원칙으로 한다.

폐쇄망이므로 외부 GitHub/Hugging Face/일반 패키지 레지스트리 의존은 사전 반입 또는 Artifactory proxy를 전제로 한다. Docker 이미지는 Kaniko 기반 빌드 흐름과 결합될 수 있다.

---

# 2. vLLM / Production Stack 기준선

## vLLM

- 현재 active validation 계열: **vLLM 0.27.x**
- 모델별로 0.26.0과 0.27.x 사이 regression 여부를 반드시 stage-gate 방식으로 확인
- 기능 지원 여부와 실제 production-safe 여부를 구분

## vLLM Production Stack

### Short-term

- custom **0.1.8 baseline** 유지
- 기존 운영 renderer를 최대한 건드리지 않고 additive extension으로 P/D 기능 검증

### Long-term

- **0.1.12+**의 upstream P/D orchestration, routing, observability primitive를 기준으로 semantic migration
- 0.1.8 custom patch를 그대로 누적하지 않고 필요한 조직별 정책만 custom layer로 재적용

상세 진행 계획:

- [vLLM Stack 진행계획](vllm-stack/2026-08-18-진행계획.md)
- [P/D Master Plan](vllm-stack/pd-disaggregation/2026-08-18-vllm-stack-pd-disaggregation-master-plan.md)
- [Node-local P/D Cell](vllm-stack/pd-disaggregation/2026-08-18-node-local-pd-cell-0.1.8-plan.md)
- [Mooncake 0.3.10-post2 폐쇄망 Source Build](vllm-stack/pd-disaggregation/2026-08-24-mooncake-0.3.10-post2-offline-build.md)

---

# 3. P/D Disaggregation — 현재 상태

Repository: `Soomin-Jung/vllm-production-stack-custom`

## PR #1

- custom vLLM Production Stack **0.1.8 baseline**
- merge 완료
- main의 기준선으로 사용
- baseline 목적과 맞지 않는 schema/PDB 과도한 변경은 되돌리고 향후 0.1.12 migration에서 재검토

## PR #2 — Node-local P/D Cell

현재 상태: **Open Draft / runtime smoke 확인됨 / merge 전**

핵심 구조:

```text
PD Cell Pod
  ├─ Cell Router
  ├─ Prefill × N
  └─ Decode × M
```

- `pdCellSpec.models[]` additive root
- Prefill/Decode는 별도 사용자 모델이 아니라 동일 served model의 phase
- Cell replica가 scheduler의 기본 scale 단위
- 한 Cell의 GPU request를 하나의 node에 원자적으로 배치
- Cell-local router가 localhost P/D backend를 연결
- 외부 Service는 Cell Router만 노출
- P/D/Router container별 metrics port를 노출해 Prometheus가 개별 scrape 가능하도록 구성
- 기본 Helm lint/template/test/pre-commit 검증 완료
- PR #2 이전 커밋 `36e45f0c277d4206ce233c8057a383e387c616c1`에서 P1:D1, LiteLLM, Anthropic 경로 확인
- 장기 context, cancellation, restart/recovery, P2:D2/P3:D1, router/global discovery, LiteLLM path, 성능 비교는 계속 검증 대상

## PR #4 — P/D values / template / router contract audit

현재 상태: **Open Draft / PR #2 위에 stacked / 실제 환경 테스트 중**

검증 순서: Qwen3.6-27B P1:D1 재검증 후 P2:D1 → P2:D2 → P1:D3, replica 0↔1

주요 반영:

- `kvTransfer` ownership을 `pdCellSpec.models[]` 단위로 고정
- `servedModelNames: [primary, alias...]`
- LMStack Router / vLLM Router / custom router 계약 분리
- env/envFrom/volumeMount/extraVolumes 상속 보완
- 비대칭 topology 지원 문서화
- P/D별 실제 parallelism과 `requestGPU` 일치 원칙
- `PROMETHEUS_MULTIPROC_DIR=/tmp`
- 최소 values + full reference values 제공
- long-context production의 KV load failure policy는 **P/D 모두 fail**을 우선 기준으로 둠

## Failure boundary

단기 P/D Cell에서는 부분 복구를 지나치게 복잡하게 만들지 않는다.

- P/D engine 일부가 unhealthy/dead → Cell 전체를 route에서 제외
- 완전 복구 후 다시 가입
- Pod Ready와 inference healthy를 동일시하지 않음

---

# 4. Mooncake / KV Transfer — 현재 blocker

현재 가장 중요한 P/D 후속 이슈다.

## 요구사항

Network B는 InfiniBand가 없으므로 node-local P/D의 KV transfer가 host/TCP 우회 경로가 아니라 **NVLink/NVSwitch를 활용하는 GPU-local 경로**를 타야 한다.

목표 Mooncake 계열:

- `mooncake-transfer-engine == 0.3.10.post2`
- 필요한 transport: `nvlink` 및 특히 `nvlink_intra`

## 확인된 상태

- 이전 CUDA runtime ABI 문제는 공식 wheel 재설치로 해소
- 그러나 기본 배포 wheel에는 원하는 `nvlink_intra` transport가 포함되지 않는 빌드 조합이 존재
- 따라서 단순 pip wheel 교체로는 현재 요구사항을 만족하지 못함

Transport 의미:

- `nvlink`: Multi-Node NVLink(MNNVL), `USE_MNNVL=ON`
- `nvlink_intra`: 동일 Node NVLink/NVSwitch, `USE_INTRA_NVLINK=ON`
- Network B H200 Node-local P/D의 목표는 `nvlink_intra`

## 다음 구현 방향

**P/D Helm PR과 분리된 별도 image-build PR**로 다룬다.

목표:

1. Mooncake `v0.3.10.post2` source를 고정하고 vLLM base image 안에서 build
2. NVLink/NVLink-intra 관련 build flag/dependency를 명시
3. 폐쇄망에서 외부 GitHub/Go/module/package fetch가 발생하지 않도록 필요한 source/dependency를 사전 반입
4. Artifactory가 proxy 가능한 Docker/YUM/APK/Python 영역과, 별도 반입이 필요한 Go/Git source dependency를 구분
5. Kaniko가 재현 가능한 Dockerfile + build context manifest 제공
6. build 후 실제 transport enumeration과 P→D KV path를 runtime에서 검증

이 작업이 완료되기 전에는 node-local P/D 성능을 최종 평가하지 않는다.

---

# 5. KV Cache / Routing 원칙

서로 다른 문제를 구분한다.

1. **P→D KV transfer**: Prefill 결과 KV를 Decode GPU 쪽으로 전달
2. **remote/shared KV cache**: 여러 replica/session 사이에서 KV 재사용
3. **cache-aware routing**: cache hit 가능성이 높은 backend로 요청을 전달

현재 검토/사용 대상:

- Mooncake
- LMCache
- NIXL
- vLLM native/offload 기능
- LMRouter / cache-aware routing 계열

Network topology에 따라 connector/transport를 다르게 선택한다.

- same-node NVLink/NVSwitch: GPU-local direct path 우선
- cross-node IB 가능: GDRDMA/IB 기반 path 우선
- Ethernet/TCP fallback은 long-context P/D의 기본 목표 경로로 보지 않음

---

# 6. MOC — Model-Ops-Controller

상세 기준:

- [MOC README](moc/README.md)

## Canonical identity

- **WADM은 폐기됨.** active component 또는 migration target으로 취급하지 않는다.
- MOC는 독립적인 **LLM E2E Operations Control Plane**이다.
- inference request data path를 직접 대체하지 않고 actual state를 수집해 safe reconcile plan을 수행한다.

## 현재 우선순위

**MOC Core Framework가 먼저다.**

```text
State Collection
  -> Canonical State / Normalization
  -> Policy Evaluation
  -> Plan Generation
  -> Safety Guard
  -> Adapter / Executor
  -> Verify / Reconcile
  -> Audit
```

Business capability는 Core 위에서 단계적으로 구현한다.

우선순위:

1. Production Stack discovery/adoption + vLLM runtime adapter
2. Kubernetes Service endpoint dynamic management
3. LiteLLM model endpoint/model-info synchronization
4. workload-aware model placement
5. time/event based scaling
6. donor/victim GPU reclaim
7. eviction/repacking / TP8 admission

PD disaggregation, multi-node deployment, KV cache/routing, stateful conversation policy 등은 Core Framework 이후 domain module 또는 integration으로 다룬다.

---

# 7. Model-specific active tracks

## Kimi-K3

상세 자료:

- [Kimi-K3 Study / Architecture](models/kimi-k3/README.md)

현재 방향:

- vLLM 0.27.x 기반 공식 지원 검증
- H200 multi-node 16/32 GPU급 deployment 검토
- Ray 의존 축소와 native multiprocess backend 검토
- KDA / Gated MLA / Stable LatentMoE / EP·TP·TEP / multi-node communication을 함께 이해
- KV cache뿐 아니라 recurrent/KDA state를 별도 관점으로 관측

## DeepSeek V4

- Flash / Pro 계열 deployment 및 DSpark 적용 검증
- vLLM 0.26.0에서 성공하던 구성과 0.27.x DeepGEMM/CUDA regression을 별도 이슈로 추적
- production promotion은 실제 H100/H200 stage-gate 검증 후 결정

## Qwen 3.x

- 현재 P/D 운영 검증 우선 모델은 Qwen3.6-27B
- Qwen 3.8 27B + MTP deployment/streaming validation 진행
- reasoning strength/profile별 서비스 동작과 speculative/MTP path 검증

## Gemma / 기타

- throughput 증가 시 repetition/looping 같은 correctness regression은 단순 TPS benchmark와 별도로 추적

---

# 8. Study canonical sources

현재 학습 자료의 canonical root:

- [Study README](study/README.md)
- [Attention](study/attention/README.md)
- [GPU Architecture](study/gpu-architecture/README.md)
- [Git & GitHub](study/git-github/README.md)
- [Speculative Decoding](study/speculative-decoding/2026-08-18-speculative-decoding-mtp-dspark.md)

학습 원칙:

- 공식/수식 결과를 암기하기보다 **왜 필요한가 → 내부에서 무엇을 보존/변환하는가 → 시스템/LLM에서 어디 쓰이는가**까지 연결
- hardware / runtime / model architecture / serving behavior를 분리하지 않고 연결
- math rendering은 repository Markdown 규칙을 지킴

추가 backlog:

- Dense FFN → sparse MoE → modern MoE 계보를 독립 학습 모듈로 정리
- math/vector/norm 학습은 기초 벡터 개념과 연결해 확장

---

# 9. 2026 Q3 Architecture / Workstreams

Canonical documents:

- [2026 Q3 Overview](roadmap/2026/Q3/README.md)
- [Q3 Architecture](roadmap/2026/Q3/architecture.md)
- [Q3 Workstreams](roadmap/2026/Q3/workstreams.md)

전체 architecture에서는 MOC를 전체 플랫폼의 일부 control plane으로 본다.

주요 축:

- API / compatibility
- Stateful conversation
- Intelligent routing
- Inference runtime
- KV/cache
- GPU / network
- Observability
- Operations control plane (MOC)

---

# 10. Reporting / recurring knowledge

- `weekly-report/`: vLLM, dependent projects, open-weight models, serving/optimization, 주요 논문/기술 변화의 주간 기록
- 단기 운영 판단은 최신 weekly/daily 정보보다 실제 stage-gate runtime test가 우선
- 오래된 ChatGPT 뉴스 대화는 canonical source로 사용하지 않음

---

# 11. Deprecated / historical decisions

아래는 현재 상태로 되살리지 않는다.

- WADM을 active project로 취급하는 설계
- Production Stack 0.1.8에 계속 모든 장기 기능을 누적하는 전략
- Node-local PD 성공을 multi-node/fabric PD 성공으로 간주
- TCP fallback이 Network B의 목표 KV transport라는 해석
- Pod Ready만으로 P/D Cell health를 판단
- Prefill과 Decode를 사용자 관점의 서로 다른 served model로 모델링

---

# 12. Chat deletion contract

과거 ChatGPT 대화방을 삭제한 뒤에는 다음 우선순위를 사용한다.

```text
1. 실제 GitHub code / PR current state
2. CURRENT_CONTEXT.md
3. 각 영역의 canonical README / design docs
4. 최신 runtime test 결과
5. Memory에 보존된 장기 사용자/협업 preference
6. 과거 chat transcript (삭제 가능)
```

즉, 과거 채팅은 **결정의 원본 로그**일 뿐 앞으로의 source of truth가 아니다.
