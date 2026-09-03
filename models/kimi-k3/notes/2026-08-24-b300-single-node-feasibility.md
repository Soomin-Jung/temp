# Kimi-K3 — B300 8-GPU Single-Node Feasibility

업데이트: 2026-08-25 KST  
상태: **Feasibility note / production validation 아님**

## 결론

Kimi-K3를 **HGX/DGX B300 8-GPU 단일 노드**에서 구동하는 것은 기술적으로 가능한 경로로 본다. 다만 이것을 현재 H200 multi-node 운영 계획의 대체안으로 간주하지 않는다.

- 단일 B300 노드의 기본 형태는 **TP8**이다.
- Kimi-K3의 native MXFP4 weight footprint는 단일 8-GPU B300의 aggregate HBM 안에 들어가는 범주이므로 low-concurrency serving 자체는 feasibility가 있다.
- 1M context는 모델이 지원한다고 해서 곧바로 production concurrency가 충분하다는 뜻이 아니다. 장문맥 KV/KDA state, CUDA Graph workspace, runtime buffer를 포함하면 동시성은 제한된다.
- 따라서 B300 단일노드는 **개발/기능 검증·저동시성 long-context endpoint**에는 의미가 있지만, 높은 동시성 또는 단일 노드 내부 P/D 분리의 기본 production topology로 보지 않는다.

## 소프트웨어 전제

2026-08-24 검토에서 잡은 working baseline은 다음과 같다.

- vLLM `>= 0.27.1` 계열
- CUDA 13 계열
- NVIDIA driver R580+ 계열
- Kimi-K3 공식/패치 반영 image를 digest로 고정
- fastsafetensors 등 large-checkpoint load path 검증

현재 Network A/B의 기존 운영 driver `575.57.08`과 `v0.27.1-cu129` 환경은 이 B300 경로의 그대로의 실행 전제가 아니다. B300을 실제 도입할 경우 driver/CUDA/container stack을 별도 승격해야 한다.

## 최초 bring-up 기준

처음부터 모든 기능을 동시에 켜지 않는다.

1. TP8 + eager + 짧은 context + seq1
2. 실가중치 load/HBM/출력 correctness
3. CUDA Graph
4. concurrency 2 → 4 → 8 soak
5. context ladder
6. FP8 KV / prefix cache / speculative decoding은 각각 별도 gate

초기 검증에서는 기능 지원 여부와 production-safe 여부를 분리한다.

## 1M context 해석

Kimi-K3의 1M context는 architecture capability다. 실제 B300 단일노드 capacity는 다음을 함께 계산해야 한다.

- MXFP4 weights
- MLA KV
- KDA recurrent state
- CUDA Graph capture/workspace
- MoE/EP/TP temporary buffers
- fragmentation/headroom
- concurrency

따라서 `1M context 지원 = 1M 요청을 여러 개 동시에 안정적으로 처리 가능`으로 해석하지 않는다.

## 현재 H200 계획과의 관계

현재 production validation의 기본 경로는 계속 **H200 ×16에서 안전성 검증 후 ×32로 확장**하는 단계다.

B300 TP8은 다음과 같이 취급한다.

```text
B300 ×8 single node
  = compact feasibility / low-concurrency candidate

H200 ×16 → ×32
  = current staged production-validation track
```

두 경로의 benchmark를 다른 조건으로 직접 순위화하지 않는다.

## Production gate

B300 단일노드를 실제 후보로 올릴 때 최소한 다음을 확인한다.

- exact GPU SKU / aggregate HBM / NVSwitch topology
- driver/CUDA/vLLM/container digest
- model revision과 weight format
- load time 및 host-memory staging
- TP8 collective/NCCL 안정성
- fixed golden prompt correctness
- graph/eager A/B
- 32K → 64K → 128K → 262K → 필요 시 1M context ladder
- concurrency별 HBM/latency/OOM
- KDA state / MLA KV correctness
- prefix cache 및 FP8 KV 회귀
- long soak에서 CUDA IMA/Xid/restart/malformed output 0 여부

## 판단 원칙

B300 단일노드는 **가능/불가능의 문제가 아니라 workload fit의 문제**다. 단일 노드에 모델을 올릴 수 있다는 사실과 production throughput·tail latency·failure-domain 요구를 충족한다는 사실을 분리해서 검증한다.
