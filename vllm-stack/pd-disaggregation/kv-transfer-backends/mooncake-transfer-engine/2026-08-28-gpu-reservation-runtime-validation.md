# Mooncake P/D Cell GPU Reservation Runtime Validation — 2026-08-28

> 결론: **PR #4의 Pod-local aggregate GPU reservation 구조로 실제 P/D KV transfer 성공 run을 확인했다.**
>
> 다만 이후 완전히 새로 생성된 P/D Cell에서도 Mooncake `nvlink_intra`
> `cudaIpcOpenMemHandle(...)=CUDA_ERROR_INVALID_CONTEXT(201)`가 재현되었다.
>
> 따라서 이 문서는 GPU reservation/common CVD/`--device-ids` 구조의 성공 검증 기록이며,
> Mooncake CUDA IPC의 **재현성/안정성 자체는 Issue #6에서 계속 추적**한다.

---

## 1. 검증 목적

기존 Kubernetes/NVIDIA Device Plugin의 container별 GPU allocation에서는
Prefill/Decode가 서로 다른 physical GPU를 각자 local ordinal 0..N-1로 보게 되었고,
Mooncake `nvlink_intra`의 `cudaIpcOpenMemHandle()`이 실패했다.

채택한 해결 구조:

```text
P/D Cell Pod
  |
  +-- gpu-reservation
  |     nvidia.com/gpu = Cell 전체 합계
  |     -> 할당 UUID를 deterministic order로 파일 기록
  |
  +-- Prefill
  |     all GPU device 접근 가능
  |     CVD = Cell reservation UUID 전체
  |     --device-ids = Prefill subset
  |
  +-- Decode
        all GPU device 접근 가능
        CVD = Cell reservation UUID 전체
        --device-ids = Decode subset
```

목표는 다음 두 조건을 동시에 만족하는 것이었다.

1. Kubernetes scheduler 수준 GPU reservation 유지
2. P/D process가 동일 CUDA ordinal namespace를 공유하면서 실제 compute GPU는 non-overlap

---

## 2. 검증 topology

```text
Node GPU       8장
P/D topology   P1D1
Prefill        TP2 / 2 GPU
Decode         TP2 / 2 GPU
Cell total     4 GPU
Protocol       Mooncake nvlink_intra
vLLM           0.26.0
Mooncake       0.3.10.post2 custom build
Device Plugin  v0.18.0
```

Chart 입력은 GPU UUID/range를 직접 지정하지 않고 기존 topology만 사용한다.

```yaml
prefill:
  count: 1
  requestGPU: 2

decode:
  count: 1
  requestGPU: 2
```

Chart 결과:

```text
gpu-reservation -> 4 GPU
prefill-0       -> CNTR_GPU_IDX=0,1
decode-0        -> CNTR_GPU_IDX=2,3
```

---

## 3. Kubernetes / container visibility 검증

### 3.1 reservation

`gpu-reservation` container가 계산된 총량 4 GPU를 요청하고
할당된 GPU UUID 파일을 정상 생성했다.

```text
expected total GPU = 2 + 2 = 4
actual reservation = 4
UUID file           = created
```

### 3.2 engine container device visibility

Prefill/Decode container 내부에서:

```bash
nvidia-smi
```

를 실행하면 node GPU 8개가 모두 보였다.

이는 Mooncake CUDA IPC가 peer GPU device를 import할 수 있도록 container runtime
단계에서 device injection을 넓힌 결과다.

### 3.3 실제 vLLM process의 CUDA namespace

interactive `kubectl exec` shell의 environment와 실제 PID1 process environment는
동일하다고 가정하면 안 된다. launcher가 process 시작 직전에 CVD를 설정하기 때문이다.

실제:

```text
/proc/1/environ
  CUDA_VISIBLE_DEVICES=<gpu-reservation에서 확보한 UUID 4개>
```

를 확인했다.

즉:

```text
Linux/container device visibility = node GPU 전체
vLLM CUDA runtime visibility       = Cell reservation GPU만
```

으로 의도한 2단계 격리가 성립했다.

---

## 4. compute partition 검증

Chart가 자동 생성한 `--device-ids`가 Prefill/Decode에 정상 주입되었다.

```text
Prefill -> 0,1
Decode  -> 2,3
```

vLLM engine 초기화와 실제 GPU memory 점유를 확인했을 때 각 engine은 지정된 subset
GPU만 사용했다.

따라서:

```text
동일 Cell-wide CVD           PASS
Prefill/Decode device index  non-overlap
실제 compute placement       PASS
```

다.

---

## 5. Mooncake initialization 검증

Prefill/Decode 양쪽에서:

```text
Using Intra-Node NVLink transport
```

를 확인했다.

KV Connector 초기화 단계에서 custom instrumentation의:

```text
EXPORT_PTR
EXPORT_ALLOC
EXPORT_HANDLE
CTX
```

로그가 정상 생성되었고 초기 registration 단계 특이사항은 없었다.

---

## 6. 실제 inference KV transfer 검증

추론 요청 시 Prefill Mooncake Transfer Engine:

```text
[REQUEST] where=submitTransferTask
[CTX] where=relocateSharedMemoryAddress:before-ipc-open
[IMPORT_SUCCESS] ...
```

가 확인되었다.

이어 다수 transfer descriptor가 처리되었고 metrics:

```text
Num successful transfers = 4
Avg xfer time            ~= 0.77 ms
P90 xfer time            ~= 0.796 ms
Avg MB per transfer      ~= 122.5 MB
Throughput               ~= 159015 MB/s
Avg descriptors          ~= 112
Num failed transfers     = 0
Num failed recvs         = 0
```

를 확인했다.

이 결과는 기존 실패 지점이었던:

```text
cudaIpcOpenMemHandle
```

를 실제로 통과하고 remote GPU mapping + copy 단계까지 진행했다는 강한 증거다.

---

## 7. Decode-side 검증

Decode INFO log에는 Prefill과 같은 transfer metrics가 보이지 않아 처음에는 수신 여부가
불명확했다.

Decode engine에만:

```text
VLLM_LOGGING_LEVEL=DEBUG
```

를 적용해 재검증한 결과:

- remote producer engine을 대상으로 KV receive 시작
- producer TP rank별 receive
- Mooncake transfer request
- remote KV receive/load 완료

를 확인했다.

따라서 HTTP response 성공만이 아니라 **Decode가 실제 Prefill KV를 소비했다**고 판정한다.

---

## 8. 최종 판정

| 검증 항목 | 상태 |
|---|---|
| Chart GPU 합계 계산 | PASS |
| reservation container GPU request | PASS |
| UUID discovery/file | PASS |
| engine all-GPU device visibility | PASS |
| PID1 Cell-wide CVD | PASS |
| 자동 CNTR_GPU_IDX | PASS |
| P/D compute GPU non-overlap | PASS |
| vLLM `--device-ids` binding | PASS |
| Mooncake `nvlink_intra` init | PASS |
| CUDA IPC import | PASS |
| actual KV send | PASS |
| actual Decode receive/load | PASS |
| failed transfer count | 0 |

따라서 **Pod-local GPU Reservation Bridge는 검증 완료**로 상태를 변경한다.

---

## 9. 검증 범위 밖 / 별도 blocker

Prefill container만 강제로 종료 후 자동 restart시키면 cold-start와 달리:

```text
cudaIpcOpenMemHandle
  -> CUDA_ERROR_INVALID_CONTEXT (201)
```

가 재현되었다.

동시에 현재 vllm-router v0.1.15은 Prefill bootstrap `engine_id`를 startup 시 cache하고
runtime refresh하지 않는 restart lifecycle 문제도 확인되었다.

따라서 다음을 구분한다.

```text
GPU reservation / namespace 설계      VALIDATED
cold-start Mooncake KV transfer        VALIDATED

independent Prefill restart recovery    NOT VALIDATED / FAIL
independent Decode restart recovery     NOT VALIDATED
dynamic Router engine-id refresh        NOT SUPPORTED in current baseline
```

이 문제는 GPU reservation 설계의 실패로 되돌리지 않고,
**P/D Cell generation / failure-domain / Mooncake worker context lifecycle** 이슈로 별도 관리한다.

---

## 9.1 Whole-cell guardian 구현

Partial Prefill/Decode restart는 현재 vllm-router/Mooncake generation state를 안전하게
이어갈 수 없으므로 **P/D Cell 전체를 하나의 failure domain으로 recycle**하도록
production-stack PR #4에 guardian을 추가했다.

구성:

```text
P/D Cell Pod
├─ gpu-reservation
├─ pd-router
├─ prefill-*
├─ decode-*
└─ pd-cell-guardian
```

guardian은 모든 핵심 container가 최초 Ready 된 뒤 각 `restartCount`를 baseline으로
저장한다.

```text
all core containers Ready
  -> baseline restartCount 저장
  -> ARMED

Prefill / Decode / Router / reservation 중 하나 restart
  -> restartCount 증가 감지
  -> 자기 Pod UID precondition으로 Kubernetes DELETE
  -> Deployment가 fresh Pod 생성
```

Kubernetes 기본 container restart를 그대로 허용하면 surviving Router/engine이 stale
Mooncake generation/CUDA IPC state를 유지할 수 있으므로, partial restart 직후 해당
Pod generation 자체를 폐기하는 방식이다.

guardian은 같은 vLLM image의 Python runtime을 재사용하며 GPU는 사용하지 않는다.

```text
NVIDIA_VISIBLE_DEVICES=void
poll interval=2s
delete grace=5s
```

RBAC:

```text
pods/get
pods/delete
```

만 namespace Role로 부여한다.

Pod 기본 ServiceAccount token 자동 mount는 끄고, guardian container에만 projected
token과 `kube-root-ca.crt`를 mount한다.

현재 남은 검증은 implementation 자체가 아니라 다음 runtime recovery path다.

```text
cold transfer PASS
  -> Prefill kill
  -> guardian detects restartCount
  -> whole Pod deleted
  -> new Pod / new UID
  -> fresh GPU reservation
  -> fresh Router/P/D/Mooncake generations
  -> actual KV transfer PASS
```

P1D1 반복 recycle과 P1D2 whole-cell recycle까지 통과하면 restart policy를
production-validated로 올린다.

---

## 10. 운영 baseline

현재 검증된 baseline:

```text
P/D Cell cold deployment
  -> aggregate GPU reservation
  -> common CUDA namespace
  -> automatic --device-ids
  -> forced Mooncake nvlink_intra
  -> actual CUDA IPC KV transfer
```

독립 engine crash가 발생하는 경우 자동 partial restart의 안전성이 확보되기 전까지는
Cell 전체 재생성을 안전한 fallback으로 본다.

성능 검증은 이제 GPU namespace 자체를 의심하는 단계가 아니라,
실제 workload topology/TP/MTP 및 restart resilience를 별도 축으로 진행한다.
