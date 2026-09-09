# vLLM Cross-Version Migration Audits

업데이트: 2026-09-10 KST

`versions/`가 tag별 사실 원장이라면 이 디렉터리는 **실제 serving platform migration decision**을 소유한다.

## Current program

Baseline:

```text
vLLM v0.26.0
```

Candidate:

```text
vLLM v0.29.0
```

Intermediate releases 0.27/0.28은 별도 production certification target이라기보다 regression bisect point와 feature provenance로 사용한다.

## Documents

| 문서 | 목적 |
|---|---|
| [`v0.26.0-to-v0.29.0.md`](v0.26.0-to-v0.29.0.md) | 전체 migration architecture, rollout/rollback gate |
| [`deployment-option-matrix.md`](deployment-option-matrix.md) | 실제 CLI/env/config의 KEEP/ADD/REMOVE/RETEST 판단 |
| [`model-runner-v1-v2.md`](model-runner-v1-v2.md) | MRV1→MRV2 default inversion과 execution-path 검증 |
| [`frontend-parser-compatibility.md`](frontend-parser-compatibility.md) | Python/Rust frontend, renderer, reasoning/tool/stream parser |
| [`hybrid-mamba-gdn.md`](hybrid-mamba-gdn.md) | Qwen/Kimi 계열 hybrid state cache, prefix/replay/spec semantics |
| [`kv-transfer-mooncake.md`](kv-transfer-mooncake.md) | KVTransferConfig, Mooncake/NIXL, source-build 필요성, P/D compatibility |
| [`model-family-compatibility.md`](model-family-compatibility.md) | Qwen3.5+, Kimi K3, DeepSeek V4, GLM 5.x validation lane |

## Migration principle

`0.29가 최신이므로 옵션을 최신 default에 맞춘다`가 아니다.

1. **Compatibility phase**: 0.26의 의도를 최대한 보존해 0.29를 실행한다.
2. **Isolation phase**: MRV2, communicator, graph policy, cache policy 등 큰 변수를 하나씩 분리한다.
3. **Optimization phase**: 0.29의 새 default/optimization을 명시적으로 채택한다.
4. **Platform phase**: router/frontend/observability/chart까지 E2E contract를 업데이트한다.

## Action labels

- `KEEP`: 기존 explicit value 유지
- `PIN`: exact version/value를 더 강하게 고정
- `ADD`: 신규 option/config 도입 권장
- `REMOVE`: 삭제/deprecated option 제거
- `RETEST`: 동일 이름이 남아도 내부 path가 바뀌어 재검증 필수
- `A/B`: migration isolation을 위해 두 path 비교
- `OPTIONAL`: 운영 정책에 따라 선택

## Definition of compatible

단순 startup success가 아니다.

```text
API shape
+ parser/streaming correctness
+ model token correctness
+ scheduler behavior
+ memory capacity/headroom
+ KV/state cache correctness
+ P/D transfer correctness
+ speculative acceptance
+ TP/DP/EP communication
+ latency/throughput SLO
+ failure/recovery behavior
```

이 전체가 acceptance surface다.
