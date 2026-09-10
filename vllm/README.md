# vLLM Upstream Version & Migration Audits

업데이트: 2026-09-10 KST

이 디렉터리는 **upstream vLLM 자체의 버전 변화와 source-level compatibility를 추적하는 원장**이다.

`vllm-stack/`이 실제 serving platform의 구현 계약과 배포 구조를 소유한다면, `vllm/`은 그 판단에 필요한 upstream 근거를 소유한다.

## 책임 경계

| 경로 | 책임 |
|---|---|
| `vllm/versions/` | 특정 vLLM release/tag의 변경점을 release note, PR, source code 기준으로 고정해 기록 |
| `vllm/migrations/` | 여러 release를 가로지르는 실행 경로, API, parser, KV transfer, scheduler, model-family compatibility와 migration 의사결정 |
| `vllm-stack/` | 실제 platform chart/router/service/runtime integration 계약 |
| `models/` | 모델 family architecture와 serving 특성 |
| `study/` | 특정 버전에 종속되지 않는 serving 원리와 mental model |

## 이 영역에서 중요하게 보는 것

단순히 `new model supported` 여부를 나열하지 않는다. E2E serving platform 관점에서 다음 변경을 우선 추적한다.

1. Python/Rust frontend, renderer/derender, HTTP/SSE/gRPC request path
2. reasoning/tool-call/streaming parser와 structured output
3. Model Runner V1/V2 선택 조건과 execution-path 변화
4. scheduler token budget, `max_num_batched_tokens`, `max_num_seqs`, async scheduling
5. CUDA Graph capture/profiling과 non-KV/KV memory accounting
6. KV cache layout, Mamba/SSM/GDN state cache, prefix caching
7. KV transfer connector interface와 P/D, E/P/D disaggregation
8. Mooncake/NIXL dependency와 transport/build contract
9. MTP/DSpark/EAGLE/DFlash 등 speculative decoding
10. Qwen3.5+, Kimi K3, DeepSeek V4, GLM 5.x 같은 최신 high-end serving target
11. H100/H200와 B200/B300 같은 hardware-specific default/optimization 차이

## 작성 원칙

각 판단은 가능하면 다음 우선순위로 검증한다.

1. tag에 포함된 실제 source code
2. 해당 변경을 도입한 merged PR의 diff와 설명
3. release note
4. 공식 docs/recipe
5. runtime validation record

Release note와 source가 다르게 읽힐 경우 **source 조건을 우선**하고, 그 차이를 문서에 명시한다.

예를 들어 v0.28의 `max_num_batched_tokens=16384` 변경은 모든 serving GPU에 일괄 적용되는 값이 아니다. 실제 `EngineArgs.get_batch_defaults()`는 GPU memory tier와 usage context를 함께 보고, OpenAI API server에서 16384를 기본으로 받는 것은 >=160 GiB급 GPU가 핵심 대상이다. 같은 release의 CUDA Graph capture 1024 역시 data-center Blackwell 조건부다.

## 현재 읽는 순서

1. [`versions/README.md`](versions/README.md)
2. [`migrations/v0.26.0-to-v0.29.0.md`](migrations/v0.26.0-to-v0.29.0.md)
3. [`migrations/deployment-option-matrix.md`](migrations/deployment-option-matrix.md)
4. [`migrations/model-runner-v1-v2.md`](migrations/model-runner-v1-v2.md)
5. [`migrations/frontend-parser-compatibility.md`](migrations/frontend-parser-compatibility.md)
6. [`migrations/hybrid-mamba-gdn.md`](migrations/hybrid-mamba-gdn.md)
7. [`migrations/kv-transfer-mooncake.md`](migrations/kv-transfer-mooncake.md)
8. [`migrations/model-family-compatibility.md`](migrations/model-family-compatibility.md)

## 기존 상세 자료

이 audit은 기존 deep dive를 대체하지 않는다.

- KV transfer lifecycle: [`../vllm-stack/pd-disaggregation/kv-transfer-backends/vllm-kv-transfer-path.md`](../vllm-stack/pd-disaggregation/kv-transfer-backends/vllm-kv-transfer-path.md)
- Mooncake: [`../vllm-stack/pd-disaggregation/kv-transfer-backends/mooncake-transfer-engine/README.md`](../vllm-stack/pd-disaggregation/kv-transfer-backends/mooncake-transfer-engine/README.md)
- Mooncake air-gap source build: [`../vllm-stack/pd-disaggregation/kv-transfer-backends/mooncake-transfer-engine/source-build-airgap.md`](../vllm-stack/pd-disaggregation/kv-transfer-backends/mooncake-transfer-engine/source-build-airgap.md)
- NIXL integration: [`../vllm-stack/pd-disaggregation/kv-transfer-backends/nixl/vllm-integration.md`](../vllm-stack/pd-disaggregation/kv-transfer-backends/nixl/vllm-integration.md)
- 기존 0.28 migration note: [`../vllm-stack/migrations/vllm-0.28.md`](../vllm-stack/migrations/vllm-0.28.md)

`vllm/`의 결론이 실제 platform contract로 채택되면 최종 운영 상태는 `vllm-stack/` 또는 `docs/context/`에 반영한다.
