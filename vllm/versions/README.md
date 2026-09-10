# vLLM Version Audit Ledger

업데이트: 2026-09-10 KST

이 디렉터리는 upstream vLLM release를 **버전별 immutable review**로 기록한다.

## 범위

| Version | Release date | Platform significance | Report |
|---|---:|---|---|
| v0.24.0 | 2026-06-29 | Streaming Parser Engine 도입기, KV/P-D 확장 | [`v0.24.0.md`](v0.24.0.md) |
| v0.25.0 | 2026-07-11 | MRV2 dense default, PagedAttention 제거, parser unification | [`v0.25.0.md`](v0.25.0.md) |
| v0.25.1 | 2026-07-14 | startup 및 fused all-reduce correctness hotfix | [`v0.25.1.md`](v0.25.1.md) |
| v0.26.0 | 2026-07-27 | 현재 migration baseline, DCP/hybrid 및 NIXL PP push | [`v0.26.0.md`](v0.26.0.md) |
| v0.27.0 | 2026-08-10 | Torch 2.13, Kimi K3/Qwen3.5, hybrid P/D 확장 | [`v0.27.0.md`](v0.27.0.md) |
| v0.27.1 | 2026-08-11 | quantized DSpark Markov head hotfix | [`v0.27.1.md`](v0.27.1.md) |
| v0.28.0 | 2026-08-26 | MRV2 E/P/D, Mamba prefix cache, conditional HW defaults | [`v0.28.0.md`](v0.28.0.md) |
| v0.29.0 | 2026-09-09 | MRV2 all-model default, parser/connector/cache policy maturation | [`v0.29.0.md`](v0.29.0.md) |

## 공통 impact 등급

- **P0 / migration blocker**: startup 실패, silent corruption, protocol incompatibility, unsupported execution path 가능
- **P1 / mandatory retest**: 결과 correctness 또는 serving SLO에 직접 영향
- **P2 / performance/operation**: 성능, memory, observability, deployment reproducibility에 영향
- **P3 / optional**: 현재 platform target과 직접 관련이 낮음

## Release note를 그대로 믿지 않는 이유

vLLM은 release note가 요약 단위이고 실제 default가 hardware, usage context, architecture, frontend 또는 Model Runner 조건에 따라 갈리는 경우가 많다.

대표 사례:

- v0.25 `MRV2 default for all dense models` → 실제 PR #44443은 hybrid/attention-free를 제외했다.
- v0.28 `max_num_batched_tokens 8192 -> 16384` → PR #51726은 >=160 GiB에서는 online/offline 모두 16384, H100/H200 tier에서는 offline 16384 / OpenAI API server 8192로 분기한다.
- v0.28 `CUDA Graph capture default 1024` → PR #49390은 data-center Blackwell에만 1024 cap을 적용한다.
- v0.29 `MRV2 default for all models` → PR #53183은 CUDA/default path를 뒤집지만 ROCm의 DeepSeek V3.2/V4 등은 V1 default 예외로 남긴다.

따라서 각 report는 release note의 표현과 실제 source semantics를 분리해 기록한다.

## Cross-version 분석

버전별 사실을 실제 migration decision으로 합친 문서는 [`../migrations/`](../migrations/README.md)에 둔다.
