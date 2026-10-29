# Attention/KV Cache/Interconnect 발표용 검증 노트 (2026-10-30)

> **Scope**: 아키텍처 별 logical cache, precision/physical allocator 차이, HW bandwidth, P/D transport를 같은 단위로 비교하기 위한 재사용 가능한 설명. **공개 저장소 원칙**: 내부 요청량·하드웨어 배치·조직·모델 alias는 기록하지 않는다.

## 1. 먼저 세 가지 비용을 분리한다

- FLOPS: floating-point 연산 처리량. Roofline: `P <= min(P_peak, B_HBM × ArithmeticIntensity)`. FLOPS peak를 실제 tokens/s로 직접 환산하지 않는다.
- Capacity [bytes]: weights / KV 또는 recurrent state / workspace / CUDA graph / NCCL / allocator margin.
- Bandwidth [bytes/s] 및 latency [s]: HBM, NVLink/PCIe, RDMA fabric은 각각 물리적으로 다른 경로다. 400 Gb/s 포트는 **이론상 50 GB/s** line-rate이며 effective application payload bandwidth는 더 낮다.

## 2. Cache 표현별 정식화

동일 precision `b` bytes/element, 레이어 수 `L`, token 수 `T`:
- MHA / GQA / MQA raw: `M = LT H_kv (d_k b_k + d_v b_v)`. `d_k=d_v=d, b_k=b_v=b`이면 `M = 2 LT H_kv d b`. MHA는 `H_kv=H_q`, MQA는 `H_kv=1`, GQA는 중간값이다. **모두 global full-attention이라면 cache O(T), prefill pairwise O(T²)**.
- DeepSeek형 MLA main latent: `M_main ≈ LT (r_KV+d_R) b`. Q/K decoupled positional channel, cached representation, precision, backend가 같다는 가정이며 indexer/other state 제외. MLA는 global token-wise state O(T)를 유지한다.
- SWA: 최근 W만 저장하는 올바른 ring/eviction 구현이라면 per-layer O(min(T,W))의 token KV.
- DSA sparse attention: 조회는 top-k에 국한될 수 있지만 future indexer가 접근 가능한 **global cache와 indexer state를 그대로 보존**할 수 있다. `k=2048`이 곧 cache length 2048이 아니다.
- Compressed sequence: layer l의 global entries `ceil(T/m_l)`. `Σ_l ceil(T/m_l) × bytes/entry_l`에 indexer, SWA, uncompressed tail을 더한다.
- Recurrent state (Linear/Delta/GDN/KDA): token-wise full KV 대신 `S ∈ R^(d_v×d_k)` 등 **sequence-length-independent** state를 보관할 수 있지만 state가 작다는 뜻은 아니고, conv/rollback/snapshot을 따로 고려한다.
- Cross-layer reuse: 실제 KV bytes를 layer별로 더하면 과대계산 가능. shared main KV, shared indexer K, shared top-k *서로 다른 재사용 객체*를 구분한다.

### 비교 시 안전한 수식
- `M_request = M_global_KV + M_local/SWA + M_recurrent + M_indexer + M_tail + M_metadata`
- `M_HBM = weights + M_request + activation/workspace + graphs + comm + reserved`
- `T_transfer >= M_transfer / B_effective + T_setup` (copy/reformat/registration/queueing 등 추가; 식은 낙관적 bound)
- `MaxConcurrency <= floor(M_usable / M_request)` 는 **capacity-only** bound. compute/bandwidth/queueing SLO에서 실제 최댓값은 낮아질 수 있다.

## 3. 모델별 2026-10-30 공식 근거

| 모델 | 공식 구조 | Cache 관점 / 주의 |
|---|---|---|
| Kimi K3 | **93 layers = 69 KDA + 24 Gated MLA**, KDA×3 후 MLA, 마지막 MLA | KDA fixed recurrent state + MLA latent cache O(T). 전부 recurrent거나 cache가 0이라는 표현 금지. [1][2] |
| GLM-5.3 | GLM-5.2와 같은 base / `GlmMoeDsaForCausalLM`; **78 layers**, `kv_lora_rank=512`, `qk_rope_head_dim=64`, `index_topk=2048`, IndexShare | MLA main latent + DSA top-k + indexer 결과 재사용. GLM-5.3-Flash와 혼동 금지. [3][4] |
| DeepSeek-V4-Pro | **61 layers**; CSA(compression≈4, top-k=1024), HCA(compression≈128) 및 recent SWA 128 | HCA는 heavily compressed representation에 **dense** attention. DSA/CSA의 sparse selection과 구분. [5][6] |
| DeepSeek-V4.1-Flash | **40 layers: 20 causal encoder + 20 decoder**; CSA2 Full/Reindex/Reuse, cross-layer global KV reuse, FP4 KV, SWA | 공식 technical report는 global HBM KV **890 bytes/token**을 보고. *global-only*, 전체 VRAM/TPOT이 아니다. persistent SSD/host cache 약 1/8은 다른 cache tier 비교. [7][8] |

**중요한 lineage:** Mamba/selective SSM과 DeltaNet/fast-weight linear attention은 독립적인 출발점이 있다. Gated Delta Networks는 gate와 delta update를 결합하며, KDA는 gated delta memory에 channel-wise decay를 추가한 계열이다. `Mamba→GDN`을 단순 직계 제품 버전처럼 묘사하지 않는다. [9][10]

## 4. 공개 산술 예제 (실제 사용량이 아닌 가상 모델)

64-layer, 32 Q heads, head_dim=128, BF16, **T=128,000 tokens**, 개별 sequence cache.
- MHA Hkv32: `2×64×128000×32×128×2 = 67.108864 GB`.
- GQA Hkv8: `16.777216 GB`.
- MQA Hkv1: `2.097152 GB`.
- 단순 GQA→MQA 변화가 raw KV byte를 8배 줄여도 실제 decode speed, system VRAM capacity, quality를 8배 개선한다는 보장은 없다.
- fp8 KV storage를 사용하더라도 raw element 차원에서 약 절반이라는 뜻이며, scale·alignment·indexer·recurrent state·graph pool은 별도다.

## 5. 발표용 자가 검증 체크리스트

1. 논문의 peak/benchmark와 공개 config가 동일 checkpoint인지 확인.
2. Memory byte의 범위: main KV, global HBM, SWA, indexer, recurrent, SSD persistent, whole VRAM.
3. 정밀도의 범위: weights vs KV vs recurrent state, packed FP4 scale overhead.
4. GPU topology: SXM vs PCIe, NVLink support, NIC/HCA per GPU, PCIe roots.
5. TP/EP/CP/PP 및 P/D 위치에 따른 shard/replication, NCCL/UCX transport.
6. TTFT/TPOT/P99, KV preemption, GPU memory profiler, fabric bandwidth를 직접 측정.
7. 정확도 검증 없이 MHA↔GQA↔MLA 또는 KV quantization을 단순 호환으로 취급하지 않는다.

## References

1. Kimi K3 paper: https://arxiv.org/abs/2607.24653
2. Kimi K3 official: https://github.com/MoonshotAI/Kimi-K3
3. GLM-5.3 official config: https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json
4. GLM-5.3 announcement: https://z.ai/blog/glm-5.3
5. DeepSeek-V4 paper: https://arxiv.org/abs/2606.19348
6. DeepSeek-V4-Pro official config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json
7. DeepSeek-V4.1-Flash technical report: https://arxiv.org/abs/2609.19969
8. DeepSeek-V4.1 official config: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json
9. Mamba: https://arxiv.org/abs/2312.00752
10. Gated Delta Networks: https://arxiv.org/abs/2412.06464
11. Multi-Query Attention: https://arxiv.org/abs/1911.02150
12. GQA: https://arxiv.org/abs/2305.13245
13. DeepSeek-V2 MLA: https://arxiv.org/abs/2405.04434
14. Kimi Linear KDA: https://arxiv.org/abs/2510.26692
15. FlashAttention: https://arxiv.org/abs/2205.14135
16. PagedAttention: https://arxiv.org/abs/2309.06180
17. H200 specs: https://www.nvidia.com/en-us/data-center/h200/
18. L40S specs: https://www.nvidia.com/en-us/data-center/l40s/
