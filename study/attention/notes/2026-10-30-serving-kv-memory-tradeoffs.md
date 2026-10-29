# 2026-10-30 기술교류회: Attention/Cache/Hardware 정량 계산 보충

> Scope: 모델의 일반적 성능 비교가 아니라 serving에서의 token-addressable KV cache, compressed KV, recurrent state, GPU/IB 이동 비용.

## 원칙

1. MHA/MQA/GQA의 `num_key_value_heads`는 저장할 KV head 수다. MLA나 recurrent layer에서는 별도 cache spec이 필요하다.
2. `M_token = Σ_l H_kv,l (d_k,l b_k,l + d_v,l b_v,l)`; `M_request = T × M_token`은 conventional KV의 논리 상한 모델이다.
3. MLA는 `M_token,layer ≈ r_kv b_latent + d_rope b_rope` (구현별 추가 텐서 제외).
4. Recurrent state는 `M_state,layer ≈ H_s × d_k × d_v × b_s + conv state`로 따로 합산한다.
5. Hybrid: `M_request(T) = T Σ_full/MLA bytes/token + Σ_recurrent bytes/state + auxiliary`.
6. DSA의 top-k는 **조회 토큰 수**를 줄이지만 저장된 KV/token history가 자동으로 top-k로 줄어드는 것은 아니다.
7. FlashAttention(exact softmax IO), PagedAttention(KV block allocation), Prefix caching(cross-request reuse)은 attention architecture와 다른 계층이다.

## 모형 비교: 93 layers, query heads=96, d_k=d_v=128, BF16, 170,000-token prompt

| KV family | KV heads | MiB/token (all layers) | GiB/170K tokens |
|---|---:|---:|---:|
| MHA | 96 | 0.004257 | 724.00 |
| GQA | 8 | 0.000355 | 60.33 |
| MQA | 1 | 0.0000443 | 7.54 |

*가상 통일 모델 비교이며 Kimi-K3 cache 용량이 아니다. 숫자는 정수 byte 공식으로 계산하여 재검증할 것.*

## 2026 Frontier 구분

- **Kimi K3:** 공식 config 93 layers=69 KDA + 24 gated MLA; `kv_lora_rank=512`, `mla_use_nope=true`; KDA recurrent state + 24-layer latent cache. https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json
- **GLM-5.3:** Z.ai는 GLM-5.2와 base 동일, 개선은 post-training이라고 설명한다. 78-layer MLA + DSA/IndexShare 계보와 혼동하지 말 것. https://z.ai/blog/glm-5.3
- **DeepSeek-V4-Pro:** CSA/HCA + local sliding-window branch. 3.2 DSA/MLA만으로 extrapolate하지 말 것. https://arxiv.org/abs/2606.19348
- **DeepSeek-V4.1-Flash:** 새 Causal Encoder–Decoder (20 encoder + 20 decoder), decoder global KV는 encoder final state로부터 투영, SWA bounded replay. 공식 발표 HBM KV 1/4, SSD 1/8은 V4-Flash 대비 비교치. https://www.deepseek.com/en/news/deepseek-v4-1-flash/

## 통신/용량 주의

- `B_network=400 Gb/s → theoretical 50 GB/s`; 실효 GPU-to-GPU는 protocol/NIC/GDRDMA/topology에 의해 감소.
- P/D는 KV/state cost를 없애지 않고 P write→fabric→D read 전송 단계를 추가.
- H200 SXM 141 GB HBM3e, 4.8 TB/s; H100 SXM 80 GB HBM3, 3.35 TB/s (NVIDIA HGX 공식 자료).
- 실효 concurrency는 cache capacity뿐 아니라 GEMM/MoE, NCCL collective, scheduler, CUDA Graph, preemption, SLO로 검증해야 한다.

## 후속 검증

실제 checkpoint와 runtime에서 per-layer CacheSpec, cache dtype, per-rank replication/sharding, KV/state page padding, prefix cache behavior, NCCL and NIC counters 확인.

## 기존 문서 참조

- [MHA/MQA/GQA](../attention/01-full-attention.md)
- [MLA](../attention/02-mla-low-rank-attention.md)
- [Linear/DeltaNet/KDA](../attention/03-linear-recurrent-attention.md)
- [Sparse/Compressed/Hybrid](../attention/04-local-sparse-compressed-hybrid.md)
- [Serving engineer](../attention/90-serving-engineer-view.md)
- [GPU fabric](../gpu-architecture/09-gpu-communication-mental-model.md)
