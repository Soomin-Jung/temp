# 01. DeepSeek-V4 Family & Latest Checkpoints

작성일: 2026-08-25

## 1. 현재 기준점

2026-08-25 기준으로 architecture deep-dive에서 기준으로 삼을 checkpoint는 다음 두 개다.

| 항목 | Flash | Pro |
|---|---:|---:|
| 최신 공개 checkpoint | `DeepSeek-V4-Flash-0731` | `DeepSeek-V4-Pro-0813` |
| Backbone lineage | Flash Preview와 동일 | Pro Preview와 동일 |
| Total params | 약 284B | 약 1.6T |
| Active params | 약 13B | 약 49B |
| Hidden size | 4096 | 7168 |
| Layers | 43 | 61 |
| Attention heads | 64 | 128 |
| KV heads | 1 | 1 |
| Routed experts | 256 | 384 |
| Active routed experts/token | 6 | 6 |
| Shared experts | 1 | 1 |
| Context | 1,048,576 | 1,048,576 |
| Index top-k | 512 | 1024 |
| HC copies | 4 | 4 |
| DSpark | 최신 checkpoint 포함 | 최신 checkpoint 포함 |

> Pro의 1.6T / 49B active와 Flash의 284B / 13B active는 V4 technical report의 공개 구조 기준이다. 공식 최신 release는 해당 Preview backbone 구조를 유지한다고 설명한다.

---

## 2. Flash-0731에서 실제로 바뀐 것

DeepSeek API changelog는 `DeepSeek-V4-Flash-0731`에 대해 다음을 명확히 한다.

- V4-Flash Preview와 **model structure / size가 동일**
- 새로 post-training 수행
- agent capability 대폭 강화
- Responses API / Codex 계열 agent integration 강화
- 공개 checkpoint에는 DSpark speculative decoding module 포함

따라서:

```text
Flash Preview backbone
        │
        ├── architecture: 그대로
        │
        ├── post-training: 재수행 / 강화
        │
        └── decoding package: DSpark 포함
        ↓
Flash-0731
```

Architecture deep-dive에서 `0731 architecture`를 별도 세대로 만들 필요는 없다.

---

## 3. Pro-0813에서 실제로 바뀐 것

공식 model card는 `DeepSeek-V4-Pro-0813`을:

- official V4-Pro release
- V4-Pro Preview model structure 기반
- enhanced agentic capabilities
- production 성능 강화
- DSpark speculative decoding module attached

로 설명한다.

따라서 Pro 역시:

```text
Pro Preview backbone
       +
latest post-training
       +
DSpark
       ↓
Pro-0813
```

으로 이해한다.

---

## 4. Flash config를 실제로 읽으면

`DeepSeek-V4-Flash-0731/config.json`의 중요한 값:

```text
hidden_size            = 4096
num_hidden_layers      = 43
num_attention_heads    = 64
num_key_value_heads    = 1
head_dim               = 512
q_lora_rank            = 1024
qk_rope_head_dim       = 64
n_routed_experts       = 256
num_experts_per_tok    = 6
n_shared_experts       = 1
moe_intermediate_size  = 2048
index_n_heads          = 64
index_head_dim         = 128
index_topk             = 512
sliding_window         = 128
hc_mult                = 4
hc_sinkhorn_iters      = 20
max_position_embeddings= 1048576
```

attention schedule:

```text
[0, 0,
 4, 128, 4, 128, ...,
 4,
 0, 0, 0]
```

의미:

- `0` → sliding attention only
- `4` → CSA
- `128` → HCA

초반/후반 일부 local-only layer와 가운데 CSA/HCA interleave가 존재한다.

---

## 5. Pro config를 실제로 읽으면

`DeepSeek-V4-Pro-0813/config.json`에서 확인되는 주요 값:

```text
hidden_size           = 7168
num_hidden_layers     = 61
num_attention_heads   = 128
num_key_value_heads   = 1
head_dim              = 512
q_lora_rank           = 1536
qk_rope_head_dim      = 64
n_routed_experts      = 384
num_experts_per_tok   = 6
n_shared_experts      = 1
moe_intermediate_size = 3072
index_n_heads         = 64
index_head_dim        = 128
index_topk            = 1024
hc_mult               = 4
```

Flash와 Pro의 중요한 scaling 차이는 단순 total parameter가 아니다.

```text
Residual width      4096 → 7168
Depth               43   → 61
Attention heads     64   → 128
Expert bank         256  → 384
Expert width        2048 → 3072
CSA retrieval k     512  → 1024
```

즉 Pro는 **depth / residual width / attention capacity / expert capacity / retrieved long-memory capacity를 동시에 scale-up**한다.

---

## 6. 왜 active parameter만 보고 Flash/Pro를 비교하면 안 되는가

MoE 모델에서 active parameter는 token당 compute를 이해하는 데 유용하지만 다음 비용은 따로 존재한다.

- attention Q projection
- compressed memory generation
- Lightning Indexer
- SWA local branch
- mHC mixing
- routing
- expert communication
- speculative drafter
- cache movement

따라서 `13B vs 49B active`만으로 3.8배 inference cost라고 계산하면 부정확하다.

실제 serving에서는:

```text
compute cost
+ HBM capacity
+ attention/cache bandwidth
+ EP communication
+ TP collective
+ DSpark draft/verify ratio
```

가 함께 throughput을 결정한다.

---

## 7. API 모델명과 checkpoint명을 분리한다

API:

```text
deepseek-v4-flash
deepseek-v4-pro
```

checkpoint:

```text
DeepSeek-V4-Flash-0731
DeepSeek-V4-Pro-0813
```

API alias는 이후 checkpoint가 교체될 수 있으므로 운영 문서에서는 항상:

```text
API alias
resolved checkpoint/version
runtime image/vLLM version
```

을 같이 기록해야 재현성이 생긴다.

---

## 8. 출처

- DeepSeek official V4 announcement: https://deepseek.com/en/news/v4-preview/
- API changelog: https://api-docs.deepseek.com/zh-cn/updates/
- Flash-0731 model card/config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- Pro-0813 model card/config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813
