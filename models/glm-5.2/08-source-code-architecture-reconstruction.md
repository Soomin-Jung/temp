# 08. Source-Code Architecture Reconstruction

작성일: 2026-08-25  
목적: 논문의 box diagram이 아니라 **current GLM-5.2 `config.json`과 Transformers source에서 실제 forward path를 복원**한다.

## 1. Source of truth

### Model config

- https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json

### Transformers implementation

- configuration: `src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py`
- modeling: `src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py`
- repository: https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm_moe_dsa

Architecture identifier:

```json
{
  "architectures": ["GlmMoeDsaForCausalLM"],
  "model_type": "glm_moe_dsa"
}
```

---

## 2. Top-level module tree

current implementation을 class hierarchy로 정리하면:

```text
GlmMoeDsaForCausalLM
├─ model: GlmMoeDsaModel
│  ├─ embed_tokens
│  ├─ rotary_emb
│  ├─ layers[0..77]: GlmMoeDsaDecoderLayer
│  │  ├─ input_layernorm
│  │  ├─ self_attn: GlmMoeDsaAttention
│  │  │  ├─ Q low-rank projections
│  │  │  ├─ KV low-rank projections
│  │  │  └─ indexer: GlmMoeDsaIndexer | None(shared layer)
│  │  ├─ post_attention_layernorm
│  │  └─ mlp
│  │     ├─ layer 0..2: GlmMoeDsaMLP
│  │     └─ layer 3..77: GlmMoeDsaMoE
│  │        ├─ gate: GlmMoeDsaTopkRouter
│  │        ├─ experts: GlmMoeDsaExperts
│  │        └─ shared_experts: GlmMoeDsaMLP
│  └─ norm
└─ lm_head
```

이 구조만 봐도 GLM-5.2 backbone의 본체는:

```text
DSA-enabled MLA attention
+ mostly MoE FFN
```

이다.

---

## 3. Full model forward

`GlmMoeDsaModel.forward()`의 핵심 흐름:

```python
inputs_embeds = embed_tokens(input_ids)
position_embeddings = rotary_emb(inputs_embeds, position_ids)

topk_indices = None

for i, decoder_layer in enumerate(layers):
    hidden_states, topk_indices = decoder_layer(
        hidden_states,
        attention_mask=...,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        past_key_values=past_key_values,
        prev_topk_indices=topk_indices,
    )

hidden_states = norm(hidden_states)
```

여기서 GLM-5.2-specific state는 `topk_indices`다.

standard Transformer loop에는 없는 값이 layer 사이를 전달된다.

```text
Layer n Full indexer
   │ returns top-k positions
   ▼
Layer n+1 Shared
   │ same top-k
   ▼
Layer n+2 Shared
   │ same top-k
   ▼
Layer n+3 Shared
```

**IndexShare가 model-level stateflow로 구현되어 있음을 직접 확인할 수 있다.**

---

## 4. Decoder layer forward

`GlmMoeDsaDecoderLayer`는 PreNorm residual 구조다.

```text
residual = h
x = RMSNorm(h)
x, topk = SelfAttention(x, prev_topk)
h = residual + x

residual = h
x = RMSNorm(h)
x = DenseMLP or MoE(x)
h = residual + x

return h, topk
```

즉 attention/MoE 자체는 바뀌었지만 residual skeleton은 비교적 전통적인 decoder block이다.

Kimi K3처럼 depth-axis residual mechanism을 별도로 재설계한 모델과 대비되는 지점이다.

---

## 5. Main attention Q path

config:

```text
hidden_size        = 6144
q_lora_rank        = 2048
num_heads          = 64
qk_nope_head_dim   = 192
qk_rope_head_dim   = 64
qk_head_dim        = 256
```

source flow:

```python
q_resid = q_a_layernorm(q_a_proj(hidden_states))
q_states = q_b_proj(q_resid)
q_states = view(B, S, 64, 256).transpose(...)
q_pass, q_rot = split(q_states, [192, 64])
q_rot = apply_rope(q_rot)
query_states = concat(q_pass, q_rot)
```

shape reconstruction:

```text
[B,S,6144]
  → [B,S,2048]
  → [B,S,64×256]
  → [B,64,S,192] + [B,64,S,64]
  → [B,64,S,256]
```

---

## 6. Main attention KV path

config:

```text
kv_lora_rank      = 512
qk_rope_head_dim  = 64
qk_nope_head_dim  = 192
v_head_dim        = 256
```

source flow:

```python
compressed_kv = kv_a_proj_with_mqa(hidden_states)
kv_pass, k_rot = split(compressed_kv, [512, 64])

k_pass = kv_a_layernorm(kv_pass)
k_rot = apply_rope(k_rot)

key_states, value_states = expand_kv(k_pass, k_rot)
```

`expand_kv()`:

```text
512 latent
   │ kv_b_proj
   ▼
64 × (192 K_noPE + 256 V)
```

그리고 shared 64-dimensional RoPE key를 각 head에 broadcast해:

```text
K = [192 noPE ; 64 RoPE] = 256/head
V = 256/head
```

로 만든다.

---

## 7. Indexer construction

각 attention layer init 시:

```python
self.skip_topk = config.indexer_types[layer_idx] == "shared"
self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(...)
```

따라서 `shared` layer에는 indexer parameter/module 자체가 없다.

이 점은 단순 runtime branch보다 강하다.

```text
Full layer   → owns indexer module
Shared layer → no indexer module, requires previous index
```

current config의 initial pattern:

```text
0 F
1 F
2 F
3 S
4 S
5 S
6 F
7 S
8 S
9 S
10 F
...
```

---

## 8. Indexer Q/K source path

Indexer config:

```text
index_n_heads  = 32
index_head_dim = 128
index_topk     = 2048
q_lora_rank    = 2048
```

### Query

main attention에서 이미 만든 `q_resid`를 재사용한다.

```python
q = wq_b(q_resid)
q = view(B,S,32,128)
```

즉:

```text
hidden 6144
 → main q_a_proj 2048
 → index wq_b 32×128
```

### Key

```python
k = k_norm(wk(hidden_states))
```

즉 hidden 6144에서 128-dimensional retrieval key를 별도로 만든다.

### RoPE

indexer Q/K도 positional subspace를 split하고 interleaved RoPE를 적용한다.

---

## 9. Index score exact dataflow

source를 거의 그대로 수식화하면:

### Per-index-head similarity

```math
S_{h,q,k}=\mathrm{ReLU}\left(\frac{Q_{h,q}K_k^T}{\sqrt{128}}\right)
```

### Query-dependent head weights

```math
w_q = W_wh_q / \sqrt{32}
```

### Aggregate retrieval score

```math
I_{q,k}=\sum_{h=1}^{32}w_{q,h}S_{h,q,k}
```

### Causal mask + top-k

```math
P_q=\operatorname{TopK}_{2048}(I_q)
```

output dtype는 int32 position index다.

---

## 10. Full vs Shared forward branch

attention forward:

```python
if self.indexer is not None:
    topk_indices = self.indexer(...)
else:
    if prev_topk_indices is None:
        raise ValueError(...)
    topk_indices = prev_topk_indices
```

따라서 shared layer는 previous Full layer가 없으면 실행 자체가 불가능하다.

이것이 current config가 초기 layer를 bootstrap Full로 두는 이유를 이해하는 데 중요하다.

---

## 11. Sparse attention backend branch

generic implementation은 attention backend에 따라 top-k를 다르게 사용한다.

### eager / SDPA

```text
top-k positions
  ↓
dense boolean mask 생성
  ↓
selected = allowed
other    = -inf
```

correctness는 맞지만 long-context sparse efficiency를 충분히 얻기 어려울 수 있다.

### sparse-aware backend

```python
indices=sparse_indices
```

를 kernel에 직접 넘긴다.

production path에서는 Flash-MLA/SFA 계열 sparse kernel이 position index를 직접 소비해야 메모리/compute 이점이 살아난다.

---

## 12. Cache path

main K/V:

```python
past_key_values.update(key_states, value_states, layer_idx)
```

indexer key:

```python
past_key_values.update_indexer(k, layer_idx)
```

즉 cache abstraction에도 DSA-specific indexer state가 추가된다.

현재 generic Transformers code에는 sparse model이 expanded K/V를 cache하는 부분에 TODO가 존재한다. 이 때문에 **HF reference memory layout을 production MLA engine memory layout으로 간주하면 안 된다.**

---

## 13. MoE router source reconstruction

source:

```python
router_logits = linear(hidden.float(), weight.float())
scores = sigmoid(router_logits)
scores_for_choice = scores + e_score_correction_bias
...
topk_indices = topk(scores_for_choice, k=8)
topk_weights = scores.gather(topk_indices)
topk_weights /= sum(topk_weights)
topk_weights *= 2.5
```

current config는 `n_group=1`이므로 group selection logic은 사실상 전체 expert bank를 그대로 통과한다.

최종 flow:

```text
h
 → fp32 router logits
 → sigmoid
 → correction bias for selection
 → top-8 expert IDs
 → original sigmoid weights gather
 → normalize
 → ×2.5
 → routed expert GEMM
 → + shared expert
```

---

## 14. MTP checkpoint와 backbone 경계

current config:

```text
num_hidden_layers        = 78
num_nextn_predict_layers = 1
```

Transformers pretrained model class에는:

```text
model.layers.78.*
```

unexpected key를 ignore하도록 하는 rule이 존재한다.

이것은 base causal-LM backbone `layers[0..77]`와 별도 MTP checkpoint layer를 구분해야 한다는 신호다.

production speculative engine은 이 extra MTP state를 별도 draft/verification flow에서 사용한다.

따라서 source reconstruction에서는:

```text
Backbone forward
≠
Full production MTP speculative forward
```

를 명시적으로 분리한다.

---

## 15. Full token forward reconstruction

하나의 token/query가 layer를 통과하는 전체 구조:

```text
                         hidden [6144]
                              │
                         RMSNorm
                              │
             ┌────────────────┴────────────────┐
             │                                 │
        Main MLA path                    DSA Index path
     Q rank 2048                         q_resid reuse
     KV rank 512                         32×128 query
     RoPE 64                             hidden→128 key
             │                                 │
             │                        global score + top2048
             │                                 │
             └───────────────┬─────────────────┘
                             ▼
                     Sparse Attention
                             │
                           o_proj
                             │
                        + residual
                             │
                         RMSNorm
                             │
             ┌───────────────┴─────────────────┐
             │                                 │
      Routed MoE top-8                  Shared Expert
       from 256 experts                    always on
             │                                 │
             └───────────────┬─────────────────┘
                             ▼
                          + residual
                             │
                         next layer
```

Shared-index layer에서는 오른쪽 DSA Index path를 실행하지 않고 이전 layer의 top-2048 position을 받는다.

---

## 16. vLLM source를 볼 때 찾아야 할 mapping

current vLLM은 DSA family 구현을 DeepSeek-V3.2 계보와 공유/재사용하는 부분이 있다.

소스 분석 시 이름만 `glm`으로 grep해서 끝내지 말고:

- `glm_moe_dsa`
- `GlmMoeDsaForCausalLM`
- `DeepseekV32`
- DSA / sparse MLA
- index cache/share
- MTP

를 같이 추적해야 한다.

모델 architecture name과 inference-engine internal implementation class가 반드시 1:1 이름으로 대응하지 않는다.

---

## 17. Debugging checklist

custom kernel/engine porting 시 shape부터 확인한다.

```text
Q latent       2048
KV latent      512
Q/K noPE       192
Q/K RoPE        64
Q/K total      256
V              256
main heads      64
index heads     32
index dim      128
top-k         2048
hidden        6144
```

그리고 semantic layout:

- main RoPE interleaved
- indexer RoPE interleaved
- full/shared index layer pattern
- indexer cache handling
- router fp32
- top-k normalization/scaling

까지 맞춰야 한다.

---

## 18. 한 문장 정리

> **GLM-5.2 source에서 가장 특이한 state는 hidden/KV가 아니라 layer 사이를 흐르는 `topk_indices`다. 이 int32 position tensor가 MLA를 DSA로 만들고, 다시 여러 layer에서 재사용되면서 GLM-5.2의 IndexShare architecture가 된다.**

## Sources

- GLM-5.2 config: https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json
- Transformers GLM-MoE-DSA modeling: https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
- Transformers GLM-MoE-DSA config: https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py
- GLM-5 Technical Report: https://arxiv.org/abs/2602.15763
- IndexCache: https://arxiv.org/abs/2603.12201
