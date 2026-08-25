# 07. Source-Code Architecture Reconstruction

작성일: 2026-08-25

## 1. 목적

이 문서는 논문 diagram을 다시 설명하는 문서가 아니다.

목표는 실제 checkpoint config와 reference/HF/vLLM source에서:

```text
input_ids
→ embedding
→ mHC residual state
→ attention/compressor/indexer
→ MoE
→ layer stack
→ LM head
→ MTP/DSpark
```

가 어떤 class/module로 구현되는지 연결하는 것이다.

---

## 2. 기준 source

### Model semantics / reference

- DeepSeek official checkpoint `inference/model.py`
- Hugging Face Transformers `modeling_deepseek_v4.py`

### Production serving

현재 vLLM main의 V4 코드는 hardware별로 분리돼 있다.

```text
vllm/models/deepseek_v4/
├── attention.py
├── compressor.py
├── sparse_mla.py
├── quant_config.py
├── common/
├── nvidia/
│   ├── model.py
│   ├── flashmla.py
│   ├── flashinfer_sparse.py
│   ├── fi_moe.py
│   ├── mtp.py
│   └── dspark.py
├── amd/
└── xpu/
```

entry point는 platform에 따라 NVIDIA / AMD / XPU implementation을 선택한다.

---

## 3. Top-level model forward

개념 call graph:

```text
DeepseekV4ForCausalLM
  ↓
DeepseekV4Model / Transformer
  ↓
Embedding
  ↓
HyperConnection expansion
  ↓
for each DecoderLayer
    ├─ mHC attention pre/post
    ├─ DeepseekV4Attention
    ├─ mHC FFN pre/post
    └─ DeepseekV4MoE
  ↓
HyperHead collapse
  ↓
RMSNorm
  ↓
LM Head
```

최신 checkpoint에서 speculative decoding을 켜면 target model 외에:

```text
DeepSeekV4MTP
or
DSparkDeepseekV4ForCausalLM
```

이 별도 drafter path로 붙는다.

---

## 4. Layer type dispatch는 config가 결정한다

checkpoint의 legacy `compress_ratios`:

```text
0   → sliding_attention
4   → compressed_sparse_attention
128 → heavily_compressed_attention
```

HF config implementation은 이를 `layer_types`로 변환한다.

따라서 한 V4 model 안에서도 layer마다 attention implementation이 다르다.

### Flash 예

```text
L0  : 0   → SWA
L1  : 0   → SWA
L2  : 4   → CSA
L3  : 128 → HCA
L4  : 4   → CSA
L5  : 128 → HCA
...
```

---

## 5. vLLM attention object 생성

현재 vLLM `DeepseekV4Attention.__init__()`은 layer id를 읽고:

```python
self.compress_ratio = max(1, config.compress_ratios[layer_id])
```

로 현재 layer mode를 결정한다.

그리고:

```text
compress_ratio == 1
    → local/SWA only

compress_ratio == 4
    → compressor 생성
    → indexer 생성

compress_ratio == 128
    → compressor 생성
    → indexer 없음
```

으로 object graph 자체가 달라진다.

코드에서도 명시적으로:

> Only C4A uses sparse attention and hence has indexer.

라고 되어 있다.

---

## 6. Attention input projection

vLLM current NVIDIA path에서 attention forward는 가장 먼저 multiple projection을 병렬 실행한다.

핵심 tensor:

```text
qr_kv
kv_score
indexer_kv_score
indexer_weights
```

main projection:

```text
hidden_states
 → fused_wqa_wkv
 → qr + kv
 → fused Q/KV RMSNorm
```

동시에 CSA/HCA layer에서는:

```text
compressor score projection
```

CSA layer에서는 추가로:

```text
indexer weights projection
indexer compressor score projection
```

을 계산한다.

vLLM은 CUDA auxiliary streams를 사용해 이 GEMM들을 overlap하려 한다.

---

## 7. CSA forward path

현재 vLLM code 기준 CSA layer의 중요한 실행 순서:

```text
hidden_states
  │
  ├─ Q/KV projection + raw KV insert
  │
  ├─ compressor(kv_score)
  │
  └─ indexer preparation
       ↓
indexer_op → top-k compressed entries
       ↓
forward_mqa
       ↓
output projection
```

실제 구현에서는:

```python
q, (indexer_inputs, _) = execute_in_parallel(
    project_query_and_cache_kv,
    [
        indexer(...),
        compressor(...),
    ],
    ...
)
```

처럼 main Q/KV path, indexer, compressor를 가능한 한 겹쳐 실행한다.

이게 논문의 `compression + sparse retrieval`이 production engine에서 실제 overlap scheduling 문제로 바뀌는 지점이다.

---

## 8. HCA forward path

HCA에서는:

```text
compress_ratio = 128
compressor != None
indexer == None
```

따라서:

```text
Q/KV projection + raw local cache insert
          ||
HCA compressor
          ↓
all available coarse compressed entries
          ↓
attention
```

이다.

CSA와 달리 top-k index retrieval이 없다.

---

## 9. Local SWA cache는 compressed cache와 별개다

vLLM V4 attention object에는 `DeepseekV4SWACache`가 별도로 존재한다.

즉 실제 cache state는 한 종류가 아니다.

CSA layer를 예로 들면 최소한 개념적으로:

```text
A. local raw SWA cache
B. compressed attention cache
C. indexer cache
```

가 존재한다.

HCA는:

```text
A. local raw SWA cache
B. heavily compressed cache
```

가 중심이다.

따라서 V4의 `KV cache size`를 계산할 때 일반 Transformer처럼:

```text
layers × 2 × heads × head_dim × tokens
```

하나의 식으로 끝낼 수 없다.

---

## 10. KV cache dtype 분기

vLLM current source는 V4에서 DeepSeek 전용 FP8 MLA-layout cache를 지원한다.

```text
fp8_ds_mla
```

이 format은 UE8M0 block-scaled FP8을 `uint8` packed layout으로 저장한다.

plain backend에서는:

- BF16 row
- per-tensor FP8 E4M3 row

도 가능하다.

즉 `--kv-cache-dtype fp8` 하나만 보고 physical layout이 같다고 생각하면 안 된다.

backend가:

```text
DeepSeek-specific packed FP8
vs
plain FP8 row
```

중 무엇을 쓰는지 확인해야 한다.

---

## 11. Output projection

V4의 main attention output은 head_dim=512로 넓다.

vLLM에서도 output은:

```text
attention output
→ inverse RoPE handling
→ wo_a grouped low-rank projection
→ wo_b
→ hidden_size
```

경로를 탄다.

`wo_a`는 group-batched matrix 형태로 설정된다.

이게 논문의 Grouped Output Projection이 engine 코드로 내려온 형태다.

---

## 12. MoE forward reconstruction

개념:

```text
hidden
 ↓
router score = sqrt(softplus(linear(hidden)))
 ↓
selection
 ├─ first 3 layers: tid2eid hash
 └─ others: top-k
 ↓
6 routed experts
 +
1 shared expert
 ↓
weighted sum / reduce
```

routed expert 내부:

```text
w1/gate
w3/up
 ↓ clamp
SwiGLU
 ↓
w2/down
```

NVIDIA serving에서는 vLLM의 standard/fused MoE뿐 아니라 V4 전용 FlashInfer/fused path가 개입할 수 있으므로 실제 profiler에서는 어떤 backend가 선택됐는지 같이 기록한다.

---

## 13. mHC source path

각 decoder layer는 attention과 MoE를 단순:

```text
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

로 호출하지 않는다.

residual state가 HC-expanded representation이므로 각 sublayer 주위에서:

```text
pre mix
→ actual sublayer input
→ sublayer
→ post/comb residual mixing
```

을 수행한다.

vLLM에는 V4 mHC 전용 kernel warmup과 kernel test도 별도로 존재한다.

따라서 mHC는 compiler가 자동으로 없애주는 metadata가 아니라 실제 inference kernel path다.

---

## 14. CUDA Graph에서 eager break가 보이는 이유

현재 vLLM `attention.py`에는 sparse indexer / MLA attention 구간을 `eager_break_during_capture`로 분리하는 코드가 있다.

이건 V4가 CUDA graph에 부적합하다는 뜻이 아니다.

의미는:

- projection/input preparation은 captured graph에 최대한 남기고
- sparse metadata/indexer처럼 dynamic behavior가 있는 부분은 controlled eager region으로 분리

한다는 것이다.

따라서 V4에서 `FULL CUDA graph`라고 해도 내부적으로 모든 op가 하나의 monolithic graph라는 뜻은 아닐 수 있다.

runtime version별 capture strategy를 profiler trace로 확인해야 한다.

---

## 15. Source를 볼 때 naming 함정

vLLM source에 `MLA`, `sparse_mla`, `FlashMLA`라는 이름이 많이 보인다.

하지만 V4 논문의 architecture semantic은 V3의 classic MLA와 동일하지 않다.

vLLM은 기존 sparse-MLA backend/cache infrastructure를 재사용하면서 V4의:

```text
shared K=V MQA
+ compressor
+ SWA
+ sparse indexer
```

를 구현한다.

따라서 **class/file name을 architecture 이름으로 그대로 역추론하지 않는다.**

---

## 16. Debugging용 call graph

```text
DeepseekV4ForCausalLM
└─ DeepseekV4Model
   └─ DecoderLayer[i]
      ├─ mHC(attention)
      │  └─ DeepseekV4Attention
      │     ├─ fused_wqa_wkv
      │     ├─ SWA cache insert
      │     ├─ DeepseekCompressor [ratio 4/128]
      │     ├─ DeepseekV4Indexer [ratio 4 only]
      │     ├─ sparse/global attention backend
      │     └─ grouped output projection
      └─ mHC(FFN)
         └─ MoE
            ├─ hash/top-k router
            ├─ routed experts
            └─ shared expert
```

Spec decode:

```text
Target DeepseekV4ForCausalLM
        ↑ verify
DSpark drafter
 ├─ target hidden features
 ├─ parallel draft backbone
 ├─ Markov head
 └─ confidence head/runtime scheduler
```

---

## 17. 주요 source 링크

### DeepSeek / HF

- Flash inference reference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/inference
- Transformers modeling: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
- Transformers config: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py

### vLLM

- `vllm/models/deepseek_v4/attention.py`
- `vllm/models/deepseek_v4/compressor.py`
- `vllm/models/deepseek_v4/nvidia/model.py`
- `vllm/models/deepseek_v4/nvidia/flashmla.py`
- `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`
- `vllm/models/deepseek_v4/nvidia/fi_moe.py`
- `vllm/models/deepseek_v4/nvidia/mtp.py`
- `vllm/models/deepseek_v4/nvidia/dspark.py`

Repository: https://github.com/vllm-project/vllm/tree/main/vllm/models/deepseek_v4
