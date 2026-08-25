# 99. Papers, Code & Glossary

작성일: 2026-08-25

## 1. Source-of-truth 우선순위

DeepSeek-V4를 조사할 때는 다음 순서로 사실을 확인한다.

```text
1. DeepSeek official technical report / announcement
2. Official model card + config + inference code
3. Framework implementation source (vLLM / Transformers / SGLang / TRT-LLM)
4. Framework recipes / engineering blogs
5. Issues / PR / benchmark reports
6. Secondary analysis
```

Architecture 숫자는 blog summary보다 checkpoint `config.json`을 우선한다.

Runtime 지원 여부는 논문보다 **현재 framework source/version**을 우선한다.

---

## 2. 필수 원문

### DeepSeek-V4 Technical Report

**DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence**

- arXiv: https://arxiv.org/abs/2606.19348

읽을 순서:

1. §2 Architecture
2. §2.1 MoE
3. §2.2 mHC
4. §2.3 Hybrid Attention
5. §2.4 Muon
6. §3 Infrastructure
7. §4 Pretraining
8. §5 Post-training

### DSpark

**DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation**

- arXiv: https://arxiv.org/abs/2607.05147

읽을 때 구분:

- parallel backbone
- Markov/RNN sequential correction
- confidence head
- verification scheduler
- production target results

---

## 3. Official model pages

### V4 announcement

- https://deepseek.com/en/news/v4-preview/

### API docs / updates

- https://api-docs.deepseek.com/
- https://api-docs.deepseek.com/zh-cn/updates/

### Flash latest

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- config: `DeepSeek-V4-Flash-0731/config.json`
- reference inference: `inference/model.py`

### Pro latest

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813
- config: `DeepSeek-V4-Pro-0813/config.json`

---

## 4. Framework code

### Hugging Face Transformers

Architecture semantics를 보기 좋다.

- docs: https://huggingface.co/docs/transformers/en/model_doc/deepseek_v4
- config: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py
- model: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py

특히 확인할 class:

```text
DeepseekV4Config
DeepseekV4HyperConnection
DeepseekV4Attention
DeepseekV4Compressor
DeepseekV4Indexer
DeepseekV4MoE
DeepseekV4Model
DeepseekV4ForCausalLM
```

### vLLM

Production-serving implementation을 본다.

Root:

- https://github.com/vllm-project/vllm/tree/main/vllm/models/deepseek_v4

핵심 파일:

```text
attention.py
compressor.py
sparse_mla.py
quant_config.py
common/ops/*
nvidia/model.py
nvidia/flashmla.py
nvidia/flashinfer_sparse.py
nvidia/fi_moe.py
nvidia/mtp.py
nvidia/dspark.py
```

### vLLM recipes

- Flash: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- Pro: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro

### vLLM architecture blog

- https://vllm-project.github.io/2026/04/24/deepseek-v4.html

특히 c4/c128 memory layout과 KV cache 계산을 이해하는 데 유용하다.

### TensorRT-LLM

- https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/deepseek_v4

backend 차이와 NVIDIA optimization 경로를 비교할 때 참고한다.

---

## 5. 계보상 같이 읽을 연구

### DeepSeek-V3

필요 개념:

- DeepSeekMoE
- auxiliary-loss-free routing
- MLA
- MTP
- FP8 training

### DeepSeek-V3.2

필요 개념:

- DeepSeek Sparse Attention
- Lightning Indexer lineage

### Hyper-Connections / mHC

V4 residual system을 이해하는 선행 연구.

### Muon

matrix optimizer / Newton-Schulz orthogonalization 계보.

---

## 6. Glossary

### SWA — Sliding Window Attention

최근 일정 window의 raw token만 exact attention하는 local path.

V4 window: 128.

### CSA — Compressed Sparse Attention

moderate compression된 long-memory에서 sparse indexer가 top-k entry를 골라 attention하는 방식.

V4 기본 compression ratio: 4.

### HCA — Heavily Compressed Attention

history를 매우 강하게 compression한 뒤 compressed memory 전체를 읽는 global coarse path.

V4 compression ratio: 128.

### DSA — DeepSeek Sparse Attention

query마다 long-memory candidate 중 중요한 일부만 선택해 main attention compute를 bounded하게 만드는 sparse-attention lineage.

V4에서는 CSA의 retrieval mechanism으로 결합된다.

### Lightning Indexer

main attention보다 가벼운 learned retrieval network.

Flash top-k 512, Pro top-k 1024.

### MQA — Multi-Query Attention

여러 query heads가 적은 수의 KV heads를 공유하는 attention.

V4는 `num_key_value_heads=1`이며 shared K=V representation을 사용한다.

### Partial RoPE

head dimension 일부에만 RoPE를 적용.

V4:

```text
head_dim=512
rope_dim=64
```

### Grouped Output Projection

wide multi-head attention output을 head group별 low-rank intermediate로 축소한 뒤 hidden_size로 projection하는 방식.

### Attention Sink

head가 특정 content token 외에 probability mass를 안정적으로 둘 수 있게 하는 learnable sink term.

### HC — Hyper-Connections

residual stream을 multiple copies로 확장하고 layer input/output 사이에서 learnable mixing을 수행하는 residual architecture.

### mHC — Manifold-Constrained Hyper-Connections

HC의 residual mixing matrix를 Sinkhorn 기반 doubly-stochastic manifold에 제약해 signal propagation 안정성을 높이는 방식.

V4 `hc_mult=4`.

### Sinkhorn-Knopp

행/열 normalization을 반복해 matrix를 doubly-stochastic form에 가깝게 만드는 iterative procedure.

### Hash-MoE

초기 MoE layer에서 token id 기반 static mapping으로 expert identity를 선택하는 bootstrap routing.

V4 첫 3 layer.

### Auxiliary-loss-free routing

expert load balance를 main training loss의 auxiliary term 대신 routing correction bias로 조정하는 DeepSeek MoE 방식.

### sqrt(softplus)

V4 learned MoE affinity activation.

```math
\sqrt{\log(1+e^x)}
```

### Clipped SwiGLU

SwiGLU의 gate/up pre-activation magnitude를 제한해 stability/low-precision robustness를 높이는 V4 expert FFN 방식.

### Muon

matrix parameter update에 orthogonalization 계열 transformation을 사용하는 optimizer family.

V4에서 대부분 matrix parameter training에 사용.

### FP4 QAT

4-bit floating-point quantization behavior를 training 과정에 반영해 low-precision inference에서도 quality를 유지하도록 하는 quantization-aware training.

### MTP — Multi-Token Prediction

현재 hidden representation에서 복수 미래 token prediction objective를 추가하는 training mechanism.

### DSpark

DeepSeek의 semi-autoregressive speculative decoding framework.

- parallel draft backbone
- cheap sequential correction
- confidence prediction
- load-aware verification scheduling

### Markov Head

DSpark에서 직전 draft token을 이용해 다음 draft position logit을 low-rank correction하는 sequential module.

### OPD — On-Policy Distillation

student 자신의 trajectory에서 teacher distribution과 KL objective를 계산하는 distillation.

V4는 multiple domain specialists를 unified model로 합칠 때 사용한다.

### Quick Instruction

기존 KV cache를 재사용하면서 special task token을 통해 auxiliary agent task를 수행하는 V4 post-training/system mechanism.

---

## 7. 자주 틀리는 표현

### 틀림

> V4는 MLA + DSA다.

### 더 정확

> V4는 shared K=V MQA 기반 local SWA와 multi-scale compressed memory를 사용하고, CSA branch에서 DSA/Lightning Indexer를 사용한다.

---

### 틀림

> Flash-0731은 Flash Preview보다 큰 architecture다.

### 정확

> 공식 설명상 target architecture/size는 동일하고 post-training과 DSpark packaging이 달라졌다.

---

### 틀림

> Pro-0813은 V4.1 architecture다.

### 정확

> 현재 공식 lineage에서는 V4-Pro Preview structure 기반의 official V4-Pro release다.

---

### 틀림

> DSpark는 MTP다.

### 정확

> MTP lineage를 활용하지만 별도의 semi-autoregressive drafter와 confidence-scheduled verification framework다.

---

### 틀림

> 1M context니까 KV cache가 1M raw tokens만큼 필요하다.

### 정확

> V4는 SWA + c4 + c128 + sparse retrieval + low-precision cache를 사용하므로 physical cache representation이 raw full-attention Transformer와 다르다.

---

## 8. 후속 연구 queue

아래는 architecture deep-dive 다음 단계다.

1. `DeepseekCompressor`의 실제 compression weight/gating 수식과 tensor shape 추적
2. FP4 indexer cache physical layout / scale granularity 계산
3. FlashMLA vs FlashInfer sparse backend dispatch 조건 정리
4. V4 mHC fused kernel call graph / Triton-CUDA implementation 분석
5. `fi_moe.py` / DeepGEMM backend와 FP4 expert execution path 비교
6. DSpark adaptive verification의 scheduler ↔ model runner call graph 추적
7. P/D disaggregation에서 CSA/HCA/indexer cache transfer object를 source 기준으로 확정
8. B300/Blackwell에서 native FP4/NVFP4 path와 H100 SM90 path 성능 비교
9. Flash/Pro 실제 memory calculator 작성
10. 1M context에서 cache bytes/token을 layer type별로 계산

이 queue는 `model architecture → engine → kernel → GPU` 순서로 내려가는 다음 deep-dive 단계다.
