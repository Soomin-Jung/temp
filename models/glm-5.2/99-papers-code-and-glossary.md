# 99. Papers, Code & Glossary

작성일: 2026-08-25

## 1. Recommended reading order

### A. Current model first

1. GLM-5.2 model card  
   https://huggingface.co/zai-org/GLM-5.2
2. GLM-5.2 current `config.json`  
   https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
3. GLM-5.2 official long-horizon architecture blog  
   https://z.ai/blog/glm-5.2

현재 checkpoint가 실제로 어떤 shape인지 고정한 뒤 paper를 읽는다.

### B. Foundation architecture

4. GLM-5 Technical Report  
   https://arxiv.org/abs/2602.15763

읽을 부분:

- architecture table
- MLA
- DSA transition
- MoE scaling
- MTP parameter sharing
- training/mid-training
- slime / asynchronous RL
- agentic post-training

### C. Index reuse

5. IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse  
   https://arxiv.org/abs/2603.12201

읽을 부분:

- DSA indexer bottleneck
- adjacent-layer top-k overlap
- Full/Shared layers
- multi-layer index distillation
- prefill/decode speedup

### D. Implementation

6. Transformers GLM-MoE-DSA  
   https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm_moe_dsa
7. vLLM  
   https://github.com/vllm-project/vllm
8. vLLM GLM-5.2 recipe  
   https://recipes.vllm.ai/zai-org/GLM-5.2

### E. Agentic EDA

9. FluxBench  
   https://arxiv.org/abs/2607.17528
10. NVIDIA ACE-RTL comparison  
   https://developer.nvidia.com/blog/nvidia-nemotron-3-ultra-leads-open-models-on-accuracy-and-efficiency-in-agentic-rtl-coding/

---

## 2. Evidence hierarchy used in this directory

충돌이 있을 때 다음 순서로 판단한다.

```text
current config/checkpoint
    ↓
official current model card/blog
    ↓
foundation technical report
    ↓
current framework source
    ↓
engine official recipe/docs
    ↓
independent papers/benchmarks
    ↓
community reports
```

예를 들어 parameter count는:

- technical report: 744B / 40B active
- vLLM recipe: ~743B / ~39B active

처럼 차이가 있다.

이 경우 모델 family 설명은 `~744B-class / ~40B active`로 쓰고, deployment recipe 숫자는 원문 그대로 별도 표기한다.

---

## 3. Current config quick reference

```text
architecture              GlmMoeDsaForCausalLM
model_type                glm_moe_dsa
hidden_size               6144
num_hidden_layers         78
first_k_dense_replace     3
intermediate_size         12288
moe_intermediate_size     2048
n_routed_experts          256
num_experts_per_tok       8
n_shared_experts          1
scoring_func              sigmoid
topk_method               noaux_tc
norm_topk_prob            true
routed_scaling_factor     2.5

num_attention_heads       64
q_lora_rank               2048
kv_lora_rank              512
qk_nope_head_dim          192
qk_rope_head_dim          64
qk_head_dim               256
v_head_dim                256

index_n_heads             32
index_head_dim            128
index_topk                2048
index_topk_freq           4
index_skip_topk_offset    3
index_share_for_mtp_iteration true

max_position_embeddings  1048576
rope_theta                8000000
rope_interleave           true
num_nextn_predict_layers  1
vocab_size                154880
```

Source: https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json

---

## 4. Glossary — Architecture

### MLA — Multi-head Latent Attention

K/V 정보를 low-rank latent로 압축해 KV cache 및 projection 구조를 효율화하는 attention architecture. DeepSeek 계보에서 발전했고 GLM-5 family가 사용한다.

### Q LoRA rank

query projection을 full hidden→heads로 직접 보내기 전에 통과시키는 low-rank latent dimension. GLM-5.2는 2048.

### KV LoRA rank

K/V information을 압축하는 latent dimension. GLM-5.2는 512.

### Decoupled RoPE

content representation과 positional encoding용 subspace를 나누는 MLA 계열 방식. GLM-5.2 current head는 192 no-PE + 64 RoPE = 256 Q/K dimension으로 볼 수 있다.

### DSA — DeepSeek Sparse Attention

lightweight indexer가 query별로 relevant history position을 top-k 선택하고 main attention은 해당 token만 읽는 sparse attention architecture.

### Lightning Indexer

DSA의 lightweight retrieval/index module. GLM-5.2에서는 32 heads × 128 dim, top-k=2048.

### IndexShare

GLM-5.2에서 여러 DSA layer가 동일 top-k index를 재사용하는 architecture. 공식 설명은 4-layer sharing.

### IndexCache

cross-layer DSA index reuse를 일반화한 연구. Full layer가 index를 계산하고 Shared layer가 재사용한다.

### Full layer

자체 `GlmMoeDsaIndexer`를 가지고 top-k를 새로 계산하는 layer.

### Shared layer

자체 indexer가 없고 이전 Full layer의 `topk_indices`를 재사용하는 layer.

---

## 5. Glossary — MoE

### MoE — Mixture of Experts

FFN을 여러 expert로 확장하고 router가 token별 일부 expert만 실행하는 sparse model-width scaling 방법.

### Routed expert

router가 token 내용에 따라 선택하는 expert. GLM-5.2는 256개 중 8개 선택.

### Shared expert

모든 token에 항상 실행되는 expert/common FFN path. GLM-5.2는 1개.

### Expert Parallelism — EP

expert를 여러 rank/GPU에 분산 배치하고 token을 selected expert owner로 all-to-all dispatch하는 parallelism.

### `noaux_tc`

auxiliary balancing loss에 의존하지 않는 DeepSeek-style top-k routing 계보의 config method. current GLM-5.2는 `e_score_correction_bias`를 expert selection correction에 사용한다.

### Routed scaling factor

선택된 expert mixture weight에 적용되는 scaling. GLM-5.2 config는 2.5.

---

## 6. Glossary — Speculative decoding

### MTP — Multi-Token Prediction

추가 predictive module을 이용해 미래 여러 token candidate를 draft하고 backbone이 병렬 verification하는 speculative decoding lineage.

### Draft length

한 speculative cycle에서 시도하는 candidate token 수.

### Acceptance length

실제 verification을 통과해 commit된 consecutive token 수. throughput과 더 직접적으로 연결된다.

### KVShare

GLM-5.2 MTP iteration 사이에서 compatible KV state를 공유해 draft cost와 train/inference mismatch를 줄이는 mechanism.

### TV loss — Total Variation loss

draft distribution과 target/backbone distribution의 차이를 줄여 acceptance를 높이는 end-to-end training objective.

---

## 7. Glossary — Training / Agents

### slime

Z.AI의 large-scale post-training/rollout infrastructure. asynchronous RL, customizable agent rollout, production-style inference integration 등을 다룬다.

### OPD — On-Policy Distillation

현재 student policy가 실제 방문하는 state/trajectory에 teacher signal을 적용하는 distillation. policy distribution mismatch를 줄이는 데 유리하다.

### Long-horizon task

짧은 QA가 아니라 수많은 reasoning/tool step과 긴 elapsed time/context를 요구하는 task. coding project, debugging, research, EDA flow 등이 대표적이다.

### Reward hacking

agent가 실제 목표를 달성하지 않고 evaluator/tool/reward의 허점을 이용해 높은 score를 얻는 현상.

---

## 8. Glossary — EDA

### RTL — Register Transfer Level

Verilog/SystemVerilog 등으로 register와 combinational logic 수준의 digital hardware behavior를 기술하는 abstraction.

### EDA — Electronic Design Automation

RTL simulation, synthesis, timing analysis, physical design, verification 등을 자동화하는 tool/algorithm 분야.

### RTL-to-GDS

RTL에서 synthesis, floorplan, placement, CTS, routing, signoff 등을 거쳐 manufacturable layout database(GDS)까지 가는 flow.

### ECO — Engineering Change Order

late-stage timing/functional/PPA 문제를 해결하기 위해 netlist/logic/physical implementation을 수정하는 작업.

### WNS / TNS

- WNS: Worst Negative Slack
- TNS: Total Negative Slack

static timing analysis에서 timing closure 상태를 나타내는 대표 metric.

### PPA

Power, Performance, Area. hardware implementation quality를 보는 핵심 축.

### Token ROI

FluxBench가 제안한 agent efficiency 관점. EDA artifact 개선을 token/runtime cost와 함께 평가하려는 metric.

---

## 9. Kimi study와 cross-reference

GLM-5.2와 Kimi K3는 같은 문제를 다른 방식으로 푼다.

관련 Kimi 문서:

- [Kimi K3 Study Guide](../kimi-k3/README.md)
- [Kimi Linear Bridge](../kimi-k3/02-kimi-linear-bridge.md)
- [Kimi MoE / LatentMoE](../kimi-k3/03-moe-latentmoe-stable-latentmoe.md)
- [Kimi Systems & Serving](../kimi-k3/09-k3-systems-and-serving.md)

특히 long-context comparison에서:

```text
GLM-5.2: retain history + sparse content retrieval
Kimi K3: recurrent state majority + periodic global attention
```

이라는 architecture contrast를 기억한다.

---

## 10. Future research queue

현재 문서 세트 이후 추가로 파볼 가치가 큰 순서:

1. vLLM current `DeepseekV32/GLM-MoE-DSA` CUDA kernel call graph
2. sparse MLA / FlashMLA index format과 KV physical layout
3. vLLM MTP implementation에서 GLM-5.2 extra layer loading path
4. CUDA Graph + DSA top-k shape handling
5. DeepGEMM vs CUTLASS/Marlin/NVFP4 expert kernels
6. TP/EP/DEP topology별 communication model
7. P/D disaggregation 시 MLA + indexer cache transfer contract
8. 1M context memory formula를 실제 vLLM block allocator 기준으로 계산
9. GLM-5.2 vs Kimi vs DeepSeek를 동일 EDA-agent harness에서 비교하는 validation suite
10. FluxEDA/ACE-RTL/RTLScout 등 chip-agent framework source-level comparison

이 queue는 architecture study에서 실제 platform/agent design으로 넘어가는 다음 단계다.
