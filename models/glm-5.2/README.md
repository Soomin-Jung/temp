# GLM-5.2 — Architecture, Agentic Training & Systems Study Guide

작성일: 2026-08-25  
분류: `models/glm-5.2`  
기준 모델: `zai-org/GLM-5.2`  
범위: GLM-5 계보, MLA, DeepSeek Sparse Attention(DSA), IndexShare/IndexCache, MoE, MTP, long-horizon RL, EDA/RTL agent relevance, source-code reconstruction, vLLM/SGLang serving

## 1. 이 디렉터리의 목적

이 문서 세트의 목표는 GLM-5.2의 benchmark 숫자를 나열하는 것이 아니다.

**왜 GLM-5.2가 `744B-class MoE + MLA + DSA + cross-layer IndexShare + MTP + long-horizon agentic RL`이라는 구조가 되었는지, 그리고 그 구조가 실제 inference engine과 AI-agent workload에서 무엇을 의미하는지**를 논문·공식 config·Transformers/vLLM 구현까지 내려가 이해하는 것이 목표다.

GLM-5.2를 가장 짧게 표현하면 다음과 같다.

```text
GLM-5 foundation
  ├─ ~744B total / ~40B active sparse MoE
  ├─ MLA
  ├─ DSA: attention 본체는 top-k=2048 token만 읽음
  ├─ MTP
  └─ large-scale agentic RL / slime
          │
          ▼
GLM-5.1
  └─ long-horizon agent capability 강화
          │
          ├──────── IndexCache research
          │          └─ adjacent DSA layers의 index overlap을 이용
          │
          ▼
GLM-5.2
  ├─ native 1M context
  ├─ IndexShare: 여러 sparse-attention layer가 indexer 결과 공유
  ├─ MTP IndexShare + KVShare + rejection sampling + TV loss
  ├─ long-horizon coding/agent RL 강화
  └─ production long-context serving co-design
```

여기에 DeepSeek 계보도 반드시 같이 봐야 한다.

```text
DeepSeek MLA ──────────────┐
DeepSeek-V3.2 DSA ────────┼─→ GLM-5 DSA backbone
                           │
IndexCache / cross-layer ─┼─→ GLM-5.2 IndexShare
index reuse research ─────┘
```

즉 GLM-5.2는 새로운 attention 하나만 추가한 모델이 아니다. **1M-token agent trajectory를 실제 비용으로 처리하기 위해 attention indexing, speculative decoding, RL workflow, serving runtime을 함께 바꾼 모델**로 보는 편이 정확하다.

---

## 2. 현재 모델을 config 기준으로 먼저 고정한다

공식 Hugging Face `config.json`과 GLM-5 technical report를 함께 보면 현재 architecture의 핵심 숫자는 다음과 같다.

| 항목 | GLM-5.2 current config | 해석 |
|---|---:|---|
| Architecture class | `GlmMoeDsaForCausalLM` | MLA + DSA + MoE 계열 decoder-only LM |
| Hidden size | 6144 | residual stream width |
| Backbone hidden layers | 78 | config의 `num_hidden_layers` |
| Dense MLP layers | first 3 | 이후 대부분 sparse MoE |
| MoE layers | 75 | technical report의 GLM-5 table과 일치 |
| Routed experts | 256 | expert bank size |
| Experts/token | 8 | token별 routed activation |
| Shared experts | 1 | 항상 실행되는 common path |
| MoE intermediate | 2048 | expert FFN width |
| Attention heads | 64 | main MLA query heads |
| Q LoRA rank | 2048 | query low-rank bottleneck |
| KV LoRA rank | 512 | compressed KV latent |
| Q/K no-PE dim | 192 | content subspace |
| Q/K RoPE dim | 64 | positional subspace |
| Effective Q/K head dim | 256 | 192 + 64 |
| V head dim | 256 | per-head value dimension |
| DSA indexer heads | 32 | lightweight retrieval/index path |
| Index head dim | 128 | indexer projection width |
| Sparse top-k | 2048 | query가 실제 attention할 token 후보 수 |
| Index sharing frequency | 4 | cross-layer index reuse cadence |
| Max position | 1,048,576 | native 1M context |
| MTP auxiliary layers | 1 checkpoint layer | multi-step inference에서 parameter reuse |
| Vocabulary | 154,880 | tokenizer vocabulary |

> **Layer count 주의:** GLM-5 technical report는 `3 dense + 75 MoE + 1 MTP`로 표기한다. 현재 HF config의 backbone은 `num_hidden_layers=78`이며 MTP checkpoint layer는 별도로 존재한다. 따라서 코드 분석에서는 막연히 “80 transformer layers”라고 합치지 않고 backbone 78 + MTP를 구분한다.

모델 크기는 자료별 counting convention에 따라 약간 다르다.

- GLM-5 technical report: **744B total / 40B active**
- current vLLM recipe: **~743B total / ~39B active**

이 문서에서는 family scale을 설명할 때 `~744B / ~40B active`로 쓰고, engine recipe의 실제 표기는 별도로 적는다.

---

## 3. GLM-5.2를 이해하는 네 축

### 3.1 Width — sparse MoE

```text
6144 hidden
   │
   ├─ shared expert ×1 ───────────────┐
   │                                  │
   └─ router → 256 experts 중 top-8 ─┼─→ sum
                                      │
                                      ▼
                                  residual
```

전체 parameter는 매우 크지만 token 하나가 routed expert 256개를 전부 실행하지 않는다. 이 구조는 parameter capacity와 token당 compute를 분리하지만, 그 대가로 expert routing과 EP all-to-all이 시스템 병목이 된다.

### 3.2 Sequence — MLA + DSA

MLA는 **KV를 low-rank latent로 압축**해 KV-cache 문제를 줄이고, DSA는 **각 query가 모든 history token을 읽지 않고 중요한 top-k token만 선택**해 attention compute를 줄인다.

```text
long history
    │
    ├─ lightweight DSA indexer ──→ top-2048 positions
    │
    └─ MLA Q/K/V ────────────────→ selected positions만 sparse attention
```

이 둘은 같은 최적화가 아니다.

- MLA: `무엇을 cache할 것인가?`
- DSA: `cache된 history 중 무엇을 실제로 읽을 것인가?`

### 3.3 Cross-layer — IndexShare

DSA는 attention 본체를 sparse하게 만들어도 indexer가 매 layer마다 전체 history를 scan하면 1M context에서 indexer 자체가 비싸진다.

GLM-5.2는 인접 layer의 top-k 선택이 상당히 겹친다는 관찰을 이용해 indexer 결과를 여러 layer에 재사용한다.

```text
Layer F : indexer 실행 → top-k
Layer S : 재사용
Layer S : 재사용
Layer S : 재사용
          ↓
next Full indexer
```

공식 설명은 “4개 layer가 하나의 indexer를 공유”하는 구조다. 현재 config에는 초기 bootstrap layer 예외까지 포함된 실제 `full/shared` pattern이 명시되어 있다.

### 3.4 Agent runtime — MTP + long-horizon training

agent workload는 단발성 chat과 다르다.

- 긴 trajectory
- 반복 tool call
- 코드/로그/EDA output 누적
- 매우 긴 output/reasoning
- 긴 wall-clock task

GLM-5.2는 이 workload에서 decode 비용도 줄이기 위해 MTP를 강화하고, 동시에 1M-context coding-agent trajectory와 RL을 강화했다.

---

## 4. 권장 학습 순서

| 순서 | 문서 | 핵심 질문 |
|---:|---|---|
| 0 | [00-lineage-and-study-map.md](00-lineage-and-study-map.md) | GLM-5.2는 어떤 연구선이 합쳐져 만들어졌는가? |
| 1 | [01-glm5-foundation-and-scaling.md](01-glm5-foundation-and-scaling.md) | 744B-class MoE foundation과 MLA/DSA는 왜 필요했는가? |
| 2 | [02-mla-dsa-and-lightning-indexer.md](02-mla-dsa-and-lightning-indexer.md) | compressed KV와 sparse token retrieval은 실제로 어떻게 동작하는가? |
| 3 | [03-indexshare-and-1m-context.md](03-indexshare-and-1m-context.md) | DSA에서도 남는 O(L²) indexer를 IndexShare가 어떻게 줄이는가? |
| 4 | [04-moe-routing-and-expert-parallelism.md](04-moe-routing-and-expert-parallelism.md) | 256×top-8 MoE의 router와 EP cost는 무엇인가? |
| 5 | [05-mtp-and-speculative-decoding.md](05-mtp-and-speculative-decoding.md) | MTP, IndexShare, KVShare가 decode throughput과 어떻게 연결되는가? |
| 6 | [06-pretraining-posttraining-and-agentic-rl.md](06-pretraining-posttraining-and-agentic-rl.md) | GLM-5→5.2의 agent capability는 architecture 외에 어떻게 학습되는가? |
| 7 | [07-chip-design-eda-agent-relevance.md](07-chip-design-eda-agent-relevance.md) | 왜 2026 EDA/RTL agent 연구에서 GLM-5.2가 자주 backend로 등장하는가? |
| 8 | [08-source-code-architecture-reconstruction.md](08-source-code-architecture-reconstruction.md) | config와 Transformers 코드로 forward path를 어떻게 재구성하는가? |
| 9 | [09-systems-and-serving.md](09-systems-and-serving.md) | NVIDIA/vLLM/SGLang에서 1M context, MoE, DSA, MTP를 어떻게 서빙해야 하는가? |
| 90 | [90-historical-deployment-context.md](90-historical-deployment-context.md) | 기존 H200 multi-node 배포 검토 기록 |
| 99 | [99-papers-code-and-glossary.md](99-papers-code-and-glossary.md) | 원문·코드·용어 색인 |

---

## 5. 이 문서 세트를 읽고 답할 수 있어야 하는 질문

1. MLA와 DSA는 각각 KV-cache와 attention FLOPs의 어느 부분을 줄이는가?
2. DSA가 `O(L·k)` sparse attention이어도 왜 indexer는 다시 long-context bottleneck이 될 수 있는가?
3. `index_topk=2048`은 1M context에서 어떤 sparsity ratio를 만드는가?
4. GLM-5.2의 IndexShare와 IndexCache 논문의 관계는 무엇인가?
5. 왜 adjacent layer가 같은 top-k token을 재사용해도 품질을 크게 잃지 않을 수 있는가?
6. current config의 `indexer_types`가 실제 forward loop에서 어떻게 `prev_topk_indices`로 이어지는가?
7. MLA의 `q_lora_rank=2048`, `kv_lora_rank=512`, `192 noPE + 64 RoPE` 분리가 의미하는 것은 무엇인가?
8. 256 experts 중 top-8만 선택하면서 shared expert를 별도로 두는 이유는 무엇인가?
9. `sigmoid + noaux_tc + e_score_correction_bias` router는 어떤 balancing 철학을 갖는가?
10. GLM-5.2 MTP의 IndexShare/KVShare가 단순한 “draft token 수 증가”와 어떻게 다른가?
11. 1M context에서 compute가 줄어들어도 왜 KV capacity와 CPU scheduler가 병목으로 남는가?
12. 왜 GLM-5.2가 RTL/EDA agent benchmark에 자주 등장한다고 해서 곧바로 “칩 설계 업계 표준 모델”이라고 말하면 안 되는가?
13. 동일 GLM-5.2를 사용해도 agent harness가 달라지면 EDA 성능이 크게 달라지는 이유는 무엇인가?
14. vLLM/SGLang에서 DSA kernel, MoE EP, MTP, prefix/KV cache, P/D disaggregation을 어떤 순서로 검증해야 하는가?

---

## 6. Sources of truth

이 디렉터리는 다음 우선순위로 사실을 고정한다.

1. **Current model config / model card**
   - https://huggingface.co/zai-org/GLM-5.2
   - https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
2. **Z.AI official technical material**
   - GLM-5 technical report: https://arxiv.org/abs/2602.15763
   - GLM-5.2 blog: https://z.ai/blog/glm-5.2
3. **Index reuse research**
   - IndexCache: https://arxiv.org/abs/2603.12201
4. **Current implementation**
   - Transformers `glm_moe_dsa`: https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm_moe_dsa
   - vLLM: https://github.com/vllm-project/vllm
   - vLLM recipe: https://recipes.vllm.ai/zai-org/GLM-5.2
5. **EDA/RTL evidence**
   - FluxBench: https://arxiv.org/abs/2607.17528
   - NVIDIA ACE-RTL evaluation: https://developer.nvidia.com/blog/nvidia-nemotron-3-ultra-leads-open-models-on-accuracy-and-efficiency-in-agentic-rtl-coding/

벤치마크나 runtime behavior가 source마다 다르면 **모델 공식 자료 → current config/code → engine 공식 자료 → third-party benchmark** 순으로 구분해 기록한다.
