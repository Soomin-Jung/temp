# DeepSeek-V4 개선판: DSpark checkpoint와 vLLM 구현 차이

작성일: 2026-08-18
대상: `DeepSeek-V4-Flash` ↔ `DeepSeek-V4-Flash-0731`, `DeepSeek-V4-Pro` ↔ `DeepSeek-V4-Pro-0813`
범위: target architecture, draft module, checkpoint layout, vLLM 지원 범위

개념적인 SD 분류는 [전통 SD, MTP, DFlash, DSpark의 구조적 차이](../../../study/speculative-decoding/00-foundations-and-method-lineage.md)를 먼저 참고한다.

---

## 1. 결론

1. 각 개선판은 같은 계열의 **target architecture 규모를 유지**한다. 그래서 main model의 layer 수, routed expert 수, token당 활성 expert 수는 preview와 같다.
2. 그렇다고 weight까지 같은 모델은 아니다. `0731`과 `0813`은 개선된 post-training target weight와 새로운 encoding/reasoning 동작을 포함한 공식 release다.
3. 가장 큰 serving 구조 차이는 개선판 checkpoint에 **DSpark drafter가 함께 들어간 것**이다. preview checkpoint는 일반 MTP depth 1을 제공한다.
4. DSpark tensor가 checkpoint에서 `mtp.*` namespace 아래 저장되더라도 일반 MTP로 실행하면 안 된다. config와 tensor 종류를 보고 `method=dspark`로 선택해야 한다.
5. vLLM의 초기 DSpark 지원은 Markov fixed-K drafting까지 구현했지만, 논문의 confidence-scheduled verification은 뒤늦게 별도 기능으로 추가됐다. 따라서 `DSpark checkpoint를 로드했다`와 `논문의 전체 DSpark runtime을 사용한다`는 같은 말이 아니다.

---

## 2. 파라미터와 expert 수가 같아 보이는 이유

DeepSeek-V4 technical report가 정의한 target 본체의 canonical 규모는 다음과 같다.

| 계열 | Transformer layer | hidden size | routed expert/layer | shared expert/layer | token당 routed top-k | target 총 파라미터 | token당 활성 파라미터 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V4 Flash | 43 | 4,096 | 256 | 1 | 6 | 284B | 13B |
| V4 Pro | 61 | 7,168 | 384 | 1 | 6 | 1.6T | 49B |

`Flash-0731`은 Flash target 구조를, `Pro-0813`은 Pro target 구조를 유지한다. 모델의 능력을 높이기 위해 layer나 expert를 추가한 architecture revision이 아니라, 같은 target shape에 post-training과 agentic capability를 개선한 release이기 때문이다.

다만 파라미터 수를 읽을 때 다음 세 숫자를 섞으면 안 된다.

- **target canonical parameters**: 논문이 정의한 main model 규모인 284B/13B, 1.6T/49B
- **checkpoint에 저장된 tensor 총량**: target 외 MTP 또는 DSpark drafter, quantization scale과 보조 tensor까지 포함할 수 있음
- **Hugging Face의 Safetensors 추정치**: dtype과 tensor layout에 따라 논문 수치와 다르게 표시될 수 있음

예를 들어 Hugging Face는 `Flash-0731`을 약 304B, `Pro-0813`을 약 1.7T로 표시한다. 이것을 target MoE 자체가 284B→304B 또는 1.6T→1.7T로 커졌다는 뜻으로 해석하면 안 된다. 개선판에 동봉된 DSpark weight와 checkpoint 계수 방식이 섞인 수치다.

> `같은 파라미터/활성 expert 수`는 target architecture가 같다는 뜻이지, checkpoint 파일과 학습 결과가 동일하다는 뜻이 아니다.

---

## 3. preview와 개선판의 실질적 차이

| 항목 | Flash preview | Flash-0731 | Pro preview | Pro-0813 |
|---|---|---|---|---|
| target 규모 | 284B / 13B active | 동일 구조 | 1.6T / 49B active | 동일 구조 |
| layer / routed expert | 43 / 256 | 동일 | 61 / 384 | 동일 |
| 기본 보조 module | MTP depth 1 | DSpark | MTP depth 1 | DSpark |
| target weight | preview | 새 official release weight | preview | 새 official release weight |
| reasoning mode | preview 동작 | `low` / `high` / `max` | preview 동작 | `low` / `high` / `max` |
| chat encoding | preview V4 encoding | 전용 `encoding/`, Jinja 미제공 | preview V4 encoding | 전용 `encoding/`, Jinja 미제공 |
| 권장 SD method | `mtp` | `dspark` | `mtp` | `dspark` |

공식 model card가 공개한 agent benchmark도 architecture가 아니라 weight/post-training이 크게 바뀌었음을 보여준다.

- Flash: Terminal Bench 2.1 `61.8 → 82.7`, DeepSWE `7.3 → 54.4`
- Pro: Terminal Bench 2.1 `72.1 → 87.9`, DeepSWE `12.8 → 62.7`

이 수치는 각 model card의 평가 설정에 따른 공개 결과이며, 임의 workload의 serving 품질이나 throughput을 보장하는 수치는 아니다.

---

## 4. checkpoint에서 확인되는 DSpark 구조

### 4.1 공통 구조

개선판 config에는 일반 MTP 정보와 별도로 DSpark 관련 field가 들어간다.

| config field | Flash-0731 | Pro-0813 | 의미 |
|---|---:|---:|---|
| `num_nextn_predict_layers` | 1 | 1 | target pretraining의 MTP depth |
| `dspark_target_layer_ids` | `[40, 41, 42]` | `[58, 59, 60]` | drafter가 입력 feature를 받는 target layer |
| `dspark_block_size` | 5 | 5 | DSpark가 학습한 production draft block 크기 |
| `dspark_markov_rank` | 256 | 512 | low-rank Markov transition 차원 |
| attention compression | 4 / 128 반복 | 4 / 128 반복 | CSA/HCA 계열 layer별 compression |

weight index를 보면 preview Flash에는 주로 `mtp.0.*`가 있지만, `Flash-0731`에는 `mtp.0.*`, `mtp.1.*`, `mtp.2.*`와 다음 DSpark 전용 tensor가 나타난다.

```text
mtp.2.markov_head.markov_w1.weight
mtp.2.markov_head.markov_w2.weight
mtp.2.confidence_head.proj.weight
```

여기서 중요한 점은 **저장 namespace와 알고리즘 의미를 분리**하는 것이다.

- `mtp.*`는 checkpoint loader 호환을 위한 module namespace다.
- Markov head, confidence head, 세 target feature layer라는 조합은 일반 DeepSeek MTP-1이 아니라 DSpark drafter다.
- 따라서 vLLM에서는 개선판에 `method=mtp`를 적용하지 말고 `method=dspark`를 사용한다.

### 4.2 학습 관계도 다르다

preview의 MTP는 main pretraining과 함께 MTP loss로 학습된다. 반면 DSpark는 완성된 target을 freeze한 상태에서 target hidden feature를 입력받도록 drafter backbone, sequential head, confidence head를 학습한다.

| 구분 | MTP | DSpark |
|---|---|---|
| target과의 학습 | main training과 공동 최적화 | target 고정 후 post-hoc training |
| block 내부 dependency | 얕은 module의 AR chain | parallel backbone + Markov/RNN correction |
| checkpoint 목적 | 학습 보조 + SD 재사용 | serving용 전용 drafter |
| verification 길이 | 보통 고정 K | fixed K 또는 confidence scheduler |

---

## 5. `DSpark-Markov`의 정확한 의미

DSpark의 parallel backbone은 K개 위치의 base logits를 한 번에 만든다. Markov head는 뒤쪽 위치를 생성할 때 직전 draft token의 low-rank transition bias를 더한다.

```math
p_k(v)=\mathrm{softmax}\left(U_k(v)+W_1[x_{k-1}]W_2[:,v]\right)
```

- `U_k`: target feature와 parallel backbone이 만든 context-rich logits
- `W_1[x_{k-1}]W_2`: 직전 draft token이 다음 token에 주는 값싼 bias
- Markov rank: Flash 256, Pro 512

따라서 DSpark-Markov는 별도 serialization 형식도, 단순 bigram LM도 아니다. **DSpark의 순차 correction head가 1차 Markov 구조인 기본 variant**다. 논문에는 더 긴 block prefix를 recurrent state로 담는 RNN variant도 있지만, 비용 대비 추가 이득이 작아 Markov를 production 기본값으로 사용한다.

---

## 6. vLLM에서 실제로 구현되는 범위

### 6.1 기본 DSpark 지원

vLLM의 공식 recipe는 NVIDIA 기준 DSpark의 공개 최소 버전을 `0.25.0`으로 안내한다. ROCm 지원은 `0.26.0`부터다.

```json
{
  "method": "dspark",
  "num_speculative_tokens": 7,
  "draft_sample_method": "greedy"
}
```

하지만 config의 `dspark_block_size=5`와 recipe의 `num_speculative_tokens=7`이 다르다. 이것만으로 recipe 오류라고 단정할 수는 없다. runtime이 학습 block보다 큰 draft budget을 허용할 수 있고, 최적 K는 acceptance와 hardware load에 따라 달라진다.

운영에서는 적어도 `K=5`와 `K=7`을 같은 workload/concurrency에서 A/B test해야 한다.

### 6.2 초기 구현은 confidence head를 사용하지 않았다

vLLM `0.27.0`과 `0.27.1`의 DeepSeek-V4 DSpark loader는 confidence head가 inference에 연결되지 않았다는 주석과 함께 해당 weight를 버린다.

즉 이 버전에서 `method=dspark`가 의미하는 것은 다음과 같다.

- parallel DSpark backbone 로드
- Markov head로 K개 후보를 순차 보정
- 설정한 K를 고정 길이로 target verification
- 논문의 request별 confidence/load-aware scheduling은 미사용

논문 전체의 DSpark는 여기에 confidence head, prefix survival probability, hardware cost profile을 이용한 adaptive verification까지 포함한다.

### 6.3 adaptive verification

vLLM에는 2026-08-14 이후 `enable_adaptive_verification` 경로가 추가됐다. 기본값은 꺼져 있으며 full CUDA graph가 필요하다. 초기 구현 기준으로 eager mode, pipeline parallel, LoRA, logprobs와 함께 사용할 수 없는 제약도 있다.

long-context workload라면 기본 profiling context가 실제 배포 분포를 대표하는지도 확인해야 한다. 예를 들어 128K 중심 서비스에서 8K profile만 사용하면 verification cost curve가 달라질 수 있다.

> runtime 버전 선택은 `DSpark method 존재 여부`, `confidence scheduler 존재 여부`, `해당 hardware backend 안정성`을 각각 확인해야 한다.

---

## 7. 배포 버전 판단

| 목적 | 판단 |
|---|---|
| NVIDIA에서 DSpark 기능 자체를 인식하는 공개 최소 버전 | vLLM `0.25.0` |
| ROCm DSpark 공개 최소 버전 | vLLM `0.26.0` |
| H100×4에서 현재 확인된 안정 기준 | vLLM `0.26.0` 사용자 실측 |
| 논문의 adaptive verification까지 필요 | 해당 기능이 들어간 후속 build를 별도 검증 |
| H100에서 `0.27.0`/`0.27.1` | DeepGEMM SM90 IMA 때문에 사용 보류 |

공개 지원표의 minimum과 production에 올릴 minimum은 다르다. H100×4 DP/EP에서는 `0.26.0`이 실제로 기동됐고 `0.27.0`/`0.27.1`이 같은 옵션에서 실패했으므로, 현재 운영 baseline은 `0.26.0`으로 잡는 것이 맞다. 상세 원인은 [vLLM 0.27.x DeepGEMM SM90 CUDA IMA 분석](2026-08-18-vllm-0.27-deepgemm-sm90-cuda-ima.md)에 정리한다.

---

## 8. 운영 검증 체크리스트

- checkpoint config에서 `dspark_*` field와 Markov/confidence tensor 존재 확인
- `method=dspark` 사용, 일반 `mtp`와 혼동하지 않기
- K=5/K=7 각각 acceptance, draft latency, verify latency 측정
- reasoning `low`/`high`/`max`, code/prose/tool-call workload를 분리 측정
- target-only 대비 per-user tok/s와 aggregate tok/s를 동시에 비교
- fixed-K와 adaptive verification을 별도 결과로 기록
- full CUDA graph capture, idle 후 재요청, batch drain/refill 반복 검증
- framework version뿐 아니라 vLLM/DeepGEMM/FlashInfer의 실제 commit 고정

---

## 9. 근거 자료

- DeepSeek-AI, [DeepSeek-V4 Technical Report](https://arxiv.org/html/2606.19348v1)
- DeepSeek-AI, [DSpark paper](https://arxiv.org/html/2607.05147v1)
- DeepSeek-AI, [DeepSpec repository](https://github.com/deepseek-ai/DeepSpec)
- Hugging Face, [DeepSeek-V4-Flash-0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- Hugging Face, [DeepSeek-V4-Flash-0731 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json)
- Hugging Face, [DeepSeek-V4-Pro-0813 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)
- Hugging Face, [DeepSeek-V4-Pro-0813 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/main/config.json)
- vLLM Recipes, [DeepSeek-V4-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)
- vLLM Recipes, [DeepSeek-V4-Pro](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro)
