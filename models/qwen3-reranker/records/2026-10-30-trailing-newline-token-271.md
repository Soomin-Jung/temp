# 2026-10-30 — Qwen3-Reranker-4B trailing token 271 누락과 점수 왜곡

- 분류: model-specific serving correctness / prompt serialization incident
- 확인일: 2026-10-30 KST
- 대상: vLLM v0.24.0, `runner=pooling`, `Qwen3ForSequenceClassification`, Qwen3-Reranker-4B
- 상태: **해결 확인** — Jinja suffix 수정 후 정상 score 회복 (사용자 실환경 확인)
- 주의: 공개 저장소 정책에 따라 실제 내부 서비스명, 주소, 사용자 식별자는 기록하지 않음

## 요약

외부 엔지니어가 공식 모델 예제와 내부 vLLM rerank API의 score 차이를 보고했다. 로컬에서 vLLM과 동일한 모델 구성 및 Jinja 템플릿을 적용하고, 마지막 토큰을 하나 제거하는 A/B 실험을 수행했다. 내부 API 입력은 로컬 대비 마지막 token ID **271** (이 tokenizer에서 `\n\n`에 해당)이 누락되어 있었으며, **그 토큰 하나를 제거한 로컬 점수가 API 점수와 거의 일치**했다.

**운영 Jinja 마지막에 `{{ "\n\n" }}`을 추가한 후 정상 점수가 복원됨을 실환경에서 확인했다.** vLLM upstream에서 제공한 기존 `qwen3_reranker.jinja`만으로는 이 배포 환경의 모델 저자 기대 score를 재현하지 못했다.

## 공식 예제 입력과 기대 출력

Query: `What is the capital of China?`

- Doc1: `The capital of China is Beijing.`
- Doc2: `Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.`

모델 저자 예제에서 제시된 raw logits: `[6.4375, -14.375]`.

- sigmoid(6.4375) ≈ 0.9984
- sigmoid(-14.375) ≈ 0.000000571

해당 값은 모델 저자 예제의 기대값이며, **동일한 checkpoint/revision/tokenizer/dtype/입력 토큰열**일 때 비교해야 한다.

## 외부 엔지니어가 수행한 비교

| Case | Doc1 tokens | Doc2 tokens | Doc1 score | Doc2 score |
|---|---:|---:|---:|---:|
| (1) 로컬, 정상 템플릿 전체 입력 | 85 | 108 | 0.9982 | 0.0000005 |
| (2) 로컬, 마지막 토큰 강제 제거 | 84 | 107 | 0.2635 | 0.1602 |
| (3) 내부 vLLM API | 84 | 107 | 0.2653 | 0.1600 |

관찰:
1. 원래 입력의 마지막 token ID는 **271**, 의미는 `\n\n`.
2. 내부 API의 최종 입력 token IDs에는 **271이 없음** (운영 측 확인).
3. 로컬에서 동일 토큰을 제거하면 내부 API의 결과가 거의 재현됨.
4. 운영 Jinja에 `{{ "\n\n" }}`을 추가하자 정상 score로 회복됨 (운영 측 확인).

해석: scoring 수학/weight loading 자체보다는 **최종 prompt suffix 및 마지막 hidden-state 위치**가 결과 차이를 설명한다. 점수의 작은 잔차는 별도 수치 차이 가능성이 있으며 이 실험만으로 단정하지 않는다.

## 적용된 수정

원본 upstream template 마지막:

```jinja2
<|im_start|>assistant
<think>
</think>
```

검증 환경 수정:

```jinja2
<|im_start|>assistant
<think>
</think>{{ "\n\n" }}
```

목표 렌더링 suffix: `</think>\n\n` (끝에 개행 두 개). **Jinja 표현식이 출력하는 두 개행**을 의도적으로 추가하며, 파일 끝에 빈 줄을 넣는 것과 동일하다고 보지 않는다.

## 원인 분리 — 확정 vs 미확정

**확정된 사실**
- 모델 저자 기대 점수와 기존 서버 점수는 크게 달랐다.
- 실제 서버 입력에서 마지막 token 271이 누락되어 있었다.
- 로컬 마지막 토큰 제거 실험이 서버 점수를 거의 재현했다.
- Jinja 끝에 `{{ "\n\n" }}`을 추가하자 운영 score가 정상화됐다.
- upstream vLLM v0.24.0 예제 템플릿을 그대로 사용하는 것만으로는 이 환경에서 충분하지 않았다.

**미확정 사항**
- 정확히 어느 단계에서 개행이 소실되었는지: Jinja 환경의 trailing newline 처리, 템플릿 파일 자체의 suffix, tokenizer, API 후처리 중 어느 것이 최초 원인인지 아직 계측으로 확정하지 않았다.
- upstream 템플릿의 모든 버전/모델에서 동일한 문제가 발생하는지는 미검증이다.
- 마지막 토큰을 언제나 무조건 추가해야 하는 일반 규칙은 아니다. **해당 모델의 기대 입력 형식과 실제 tokenizer 결과를 기준으로 검증**해야 한다.
- `strip()` 호출이 실제 원인이라는 증거는 확보되지 않았다. Jinja의 기본 `keep_trailing_newline=False`는 유력한 조사 가설일 뿐 확정 원인으로 기록하지 않는다.

## 재현/회귀 검증 절차

1. 동일 tokenizer/model revision에서 Jinja 렌더링 직후 `repr(prompt[-40:])` 기록.
2. tokenization 직후 `input_ids[-10:]`, `len(input_ids)` 기록.
3. engine 전달 직전의 최종 `prompt_token_ids[-10:]` 및 길이 기록.
4. 위 두 예제에서 85/108 tokens, 마지막 token 271 확인.
5. 활성화된 score와 `use_activation=false` raw score를 각각 비교.
6. vLLM 또는 템플릿 업그레이드 시 같은 fixture를 regression gate로 유지.

검증 환경에서만 `VLLM_LOGGING_LEVEL=DEBUG`, `--enable-log-requests`, `--max-log-len 512`을 사용해 입력 토큰 ID를 확인할 수 있다. 로그에 실제 문서 내용이 포함될 수 있으므로 운영 전체에 상시 활성화하지 않는다.

## 구현 근거

- vLLM v0.24.0 `vllm/model_executor/models/config.py`: `Qwen3ForSequenceClassificationConfig`
- vLLM v0.24.0 `vllm/model_executor/models/adapters.py`: `load_weights_using_from_2_way_softmax`, `W_score = W_yes - W_no`
- vLLM v0.24.0 `vllm/model_executor/layers/pooler/seqwise/heads.py`: classifier linear 연산
- vLLM v0.24.0 `vllm/model_executor/layers/pooler/activations.py`: sigmoid
- vLLM v0.24.0 `vllm/entrypoints/pooling/scoring/io_processor.py`: Jinja 렌더링 및 토큰화

## 출처

- [Qwen3-Reranker-4B 공식 모델 카드](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
- [vLLM v0.24.0 Qwen3 Reranker 템플릿](https://github.com/vllm-project/vllm/blob/v0.24.0/examples/pooling/score/template/qwen3_reranker.jinja)
- [vLLM v0.24.0 adapters.py](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/models/adapters.py)

기록 근거: 외부 엔지니어가 전달한 A/B 수치 및 운영자의 2026-10-30 수정·정상화 확인. 실제 요청/로그 전문은 이 공개 문서에 보관하지 않음.
