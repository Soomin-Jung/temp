# Qwen3-Reranker-4B — Serving Profile & Score Fidelity

업데이트: 2026-10-30 KST

## Scope

Qwen3-Reranker-4B의 원본 생성형 yes/no scoring과 vLLM v0.24.0 pooling/classification 경로를 비교한다. 이 문서는 **검증된 배포 입력 계약**을 기록하며, 특정 시점의 실측 결과는 [2026-10-30 trailing newline incident](records/2026-10-30-trailing-newline-token-271.md)를 따른다.

## 배포 프로파일 (검증 환경)

- vLLM: **v0.24.0**
- runner: `pooling`
- max-model-len: `32768` (32K)
- tensor parallelism: `4`
- dtype: `auto`
- architecture override: `Qwen3ForSequenceClassification`
- `classifier_from_token: ["no", "yes"]`
- `is_original_qwen3_reranker: true`
- `--chat-template`: 모델별 커스텀 `chat_template.jinja`
- API: rerank / score 계열

원본 upstream 템플릿: [vLLM v0.24.0 qwen3_reranker.jinja](https://github.com/vllm-project/vllm/blob/v0.24.0/examples/pooling/score/template/qwen3_reranker.jinja). **검증 환경에서는 이 템플릿을 그대로 사용했을 때 기대 점수가 나오지 않았으며, 렌더링된 suffix의 끝 개행을 명시적으로 보존하는 수정이 필요했다.** 이 결과를 모든 tokenizer/vLLM 버전의 일반적인 결함으로 확대하지 않는다.

### 필수 검증: 템플릿 마지막 suffix

기존 템플릿 마지막 부분의 `</think>` 바로 뒤에 Jinja 표현식을 추가한다.

```jinja2
<|im_start|>assistant
<think>
</think>{{ "\\n\\n" }}
```

위 코드 블록에서 의도한 Jinja 문자열은 **`{{ "\n\n" }}`** (백슬래시 n 두 번)이다. 실제 파일에서는 렌더링 결과가 `</think>\n\n`으로 끝나야 한다. 단순히 파일 끝에 Enter를 추가하는 것과 동일하다고 가정하지 않는다.

검증 체크:
1. `repr(rendered_prompt[-40:])`로 끝 문자열 확인
2. 실제 tokenizer의 마지막 token ID가 **271**인지 확인
3. 공식 예제 두 문서의 token count가 **85, 108**인지 확인
4. 점수 및 ranking 회귀 테스트

## 점수 계산 경로

- `models/config.py`: `Qwen3ForSequenceClassificationConfig`가 `text_config.method = "from_2_way_softmax"` 및 classifier token 설정
- `models/adapters.py`: 원본 LM head의 두 token row를 가져와 `W_score = W_yes - W_no`로 변환하고 score layer에 로딩
- `pooler/seqwise/methods.py`: 마지막 유효 토큰의 hidden state 선택
- `pooler/seqwise/heads.py`: `self.classifier(pooled_data)`로 raw score 계산
- `pooler/activations.py`: 단일 label에 대해 sigmoid 적용 (activation enabled인 경우)

```text
score = sigmoid(h_last @ (W_yes - W_no))
```

두 경로가 수학적으로 동등하더라도 **입력 token IDs가 다르면 마지막 hidden state가 달라져 score가 크게 달라질 수 있다.**

## 운영 주의

- upstream 템플릿이라는 사실만으로 모델 저자 예제와 동일한 token sequence를 보장하지 않는다.
- 새 vLLM 버전 또는 tokenizer/모델 리비전 적용 시 trailing suffix의 token ID, 길이, raw/activated score를 재검증한다.
- 요청 프롬프트와 토큰 ID는 민감할 수 있으므로 상세 로깅은 격리된 테스트 환경에서만 사용한다.

## References

- [Qwen/Qwen3-Reranker-4B model card](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
- [vLLM v0.24.0 upstream Jinja](https://github.com/vllm-project/vllm/blob/v0.24.0/examples/pooling/score/template/qwen3_reranker.jinja)
- [vLLM v0.24.0 adapters.py](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/models/adapters.py)
- [Incident / runtime evidence](records/2026-10-30-trailing-newline-token-271.md)
