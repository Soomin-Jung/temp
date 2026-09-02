# API Routing Contract — Chat / Messages / Responses

업데이트: 2026-09-03 KST  
상태: architecture decision / implementation input

이 문서는 여러 OpenAI/Anthropic-compatible endpoint를 하나의 proxy path로 억지로 통합하지 않고, **API semantics별 책임을 분리하는 routing contract**를 정의한다.

## 1. 현재 결정

권고 data path:

```text
Client
  -> API Gateway
  -> custom vLLM Proxy
       |
       |-- /v1/completions
       |-- /v1/chat/completions
       |      -> LiteLLM
       |      -> hosted_vllm/<MODEL>
       |
       |-- /v1/messages
       |      -> LiteLLM
       |      -> anthropic/<MODEL>
       |
       |-- /v1/responses
              -> Agentic API
              -> LMStack Router
              -> selected vLLM backend
```

핵심은 **endpoint path가 책임 경계**라는 점이다.

## 2. 왜 한 backend wrapper로 통일하지 않는가

각 endpoint는 단순 URL alias가 아니다.

### Chat / Completions

- stateless request/response 중심
- 기존 LiteLLM model registry/alias를 그대로 활용하기 쉬움
- 현재 운영 path의 regression surface가 작음

### Anthropic Messages

- Anthropic message/tool/content block semantics
- 기존 LMStack Router 0.1.9에는 `/v1/messages` route가 없음
- LiteLLM Anthropic wrapper가 현재 compatibility owner

### Responses

- typed item/event semantics
- reasoning/tool lineage
- `previous_response_id`
- SSE/WebSocket
- durable state / tool orchestration 가능성

따라서 Responses를 기존 chat translator에 억지로 넣으면 fidelity loss를 찾기 어려워진다.

## 3. LMStack Router 0.1.9의 정확한 경계

Production Stack tag `vllm-stack-0.1.9`은 `/v1/responses` endpoint를 처음 추가한다.

source:
https://github.com/vllm-project/production-stack/blob/vllm-stack-0.1.9/src/vllm_router/routers/main_router.py

구현은:

```python
@main_router.post("/v1/responses")
async def route_v1_responses(request, background_tasks):
    return await route_general_request(
        request,
        "/v1/responses",
        background_tasks,
    )
```

이다.

`route_general_request`는 request body를 JSON으로 읽어 `model`을 추출해 backend를 고른 뒤, 원래 body를 선택한 backend의 같은 endpoint로 전달한다.

즉 0.1.9에서 Router의 Responses 역할은 대체로:

```text
JSON parse
  -> model extraction
  -> routing
  -> raw-ish body forwarding
```

이다.

Responses 전체 schema를 Router가 typed Pydantic model로 재구성하는 구조가 아니므로, Agentic API 뒤에 놓을 때 오히려 유리하다.

단 다음은 그대로 검증해야 한다.

- request rewriter 활성화 여부
- alias rewrite
- header filtering
- streaming response
- error/status propagation
- Responses-specific SSE event fidelity

## 4. 0.1.8 / 0.1.9 / current main 차이

### 0.1.8

- `/v1/responses` 없음
- `/v1/messages` 없음

### 0.1.9

- `/v1/responses` 추가
- `/v1/messages` 없음

### newer upstream

현재 upstream main에는 `/v1/messages` route도 존재한다.

따라서 "LMStack Router는 Messages를 지원하지 않는다"는 말을 일반화하면 안 된다.

정확한 표현:

> pinned 0.1.9에서는 `/v1/messages`가 없으므로 현재 Messages lane은 LiteLLM이 담당한다. 이후 stack/router upgrade 때 route availability와 semantic fidelity를 다시 비교한다.

## 5. Agentic API의 multi-backend 경계

Agentic API standalone server는 현재 하나의 `llm_api_base`를 application state로 가진다.

즉 다음 형태의 per-model backend map을 Agentic API 자체가 직접 소유하는 구조는 아니다.

```text
model-A -> api-base-A
model-B -> api-base-B
model-C -> api-base-C
```

대신 upstream base를 **multi-model router**로 두고 request의 `model`을 통과시키는 구조가 맞다.

Agentic API Kubernetes docs도 여러 모델을 serve하는 Router 뒤에 놓는 형태를 설명한다.

따라서:

```text
Agentic API
   -> one logical LLM_API_BASE
   -> LMStack Router
        -> model A vLLM
        -> model B vLLM
        -> model C vLLM
```

를 기본으로 한다.

이건 limitation을 workaround하는 임시 hack이 아니라 책임 분리상 자연스럽다.

- Agentic API: Responses state/tool semantics
- LMStack Router: model/backend selection
- vLLM: inference

## 6. vLLM Proxy가 가져야 할 최소 routing logic

proxy는 API semantics를 소유하지 않는다.

최소 path dispatch만 한다.

```text
/v1/chat/completions -> LiteLLM
/v1/completions      -> LiteLLM
/v1/messages         -> LiteLLM
/v1/responses        -> Agentic API
```

proxy가 하지 말아야 할 것:

- Responses item model 재정의
- tool execution
- conversation persistence
- model별 backend URL hardcode
- P/D topology 인지
- durable state owner 역할

## 7. Model identity contract

같은 physical model이라도 external logical model identity와 backend wrapper는 분리한다.

예:

```text
<MODEL>
  -> LiteLLM hosted_vllm/<MODEL>
  -> Chat/Completions

<MODEL>-anthropic
  -> LiteLLM anthropic/<MODEL>
  -> Messages

<MODEL>
  -> Agentic API request.model
  -> LMStack Router discovery
  -> vLLM /v1/responses
```

Responses lane에서 suffix alias를 강제로 추가할 필요는 없다. Agentic API가 원래 `model`을 보존하고 router가 backend discovery를 담당하게 한다.

## 8. Validation Matrix

### direct baseline

```text
Client
 -> Agentic API
 -> vLLM
```

### router path

```text
Client
 -> Agentic API
 -> LMStack Router
 -> vLLM
```

둘의 output/event sequence가 같아야 한다.

### full path

```text
Client
 -> Gateway
 -> vLLM Proxy
 -> Agentic API
 -> LMStack Router
 -> vLLM
```

검증:

- non-streaming Responses
- SSE
- WebSocket
- reasoning item
- function/custom tool
- tool call ID
- previous_response_id
- model alias
- backend failover
- cancel/disconnect
- 4xx/5xx body/status
- content-type/header

## 9. Migration 판단

이 routing contract의 목적은 모든 API를 한 제품에 몰아넣는 것이 아니다.

```text
legacy/stateless compatibility
  -> LiteLLM

stateful/agentic Responses
  -> Agentic API

model routing
  -> LMStack Router

inference
  -> vLLM
```

각 layer는 자신이 잘하는 semantics만 소유한다.

향후 LMStack Router나 Agentic API가 기능을 더 지원해도, **기능 존재만으로 layer를 제거하지 않고 fidelity·failure·operability 기준으로 판단**한다.

## 10. 공식/소스 근거

- Production Stack 0.1.9 Responses route  
  https://github.com/vllm-project/production-stack/blob/vllm-stack-0.1.9/src/vllm_router/routers/main_router.py
- Production Stack 0.1.9 request forwarding  
  https://github.com/vllm-project/production-stack/blob/vllm-stack-0.1.9/src/vllm_router/services/request_service/request.py
- Agentic API deployment  
  https://github.com/vllm-project/agentic-api/blob/main/docs/deploying/kubernetes.md
- Agentic API server config  
  https://github.com/vllm-project/agentic-api/blob/main/crates/agentic-server/src/app.rs
