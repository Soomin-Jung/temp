# Stateful Conversation Architecture for OpenAI Responses / Anthropic-compatible APIs

업데이트: 2026-08-24 KST  
상태: Architecture requirement / design context

## 문제 정의

멀티 replica LLM serving 환경에서 `/v1/responses`의 `previous_response_id` 같은 상태 연속성을 inference process-local memory에 맡기면 replica round-robin, process restart, rollout, scale-in 시 대화 상태가 사라진다.

따라서 stateful conversation은 단순한 "chat history 저장" 문제가 아니라 **serving plane 상위 계층에서 durable conversation execution state를 관리하는 문제**로 본다.

## 현재 serving 관점

```text
Client
  -> API Gateway / compatibility proxy
  -> Router / model gateway
  -> vLLM replicas
```

vLLM replica의 process-local state는 durable source of truth로 간주하지 않는다. Router affinity만으로는 장애·재시작 후 복구를 보장할 수 없으며, affinity는 최적화일 수는 있어도 persistence 자체가 아니다.

## 보존해야 하는 state

텍스트 message만 저장하면 충분하지 않다. 최소한 다음 계층을 보존할 수 있어야 한다.

- conversation / response identity
- structured input/output items
- assistant message / reasoning item
- tool call ID, tool name, arguments
- tool result / status / execution ordering
- parent response / branching relationship
- model / sampling / tool schema 등 request metadata
- multimodal item reference 또는 재현 가능한 payload reference
- retry / idempotency / partial execution에 필요한 상태
- streaming 중 완료된 item과 terminal state

즉 storage schema는 특정 provider의 `content: string`만 가정하면 안 된다.

## 핵심 architectural contract

### 1. Durable Conversation Store

Conversation/Response state는 replica와 독립적인 shared store에 저장한다.

저장 계층 후보는 제품 선택보다 계약이 중요하다.

- durable DB/object store
- explicit TTL / retention
- response ID / parent ID index
- idempotent update
- partial/failed execution state
- schema versioning

### 2. Conversation State Facade

Client-facing OpenAI Responses / Anthropic Messages semantics와 backend inference runtime을 분리한다.

Facade의 책임:

1. request validation
2. previous response / session state resolve
3. backend 요청에 필요한 context reconstruction
4. router/model selection
5. output item normalization
6. durable commit
7. retry/idempotency 판단

### 3. Router Affinity

가능하면 같은 continuation을 동일 replica로 보내는 affinity는 성능 최적화에 활용할 수 있다.

하지만 다음 이유로 correctness layer가 되어서는 안 된다.

- replica restart
- rolling deployment
- scale down
- model reassignment
- router state loss

따라서 `affinity != durability`를 기본 원칙으로 한다.

## OpenAI / Anthropic compatibility

OpenAI `/v1/responses`와 Anthropic `/v1/messages`는 payload shape와 state semantics가 다르므로 translation layer에서 identity를 잃지 않아야 한다.

특히 다음을 검증한다.

- reasoning/content/tool-call 분리
- tool call ID와 result linkage
- streaming delta ordering
- terminal event
- retry 시 duplicate tool execution 방지
- model switch 시 state compatibility

## 구현 판단 기준

특정 제품을 선택하기 전에 아래 질문에 답해야 한다.

1. replica가 완전히 사라져도 continuation 가능한가?
2. process-local response ID에만 의존하지 않는가?
3. tool call/result가 구조화된 상태로 복원되는가?
4. streaming 도중 장애 후 state가 어느 지점까지 commit됐는지 알 수 있는가?
5. 동일 요청 retry에서 duplicate side effect를 막을 수 있는가?
6. OpenAI와 Anthropic API를 같은 canonical state model로 수용 가능한가?

## 제품/기술 검토에서 얻은 결론

과거 검토에서 LiteLLM, Semantic Router 계열, Llama Stack, Open Responses 계열 등을 후보로 봤지만 제품 이름 자체를 canonical decision으로 고정하지 않는다.

특히 주의할 점:

- session affinity 기능을 durable state store로 오해하지 않는다.
- request/response logging DB 또는 object storage를 replay-capable conversation store와 동일시하지 않는다.
- "API를 지원한다"와 "multi-replica restart 후에도 stateful continuation이 복원된다"를 별도 검증한다.

현재 설계의 source of truth는 **외부 durable state + state facade + optional affinity** 구조다.

## 검증 시나리오

최소 golden scenarios:

1. response 생성 후 다른 replica로 continuation
2. response 생성 후 원 replica kill → continuation
3. router restart 후 continuation
4. rolling update 중 continuation
5. tool call 직후 backend failure → retry
6. streaming 중간 disconnect → resume/retry semantics 확인
7. multi-turn + reasoning + tool call + tool result chain
8. model alias/backend 변경 후 state compatibility

## MOC와의 경계

MOC는 conversation content 자체의 primary store가 아니다.

MOC가 관여할 수 있는 영역은 다음과 같다.

- stateful workload의 safe drain 여부
- session/conversation-aware rollout policy
- backend retirement 전 active affinity/session signal 반영
- state facade / store health를 deployment verification signal로 사용

Conversation semantics와 durable state ownership은 API/state layer가 가진다.
