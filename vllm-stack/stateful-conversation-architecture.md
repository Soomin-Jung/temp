# Stateful Conversation Architecture for OpenAI Responses / Codex

업데이트: 2026-09-02
상태: Architecture decision context / candidate validation

## 결론 요약

1. **현재 Codex 소스 기준 기본 inference wire protocol은 Responses API다.** provider-relative endpoint는 `/responses`이며, API key 인증이면 실효 기본 endpoint는 `https://api.openai.com/v1/responses`, ChatGPT 계열 인증이면 `https://chatgpt.com/backend-api/codex/responses`다.
2. **현재 Codex를 직접 연결하려면 Responses wire protocol이 필수다.** `wire_api = "chat"`은 제거됐고 설정 단계에서 거부된다. 단, 전체 플랫폼의 모든 workload를 Chat Completions에서 즉시 이관해야 한다는 뜻은 아니다.
3. **Responses protocol 지원과 durable state 지원은 다른 문제다.** Codex의 HTTP fallback은 `store=false`와 전체 item history로도 동작할 수 있다. 반면 WebSocket incremental continuation, 일반 client의 `previous_response_id`/conversation 사용, multi-replica failover까지 보장하려면 state owner가 필요하다.
4. vLLM 0.28.0은 opt-in in-memory Responses store를 제공하지만, replica-local dictionary이고 eviction이 없어 source 자체가 memory leak 가능성을 경고한다. **개발·단일 replica 기능이지 production durable store가 아니다.**
5. [vLLM Agentic API](https://github.com/vllm-project/agentic-api)는 기존에 정의한 `Conversation State Facade + Durable Store`를 구현하는 가장 직접적인 후보다. 그러나 tenant/persisted-state authorization, retention, 운영 hardening이 완료되기 전에는 shared production 표준으로 확정하지 않는다.
6. 권고 배치는 **model별 sidecar가 아니라 inference routing plane 앞의 model-agnostic state/tool orchestration tier**다. shared session에는 PostgreSQL을 사용하며, multi-model backend selection은 하나의 logical `LLM_API_BASE` 뒤 LMStack Router가 담당한다.
7. 현재 API lane의 구체적인 routing contract는 [API Routing Contract](api-routing-contract.md)를 따른다. Chat/Completions와 Messages는 기존 LiteLLM compatibility lane, Responses는 Agentic API → LMStack Router lane으로 분리한다.
8. LMStack Router 0.1.9와 P/D Cell의 source-level 조건 및 Kubernetes Service가 exact replica routing에 미치는 영향은 [Agentic API + LMStack Router + P/D Cell Routing Review](reviews/2026-09-02-agentic-lmrouter-pd-routing.md)에서 검증한다.

## 질문별 답

| 질문 | 판정 |
|---|---|
| Codex의 기본 endpoint는? | endpoint code는 base URL에 `/responses`를 붙인다. 기본 API key 경로는 `https://api.openai.com/v1/responses`, ChatGPT 로그인 경로는 `https://chatgpt.com/backend-api/codex/responses`다. |
| Codex에 Responses API가 필수인가? | **예, 현재 upstream Codex wire contract에는 필수다.** Chat Completions-only provider는 직접 연결할 수 없고 Responses facade/translator가 필요하다. |
| downstream vLLM도 stateful Responses여야 하는가? | **아니다.** state facade가 history를 rehydrate하면 inference backend는 stateless일 수 있다. 다만 현재 Agentic API는 upstream의 stateless `/v1/responses`를 사용하므로 Responses schema는 필요하고, Chat 변환으로 대체하면 full fidelity를 별도 입증해야 한다. |
| durable server-side state도 항상 필수인가? | **아니다.** client가 전체 item history를 재전송하는 stateless HTTP Responses는 가능하다. 다만 ID-only continuation, restart/failover, shared conversations, background/resume를 제공하려면 durable state가 필요하다. |
| Chat Completions는 폐기해야 하는가? | **아니다.** legacy/stateless chat lane은 유지 가능하다. Codex 및 agentic workload에 Responses lane을 추가하고 capability별로 분리한다. |
| `/v1/responses`가 열리면 완전 지원인가? | **아니다.** typed items, streaming event ordering, tool loop, state lineage, WebSocket, retry/idempotency, multi-replica recovery까지 별도 검증해야 한다. |

## 1. 용어와 판단 축

다음 세 축을 섞지 않는다.

### 1.1 Wire protocol

- request/response schema가 `/v1/responses`인가 `/v1/chat/completions`인가
- output이 단일 assistant message인지 typed item sequence인지
- SSE/WebSocket event contract가 보존되는지

### 1.2 Conversation durability

- `previous_response_id` 또는 `conversation_id`를 어느 replica에서도 resolve할 수 있는지
- process restart, rollout, scale-in 후에도 이어지는지
- partial execution, retry, branching, retention이 정의됐는지

### 1.3 Agent execution

- tool declaration을 보존하는지
- gateway-owned, client-owned, provider-owned tool의 실행 주체가 명확한지
- tool side effect의 retry/idempotency가 보장되는지

`Responses-compatible`, `stateful`, `agentic`은 서로 자동으로 따라오는 동의어가 아니다.

## 2. Codex 소스 기준 분석

검토 snapshot은 `openai/codex` main의 commit [`eb10d91`](https://github.com/openai/codex/tree/eb10d91e48ccbd0930427461fb392337addb1ac0)이다.

### 2.1 기본 endpoint 계산

Codex의 provider 설정은 base URL과 provider-relative path를 분리한다.

| 인증/설정 | base URL | path | 실효 endpoint |
|---|---|---|---|
| OpenAI API key | `https://api.openai.com/v1` | `/responses` | `https://api.openai.com/v1/responses` |
| ChatGPT 계열 인증 | `https://chatgpt.com/backend-api/codex` | `/responses` | `https://chatgpt.com/backend-api/codex/responses` |
| custom provider | 설정한 `base_url` | `/responses` | 예: `http://agentic-api:9000/v1/responses` |

근거:

- [`WireApi`는 `Responses`만 허용하고 `chat`을 거부](https://github.com/openai/codex/blob/eb10d91e48ccbd0930427461fb392337addb1ac0/codex-rs/model-provider-info/src/lib.rs#L40-L89)
- [인증 방식별 default base URL 선택](https://github.com/openai/codex/blob/eb10d91e48ccbd0930427461fb392337addb1ac0/codex-rs/model-provider-info/src/lib.rs#L292-L310)
- [기본 Responses endpoint의 provider-relative path가 `/responses`](https://github.com/openai/codex/blob/eb10d91e48ccbd0930427461fb392337addb1ac0/codex-rs/codex-api/src/endpoint/responses.rs#L26-L46)
- [built-in OpenAI provider는 Responses와 WebSocket을 활성화](https://github.com/openai/codex/blob/eb10d91e48ccbd0930427461fb392337addb1ac0/codex-rs/model-provider-info/src/lib.rs#L385-L420)

따라서 custom provider에는 `/v1`까지 포함한 base URL을 주고 Codex가 `/responses`를 붙이게 해야 한다. `/v1/responses` 전체를 base URL로 주면 path가 중복될 수 있다.

### 2.2 Codex가 실제로 사용하는 Responses 기능

Codex request는 단순히 Chat payload를 다른 URL로 보내는 형태가 아니다.

- typed `ResponseItem` history
- `instructions`, `reasoning`, `text` output format
- function/custom/namespace를 포함한 tool declarations와 parallel tool calls
- `include=["reasoning.encrypted_content"]`
- `prompt_cache_key`, service tier, client metadata
- typed SSE events: text delta, reasoning summary/content delta, output item, tool input delta, completion/usage
- provider가 지원하면 Responses WebSocket 우선, 실패 시 HTTP Responses fallback

Codex가 만드는 일반 request는 [`store=false`, `stream=true`](https://github.com/openai/codex/blob/eb10d91e48ccbd0930427461fb392337addb1ac0/codex-rs/core/src/client.rs#L990-L1031)다. 이것만 보고 state가 필요 없다고 결론 내리면 안 된다.

WebSocket path에서는 connection을 turn 단위로 재사용하고, 이전 request와 공통인 input을 제외한 delta와 `previous_response_id`를 전송한다. prewarm도 `response.create(generate=false)` 후 받은 response ID를 다음 request에 재사용한다. 반대로 WebSocket을 쓸 수 없거나 state를 잃으면 full request를 HTTP Responses로 재전송할 수 있다.

즉 Codex에는 두 실행 형태가 공존한다.

| 형태 | client payload | server-side lineage 의존 | 의미 |
|---|---|---:|---|
| HTTP Responses fallback | 전체 item history, `store=false` | 낮음 | stateless facade로도 기본 실행 가능 |
| Responses WebSocket | delta input + `previous_response_id` | 높음 | persistent transport, prewarm, 작은 전송량, incremental continuation |

Codex source도 [WebSocket을 우선하고 HTTP Responses로 fallback](https://github.com/openai/codex/blob/eb10d91e48ccbd0930427461fb392337addb1ac0/codex-rs/core/src/client.rs#L2010-L2055)하며, incremental request는 [`previous_response_id`와 새 item만 전송](https://github.com/openai/codex/blob/eb10d91e48ccbd0930427461fb392337addb1ac0/codex-rs/core/src/client.rs#L1828-L1871)한다.

## 3. Chat Completions 대비 Responses의 실질적 이점

OpenAI는 신규 text generation에 Responses를 권장하고, Chat Completions는 계속 지원한다. 다음 표는 protocol 자체의 이점과 별도 구현이 필요한 기능을 구분한다.

| 영역 | Chat Completions 중심 | Responses 사용 시 | self-hosted에서 필요한 구현 |
|---|---|---|---|
| 상태 전달 | 보통 client가 message history 전체 관리 | `previous_response_id`, Conversations 호환, branching | durable response/conversation store 또는 full-history fallback |
| 응답 모델 | assistant message 중심 | message, reasoning, tool call/result, compaction 등 typed items | item fidelity와 ordering 보존 |
| reasoning | provider별 message extension에 의존 | reasoning item/summary/encrypted content를 정식 흐름으로 유지 | model parser와 reasoning item round-trip |
| tools | function calling 중심 | function/custom, built-in tool, MCP 등 더 넓은 tool surface | tool ownership·executor·approval·credential boundary |
| streaming | token/message delta SSE | item lifecycle과 tool/reasoning을 포함한 semantic events | event ordering, terminal event, reconnect semantics |
| transport | 주로 HTTP/SSE | HTTP/SSE에 더해 persistent WebSocket incremental mode | WS upgrade, connection state, fallback |
| long-running work | application job wrapper 필요 | background, retrieve/cancel, webhook 계열과 결합 가능 | durable execution state와 worker lifecycle |
| context 관리 | client-side truncation/summarization | compaction item과 context management | compactor와 retention policy |
| structured output | `response_format` 계열 | `text.format`과 typed output | backend parser/schema fidelity |

추가로 reasoning model은 Responses에서 더 나은 intelligence/performance를 제공한다고 OpenAI가 명시한다. 다만 이것은 OpenAI model/service의 특성이며, **open-weight model을 `/v1/responses`로 감싼다고 모델 품질이 자동 향상되는 것은 아니다.**

또한 `previous_response_id`로 client→gateway 전송 bytes를 줄일 수 있어도, gateway가 매 turn 전체 history를 재구성해 vLLM에 보내면 model input token과 prefill 비용은 그대로다. token/latency 절감은 prefix cache, cache-aware routing, compaction과 별도로 측정해야 한다.

공식 참고:

- [Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses)
- [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)
- [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)
- [WebSocket mode](https://developers.openai.com/api/docs/guides/websocket-mode)
- [Using tools](https://developers.openai.com/api/docs/guides/tools)
- [Compaction](https://developers.openai.com/api/docs/guides/compaction)

## 4. vLLM 0.28.0의 Responses state 한계

검토 snapshot은 vLLM tag `v0.28.0`, commit [`2cf0a69`](https://github.com/vllm-project/vllm/tree/2cf0a6915ce544dc493a0990f2ea38d81601128a)다.

vLLM에는 `/v1/responses`와 opt-in store가 존재한다. 그러나 [`OpenAIServingResponses`](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/openai/responses/serving.py#L203-L239)의 실제 구현은 다음과 같다.

- `VLLM_ENABLE_RESPONSES_API_STORE` 기본값은 false
- false일 때 일반 `store=true`는 저장하지 않고 false로 바꿔 처리
- true일 때 `response_store`, `msg_store`, `event_store`는 process-local Python dictionary
- response/message/event를 제거하지 않아 source가 memory leak 가능성을 경고
- `previous_response_id`가 현재 process dictionary에 없으면 not found
- restart/rollout/scale-in 후 복구, replica 간 공유, TTL/retention, tenant authorization 없음

따라서 다음과 같이 판정한다.

| 환경 | vLLM native store 판정 |
|---|---|
| local 개발·단일 replica | 기능 확인용으로 사용 가능 |
| sticky routing이 있는 복수 replica | 정상 시 hit 가능하나 replica 장애 시 state loss |
| rolling update / autoscaling / failover | 부적합 |
| shared multi-tenant production | 부적합 |

`VLLM_ENABLE_RESPONSES_API_STORE=1`은 full stateful support의 해법이 아니라 test fixture에 가깝다.

## 5. LiteLLM의 위치와 한계

검토 snapshot은 `BerriAI/litellm` main의 commit [`31ca4dd`](https://github.com/BerriAI/litellm/tree/31ca4ddf32ae6ed16d8518c82bd2608c248fcf5a)다.

현재 LiteLLM은 `/v1/responses` surface와 `previous_response_id` 기반 deployment affinity를 제공한다. response ID에서 deployment model ID를 복원해 같은 deployment를 우선 선택하는 [affinity check](https://github.com/BerriAI/litellm/blob/31ca4ddf32ae6ed16d8518c82bd2608c248fcf5a/litellm/router_utils/pre_call_checks/deployment_affinity_check.py#L415-L454)도 있다.

하지만 이것은 durable conversation store가 아니다.

- affinity는 provider/engine이 가진 state로 다시 보내기 위한 routing 보조다.
- pinned vLLM replica가 죽으면 그 process-local state도 사라진다.
- Responses request를 Chat provider 경로로 변환하는 코드에는 [`previous_response_id`를 무시한다는 경고](https://github.com/BerriAI/litellm/blob/31ca4ddf32ae6ed16d8518c82bd2608c248fcf5a/litellm/completion_extras/litellm_responses_transformation/transformation.py#L577-L581)가 있다.

따라서 LiteLLM은 model alias, provider selection, quota/routing tier로 계속 사용할 수 있지만, state source of truth로 간주하지 않는다. Agentic API downstream에 둘 경우 `/v1/responses` typed items, SSE events, tool shapes, status/error가 변형 없이 통과하는지 version-pinned conformance test가 필요하다.

## 6. vLLM Agentic API 검토

검토 snapshot은 `vllm-project/agentic-api` main의 commit [`f20cd2b`](https://github.com/vllm-project/agentic-api/tree/f20cd2bdad3f3532a5ec9df34a98819c39652681)다. README 기준 0.5.0 wheel은 build artifact로 제공되지만 아직 PyPI 배포 전이다.

### 6.1 기존 architecture contract와의 대응

| 필요한 책임 | Agentic API 현재 구현 |
|---|---|
| Responses facade | `POST /v1/responses`, SSE, 동일 path WebSocket upgrade |
| state resolve/rehydration | `previous_response_id`, conversation, context management를 executor path로 처리 |
| durable store | SQLite 기본, shared session은 PostgreSQL 사용 안내 |
| tool orchestration | gateway/client/provider ownership 분리, web search/file search/MCP 계열 |
| Codex compatibility | Responses WebSocket, namespace/custom tool preservation, Codex harness/validation |
| stateless fast path | state가 필요 없는 `store=false` request는 configured upstream으로 pass-through |
| compaction | `/v1/responses/compact`와 context management |

주요 근거:

- [HTTP stateful/stateless path와 WebSocket API](https://github.com/vllm-project/agentic-api/blob/f20cd2bdad3f3532a5ec9df34a98819c39652681/docs/api/index.md#L49-L85)
- [Codex custom provider 예시](https://github.com/vllm-project/agentic-api/blob/f20cd2bdad3f3532a5ec9df34a98819c39652681/README.md#L266-L305)
- [shared session에서 PostgreSQL 사용](https://github.com/vllm-project/agentic-api/blob/f20cd2bdad3f3532a5ec9df34a98819c39652681/README.md#L102-L108)
- [tool ownership model](https://github.com/vllm-project/agentic-api/blob/f20cd2bdad3f3532a5ec9df34a98819c39652681/README.md#L363-L373)

### 6.2 채택 전 blocker

| Blocker | 영향 | 요구 Gate |
|---|---|---|
| tenant/persisted-state authorization 미완료 | 인증된 principal이 다른 state ID를 읽지 못한다는 보장이 아직 없음 | object-level authorization 또는 신뢰 가능한 외부 enforcement |
| retention/RBAC가 MVP 범위 밖 | 삭제·보존·감사 정책 미확정 | TTL, legal/ops retention, hard delete, backup/restore |
| production hardening이 roadmap 진행 중 | 장애·upgrade 운영 데이터 부족 | multi-replica soak, migration, graceful drain, load test |
| 새 project/배포 artifact 성숙도 | version/API 변화 가능성 | commit/version pin, SBOM/source build, upgrade playbook |
| downstream fidelity | LiteLLM/router translation에서 typed semantics 손실 가능 | direct와 full-path golden test 비교 |
| server-side tool 운영 | egress, credential, approval, side effect 위험 | allowlist, secret boundary, audit, idempotency |

Agentic API는 OIDC authentication과 principal 추출을 구현했지만, 문서가 [tenant와 persisted-state authorization을 후속 작업](https://github.com/vllm-project/agentic-api/blob/f20cd2bdad3f3532a5ec9df34a98819c39652681/docs/design/oidc-bearer-authentication.md#L54-L56)으로 명시한다. 따라서 현재 상태로 신뢰 경계가 다른 여러 사용자가 response/conversation ID를 공유하는 production surface에 바로 노출하지 않는다.

### 6.3 판정

**`POC 우선 후보`, 아직 `production standard`는 아님.**

기존 custom vLLM proxy에 response store와 tool executor를 별도 구현하기 전에 Agentic API를 검증한다. blocker가 해소되지 않으면 custom policy/auth facade 앞에서 single-tenant 또는 강하게 격리된 scope로만 제한하거나, 필요한 state authorization을 보완한 fork/extension을 검토한다.

## 7. 권고 배치

### 7.1 논리 구조

```text
Codex / Responses clients
  -> API Gateway / custom auth-policy layer
  -> Agentic API replicas
       <-> PostgreSQL
  -> LMStack Router / Responses-transparent model router
  -> vLLM Service / Router
  -> vLLM engine replicas
```

Agentic API는 model별로 배치하지 않는다. 하나의 model-agnostic tier가 request의 `model`을 유지한 채 downstream routing plane에 전달한다.

### 7.2 현재 스택에 적용하는 안전한 migration

```text
                         +-> Agentic API -> [LiteLLM*] -> vLLM Router -> engines
Client -> API Gateway ---|
                         +-> LiteLLM ------------------> vLLM Router -> engines
                             legacy Chat/Messages lane

* Responses conformance를 통과했을 때만 유지
```

1. `/v1/chat/completions`는 기존 LiteLLM path를 유지한다.
2. `/v1/responses`와 WebSocket upgrade만 Agentic API로 분기한다.
3. 최초 POC는 Agentic API → vLLM direct 또는 가장 얇은 router path로 baseline을 만든다.
4. 이후 LiteLLM을 사이에 넣고 동일 golden test 결과가 완전히 같은지 비교한다.
5. LiteLLM이 특정 item/event/tool shape를 변형하면 Responses lane에서는 우회하고 legacy/provider routing lane에만 남긴다.

Gateway/Ingress는 다음을 지원해야 한다.

- `/v1/responses`의 HTTP POST와 WebSocket upgrade를 동일 path에서 구분
- SSE buffering 비활성화와 충분한 idle timeout
- disconnect/cancel propagation
- inbound identity를 state authorization에 연결
- service credential은 downstream에서만 사용하고 client에 노출하지 않음

### 7.3 Codex custom provider 예시

Agentic API가 WebSocket까지 제공할 때:

```toml
[model_providers.agentic-api]
name = "Agentic API"
base_url = "http://agentic-api:9000/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = true
```

HTTP/SSE만 지원하는 custom facade라면 `supports_websockets = false`로 명시해 불필요한 WS 시도와 fallback을 피한다. built-in OpenAI provider의 base URL만 덮어쓰는 구성은 built-in capability가 WebSocket true이므로, 내부 provider의 실제 capability를 명시하는 custom provider가 더 안전하다.

## 8. “Responses API 완전 지원” 정의

다음 level을 모두 통과해야 `FULL`로 표시한다.

| Level | 범위 | 최소 성공 기준 |
|---|---|---|
| L0 Surface | endpoint/schema | `/v1/responses` non-streaming 요청/응답 |
| L1 Protocol | typed semantics | reasoning, tool call/result, multimodal, structured output item round-trip |
| L2 State | lineage/durability | previous ID, branching, restart, cross-replica, retention |
| L3 Transport | SSE/WS | event ordering, terminal event, WS prewarm/incremental/fallback |
| L4 Execution | tools/background | client/gateway/provider tool ownership, retry/idempotency, cancel/resume |
| L5 Operations | security/HA | tenant authorization, audit, backup/restore, rolling upgrade, SLO/metrics |

현재 예비 판정:

| Component | L0/L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| vLLM 0.28.0 direct | 기본 Responses 생성 가능 | in-memory opt-in only | SSE 중심 | model/provider 기능 범위 | durable HA 부적합 |
| LiteLLM | provider별 translation/pass-through | affinity, durability 아님 | full-path 검증 필요 | provider별 편차 | routing tier로 평가 |
| Agentic API current main | 강한 후보 | SQLite/PostgreSQL | SSE + WS | tool/compaction/background 후보 | authorization/retention/hardening blocker |
| OpenAI service | reference contract | managed | SSE + WS | managed feature set | service contract에 따름 |

## 9. 검증 계획

### Phase 0 — source/config freeze

- Codex, Agentic API, LiteLLM, vLLM commit/image digest 고정
- endpoint와 capability manifest 명시
- public egress/credential 없이도 재현 가능한 fixtures 준비

### Phase 1 — direct baseline

- Codex → Agentic API → vLLM direct
- HTTP/SSE와 WebSocket 각각 실행
- text, reasoning, structured output, function/custom/namespace tool round-trip
- `store=false` stateless path와 `previous_response_id` stateful path 비교

### Phase 2 — durability/failure

1. response 생성 후 다른 Agentic replica로 continuation
2. 원 Agentic replica kill 후 continuation
3. vLLM replica kill/교체 후 continuation
4. router와 Agentic API rolling restart
5. branch 두 개 동시 continuation
6. stream 중 disconnect/cancel/retry
7. tool call 직후 장애와 duplicate side effect 방지
8. PostgreSQL backup/restore와 schema migration rehearsal

### Phase 3 — full path

- Codex → Gateway → Agentic API → LMStack Router normal path → engine
- direct baseline과 request item, event sequence, tool call ID, terminal status 비교
- 170K input / 2K output workload에서 client bytes, gateway hydration, vLLM input tokens, TTFT/ITL 분리 측정
- prefix cache hit/miss와 session affinity를 별도 axis로 측정
- P/D Cell은 global forwarding과 별개로 cell-local router의 `/v1/responses`, `max_output_tokens`, KV transfer, SSE contract를 검증

LMStack Router 0.1.9 및 P/D/Kubernetes routing의 상세 판정은 [Agentic API + LMStack Router + P/D Cell Routing Review](reviews/2026-09-02-agentic-lmrouter-pd-routing.md)를 따른다.

### Phase 4 — security/operations

- principal A가 principal B의 response/conversation ID로 접근할 때 거부
- response ID enumeration/guessing, log redaction, secret non-persistence
- retention/TTL/hard delete, orphan cleanup
- Postgres saturation/failover, connection pool, disk growth
- SSE/WS drain과 rollout 중 active session 보호

## 10. 최종 architectural contract

```text
Durable Conversation Store
  + Conversation State Facade
  + typed Responses fidelity
  + explicit tool ownership
  + optional Router Affinity
```

- `affinity != durability`
- `Responses endpoint != stateful support`
- `state hydration != model token reduction`
- `protocol support != tool execution`
- `authentication != object-level state authorization`

MOC는 conversation primary store가 아니다. stateful workload의 safe drain, rollout policy, backend retirement, state facade/store health를 control-plane signal로 사용할 수는 있지만 conversation semantics와 durable state ownership은 API/state layer가 가진다.
