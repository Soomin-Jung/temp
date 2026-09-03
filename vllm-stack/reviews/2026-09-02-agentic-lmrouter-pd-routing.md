# Agentic API + LMStack Router + P/D Cell Routing Review

업데이트: 2026-09-02
상태: Source-level architecture review / implementation gate

> 이 문서는 특정 source revision을 검증한 **point-in-time review**다. 현재 routing/state 결정은 [API Routing Contract](../api-routing-contract.md)와 [Stateful Conversation Architecture](../stateful-conversation-architecture.md)를 우선한다.

## 1. 최종 판정

제안 구조는 **Conditional GO**다.

```text
Client
  -> Gateway
  -> vLLM Proxy
       /v1/completions, /v1/chat/completions
         -> LiteLLM hosted_vllm wrapper

       /v1/messages
         -> LiteLLM Anthropic wrapper

       /v1/responses (POST HTTP/SSE, GET WebSocket Upgrade)
         -> Agentic API
              <-> PostgreSQL
              -> reconstructed/stateless POST /v1/responses
              -> LMStack Router
              -> vLLM or validated P/D Cell
```

관심사 분리는 타당하다.

- API path/protocol selection: vLLM Proxy
- durable response/conversation state: Agentic API
- multi-model inference selection: LMStack Router
- provider compatibility wrapper: LiteLLM의 기존 Chat/Messages lane
- P/D orchestration: cell-local router

다만 다음 세 문장은 수정해야 한다.

1. LMStack Router 0.1.9는 body를 **전혀 검증하지 않는 것**이 아니라, 정상 round-robin path에서 Responses schema를 거의 해석하지 않는 **opaque-ish proxy**다.
2. fixed image tag는 `latest`보다 안전하지만 immutable source proof가 아니다. **digest와 source commit까지 연결**해야 한다.
3. P/D Cell이 안전한 이유는 `previous_response_id`가 제거돼서가 아니라, **cell-local router가 exact Responses P/D contract를 지원할 때만** 안전하다.

## 2. 주장별 판정표

| # | 주장 | 판정 | 수정/조건 |
|---:|---|---|---|
| 1 | Agentic API는 multi backend base URL을 지원하지 않는다 | **대체로 맞음** | 하나의 logical `llm_api_base`만 가진다. logical upstream이 multi-model router일 수는 있다. |
| 1 | LMStack Router 0.1.9는 body 검증 없이 전달한다 | **방향은 맞고 표현은 과함** | JSON parse, `model` 필수 검사, callback/rewriter/alias/routing inspection이 있다. plain path는 original body bytes를 전달한다. |
| 2 | path split + Agentic + LMStack Router로 해결 | **맞음** | POST/SSE뿐 아니라 같은 `/v1/responses`의 GET WebSocket Upgrade와 사용 중인 compact/lifecycle surface를 route해야 한다. |
| 3 | 운영 image는 fixed tag로 pin | **좋은 현재 상태** | mutable tag 위험과 dev-tag source 불명확성을 제거하려면 digest/source SHA/SBOM을 남긴다. |
| 4 | P/D Cell은 내부 router가 있으므로 global router는 상관없음 | **global layer 관점은 맞음, 전체 판정은 조건부** | cell-local router가 `/v1/responses`, `max_output_tokens`, KV transfer, SSE를 지원해야 한다. |
| 5 | Service를 호출하는 현재 구조에서 Pod-level KV-aware routing은 의미 없음 | **핵심은 맞음** | model/service-level routing은 여전히 의미 있다. exact replica KV routing만 kube-proxy에 선택권을 빼앗긴다. |

## 3. Agentic API의 upstream model

검토 snapshot은 `vllm-project/agentic-api` commit [`98325a8`](https://github.com/vllm-project/agentic-api/tree/98325a81b645c5fa908dc4ee956243c06d0f622e)이다.

구성에는 단일 string인 `llm_api_base`가 있고 CLI/environment도 `--llm-api-base`/`LLM_API_BASE` 하나를 받는다. model별 base URL registry는 없다.

```toml
llm_api_base = "http://logical-upstream"
```

따라서 여러 model이면 다음 중 하나가 필요하다.

1. Agentic API를 model별로 나누기
2. Agentic API 뒤에 multi-model logical upstream 배치

state runtime과 model lifecycle을 분리하려면 2가 적합하다. 단일 base URL은 단일 physical backend를 뜻하지 않는다.

### Agentic의 client-facing surface

Agentic source는 같은 path에 다음 route를 등록한다.

```text
POST /v1/responses  -> HTTP JSON or SSE
GET  /v1/responses  -> WebSocket Upgrade
POST /v1/responses/compact
```

근거는 [`app.rs` route registration](https://github.com/vllm-project/agentic-api/blob/98325a81b645c5fa908dc4ee956243c06d0f622e/crates/agentic-server/src/app.rs#L247-L255)이다.

Gateway/vLLM Proxy가 POST만 path-route하면 HTTP fallback은 동작해도 Codex WebSocket은 끊긴다. method와 Upgrade를 함께 고려해야 한다.

### Agentic 내부의 proxy/executor split

Agentic은 모든 request를 무조건 DB hydration하지 않는다. `store`, `previous_response_id`, conversation, compaction, gateway-owned tool 같은 조건이 있으면 executor path를 사용하고, stateless fast path는 upstream으로 proxy한다. 즉 downstream router는 다음 두 종류를 모두 견뎌야 한다.

- client body에 가까운 stateless request
- Agentic이 typed state/tool context를 처리해 만든 upstream request

## 4. LMStack Router 0.1.9의 `/v1/responses`

공식 upstream tag `vllm-stack-0.1.9`는 commit [`20a6580`](https://github.com/vllm-project/production-stack/tree/20a658044af0dea70e9e0136494bb9979cfd9fab)이며 `/v1/responses` route를 포함한다. endpoint를 추가한 [PR #691](https://github.com/vllm-project/production-stack/pull/691)은 Responses-specific implementation이 아니라 generic route 연결이다.

```python
@main_router.post("/v1/responses")
async def route_v1_responses(request, background_tasks):
    return await route_general_request(
        request,
        "/v1/responses",
        background_tasks,
    )
```

### 4.1 실제로 검사하는 것

`route_general_request()`는 다음을 수행한다.

1. body bytes를 JSON으로 parse한다.
2. `model` field가 없으면 400을 반환한다.
3. optional `pre_request` callback을 실행한다.
4. optional request rewriter를 실행한다.
5. model alias가 있으면 body의 `model`을 바꾸고 reserialize한다.
6. model을 제공하는 backend만 filter한다.
7. routing logic에 따라 backend URL을 선택한다.

그러므로 “아무 body validation도 없다”는 정확하지 않다.

### 4.2 거의 opaque하게 전달되는 조건

다음 조건이면 plain general path는 original `request_body` bytes를 `data=body`로 selected backend의 같은 endpoint에 보낸다.

- request rewriter 비활성
- model alias rewrite 비활성
- semantic-changing callback 없음
- round-robin 같은 Responses body 비의존 routing
- legacy disaggregated-prefill path가 아님

이 조건에서는 Agentic이 만든 typed `input`, tools, reasoning, metadata를 LMStack Router가 Responses object로 변환하지 않는다. 이번 구조에서 LiteLLM보다 선호할 이유가 여기에 있다.

### 4.3 response path의 주의점

0.1.9는 backend content를 `iter_any()`로 읽어 downstream `StreamingResponse`에 yield한다. SSE event boundary를 재작성하지 않는 점은 유리하다. 그러나 완전한 zero-copy transparent transport는 아니다.

- request body를 JSON parse한다.
- `StreamingResponse` wrapper와 backend header copy를 거친다.
- non-stream request도 streaming wrapper를 통과한다.
- `full_response = bytearray()`가 streaming/non-stream 모두에서 chunk를 계속 누적하는 코드 path가 있어, 긴 stream의 memory 사용을 검증해야 한다.
- response status/error/header, client disconnect propagation은 conformance test 대상이다.

따라서 판정은 다음과 같다.

```text
OpenAI Responses full implementation       no
POST /v1/responses model-based forwarding  yes
normal path body fidelity                  good, conditional
SSE byte forwarding                        yes, validate buffering/failure
Responses WebSocket                        no, not needed downstream
durable previous_response_id               no, Agentic responsibility
```

## 5. 0.1.9에서 피해야 할 routing mode

### 5.1 KV-aware

0.1.9 `KvawareRouter`는 tokenization 입력으로 `request_json.get("prompt", "")`를 사용하고 source에도 Chat Completions 지원 TODO가 남아 있다. Responses request는 주로 `input`을 사용하므로 0.1.9 KV-aware logic은 Responses-aware가 아니다.

초기 lane은 다음처럼 둔다.

```text
routing logic: round-robin
request rewrite: off
semantic cache: off
custom callback: minimal/off
model alias: off where possible
```

### 5.2 legacy disaggregated prefill

0.1.9 `DisaggregatedPrefillRouter`이면 generic forwarding보다 먼저 P/D 전용 branch로 빠진다. 이 branch는 request의 `max_tokens`를 1로 바꾸고 prefill/decode를 호출하는 Chat/Completions-oriented logic이다. Responses가 사용하는 token limit는 `max_output_tokens`이므로 그대로 사용할 수 없다.

## 6. `/v1/messages` 판정

공식 0.1.9 tag에는 `/v1/messages` route가 없다. Messages route는 0.1.9 이후 변경에서 추가됐다. 그러므로 0.1.9 기반에서 `/v1/messages`를 기존 LiteLLM Anthropic wrapper lane으로 분리하는 결정은 타당하다.

장기적으로 LMStack Router를 upgrade하더라도 endpoint 존재만으로 Anthropic Messages 완전 호환을 판정하지 않는다. error body, beta header, SSE event, tool/thinking block fidelity를 다시 검증한다.

## 7. Image tag와 source provenance

`latest` 대신 fixed tag를 사용하는 것은 필수적인 개선이다. 다만 tag는 다음 이유로 충분하지 않다.

- registry 정책에 따라 같은 tag가 다른 manifest로 이동할 수 있다.
- `0.1.9-dev` 계열은 official release tag와 source snapshot이 다를 수 있다.
- chart version, router package version, image tag가 자동으로 동일 commit을 뜻하지 않는다.
- local/custom repository tag 이름이 upstream official tag와 같은 SHA를 가리킨다는 보장이 없다.

배포 기록에 다음을 함께 남긴다.

```text
image repository
image tag
image digest (sha256)
source repository
source commit SHA
build recipe / SBOM / provenance
verified endpoints and contract-test result
```

runtime에서는 Pod의 `imageID`와 router `/version` 결과를 확인하되, `/version` 문자열만 source proof로 사용하지 않는다.

## 8. P/D Cell: global router와 cell-local router를 분리한다

### 8.1 맞는 부분

Agentic이 state를 resolve해 full stateless request를 만들면 global LMStack Router는 `previous_response_id`를 이해할 필요가 없다. global router가 model을 보고 P/D Cell Service를 고르는 것은 정상이다.

```text
Agentic
  -> global model router
  -> selected P/D Cell Service
  -> cell-local P/D router
  -> prefill/decode engines
```

### 8.2 빠진 조건

cell-local router는 단순 forwarding이 아니라 P/D orchestration을 한다. 따라서 다음을 알아야 한다.

- Responses endpoint
- `max_output_tokens` prefill limiting
- original decode request preservation
- connector별 `kv_transfer_params`
- Responses SSE/status/error forwarding

state field를 제거했는지는 이 contract와 별개다.

### 8.3 두 router implementation을 혼동하면 안 된다

| implementation | Responses forwarding | Responses-aware P/D | 판정 |
|---|---:|---:|---|
| LMStack Router official 0.1.9 normal route | yes | no | global thin router에 적합 |
| LMStack Router 0.1.9 legacy P/D route | endpoint는 intercept | no, `max_tokens` 중심 | Responses P/D에 부적합 |
| later `disaggregated_prefill_orchestrated` LMStack work | 별도 revision | revision별 확인 | 0.1.9 feature로 간주 금지 |
| `vllm-project/router` v0.1.15 | yes | `max_output_tokens` branch와 단위 테스트 존재 | exact image digest와 E2E 검증 후 사용 |

PR #4가 pin한 `vllm-project/router` v0.1.15 tag는 source commit
[`1fbcde7`](https://github.com/vllm-project/router/tree/1fbcde7443d75b36befb61bc081f64c2a1f13a4b)이다.
이 revision의 `VllmPDRouter::prepare_prefill_request()`는 `/v1/responses`에서 Prefill request copy의
`max_output_tokens`를 1로 설정하고 `max_tokens`를 주입하지 않는다. 해당 동작을 검증하는 단위 테스트도 같은
revision에 포함된다. 따라서 **source-level Responses P/D token-limit contract는 충족**한다.

다만 source branch/tag 확인과 deployable artifact 검증은 분리한다. 실제 내부 image digest가 이 commit으로 빌드됐는지
증명하고, Responses SSE·error·cancellation·Mooncake handoff까지 E2E를 통과시킨 뒤 production contract로 확정한다.

## 9. Kubernetes Service와 KV-aware routing

global/model router가 multi-replica ClusterIP Service를 호출하면 다음 두 선택이 연속으로 일어난다.

```text
L7 router: model-a Service 선택
Kubernetes L4 dataplane: Service endpoint Pod 선택
```

model A/B 선택은 여전히 유효하다. 그러나 L7 router가 Pod A의 KV hit를 계산해도 destination이 model-a Service VIP라면 kube-proxy가 Pod B를 선택할 수 있다.

정확한 Pod-level KV-aware routing에는 다음이 필요하다.

- EndpointSlice/Pod IP discovery
- selected Pod IP로 direct dispatch
- Pod별 connection pool
- ready/terminating watch와 pool eviction
- queue/KV/health telemetry
- Pod churn과 retry contract

상세 동작은 [Kubernetes Service와 EndpointSlice](../../study/networking/10-kubernetes-service-endpoints.md), [L4/L7과 LLM serving routing](../../study/networking/11-l4-l7-llm-serving-routing.md)을 참고한다.

## 10. vLLM Proxy에 필요한 path/transport 개발

단순 path table 외에 transport policy도 분리한다.

| route | upstream | transport/policy |
|---|---|---|
| POST `/v1/completions` | LiteLLM hosted_vllm | 기존 inference retry policy |
| POST `/v1/chat/completions` | LiteLLM hosted_vllm | 기존 JSON/SSE policy |
| POST `/v1/messages` | LiteLLM Anthropic wrapper | Anthropic header/event/error fidelity |
| POST `/v1/responses` | Agentic API | JSON/SSE, buffering off, blind retry off |
| GET `/v1/responses` + Upgrade | Agentic API | WebSocket pass-through, long-lived drain |
| POST `/v1/responses/compact` | Agentic API, 사용 시 | large body/long timeout, state-aware retry |

필수 구현 항목:

- method/path/Upgrade-aware route
- SSE buffering 비활성화와 flush 보존
- WebSocket Upgrade/close/ping-pong 통과
- 충분하지만 무한하지 않은 response header/idle/total/drain timeout
- downstream disconnect → Agentic cancel/cleanup 전파 검증
- partial response/tool execution 뒤 automatic POST replay 금지
- request/body limit와 header allowlist
- route별 request ID, trace, status/error preservation
- public `/v1/responses`와 Agentic downstream inference URL을 분리해 recursion 방지

## 11. Production gate

### correctness

- Agentic replica A에서 생성, A kill 후 B에서 continuation
- typed message/reasoning/tool call/output rehydration
- HTTP non-stream, SSE, WebSocket 각각 golden test
- target Codex version의 tool/custom/namespace contract
- model unknown/unavailable와 upstream error fidelity
- P/D Cell에서 `max_output_tokens`, KV transfer, terminal event 확인

### failure/retry

- response header 전 failure와 partial SSE 뒤 failure 구분
- tool side effect 뒤 blind retry 없음
- WebSocket reconnect와 HTTP fallback state 일치
- DB failover/commit ambiguity
- router/vLLM rolling drain

### performance

- hydration items/bytes/latency
- router TTFT/stream buffer memory
- first complete SSE event/ITL
- KV hit/miss와 prefill latency
- Service vs Pod-direct connection distribution

## 12. 최종 architecture decision

다음 구조를 1차 production candidate로 채택할 수 있다.

```text
protocol path -> vLLM Proxy lane selection
Responses state -> Agentic API + PostgreSQL
multi-model routing -> LMStack Router 0.1.9 normal round-robin path
P/D routing -> separately pinned Responses-aware cell router
Pod-level KV-aware routing -> follow-up, direct endpoint ownership required
```

결론적으로 추가 개발 범위는 “Agentic API 배포 + vLLM Proxy path routing”에서 시작해도 맞다. 다만 실제 production 완료 정의에는 **WebSocket route, transport/retry policy, exact image provenance, P/D cell-local contract test**가 포함돼야 한다.
