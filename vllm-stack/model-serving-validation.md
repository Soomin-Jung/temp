# Model Serving Validation Contract

업데이트: 2026-08-19 KST  
상태: 구현 요구사항 / Golden Test 계약

## 목적

배포된 모델의 `base URL`, API key, model name을 입력받아 연결성부터 streaming, reasoning, tool call까지 단계적으로 검증하는 공통 도구의 계약을 정의한다. 단순히 HTTP 200을 확인하는 도구가 아니라 실패 지점을 네트워크, 인증, model discovery, API schema, inference 의미로 분류하는 것이 목표다.

## 입력

| 입력 | 필수 | 처리 원칙 |
|---|---:|---|
| Base URL | 예 | trailing slash와 `/v1` 중복을 정규화하되 사용자가 준 원본도 결과에 표시 |
| API key | 환경에 따라 | 메모리에서만 사용하고 log/report에는 앞뒤 일부를 포함해 어떤 형태로도 노출하지 않음 |
| Model name | 예 | `/v1/models`의 실제 identity와 비교 |
| Timeout | 아니오 | connect/read/stream idle timeout을 분리 |
| Endpoint profile | 아니오 | OpenAI Chat, Responses, Anthropic compatibility 등을 명시적으로 선택 |

## 검증 단계

| 단계 | 요청 | 성공 기준 | 대표 실패 분류 |
|---:|---|---|---|
| 0 | URL parse / TCP / TLS | 유효 URL, 연결 및 TLS handshake | invalid URL, DNS, refused, timeout, certificate |
| 1 | `GET /health` | service liveness 응답 | ingress path, proxy, engine not ready |
| 2 | `GET /v1/models` | 인증 성공, 대상 model 발견 | 401/403, discovery mismatch, model alias mismatch |
| 3 | 최소 chat completion | 유효 schema와 non-empty assistant output | model not found, schema, inference failure |
| 4 | streaming chat | 정상 delta sequence와 종료 marker | buffering, malformed SSE, premature close |
| 5 | reasoning case | content/reasoning field 계약 충족 | parser mismatch, empty visible answer |
| 6 | tool call case | tool name, arguments JSON, finish reason 검증 | parser/schema/stream assembly mismatch |
| 7 | 선택 endpoint | `/v1/responses`, completion, Anthropic 계열 | unsupported와 broken을 구분 |

`/health`는 model parameter를 받는 endpoint로 가정하지 않는다. 서비스가 해당 endpoint를 제공하지 않으면 `unsupported/skip`으로 표시하고 `/v1/models`와 실제 inference 결과로 건강 상태를 판정한다.

## Negative Test

최소한 아래 세 실패가 서로 다른 메시지와 exit code로 구분되어야 한다.

1. 존재하지 않거나 연결할 수 없는 Base URL
2. 잘못된 API key
3. 존재하지 않는 model name

추가로 잘못된 tool schema, 지나치게 짧은 timeout, stream 중단을 재현할 수 있으면 proxy와 engine 장애를 더 빠르게 분리할 수 있다.

## Endpoint 의미

- `/v1/chat/completions`: stateless chat inference의 기본 golden path
- `/v1/responses`: server-side response/conversation state를 제공하는 계층이 있을 때만 stateful continuation을 검증
- `previous_response_id`: inference engine 단독 기능이라고 가정하지 않으며 conversation store/proxy 계약과 함께 검사
- Anthropic compatibility: OpenAI schema 성공과 별개 capability로 보고 request/response translation을 독립 검증

## 결과 형식

```text
Target       PASS  base URL normalized
Connectivity PASS  TLS / latency
Auth         PASS  /v1/models
Model        PASS  requested identity found
Chat         PASS  schema / content
Streaming    PASS  chunks / terminal event
Reasoning    WARN  provider-specific field not exposed
Tool Call    PASS  name / JSON arguments / finish reason
```

각 항목은 `PASS`, `FAIL`, `WARN`, `SKIP` 중 하나를 사용한다. 전체 실패 시에도 완료된 앞 단계와 raw status code, latency, response content-type은 남기되 API key와 민감 header/body는 redaction한다.

## 운영 연결

- 모델 배포 PR/rollout의 post-deploy gate로 재사용한다.
- Model Profile과 함께 기대 capability를 선언해 지원하지 않는 endpoint를 실패로 오판하지 않는다.
- 같은 test vector를 LiteLLM, Global Router, model-specific Service, engine direct path에 적용하면 어느 계층에서 계약이 깨졌는지 분리할 수 있다.
