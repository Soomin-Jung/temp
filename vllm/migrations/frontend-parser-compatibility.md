# Frontend / Renderer / Parser Compatibility Deep Dive

업데이트: 2026-09-10 KST

## 1. 목적

vLLM version migration에서 frontend/parser는 흔히 engine 성능 검증 뒤에 붙는 부가 기능으로 취급되지만, 최신 frontier model serving에서는 그렇지 않다.

실제 API path는 대략 다음과 같다.

```text
HTTP / SSE request
 -> protocol-specific request validation
 -> chat template / renderer
 -> tokenizer
 -> Engine request
 -> generated token IDs / text deltas
 -> reasoning/tool Streaming Parser Engine
 -> protocol-specific response renderer
 -> SSE / JSON response
```

따라서 engine token generation이 정확해도 parser나 renderer가 틀리면 API contract는 깨진다.

특히 다음 workload에서는 parser가 **serving correctness의 일부**다.

- reasoning model
- auto tool calling
- strict structured output
- Responses API
- Anthropic Messages adapter
- streaming function calls
- parallel tool calls

---

## 2. Streaming Parser Engine architecture

v0.24 PR #45413에서 modern parser engine의 핵심 구조가 들어왔다.

```text
delta_text + delta_token_ids
    |
    v
TokenIDScanner
    |
    v
IncrementalLexer
    |
    v
Parser State Machine
    |
    v
SemanticEvent[]
```

Source:
- PR #45413: https://github.com/vllm-project/vllm/pull/45413

`StreamingParserEngine`은 request-local state를 유지한다.

대표 state 개념:

```text
CONTENT
REASONING
TOOL_PREAMBLE
TOOL_NAME
TOOL_ARGS
TOOL_BETWEEN
```

즉 parser는 단순히 최종 문자열에서 `<think>`나 `<tool_call>`을 split하는 postprocessor가 아니다.

---

## 3. 왜 token ID와 text를 둘 다 보는가

Special token은 detokenized text 경계와 항상 1:1로 대응하지 않는다.

Streaming에서는 예를 들어:

```text
chunk 1: "<tool"
chunk 2: "_call>"
```

같은 문자열 split뿐 아니라 tokenizer의 special token ID 자체가 semantic boundary를 나타낼 수 있다.

Parser Engine은:

- token ID scanner
- incremental text lexer

를 같이 사용해 boundary를 복원한다.

따라서 regression test는 문자열 chunk 크기를 고정하면 안 된다. **동일 token stream을 여러 delta segmentation으로 replay**해야 한다.

---

## 4. Tool argument streaming이 특별한 이유

Tool argument는 JSON object라서 streaming 중 아직 닫히지 않은 brace/string escape를 그대로 downstream에 내보내면 invalid partial JSON이 될 수 있다.

Parser engine은 argument buffer에서 safe watermark를 관리한다.

예:

```json
{"city":"Seo
```

와 다음 delta:

```text
ul"}
```

이 합쳐져야 정상 tool arguments가 된다.

따라서 테스트 대상은 최종:

```json
{"city":"Seoul"}
```

하나가 아니라 각 SSE delta의:

- tool call index
- function name
- argument fragment
- finish state

까지 포함한다.

---

## 5. Reasoning parser vs Tool parser

CLI에서는 보통 두 개의 별도 option처럼 보인다.

```text
--reasoning-parser
--tool-call-parser
```

하지만 modern implementation에서는 두 adapter가 **동일 Parser Engine을 공유**할 수 있다.

이 coupling 때문에 parser registry에서 이름만 존재하는지 검사하는 것으로는 충분하지 않다.

대표 사례가 PR #52830이다.

---

## 6. PR #52830: shared parser engine의 실제 failure mode

PR #52830은 structured output에서 reasoning/tool parser가 같은 parser engine을 사용할 때 request-specific reasoning configuration이 조용히 사라지는 문제를 수정했다.

문제 흐름:

```text
Reasoning parser + Tool parser
        |
        +-- same ParserEngine shared
                 |
                 v
ParserManager.get_parser()
                 |
                 +-- raw shared ParserEngine 반환
                        reasoning_parser = None
                 |
                 v
Chat / Responses path
 -> reasoning disabled로 잘못 해석
 -> enable_thinking=False 같은 request kwargs 전달 누락
```

그 결과 structured output backend는 model에게 thinking을 끄라고 했음에도 parser default thinking mode를 가정할 수 있었다.

최악의 경우:

```text
reasoning boundary를 기다림
 -> schema constraint가 제대로 시작되지 않음
 -> tool call 없음
 -> max_tokens까지 비정상 생성
```

Source:
- PR #52830: https://github.com/vllm-project/vllm/pull/52830

이건 parser bug가 **structured generation correctness와 token budget까지 영향을 주는 사례**다.

---

## 7. v0.26 -> v0.29 reasoning parser registry 변화

v0.26 registry에 주요 target 기준으로 이미 존재:

```text
qwen3
deepseek_v4
glm45
glm47
kimi_k2
```

v0.29에서 추가되는 대표 parser:

```text
kimi_k3
ling3
hy_v4
muse_glimmer
```

또 `mistral` 등 일부 parser 구현은 Parser Engine adapter 쪽으로 이동한다.

Source:
- v0.26: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/reasoning/__init__.py
- v0.29: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/reasoning/__init__.py

### Platform implication

Parser name은 API compatibility key로 관리한다.

```yaml
modelProfile:
  reasoningParser: qwen3
```

처럼 선언하되, version update 시 parser implementation SHA가 바뀌었다면 regression suite를 다시 실행한다.

---

## 8. Tool parser registry 변화

v0.26 주요 target:

```text
deepseek_v4
glm45
glm47
qwen3_coder
qwen3_xml
kimi_k2
```

v0.29 신규/확장:

```text
kimi_k3
ling3
hy_v4
muse_glimmer
dots
```

Source:
- v0.26: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/tool_parsers/__init__.py
- v0.29: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/tool_parsers/__init__.py

중요한 점은 Qwen3 계열도 하나의 universal parser가 아니라 model/template 특성에 따라:

```text
qwen3_coder
qwen3_xml
mimo
```

등의 registration을 공유/분기한다는 것이다.

Qwen3.5+/3.6/3.8 deployment에서 model card가 추천하는 parser와 실제 tokenizer/template output을 항상 함께 검증한다.

---

## 9. GLM parser migration

PR #45915는 GLM4.7/GLM5.1/GLM5.2를 Streaming Parser Engine path로 옮기는 중요한 변경이다.

PR 작성자는 최소 다음 matrix를 검증했다.

```text
thinking = true / false
stream   = true / false
```

Source:
- https://github.com/vllm-project/vllm/pull/45915

GLM 5.x에서는 sparse attention engine correctness와 parser correctness를 별도 gate로 둔다.

예를 들어 engine output token이 정상인데 최종 API content가 비어 있다면 attention kernel보다 parser boundary부터 확인하는 것이 맞다.

---

## 10. Kimi K3 parser

Kimi K3는 v0.27에 full-stack으로 들어왔고 v0.29에 reasoning/tool parser registration이 정식화된다.

K3는 reasoning marker와 tool calling marker를 함께 쓰기 때문에 다음을 반드시 테스트한다.

```text
reasoning only
reasoning -> final content
reasoning -> tool call
tool call -> tool result -> final content
multiple tool calls
reasoning_effort variants
stream / non-stream
```

v0.29에는 Kimi reserved marker가 일반 content에 노출되는 것을 막는 수정과 `reasoning_effort=none` 관련 변화도 포함된다.

따라서 Kimi parser는 단순 `kimi_k3` option startup 성공으로 인증할 수 없다.

---

## 11. Qwen parser

Qwen3 계열은 parser-engine adapter를 적극 사용한다.

현재 platform에서는 Qwen3.5 이상이 주요 target이므로 다음을 모델 release마다 고정한다.

```text
model revision
tokenizer revision
chat template hash
reasoning parser
tool parser
special token IDs
thinking enabled default
```

특히 model repo의 chat template가 vLLM image update와 무관하게 변경될 수 있으므로 `revision=main` 같은 floating reference는 production에서 피한다.

---

## 12. DeepSeek V4 parser

DeepSeek V4는 engine/kernel 변화가 빠르지만 frontend도 독립적으로 변한다.

v0.26에도:

```text
reasoning_parser = deepseek_v4
tool_parser      = deepseek_v4
```

계열 adapter가 존재한다.

0.29 migration에서 parser 이름이 그대로라는 이유로 테스트를 생략하지 않는다.

다음은 version-independent regression corpus로 보존한다.

- reasoning completion
- tool required
- tool auto
- strict structured tool call
- streaming argument fragments
- finish_reason

---

## 13. Python frontend와 Rust frontend

v0.24~0.29 사이 Rust frontend는 매우 빠르게 기능을 확장한다.

주요 surface:

```text
auth/CORS
rendering
tokenize/detokenize
abort
pause/resume
health/discovery
gRPC path
tool parser integration
model-specific renderer
```

그러나 Python frontend와 Rust frontend가 모든 version에서 완전한 behavioral parity라고 가정하면 안 된다.

### E2E platform 관점

frontend를 engine image 내부 구현 세부로 숨기지 말고 deployment profile의 explicit dimension으로 기록한다.

```text
frontend = python | rust
```

그리고 API acceptance를 frontend별로 수행한다.

---

## 14. Renderer와 parser의 경계

Renderer는 input direction, parser는 output direction으로만 단순 분리되지 않는다.

대략:

```text
Renderer
 -> chat message -> prompt/tokens
 -> request-level thinking/tool configuration

Parser
 -> output tokens/text -> reasoning/tool semantic events
```

인데 structured-output backend는 parser 정보를 generation constraint 생성에도 사용한다.

즉:

```text
input render config
<-> structured-output config
<-> output parser config
```

세 영역이 일관되어야 한다.

PR #52830이 바로 이 coupling failure를 보여준다.

---

## 15. `renderer_num_workers`

v0.29에도 renderer worker pool이 존재하고 기본값은 1이다.

이 pool은 blocking tokenizer/render operation을 host side에서 병렬화한다.

현재처럼 worker를 늘린 profile에서는 GPU throughput뿐 아니라 다음을 본다.

```text
API process CPU
thread count
context switches
tokenization latency
first-byte latency
SSE inter-delta gap
```

Parser가 stateful request object를 갖더라도 renderer worker 수와 parser instance lifetime을 혼동하지 않는다.

---

## 16. API별 acceptance matrix

| API | Non-stream | Stream | Reasoning | Tool call | Structured output | Stateful |
|---|---:|---:|---:|---:|---:|---:|
| `/v1/chat/completions` | O | O | O | O | O | request-level |
| `/v1/responses` | O | O | O | O | O | external conversation/state layer와 연동 |
| `/v1/messages` adapter | O | O | model-dependent | O | adapter-dependent | platform layer |
| render endpoint | O | - | N/A | prompt rendering | N/A | N/A |

각 API는 최종 text만 비교하지 말고 schema를 비교한다.

예:

```text
id
object/type
choices/output items
delta type
reasoning field
tool_call id/name/arguments
finish reason
usage
```

---

## 17. Streaming replay test 설계

하나의 golden token sequence에 대해 delta segmentation을 바꿔 replay한다.

예:

```text
Case A: 1 token / delta
Case B: 4 tokens / delta
Case C: marker 직전 split
Case D: marker 내부 split
Case E: tool JSON string escape 직후 split
Case F: final brace 직전 split
```

Expected:

```text
semantic event sequence가 equivalent
final structured response가 identical
```

이 테스트는 GPU 없이 parser unit/integration test로 CI에서 빠르게 돌릴 수 있다.

---

## 18. Golden corpus 구성

모델 family마다 최소 20~50개의 compact prompt를 만든다.

### Reasoning

- short reasoning
- long reasoning
- thinking disabled
- no explicit reasoning end
- reasoning immediately followed by tool

### Tools

- required single tool
- named tool
- auto tool
- no tool selected
- parallel tools
- escaped JSON
- unicode/Korean argument
- nested object/array

### Streaming

- tiny delta
- large delta
- disconnect before finish
- finish exactly at structural token

### Structured output

- JSON schema
- tool + schema combination
- thinking on/off

---

## 19. Observability

추천 metric/log fields:

```text
model
frontend_impl
renderer_type
reasoning_parser
tool_parser
stream
thinking_enabled
structured_output_mode
parser_error_type
parser_fallback
semantic_event_count
tool_call_count
```

Parser error를 500 error 하나로 뭉개면 model generation failure와 구분하기 어렵다.

---

## 20. Rollout gate

새 vLLM version을 production에 올리기 전 parser lane의 pass 조건:

1. golden non-stream pass
2. streaming replay pass
3. strict tool pass
4. thinking on/off pass
5. malformed-output failure mode 확인
6. API schema parity 확인
7. Python/Rust frontend를 바꾸는 경우 각각 별도 pass

---

## 21. 최종 판단

최신 model serving에서 parser는 **model architecture의 frontend half**에 가깝다.

Engine이 올바른 token을 생성하는 것과 API가 올바른 reasoning/tool response를 반환하는 것은 별도 문제다.

따라서 v0.26 -> v0.29 migration에서는 parser/renderer를 독립 compatibility domain으로 인증하고, model revision과 함께 version pinning 대상으로 관리한다.

---

## Sources

- Streaming Parser Engine PR #45413: https://github.com/vllm-project/vllm/pull/45413
- GLM parser PR #45915: https://github.com/vllm-project/vllm/pull/45915
- shared parser reasoning adapter bugfix PR #52830: https://github.com/vllm-project/vllm/pull/52830
- v0.26 reasoning registry: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/reasoning/__init__.py
- v0.29 reasoning registry: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/reasoning/__init__.py
- v0.26 tool registry: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/tool_parsers/__init__.py
- v0.29 tool registry: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/tool_parsers/__init__.py
