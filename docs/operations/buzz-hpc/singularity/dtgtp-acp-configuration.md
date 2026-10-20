# Buzz Desktop: DTGPT 연결 / Codex·Claude ACP / 초기 API Key 실패 분석

검토: 2026-10-20. 공식 소스 및 문서 정적 검토. **SIF 재빌드·DTGPT 실호출은 아직 검증되지 않음.** 호스트명·모델명·자격증명은 예시. 비밀정보는 공용 NFS/Definition/Git에 커밋 금지.

## 핵심 구분

| Buzz Desktop에서 선택하는 Harness | 실제 동작 | 요구 HTTP API | 사용자 설정 |
|---|---|---|---|
| Buzz (buzz-agent) | Rust ACP Agent, OpenAI-compatible SDK | /v1/chat/completions 또는 /v1/responses | Agent Defaults / OPENAI_COMPAT_* 환경변수 |
| Codex | buzz-acp → codex-acp → Codex App Server | **Responses API** (통상 /v1/responses), 모델 호환성 필요 | ~/.codex/config.toml, CODEX_HOME, model_provider |
| Claude Code | buzz-acp → claude-agent-acp → Claude Agent SDK | **Anthropic Messages API** (통상 /v1/messages), 단순 OpenAI chat 호환만으로 불충분 | ~/.claude/settings.json, CLAUDE_CONFIG_DIR |

- [현재 공식 codex-acp 설치 문서](https://github.com/agentclientprotocol/codex-acp): \`@agentclientprotocol/codex-acp\`는 Codex CLI와 호환되는 \`@openai/codex\` 의존성을 갖는다. Definition에는 CLI도 직접 설치해 \`codex\` 명령을 보장.
- [현재 공식 Claude ACP](https://github.com/agentclientprotocol/claude-agent-acp): 패키지 최소 Node 22, Definition은 Node 24.
- Buzz 런타임 목록은 [discovery/catalog.rs](https://github.com/block/buzz/blob/main/desktop/src-tauri/src/managed_agents/discovery/catalog.rs), ACP adapter는 [buzz-acp README](https://github.com/block/buzz/blob/main/crates/buzz-acp/README.md).

## 1. SIF 빌드 후 런타임 확인

\`\`\`bash
singularity build Buzz.sif Buzz.def
singularity exec --cleanenv Buzz.sif bash -c '
  locale
  node --version && npm --version
  command -v codex && codex --version
  command -v codex-acp && codex-acp --version
  command -v claude-agent-acp && claude-agent-acp --version
  command -v buzz-acp && command -v buzz-agent
'
\`\`\`

**내부 빌드망에서 NodeSource / npmjs.org / apt 외부 연결이 안 되면** 사내 미러 또는 빌드 서버에서 의존성 tarball을 먼저 수집해야 한다. 공식 패키지 버전은 \`Buzz.def\`에 고정돼 있으며, 내부 승인 시 해당 의존성까지 미러링한다. 런타임 X 호스트는 npm에 접근하지 않아도 된다. npm을 사용자 홈에 설치하지 않도록 공용 SIF \`/opt/buzz-agent-npm\`에 설치.

**한글 출력 UTF-8**은 \`en_US.UTF-8\` locale을 생성해 \`LANG\`, \`LC_ALL\`, \`LANGUAGE\`로 기본 지정한다. fonts-noto-cjk를 설치했으나 터미널/클라이언트의 locale, 폰트, 인코딩이 상이하면 추가 검증 필요.

## 2. Codex — 사용자의 ~/.codex/config.toml 이용

Codex ACP는 사용자 HOME 바인드로 \`~/.codex/config.toml\`을 읽는다. 설정은 **Buzz의 OpenAI-compatible 모델 설정과 별도**이다.

\`\`\`toml
# ~/.codex/config.toml (사용자별)
model = "<DTGPT Responses 호환 served-model-name>"
model_provider = "dtgtp"

[model_providers.dtgtp]
name = "DTGPT"
base_url = "https://<DTGPT-RESPONSES-GATEWAY>/v1"
env_key = "DTGPT_API_KEY"
wire_api = "responses"
requires_openai_auth = false
\`\`\`

※ 문서에서는 "dtgtp"를 provider ID 예시로 사용한다. 실제 DTGPT 표시명과 무관하며 \`model_provider\`/테이블 이름은 반드시 동일해야 한다.

사용자 본인 Shell/credential store에서만:

\`\`\`bash
export DTGPT_API_KEY='<사용자별 API Key>'
# 기존 방법을 유지
source /nfs/apps/buzz/env.sh
buzz
\`\`\`

- NFS 공용 env.sh에는 API key 미기입. 런처의 \`--cleanenv\`로 인한 유실을 막기 위해 승인된 \`DTGPT_API_KEY\` 및 \`CODEX_HOME\`만 \`APPTAINERENV_*/SINGULARITYENV_*\`로 전달한다.
- Codex ACP가 인증을 재요구한다면 \`CODEX_API_KEY\` 또는 \`OPENAI_API_KEY\`를 요구하는 버전/인증 플로우일 수 있다. 먼저 user config의 \`env_key\` 사용 여부를 \`codex\` CLI에서 검증. **Codex 계정 로그인은 DTGPT API 키와 다름**. 임의 OpenAI API 인증 플로우로 보내지 말 것.
- 기본 \`wire_api=responses\`. DTGPT가 **Chat Completions만 제공하면 Codex ACP 경로는 호환성이 보장되지 않음**. \`POST /v1/responses\`에서 streaming, tool calls, reasoning, usage 등을 실제 Codex 요청으로 검증.
- \`codex-acp\`의 \`CODEX_CONFIG\` 환경변수는 런타임에 설정을 덮어쓸 수 있으므로 config.toml 값을 의도대로 적용하는지 함께 확인.
- \`~/.codex\` 및 Git 작업 디렉터리는 Singularity HOME/CWD 바인드에서 보여야 한다.

출처: [Codex config reference](https://developers.openai.com/codex/config-reference), [Codex ACP](https://github.com/agentclientprotocol/codex-acp).

## 3. Claude — 사용자의 ~/.claude/settings.json 이용

Claude ACP는 Claude Agent SDK를 사용하므로 **Anthropic Messages API**로 접근 가능한 DTGPT 엔드포인트가 필요하다.

\`\`\`json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://<DTGPT-ANTHROPIC-COMPAT-GATEWAY>",
    "ANTHROPIC_MODEL": "<DTGPT Anthropic Messages 호환 모델>"
  }
}
\`\`\`

보통 SDK가 \`/v1/messages\`를 붙이므로 \`ANTHROPIC_BASE_URL\`에 \`/v1/messages\`까지 적지 않는다. 해당 게이트웨이의 실제 요청 path와 SDK의 path 결합을 검증한다.

비밀키는 설정 파일에 넣지 말고 개인 사용자 쉘에서 전달:

\`\`\`bash
export ANTHROPIC_AUTH_TOKEN='<DTGPT bearer token>'
# DTGPT가 Anthropic x-api-key 인증만 받는다면 ANTHROPIC_API_KEY 사용
source /nfs/apps/buzz/env.sh
buzz
\`\`\`

- \`ANTHROPIC_AUTH_TOKEN\`: \`Authorization: Bearer ...\` 방식. \`ANTHROPIC_API_KEY\`: Anthropic \`x-api-key\` 방식. 게이트웨이 인증 스키마에 맞춰 선택한다.
- \`CLAUDE_CONFIG_DIR\`를 설정하면 \`~/.claude\` 이외의 설정을 읽는다. 런처는 사용자가 지정한 변수를 컨테이너로 전달한다.
- **사용자 settings.json의 \`env\` 값과 실행 환경변수의 우선순위**를 확인할 것. settings.json에 동일한 \`ANTHROPIC_AUTH_TOKEN\`이 들어 있으면 SIF로 전달한 값이 적용되지 않을 수 있다. 중복 선언 지양.
- DTGPT의 Anthropic Messages compatibility가 모델명 alias, thinking/tool use, stream 이벤트 등을 처리하는지 직접 검증. Anthropic wrapper 경로는 OpenAI-compatible \`/v1/chat/completions\`와 **호환하지 않는다**.

출처: [Claude Code environment variables](https://code.claude.com/docs/en/env-vars), [Claude Settings](https://code.claude.com/docs/en/settings), [Claude ACP](https://github.com/agentclientprotocol/claude-agent-acp).

## 4. Buzz onboarding: "Connect with an API key" 실패의 실제 원인

[desktop/src-tauri/src/commands/agent_models.rs](https://github.com/block/buzz/blob/main/desktop/src-tauri/src/commands/agent_models.rs) 구현은 다음과 같다.

\`\`\`
base = OPENAI_COMPAT_BASE_URL || "https://api.openai.com/v1"
GET base + "/models"
Authorization: Bearer <OPENAI_COMPAT_API_KEY>
success requires 2xx and JSON {data:[{id:"..."}]} with nonempty compatible models
\`\`\`

실패해도 UI에는 일반적인 인증 실패 문구가 나타날 수 있다. \`OPENAI_COMPAT_BASE_URL\`을 빠뜨리면 **정말로 OpenAI의 공개 API**를 호출한다.

또 \`openai\` provider는 GPT/o/chatgpt 접두어 모델만 표시하고, \`openai-compat\`는 그 GPT 모델 필터를 적용하지 않는다. 사내 Qwen 모델에서는 **OpenAI-compatible 선택 필수**.

\`\`\`bash
# \`source env.sh\` 후 호스트나 SIF 내부에서 실행
echo "DTGPT model listing URL = \${OPENAI_COMPAT_BASE_URL%/}/models"
curl -sS -i -H "Authorization: Bearer \${OPENAI_COMPAT_API_KEY}" \\
    "\${OPENAI_COMPAT_BASE_URL%/}/models"
\`\`\`

진단 순서: ① 실제 URL \`/v1/models\` ② DNS/CA ③ HTTP 401/403/404/422 ④ JSON data 배열과 id ⑤ 목록 필터 ⑥ 선택한 Runtime 별 환경변수.

\`OPENAI_COMPAT_BASE_URL\`과 \`OPENAI_COMPAT_API_KEY\`는 **Buzz 자체 buzz-agent용**이며 Codex와 Claude에서는 각각 다른 설정 시스템이 적용된다.

## 5. Desktop과 Agent 로그

### 기본 구동 명령 및 URL 출력

\`\`\`bash
source /nfs/apps/buzz/env.sh
BUZZ_DESKTOP_LOG_TO_FILE=1 buzz
\`\`\`

업데이트된 \`nfs-client/buzz-desktop\`은 **실행 엔진, NFS SIF 경로, 기본 Relay URL**, 설정된 경우 OpenAI-compatible 및 Anthropic Base URL을 시작 시 출력한다. API Key 문자열은 출력하지 않는다.

예상 출력:

\`\`\`
Buzz: engine=singularity, SIF=/nfs/apps/buzz/current/Buzz.sif
Buzz: launch relay=wss://buzz.hpc.example.internal (saved Buzz workspace may override)
Buzz: collecting launcher/stdout/stderr at ~/.local/state/buzz/desktop-...log
\`\`\`

\`BUZZ_RELAY_URL\`은 시작 시 기본값. **Desktop이 이전에 저장한 Workspace URL이 우선할 수 있으므로 이 출력만으로 실제 Relay 연결 대상을 100% 확정할 수 없다.** Buzz UI의 활성 Community/Relay URL도 확인한다. 실제 런타임 결정 우선순위: workspace override > \`BUZZ_RELAY_URL\` > build-time default > \`ws://localhost:3000\`.

릴리스 GUI는 콘솔에 로그를 거의 출력하지 않을 수 있다. \`BUZZ_DESKTOP_LOG_TO_FILE=1\`은 **Desktop stdout/stderr를 수집하는 기능**이지 모든 HTTP 요청을 자동 추적하는 옵션이 아니다.

### Local Agent logs (생성·실행된 다음)

Buzz의 app data 하위 \`agents/logs\`를 사용한다.

\`\`\`bash
find "\${XDG_DATA_HOME:-$HOME/.local/share}" -path '*/agents/logs/*.log' -type f -print 2>/dev/null
# 아래 경로는 빌드 및 사용자의 XDG_DATA_HOME에 따라 다를 수 있음
tail -f "\$HOME/.local/share/xyz.block.buzz.app/agents/logs/"*.log
\`\`\`

Agent를 아직 만들지 않은 onboarding 단계라면 이 로그 디렉터리도 없을 수 있다.

### 상세 통신 진단

\`\`\`bash
# 기본 모델 탐색 / 인증 실패는 정해진 모델 URL을 curl로 재현
curl -vk "\${OPENAI_COMPAT_BASE_URL%/}/models" \\
  -H "Authorization: Bearer \${OPENAI_COMPAT_API_KEY}"  # -k는 진단용으로만
\`\`\`

인증서 문제가 확인되면 \`-k\`를 운영에 사용하지 말고 기업 CA를 SIF의 \`/usr/local/share/ca-certificates\`에 추가한 후 \`update-ca-certificates\`로 다시 빌드한다. **공유 로그에 자격증명 또는 HTTP header trace를 남기지 않는다.**

Codex adapter 전용 로그 필요 시 사용자별 \`APP_SERVER_LOGS\`, Claude adapter는 \`CLAUDE_AGENT_LOGS\` (현재 adapter source 기준) 사용 가능. 이들 변수는 SIF 실행 환경에 공급하고 실제로 기록되는 경로를 확인한다.

## 6. 별도 점검 사항

- npm 글로벌 패키지 버전을 SIF에 고정: \`@agentclientprotocol/claude-agent-acp@0.83.0\`, \`@agentclientprotocol/codex-acp@2.1.1\`, \`@openai/codex@0.159.1\`. 공개 패키지의 최신 버전은 수시로 바뀌므로 자동 latest 설치 대신 테스트 후 올린다.
- Singularity \`--cleanenv\`: USER의 \`DTGPT_API_KEY\`, \`ANTHROPIC_AUTH_TOKEN\`, \`CLAUDE_CONFIG_DIR\`, \`CODEX_HOME\`이 사라지지 않도록 NFS 런처 allowlist에 추가했다.
- 권한: 유저별 \`~/.claude\`, \`~/.codex\`, XDG state, HOME Keyring 분리. SIF는 읽기 전용.
- 호스트/컨테이너에서 GPU 필요 없음, DTGPT 모델 추론은 LLM Serving 시스템이 담당.
- 네트워크: **각 X 사용자 세션이 시작하는 SIF**에서 Relay WSS, DTGPT HTTPS 및 필요 시 GHES HTTPS에 도달 가능해야 한다.
