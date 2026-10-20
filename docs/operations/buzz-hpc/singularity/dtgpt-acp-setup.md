# Buzz Desktop + DTGPT + Claude/Codex ACP + 한국어 GUI 운영 가이드

기준: 2026-10-20 KST. 업스트림 소스 검토 + Definition/런처 정적 확인. 실제 DTGPT API, HPC SIF/IME/사내 Relay에서 E2E 검증은 **미완료**.

## 0. 핵심 결론

- **Buzz Relay와 LLM API는 별개**: `BUZZ_RELAY_URL=wss://buzz.internal.com`은 협업 이벤트용. DTGPT 인퍼런스는 아래 Agent별 설정으로 별도 요청한다.
- **클라이언트 사이드**란 사용자가 GUI를 실행하는 X 세션 Linux 호스트라는 뜻. SIF 내부의 Codex/Claude ACP subprocess가 DTGPT를 호출한다. K8s agent Pod 필요 없음.
- Self-hosted Community는 **별도 BuilderLab 배포 불필요**. BuilderLab은 Buzz의 별도 호스팅 SaaS로, `Create community` 버튼이 호스팅 인증을 열 수 있다. 사내 Relay URL로 **Join / Connect to an existing community / Self-hosted relay** 경로를 사용해야 한다.
- 사내 `https://buzz.internal.com`의 `This relay is empty`는 Relay HTTP 랜딩 화면이다. 일반적으로 사용자·조직 관리용 포털을 의미하지 않는다. Desktop에서 Nostr 이벤트로 채널 생성; 관리자는 Relay 쪽 operator/owner/멤버십/초대 처리.
- **Codex** 설정은 OpenAI **Responses** (`/v1/responses`); **Claude Code**는 **Anthropic Messages** (`/v1/messages`); **buzz-agent**는 OpenAI **Chat Completions** 또는 Responses. 한 API로 혼동하면 안 된다.

## 1. 빌드된 SIF에 포함되는 런타임

[Definition](Buzz.def)에 Ubuntu 24.04, libffi8, WebKitGTK, Node 24.18.0, npm, Codex ACP 2.2.2, Codex CLI 0.160.1, Claude ACP 0.94.0, Claude CLI 2.1.303, IBus/ibus-hangul, `ko_KR.UTF-8` 및 `en_US.UTF-8` Locale, 한글 폰트를 추가한다.

`@agentclientprotocol/codex-acp`는 호환되는 `@openai/codex` CLI를 transitive dependency로 포함한다. Codex CLI도 ACP와 일치하는 버전으로 전역 설치하고 PATH의 `/opt/acp/bin/codex`로 노출됨을 SIF `%test`에서 확인한다. 빌드 서버 npm 레지스트리에서만 패키지를 설치하며 사용자가 GUI에서 ACP 선택할 때 `npx` 다운로드를 수행하지 않는다.

```bash
singularity exec --cleanenv Buzz.sif bash -lc '
  node --version; npm --version
  command -v codex-acp; command -v claude-agent-acp
  codex-acp --version; claude --version
  locale -a | grep -Ei "en_US|ko_KR"
  ldd /usr/bin/buzz-desktop | grep "not found" && exit 1 || true
'
```

> `singularity exec ... bash -l`이 사이트 전역 프로파일에 따라 PATH를 교체하면 직접 `/opt/acp/bin/codex-acp` 등 절대 경로를 사용하라. Buzz Desktop에서 런타임 탐지를 못하면 Desktop 설정에서 해당 binary를 지정하고, `$PATH`와 `/opt/acp/bin` 노출을 검사한다.

## 2. OpenAI-compatible DTGPT: Buzz 내장 buzz-agent

Buzz onboarding의 **Connect with an API key**는 통신 URL이 아니라 API 접근 방식 선택이며, 소스의 `harnessConnectionOptions.ts`는 API 선택 시 `buzz-agent`를 우선시한다. 이 옵션이 Codex/Claude Code의 모델 공급자를 직접 바꾸지는 않는다.

**필수 설정**:

```bash
# 사용자 세션 내 source할 개인 설정 파일 또는 승인된 Credential 공급자에서 설정
export BUZZ_AGENT_PROVIDER="openai"
export OPENAI_COMPAT_BASE_URL="https://dtgpt.example.internal/v1"
export OPENAI_COMPAT_MODEL="<DTGPT-served-model-id>"
export OPENAI_COMPAT_API="chat"              # /v1/chat/completions
# export OPENAI_COMPAT_API="responses"      # DTGPT가 /v1/responses를 지원할 때만
export OPENAI_COMPAT_API_KEY="<USER_PRIVATE_KEY>" # 공유 env.sh에 기입 금지
```

**중요**: Buzz Desktop `agent_models.rs` 소스에서 `OPENAI_COMPAT_BASE_URL`이 없으면 `https://api.openai.com/v1` 기본값으로 `GET /models`를 호출한다. `buzz-agent` 자체도 해당 기본값을 갖는다. `env.sh`와 사용자별 설정에 올바른 URL이 없다면 인증 실패/외부 접속을 유발한다. 사용자 UI의 Global Agent Config/Agent Env Vars에 `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY`를 설정하는 방법도 있다.

**최초 연결 진단** (API 키를 쉘 히스토리나 로그에 남기지 말 것):

```bash
# 이미 개인 키가 환경변수로 주입됐다는 전제
curl --silent --show-error --fail-with-body \
  -H "Authorization: Bearer $OPENAI_COMPAT_API_KEY" \
  "$OPENAI_COMPAT_BASE_URL/models" | head -c 1000
# 실제 추론은 선택한 프로토콜 /chat/completions 또는 /responses의
# 유효한 tool call 응답까지 테스트해야 한다.
```

401/403 = 키/권한, 404 = base path/API 버전/모델 또는 경로, 인증성공 후 일부 모델만 실패 = 지원 기능 차이. `BUZZ_RELAY_URL`를 바꿔도 LLM URL은 바뀌지 않는다.

## 3. Codex ACP -> DTGPT (Responses API)

**가장 중요한 조건:** 현재 Codex config reference의 `model_providers.<id>.wire_api`는 `responses`만 지원한다. DTGPT가 Chat Completions만 지원하면 Codex를 그쪽으로 바로 연결할 수 없다. 내부 `agentic-api`를 통해 `/v1/responses`로 요청을 라우팅하거나 완전한 Responses 호환 Gateway를 준비해야 한다.

사용자의 **`~/.codex/config.toml`** 샘플:

```toml
model_provider = "dtgpt"
model = "YOUR_DTGPT_RESPONSES_MODEL"
model_reasoning_effort = "medium"

[model_providers.dtgpt]
name = "DTGPT"
base_url = "https://dtgpt.example.internal/v1"
env_key = "DTGPT_API_KEY"
wire_api = "responses"
```

`DTGPT_API_KEY`는 개인별 Credential 경로에서 공급한다. `model_provider = dtgpt`와 `[model_providers.dtgpt]` 이름이 반드시 일치해야 하며, `base_url`에는 일반적으로 `/v1`까지 포함하고 `/responses`를 **중복** 작성하지 않는다.

```bash
export DTGPT_API_KEY="<YOUR_PRIVATE_DTGPT_KEY>"
# GUI를 띄우기 전 런처의 허용 전달 변수에 DTGPT_API_KEY 추가 여부 확인!
# 기본 런처는 사용자의 임의 비밀변수를 자동으로 컨테이너에 들여보내지 않음.
```

**Credential 전달 방법**: `DTGPT_API_KEY`처럼 사내에서 임의로 지정한 이름은 `--cleanenv`에서 자동으로 전달되지 않는다. 사용자는 `CODEX_API_KEY`를 `env_key`로 설정해 현재 런처의 전달목록을 재사용하거나, 별도 승인된 런처에서 `APPTAINERENV_DT... / SINGULARITYENV_DT...`로 명시적으로 전달해야 한다. **권장 기본**은 다음 형태:

```toml
[model_providers.dtgpt]
name = "DTGPT"
base_url = "https://dtgpt.example.internal/v1"
env_key = "CODEX_API_KEY"
wire_api = "responses"
```

`CODEX_API_KEY`는 `codex-acp`에서도 인식하는 자격증명이다. 추가로 로그인 화면이 표시되는 경우 **API-key 인증 또는 구성된 커스텀 Provider**를 사용하고 OpenAI.com ChatGPT 구독 로그인 경로로 가지 않는다. `CODEX_CONFIG`, `MODEL_PROVIDER`는 ACP 런타임 override를 추가로 제공하지만 기본은 개인 `config.toml`이 관리하도록 둔다.

**주의**: Codex가 `/v1/responses`로 보낸 request format, tool call, streaming, reasoning/compaction을 DTGPT agentic-api가 정확히 받아들이는지 확인한다. 단순 200 성공은 충분하지 않다. Codex의 auth 및 모델 선택 UX가 gateway custom provider로만 완전 동작하는지는 ACP/GUI E2E 검증 대상이다.

## 4. Claude ACP -> DTGPT (Anthropic Messages API)

Claude ACP는 Claude Agent SDK를 사용한다. `~/.claude/settings.json`에 아래 **비밀이 아닌** 연결 정보만 둘 수 있다.

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://dtgpt.example.internal",
    "ANTHROPIC_MODEL": "YOUR_DTGPT_ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "YOUR_DTGPT_ANTHROPIC_MODEL"
  }
}
```

개인별 API 키는 별도로:

```bash
export ANTHROPIC_AUTH_TOKEN="<YOUR_PRIVATE_DTGPT_KEY>"
# 또는 DTGPT Gateway가 x-api-key 형식만 지원하면 ANTHROPIC_API_KEY 사용.
```

`ANTHROPIC_BASE_URL`은 **Anthropic `/v1/messages` 앞의 URL**로 지정한다. Gateway 구현에 따라 `/anthropic` 등의 prefix를 붙이는 경우가 있으므로 실 API 라우팅에 맞춘다. 동일 DTGPT라도 OpenAI Chat Completions Endpoint로 Claude ACP가 바로 동작하는 건 아니다.

Claude ACP/CLI 일부 계정 로그인·토큰 확인이 vendor endpoint를 요구할 수 있어, 사용자별 `settings.json`을 읽는 것과 API 호출이 내부로 향하는 것을 E2E로 검증한다. 외부 네트워크를 차단한 상태에서 실제 한 턴 실행을 권장한다.

## 5. Self-hosted Community / BuilderLab / Empty Relay

`desktop/src-tauri/src/builderlab.rs`가 `https://app.builderlab.xyz/api/goose`를 고정 사용한다. **사내에서 별도로 BuilderLab을 배포할 필요는 없음**; 호스팅 서비스용 버튼을 사용하지 말 것.

Desktop 버전에 따라 다음과 같이 표기될 수 있다:

1. 기존 커뮤니티 연결 / Join a community / I'm a member or admin
2. `wss://buzz.internal.com`을 입력 (인증서 검증 필요)
3. Relay 소유자라면 본인 Nostr 공개키가 Helm `ownerPubkey`에 등록된 공개키인지 확인. 다른 공개키로 접속하면 Owner 권한이 없다.
4. Relay 팀에서 `relay.requireRelayMembership`, Owner 부트스트랩, `buzz-admin add-member` 등 가입 허용 정책과 초대 API 권한을 확인.
5. Desktop 접속 후 채널/프로젝트 생성; 웹 Relay 랜딩 페이지는 일반 설정 포털이 아니다.

Buzz upstream [self-hosted onboarding issue #2816](https://github.com/block/buzz/issues/2816)는 'I own the community'가 호스팅으로만 이동하던 UX를 보고한다. 릴리스별로 Self-hosted 버튼 유무가 바뀔 수 있다. [Invite UI issue #4209](https://github.com/block/buzz/issues/4209)는 self-hosted의 초대 UI가 호스팅과 동일하지 않을 수 있음을 보고한다.

**Relay팀 체크리스트**: Ingress WSS/HTTP 동시 연결, `relayUrl`, `ownerPubkey`, `requireRelayMembership`, admin CLI 초기화/초대 인증, Postgres 영속성 및 Relay readiness, CORS가 Tauri origin을 허용하는지 등을 확인한다.

## 6. 사용자별 로그·현재 실제 URL 확인

기존에는 SIF에서 GUI stdout/stderr가 터미널에 출력되지 않아 진단이 어려웠다. 새 `nfs-client/buzz-desktop`은:

- `${XDG_STATE_HOME:-$HOME/.local/state}/buzz/desktop.log`: Desktop 프로세스 표준출력/표준에러 + 기동 시 Relay/LLM Base URL (UID 전용 0600)
- `.../buzz/codex-acp/`: `APP_SERVER_LOGS` Codex ACP 앱 서버 로그
- `${XDG_DATA_HOME:-$HOME/.local/share}/xyz.block.buzz.app/logs/`: Buzz Local Agent `buzz-acp` 실행 로그 (**정확한 앱 디렉터리는 app identifier 또는 XDG 설정에 따라 달라질 수 있음**)

```bash
source /nfs/apps/buzz/env.sh
buzz
tail -f "${XDG_STATE_HOME:-$HOME/.local/state}/buzz/desktop.log"
find "${XDG_DATA_HOME:-$HOME/.local/share}" -type f -path '*/logs/*.log' | tail -20
# 앱 실행 중 터미널 출력이 필요할 때
BUZZ_LOG_FOREGROUND=1 buzz
```

코드의 Relay URL 우선순위는 **workspace override > `BUZZ_RELAY_URL` > build-time URL > `ws://localhost:3000`**. 런처 로그의 Relay URL은 환경변수 기준이므로 Desktop UI에서 직접 선택한 workspace URL이 우선 적용될 수 있다. GUI 개발자 콘솔/실제 네트워크 연결까지 점검하면 최종 라우팅 확인 가능.

`RUST_LOG=buzz_acp=debug,buzz_agent=debug`는 ACP 하위 프로세스 로그를 늘릴 수 있으나 기본 Desktop 자체의 완전한 네트워크 패킷 트레이스를 보장하지 않는다. 토큰/프롬프트 및 사내 기밀 데이터가 로그에 포함될 수 있으므로 공유 금지/보관기간 제한.

## 7. UTF-8 및 한글 입력

Definition은 `en_US.UTF-8`, `ko_KR.UTF-8` Locale을 생성하고, IBus + Hangul input engine + Noto CJK/Nanum 폰트를 포함한다. 런처는 호스트에서 지정한 `LANG` 및 `LC_ALL`을 전달하며 SIF에서 유효한 UTF-8 여부를 확인하고 잘못된 값이면 `en_US.UTF-8`로 fallback한다.

```bash
# 한글 UI/입력 환경의 한 예
export LANG=ko_KR.UTF-8
export GTK_IM_MODULE=ibus
export QT_IM_MODULE=ibus
export XMODIFIERS=@im=ibus
source /nfs/apps/buzz/env.sh
buzz
```

Host X session의 IBus 데몬을 사용자 D-Bus를 통해 공유하는 것을 우선한다. 호스트에 IBus가 없거나 세션에 데몬이 없으면 **선택적으로** 아래 사용 (한 사용자 X session에서 중복 시작 주의):

```bash
export BUZZ_START_IBUS=1
buzz
```

`ibus-daemon -drx`는 SIF 내부 GUI 실행 전에 시작된다. 다만 실제 X 서버 인증, 사용자 D-Bus session/socket bind, 키보드 엔진 및 compositor가 연결되지 않으면 입력이 되지 않을 수 있으므로 실제 호스트에서 한글 조합/지움/자동완성 테스트가 필요하다.

## 근거

- [Buzz onboarding connection methods](https://github.com/block/buzz/blob/main/desktop/src/features/onboarding/ui/harnessConnectionOptions.ts)
- [Buzz OpenAI model discovery URL](https://github.com/block/buzz/blob/main/desktop/src-tauri/src/commands/agent_models.rs)
- [Buzz Agent OpenAI compatible config](https://github.com/block/buzz/blob/main/crates/buzz-agent/README.md)
- [Buzz BuilderLab constant](https://github.com/block/buzz/blob/main/desktop/src-tauri/src/builderlab.rs)
- [Buzz local agent logs](https://github.com/block/buzz/blob/main/desktop/src-tauri/src/managed_agents/storage.rs)
- [Codex ACP package](https://github.com/agentclientprotocol/codex-acp), [Claude ACP package](https://github.com/agentclientprotocol/claude-agent-acp)
- [Codex config reference](https://developers.openai.com/codex/config-reference), [Claude Code LLM gateway](https://docs.anthropic.com/en/docs/claude-code/llm-gateway)
