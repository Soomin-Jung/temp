# Buzz HPC 브라우저·OIDC·GitHub Enterprise Server 설계 검토

업데이트: 2026-10-19 KST · 상태: **업스트림 소스 정적 검토 / HPC 실제 런타임 미검증**  
기준: `block/buzz` commit 3debf18f1448334aa7bc202d328d93bb9651e168, `block/buzz-app` 현재 공개 구조.

## 결론

1. **내부 DNS는 프로토콜 필수가 아니다**. Helm chart의 `relayUrl` 자체는 필수이며, 공유 브라우저 서비스는 내부 FQDN과 HTTPS/WSS를 권장한다. 기업 OIDC/NIP-FI를 사용하면 정확한 HTTPS host/audience 매칭 때문에 FQDN을 사실상 표준으로 고정하는 것이 유리하다.
2. **TLS 파일을 Relay에 직접 마운트하지 않는다**. Ingress TLS Termination 시 Kubernetes `kubernetes.io/tls` Secret(`tls.crt`/`tls.key`)을 Controller가 참조하며, Relay는 HTTP :3000으로 동작한다. 사내 CA는 사용자 브라우저·Agent·Ingress egress 각각에 필요할 수 있다.
3. **공식 Relay 이미지의 Web UI는 전체 협업 앱이 아니다**. `web/src/app/routes.ts`에는 초대 링크, Repo 목록·상세·파일 Browser만 있고, `ReposPage.tsx`는 Desktop 실행 유도 UI를 제공한다. 편집/Agent 관리/서버형 GHES 통합을 기대하면 안 된다.
4. **별도 `block/buzz-app`은 공유 React Frontend지만 독립 Production Browser Backend/Sign-in 완제품이 아니다**. `docs/contributing.md`는 static frontend build에 standalone browser login/backend가 없음을 명시. `browser-host/`는 개발용 Node Broker다.
5. **Buzz 서버는 GitHub Enterprise Server에 계정을 연결하여 Private Repo를 관리하는 네이티브 Connector가 아니다**. NIP-34 외부 저장소 링크 메타데이터는 가능하나, GHES Git/API 인증·PR/Issue 동기화·사용자별 권한 위임은 별도 Agent/Connector가 수행해야 한다.
6. **OIDC 자체 로그인 완제품은 아니다**. Relay NIP-FI Issuer/JWKS Assertion 검증은 구현되어 있지만 사용자 로그인, Assertion issuer/Identity Adapter, 브라우저 Nostr 신원 연결은 별도 제공이 필요하다. NIP-FI 자체에서 일반 OIDC ID Token은 거부한다.
7. **기본 Helm Chart에는 Agent Runtime Deployment, GHES Connector, Identity Adapter가 없다.** 별도 Kubernetes 워커/Service/Secret 구성이 필요하다.

## 후속 결정: Browser-only 대신 X 세션 + Local Agent

기존 검토는 **브라우저에서 모든 개발/Agent 관리를 수행하려는 요구**를 평가한 결과이다. 대안으로 **Linux X 세션에서 공식 Desktop을 실행**하면 Browser UI 및 Desktop을 대체할 Control Plane을 직접 개발할 필요가 줄어든다.

- Desktop 및 Local `buzz-acp`/Coding Agent는 **사용자의 Linux X 세션 호스트**에서 실행한다. 버전별 Desktop AppImage만 NFS 공용 설치하고 사용자별 `~/.buzz`/XDG 데이터, OS Keyring, Git workspace를 분리한다.
- Kubernetes에는 기존 Helm 구성인 **Buzz Relay + Postgres + Redis + S3**를 배포한다. 사용자가 Pod를 만들거나 Kubernetes RBAC을 획득할 필요는 없다.
- Local Agent는 사내 vLLM/LiteLLM API를 직접 호출하고 **사용자별 git/gh 인증**으로 GHES에 접근할 수 있다. 단, Buzz의 네이티브 GHES 프로젝트 UI, 전체 사내 SSO는 여전히 제공되지 않는다.
- 이 대안의 선결 조건은 Linux 호스트 배포판, glibc/GLib/FUSE, NFS exec, GUI 세션 Keyring 및 X 서버의 사내 네트워크 접근이다.
- 실무 절차: **[Local Agent + NFS Desktop 배포 가이드](local-agent-nfs-desktop.md)**. 이는 원격 Kubernetes Agent Pod 아키텍처를 권장 기본값에서 제외한다는 뜻이지, 해당 옵션이 불가능하다는 뜻은 아니다.

## 인증·접속 계층

| 계층 | 기능 | 담당 |
| --- | --- | --- |
| HTTPS/TLS | 사내 DNS와 서버 신원 확인 | Ingress TLS Secret + 기업 CA |
| Web SSO (선택) | OIDC IdP로 접속 사용자 1차 인증 | oauth2-proxy 등 Ingress 앞단 외부 제품 |
| Nostr | NIP-42 WebSocket, NIP-98 REST 서명 | Buzz Client + Relay |
| NIP-FI (선택) | Issuer별 JWT/JWS Assertion 검증 | Buzz Relay + 별도 Identity Adapter/Issuer |
| GHES 인증 | Private Repo Clone/Push, PR/Issue REST API | 별도 GitHub App/사용자 Credential + Agent Service |

**SSO 프록시 통과만으로 Nostr Identity를 얻지 않는다.** Browser Web UI의 `web/src/shared/lib/nostr-signer.ts`는 영속 가입에 NIP-07 확장을 요구하고, 읽기 fallback은 탭 수명 임시 키이다.

### NIP-FI와 브라우저 간 충돌

- `BUZZ_NIP_FI_MODE=off`가 기본. `enforce`/`shadow`는 `BUZZ_NIP_FI_ISSUERS`, `BUZZ_NIP_FI_COMMUNITIES`, `BUZZ_NIP_FI_MAX_CONNECTION_LIFETIME_SECS` 설정과 JWKS 접근성이 필요하다.
- Relay NIP-FI Enforce는 WebSocket Upgrade 요청의 `Nostr-Federated-Identity: Bearer <compact-JWS>`를 사용한다.
- 브라우저 표준 `new WebSocket(url)` API로 임의 HTTP Header를 붙일 수 없다. 공식 번들 Web Client도 별도 Assertion 전송 경로가 없다. 따라서 지금 그대로 Enforce를 켜면 Browser 호환성을 보장할 수 없으며, 보안 검증된 별도 Transport/Proxy 설계와 사용자키 매핑 구현이 필요하다.
- NIP-FI는 OIDC ID Token을 직접 받아들이지 않는다. `nip-fi+jwt` 또는 엄격한 `at+jwt` 정책과 사용자 Nostr key binding을 전제로 한다.
- NIP-FA는 Adapter가 OIDC/SAML/LDAP를 사용해도 클라이언트에 특정 IdP 구현을 노출하지 않는 계약이다. `docs/federated-identity-adapter.md`에는 Desktop 쪽 로그인 완료 경로 상당 부분이 `Planned`로 표기되어 있다.
- 폐쇄망 NIP-FI 도입 시 JWKS URI는 **내부 접속 주소**와 신뢰할 수 있는 TLS/서비스메시 인증 채널을 사용해야 하며, 인증서/Issuer Audience/Host/Key Rotation 검증이 필요하다.

## GitHub Enterprise Server: 서버 연동 방식

대상은 `github.com`으로 접속하는 Enterprise Cloud가 아니라 `https://ghe.example.internal`을 사용하는 온프레미스 GHES이다.

| 요구 기능 | 현재 Buzz | 추가 필요 작업 |
| --- | --- | --- |
| GHES Repo 링크 메타데이터 | `clone` 태그 지원 | 사용자 접속 시 GHES 링크만 보임 |
| Public GitHub UI Reference | `block/buzz-app` Github Plugin 제공 | 플러그인은 `github.com`, `api.github.com` 하드코딩·무인증 요청 |
| GHES Private Repository 서버 Clone | 공식 Relay 내 네이티브 지원 확인 안 됨 | 별도 Agent Worker에서 Git HTTPS+토큰 |
| GHES PR/Issue/CI | Buzz 자체 Nostr Git/Workflow 모델과 별개 | GHES App Connector + Webhook/REST Mapper |
| 사용자별 계정 연결 | 사용자별 GHES OAuth 연동 흐름 확인 안 됨 | OAuth App/User token consent, 안전한 토큰 저장 및 revocation |
| 조직 서비스계정 기반 자동화 | Buzz 기본 기능 아님 | GHES GitHub App installation token 권장 |
| 브라우저에서 Agent 실행 제어 | 번들 Web UI에는 없음 | 새 Browser Console + Agent Control Plane 필요 |

운영 적합한 기본 토폴로지:

`Browser → (OIDC Gateway 선택) → Buzz Relay → Nostr Events → Agent Worker Pod → GHES App/REST/Git → Internal GHES`

- Agent Worker는 `buzz-acp`/`buzz-agent` 등의 별도 실행 환경, 내부 LLM API, `git`/`gh`, GHES CA, Secret/RBAC, repo workspace를 요구한다.
- Buzz Relay Helm 차트의 이미지 하나를 Pull해도 Agent Worker는 자동 생성되지 않는다. `docs/remote-agents.md`는 Kubernetes Agent Pod provider 모델을 설명하지만 기본 차트의 구성 요소가 아니다.
- GHES의 System of Record를 유지하고, Worker만 Push/PR/Issue를 수행하도록 한다. 사용자가 버즈 웹 앱에서 자동화 과정을 제어하려면 별도 UI와 Control API가 필요하다.
- GitHub App 토큰은 서버 단위로 모든 Repo의 권한을 갖게 하지 말고 설치 범위/Repo/권한을 최소화한다. 각 사용자 권한 위임과 조직 App 권한 모델은 분리 검토한다.

## 소규모 PoC에서 확인할 질문

1. Browser Web UI에서 요구하는 기능이 **읽기/Repo 조회**인지, **메시징+Agent 작업 지시/개발/PR 관리**인지? 후자면 기본 `block/buzz` Helm만으로 불가.
2. GHES Git/REST API 호스트, App/PAT 사용정책, Webhook 도달성, 내부 CA, 사용자별 권한 격리 요구사항 확정.
3. ID Provider(OIDC)의 Token Issuer/Audience, Callback, JWKS, Nostr 키 Provisioning / revoke 정책 결정.
4. 기존 사내 Ingress Controller에서 WSS upgrade, JWT/OIDC 프록시, `Host` 원본 전달, TLS Termination, timeouts 확인.
5. `BUZZ_NIP_FI_MODE=off` + 제한된 업무 PoC부터 시작; `shadow`/`enforce` 도입은 브라우저 인증 경로 구현 후 별도 검증.

## 근거 (upstream source)

- [Browser routes](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/web/src/app/routes.ts), [Desktop 유도 버튼](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/web/src/features/repos/ui/ConnectButton.tsx), [Web Nostr Signer](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/web/src/shared/lib/nostr-signer.ts)
- [Shared buzz-app README](https://github.com/block/buzz-app), [browser host limitations](https://github.com/block/buzz-app/blob/main/docs/contributing.md#shared-logic-and-host-boundaries), [GitHub plugin parser](https://github.com/block/buzz-app/blob/main/src/bundled/github/references.ts), [GitHub API](https://github.com/block/buzz-app/blob/main/src/bundled/github/data.ts)
- [Buzz NIP-FI config](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/crates/buzz-relay/src/nip_fi_config.rs), [NIP-FI](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/docs/nips/NIP-FI.md), [NIP-FA](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/docs/nips/NIP-FA.md)
- [Remote Agent Kubernetes provider spec](https://github.com/block/buzz/blob/3debf18f1448334aa7bc202d328d93bb9651e168/docs/remote-agents.md), [GHES REST authentication](https://docs.github.com/en/enterprise-server@3.19/rest/authentication/authenticating-to-the-rest-api)
- [Standard Browser WebSocket constructor](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/WebSocket)
