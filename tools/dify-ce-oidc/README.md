# Dify CE Console OIDC integration (reference 1.17.1)

Updated: 2026-11-24 KST  
Status: implementation candidate, upstream source inspected; target integration/build/security review pending.

## Scope

Standalone OIDC relying party for single-workspace Community Edition Console sign-in. Does not enable or modify Enterprise SSO, license gates, branding, RBAC, or WebApp authentication. The [Dify License](https://github.com/langgenius/dify/blob/1.17.1/LICENSE) restricts multi-workspace use without written permission and preserves frontend branding. Review applicable license/contract before use.

This is an overlay for the public CE source, not a reverse proxy that simply trusts X-User headers or a replacement for Dify session tokens. No new relational DB schema. Existing account ID, workspace membership, app ownership and permissions remain intact. Pre-provisioned identities only; no automatic email linking, JIT user provisioning, group/role sync, SAML, Enterprise SSO enforcement, WebApp end-user SSO, SCIM or IdP logout. Existing local login remains available for break-glass.

## Source audit

| Source (CE 1.17.1) | Role |
|---|---|
| [controllers/console/auth/oauth.py](https://github.com/langgenius/dify/blob/1.17.1/api/controllers/console/auth/oauth.py) | Existing OAuth redirect and Dify Console cookie issuance |
| [libs/oauth.py](https://github.com/langgenius/dify/blob/1.17.1/api/libs/oauth.py) | GitHub/Google implementations and OAuth metadata state; not reused for OIDC security transaction |
| [extensions/ext_application_services.py](https://github.com/langgenius/dify/blob/1.17.1/api/extensions/ext_application_services.py) | Provider registration and account OAuth orchestration; existing gateway not extended because callback lacks our nonce/PKCE transaction contract |
| [services/account_service.py](https://github.com/langgenius/dify/blob/1.17.1/api/services/account_service.py) | AccountService.login, update_login_info, issue_token_pair and native refresh store |
| [libs/token.py](https://github.com/langgenius/dify/blob/1.17.1/api/libs/token.py) | Access/refresh/CSRF cookie setters; preserve existing cookie policy |
| [extensions/ext_login.py](https://github.com/langgenius/dify/blob/1.17.1/api/extensions/ext_login.py) | Existing subsequent Console request authentication |
| [extensions/ext_redis.py](https://github.com/langgenius/dify/blob/1.17.1/api/extensions/ext_redis.py) | Prefix-aware Redis set(nx/ex), atomic getdel |
| [models/account.py](https://github.com/langgenius/dify/blob/1.17.1/api/models/account.py) | AccountStatus, TenantStatus, TenantAccountJoin |
| [controllers/console/__init__.py](https://github.com/langgenius/dify/blob/1.17.1/api/controllers/console/__init__.py) | Explicit resource imports and /console/api namespace |
| [web/app/signin/normal-form.tsx](https://github.com/langgenius/dify/blob/1.17.1/web/app/signin/normal-form.tsx) | Existing sign-in UI; adds independent runtime-discovered OIDC button |
| [libs/passport.py](https://github.com/langgenius/dify/blob/1.17.1/api/libs/passport.py) | Existing PyJWT dependency and Dify Passport signing, separate from IdP ID Token |

The overlay uses existing httpx, PyJWT, cryptography, Redis and SQLAlchemy runtime dependencies. Confirm exact installed PyJWT/cryptography versions and lockfile in the real target environment before build.

## Flow

~~~mermaid
sequenceDiagram
  participant B as Browser
  participant C as CE Console API
  participant R as Redis
  participant I as OIDC IdP
  participant D as Dify PostgreSQL
  B->>C: GET /console/api/ce-oidc/login
  C->>I: Discovery (configured HTTPS issuer)
  C->>R: SET NX EX 300 (state + browser-bound key)
  C-->>B: HttpOnly transaction cookie + 302 (state, nonce, S256)
  B->>I: Authorization request
  I-->>B: Redirect code + state
  B->>C: GET /console/api/ce-oidc/callback
  C->>R: GETDEL (single-use transaction)
  C->>I: POST token (code, verifier, client authentication)
  C->>I: GET JWKS
  C->>C: Validate ID Token signature/iss/aud/exp/iat/nonce/azp
  C->>C: Map configured issuer + sub to approved account UUID
  C->>D: Confirm active account and single approved workspace
  C->>C: AccountService.login and native cookie setters
  C-->>B: Dify Access, Refresh, CSRF cookies + 302 /signin
~~~

## Deployment prerequisites and configuration

- Target CE release 1.17.1. Port explicitly for any other tag. Existing migrated account must be active and have exactly one normal workspace membership matching the configured workspace ID. No JIT membership changes.
- Redis >= 6.2 with GETDEL supported in the actual cluster/ACL configuration. All API replicas use the same Redis namespace.
- Same-origin HTTPS Console API and Web. Set CONSOLE_API_URL and CONSOLE_WEB_URL explicitly; API URL is the base before /console/api. Dify native cookie domain and HTTPS settings must be valid. TLS termination must preserve public URL; IdP must be able to redirect browser to it.
- Confidential OIDC client supporting Authorization Code, query response mode, PKCE S256, RS256 ID Token, client_secret_basic or client_secret_post. This implementation intentionally requires discovery to advertise S256 and RS256; all authorization/token/JWKS endpoints must use the issuer's HTTPS origin. Additional audiences or provider endpoints on other hosts are rejected. Validate actual Keycloak/Entra/other IdP metadata; not every provider configuration is supported without adaptation.
- Register exact callback: https://dify.example/console/api/ce-oidc/callback (adjust for approved base path). Disable implicit/password grants. Do not use wildcard redirect URI. Do not enable offline_access; IdP tokens are not persisted.
- API pods need trusted CA and network egress only to the intended IdP endpoints; keep TLS verification on. Configure HTTPX environment (SSL_CERT_FILE, HTTPS_PROXY, NO_PROXY) if needed. Use a trusted CA bundle that includes necessary roots. Avoid sensitive request/query logging in all ingress/proxy/tracing layers.

API env example (place actual secret in Kubernetes Secret or approved secret manager; placeholders are not real values):

~~~ini
CONSOLE_API_URL=https://dify.example
CONSOLE_WEB_URL=https://dify.example
CE_OIDC_ENABLED=true
CE_OIDC_ISSUER=https://idp.example/realms/example
CE_OIDC_CLIENT_ID=dify-console
CE_OIDC_CLIENT_SECRET=<from-secret-manager>
CE_OIDC_TOKEN_AUTH_METHOD=client_secret_basic
CE_OIDC_TENANT_ID=<existing-workspace-uuid>
CE_OIDC_BINDINGS_FILE=/run/dify-oidc/bindings.json
~~~

For K8s, inject client secret via env.valueFrom.secretKeyRef; mount the binding file from a protected Secret or equivalent read-only volume on API pods. Do not use a public ConfigMap or repo for real subjects/account IDs. Avoid subPath mounts if relying on projected volume updates. Check file changes propagate to every replica. Roll changes through a controlled deployment, do not rely on inconsistent replica-local identity maps. The frontend discovers enablement at runtime via /console/api/ce-oidc/config, so no client secret in frontend build or browser.

### Provision existing accounts

The authoritative identity is issuer + subject, not an email address. Configure the same issuer string (including trailing slash semantics) as the IdP discovery issuer. Obtain subject for the registered client from the trusted IdP or approved identity inventory (pairwise subject may differ by client). Match against existing Dify account UUID using an administrator-approved mapping, not a user-supplied email. Store mappings as a secret, validate changes by a second operator.

~~~json
{
  "issuer": "https://idp.example/realms/example",
  "subjects": {
    "example-idp-subject": "00000000-0000-0000-0000-000000000001"
  }
}
~~~

The overlay rejects duplicate JSON keys and duplicate account assignments. Unknown subject, wrong issuer, missing/inactive account or unmatched workspace fail closed. No role elevation or automatic account creation. Maximum binding file 4 MiB; for very large installations design a governed indexed identity table and migration rather than raising bounds blindly.

Read-only inventory SQL for an administrator (keep results internal):

~~~sql
SELECT a.id, a.email, a.status, j.tenant_id, j.role, t.status AS tenant_status
FROM accounts a JOIN tenant_account_joins j ON a.id=j.account_id
JOIN tenants t ON t.id=j.tenant_id
WHERE a.status='active' AND t.status='normal'
ORDER BY a.id, j.tenant_id;
~~~

## Apply, test, build

From separate temp and upstream Dify checkouts:

~~~bash
git clone https://github.com/langgenius/dify.git
git -C dify checkout 1.17.1
git -C dify switch -c ce-oidc-1.17.1
python3 temp/tools/dify-ce-oidc/apply.py dify --check
python3 temp/tools/dify-ce-oidc/apply.py dify
git -C dify status --short
git -C dify diff --check
git -C dify diff
# Compile and run protocol tests using the target API Python environment:
python -m compileall -q temp/tools/dify-ce-oidc
python temp/tools/dify-ce-oidc/test_protocol.py
~~~

Run with installed httpx, PyJWT and cryptography (use the Dify API's pinned environment). The installer verifies tag commit and clean checkout, validates exact upstream anchors, copies two Python modules and patches the Console import and sign-in UI. The new files are untracked until staged. Review them too. Commit local changes to an internal fork, preserve Dify license/branding notices. Rebuild the API and Web images according to the checked-out tag's build instructions and use unique immutable image tags/digests. No worker-specific patch required; if using the same API image for workers, retain normal deployment entrypoints. Test the runtime config endpoint and route registration after build. For deployment, set CE_OIDC_ENABLED=false until tested. Do not replace EE image or modify Enterprise licensing.

The installer is intentionally not auto-rebaseable. For later tags compare account/session API, Redis helper, Console routes, UI anchors and feature flags; run the full regression suite.

### Acceptance matrix

Unit tests in test_protocol.py cover PKCE RFC 7636 vector, state/browser key binding, authorization parameters, valid signed token, forged signature, wrong issuer/audience/azp/nonce, expired/future-issued token, bad at_hash, wrong algorithm/kid, duplicate kid, metadata endpoint origin/PKCE downgrade, token exchange and identity mapping.

Integration tests (must run against a real isolated CE deployment and IdP):

| Test | Expected |
|---|---|
| OIDC disabled, wrong edition | No button; direct endpoint refuses login |
| Normal pre-bound user | Existing account UUID, native access/refresh/CSRF cookies, no new account or workspace |
| Second API replica handles callback | Shared Redis transaction works |
| Callback without cookie/state, duplicate params, stale state | 401, no session issued |
| Repeated/concurrent callback | At most one successful state consumption; Redis GETDEL atomic |
| Wrong code_verifier, wrong nonce, forged JWT, key rotation | Reject invalid, accept valid current signing key |
| Wrong issuer, wrong client, multi-audience | Reject |
| Unbound sub, same email but different sub, banned/closed/pending user | Reject |
| Missing/archived workspace, multiple normal memberships | Reject |
| Permission test owner/editor/normal and anonymous | Existing permissions unchanged; no new access grant |
| Cookie flags | Transaction Secure/HttpOnly/SameSite=Lax; Dify native cookie policy preserved |
| Refresh/logout | Existing Dify flow works; local logout clears native cookies |
| IdP outage/Redis outage | Fail closed; break-glass local login still available |
| CE upgrade | Reapply only after source audit and regression tests |

Also check callback response after an expired pre-existing Dify cookie; verify application-wide request hooks do not require a Console session on these public routes. Test custom reverse proxy path, app domain, rate limiting, no callback query/body logging, and license compliance. Never deploy a successful protocol test alone as proof of a working full build.

## Security/operational limitations

- Native Dify session lifetime is separate from IdP lifetime. Removing a binding blocks new OIDC logins, not necessarily already-issued Dify sessions. IdP user disable/logout is not propagated automatically. Enforce offboarding in Dify and revoke refresh/session state using approved mechanisms; already-issued JWT lifetime and loader behavior require verification. Shorten local token lifetimes under organizational policy, and design back-channel logout/revalidation separately if required.
- CE login route only. Does not protect published App/WebApp anonymous access or replace app RBAC. If EE originally enforced SSO for WebApps, that is a distinct migration gap.
- No global SSO-only enforcement. Keep emergency local admin login, preferably with additional ingress protection. Do not disable password fallback until security and availability tests pass. Existing password endpoints retain their original policy.
- State uses Redis TTL 300 seconds, one browser cookie per active flow, atomic GETDEL; starting a new login in the same browser invalidates access to the prior flow. Rate-limit login/callback per client/IP at ingress and cap provider request/response sizes. Discovery and JWKS are fetched each callback; no unsafe stale key cache, but increased provider dependency. No provider-supplied redirects followed; endpoints same-origin pinned by configuration. Production egress restrictions are essential.
- No raw tokens or raw exceptions logged by overlay; HTTP/access/trace middleware may independently log sensitive callback URLs or token traffic and must be configured to redact. Protect binding and client secrets.
- This is a minimal strict implementation (RS256, single audience, same-origin metadata and Console, single tenant). Extend using standards and tests rather than removing verification checks for provider compatibility.
- Source verification and test creation are completed; upstream checkout/container execution, real IdP login, frontend build and security review are NOT completed here. Local execution environment cannot fetch the repository over the network. No production-ready claim.

## Rollback

Set CE_OIDC_ENABLED=false on all API replicas; the config route hides the button and login/callback refuse use. Existing sessions issued by Dify may remain valid until explicit revocation/expiry. Remove overlay by reverting the patch commit in the internal Dify fork, rebuild API/Web and deploy approved base images. No new DB table or data rewrite to reverse; ordinary login audit timestamps and native session state remain. Retain original Dify login path throughout.

## References

- [OIDC Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html)
- [OIDC Discovery 1.0](https://openid.net/specs/openid-connect-discovery-1_0.html)
- [RFC 7636: PKCE](https://www.rfc-editor.org/rfc/rfc7636)
- [RFC 9700: OAuth Security BCP](https://www.rfc-editor.org/rfc/rfc9700)
- [PyJWT API](https://pyjwt.readthedocs.io/en/stable/api.html)
- [Redis GETDEL](https://redis.io/docs/latest/commands/getdel/)
