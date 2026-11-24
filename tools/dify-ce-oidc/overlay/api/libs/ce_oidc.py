"""Standalone CE OIDC relying-party protocol helpers; no Enterprise imports."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus, urlencode, urlsplit
from uuid import UUID

import httpx
import jwt

TTL = 300
OPAQUE = re.compile(r"^[A-Za-z0-9_-]{43}$")
ALGORITHM = "RS256"


class OIDCError(Exception):
    """Public errors must never include raw token/provider response."""


def origin(url: str) -> tuple[str, str, int]:
    p = urlsplit(url)
    if (p.scheme != "https" or not p.hostname or p.username is not None
            or p.password is not None or p.fragment):
        raise OIDCError("HTTPS URL required")
    return p.scheme, p.hostname, p.port or 443


@dataclass(frozen=True)
class Settings:
    issuer: str
    client_id: str
    client_secret: str
    redirect_uri: str
    console_url: str
    bindings: Path
    tenant_id: str
    auth_method: str = "client_secret_basic"

    @classmethod
    def from_env(cls, console_api: str, console_web: str) -> Settings:
        issuer = os.environ["CE_OIDC_ISSUER"]
        origin(issuer)
        if urlsplit(issuer).query:
            raise OIDCError("issuer query forbidden")
        origin(console_api)
        origin(console_web)
        if urlsplit(console_api).query or urlsplit(console_web).query:
            raise OIDCError("console URL query forbidden")
        # This overlay deliberately supports the same-origin console topology.
        if origin(console_api) != origin(console_web):
            raise OIDCError("same-origin console API and web required")
        client_id = os.environ["CE_OIDC_CLIENT_ID"]
        client_secret = os.environ["CE_OIDC_CLIENT_SECRET"]
        method = os.getenv("CE_OIDC_TOKEN_AUTH_METHOD", "client_secret_basic")
        if not client_id or not client_secret or method not in (
            "client_secret_basic", "client_secret_post"
        ):
            raise OIDCError("invalid client configuration")
        return cls(
            issuer=issuer, client_id=client_id, client_secret=client_secret,
            redirect_uri=console_api.rstrip("/") + "/console/api/ce-oidc/callback",
            console_url=console_web.rstrip("/"),
            bindings=Path(os.environ["CE_OIDC_BINDINGS_FILE"]),
            tenant_id=str(UUID(os.environ["CE_OIDC_TENANT_ID"])),
            auth_method=method,
        )


def opaque() -> str:
    return secrets.token_urlsafe(32)


def state_key(state: str, browser: str) -> str:
    if not OPAQUE.fullmatch(state) or not OPAQUE.fullmatch(browser):
        raise OIDCError("invalid transaction")
    return "ce_oidc:transaction:" + hashlib.sha256(
        (state + ":" + browser).encode("ascii")
    ).hexdigest()


def pkce_challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")


def new_transaction() -> tuple[str, str, dict[str, str]]:
    return opaque(), opaque(), {"nonce": opaque(), "verifier": opaque()}


def _json(response: httpx.Response) -> dict:
    response.raise_for_status()
    # Response size cap for parsing; transport-level limits also recommended.
    if len(response.content) > 1024 * 1024:
        raise OIDCError("oversized provider response")
    data = response.json()
    if not isinstance(data, dict):
        raise OIDCError("invalid provider response")
    return data


def http_client() -> httpx.Client:
    # TLS verification stays enabled. Configure trusted corporate CA via SSL_CERT_FILE.
    return httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0), follow_redirects=False)


def discover(client: httpx.Client, cfg: Settings) -> dict:
    meta = _json(client.get(
        cfg.issuer.rstrip("/") + "/.well-known/openid-configuration",
        headers={"Accept": "application/json"},
    ))
    if meta.get("issuer") != cfg.issuer:
        raise OIDCError("issuer mismatch")
    for name in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
        endpoint = meta.get(name)
        # Narrow policy for this overlay: same origin as configured issuer only.
        if not isinstance(endpoint, str) or origin(endpoint) != origin(cfg.issuer):
            raise OIDCError("unexpected endpoint origin")
    if "S256" not in meta.get("code_challenge_methods_supported", []):
        raise OIDCError("provider must advertise PKCE S256")
    if ALGORITHM not in meta.get("id_token_signing_alg_values_supported", []):
        raise OIDCError("provider must support RS256")
    if cfg.auth_method not in meta.get("token_endpoint_auth_methods_supported", ["client_secret_basic"]):
        raise OIDCError("unsupported token endpoint authentication")
    return meta


def authorization_url(cfg: Settings, meta: dict, state: str, txn: dict) -> str:
    endpoint = meta["authorization_endpoint"]
    params = {
        "response_type": "code", "response_mode": "query",
        "client_id": cfg.client_id, "redirect_uri": cfg.redirect_uri,
        "scope": "openid", "state": state, "nonce": txn["nonce"],
        "code_challenge": pkce_challenge(txn["verifier"]),
        "code_challenge_method": "S256",
    }
    # Discovery URLs with an existing query are not supported by this overlay.
    if urlsplit(endpoint).query:
        raise OIDCError("authorization endpoint query unsupported")
    return endpoint + "?" + urlencode(params)


def exchange(client: httpx.Client, cfg: Settings, meta: dict, code: str, txn: dict) -> dict:
    if not code or len(code) > 8192:
        raise OIDCError("invalid code")
    data = {
        "grant_type": "authorization_code", "code": code,
        "redirect_uri": cfg.redirect_uri, "code_verifier": txn["verifier"],
    }
    auth = None
    if cfg.auth_method == "client_secret_basic":
        auth = httpx.BasicAuth(quote_plus(cfg.client_id), quote_plus(cfg.client_secret))
    else:
        data.update(client_id=cfg.client_id, client_secret=cfg.client_secret)
    tokens = _json(client.post(meta["token_endpoint"], data=data, auth=auth,
                               headers={"Accept": "application/json"}))
    if not isinstance(tokens.get("id_token"), str) or len(tokens["id_token"]) > 65536:
        raise OIDCError("missing or oversized ID token")
    return tokens


def validate_id_token(cfg: Settings, tokens: dict, jwks: dict, nonce: str) -> str:
    token = tokens["id_token"]
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    if (header.get("alg") != ALGORITHM or not isinstance(kid, str) or not kid
            or "crit" in header):
        raise OIDCError("unexpected token header")
    keys = jwks.get("keys")
    if not isinstance(keys, list):
        raise OIDCError("invalid JWKS")
    matches = [k for k in keys if isinstance(k, dict) and k.get("kid") == kid
               and k.get("kty") == "RSA" and k.get("use", "sig") == "sig"
               and k.get("alg", ALGORITHM) == ALGORITHM
               and ("key_ops" not in k or "verify" in k["key_ops"])]
    if len(matches) != 1:
        raise OIDCError("ambiguous or unknown signing key")
    key = jwt.PyJWK.from_dict(matches[0], algorithm=ALGORITHM).key
    if key.key_size < 2048:
        raise OIDCError("weak RSA key")
    claims = jwt.decode(
        token, key, algorithms=[ALGORITHM], issuer=cfg.issuer,
        audience=cfg.client_id, leeway=30,
        options={"require": ["iss", "sub", "aud", "exp", "iat", "nonce"]},
    )
    aud = claims["aud"]
    audiences = [aud] if isinstance(aud, str) else aud
    # Strict audience policy: no additional audiences trusted by this client.
    if audiences != [cfg.client_id]:
        raise OIDCError("unexpected additional audience")
    if "azp" in claims and claims["azp"] != cfg.client_id:
        raise OIDCError("authorized party mismatch")
    received_nonce = claims["nonce"]
    if not isinstance(received_nonce, str) or not hmac.compare_digest(
        received_nonce.encode("utf-8"), nonce.encode("ascii")
    ):
        raise OIDCError("nonce mismatch")
    if "at_hash" in claims:
        access = tokens.get("access_token")
        if not isinstance(access, str):
            raise OIDCError("missing access token for at_hash")
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(access.encode("ascii")).digest()[:16]
        ).rstrip(b"=").decode("ascii")
        if not isinstance(claims["at_hash"], str) or not hmac.compare_digest(
            claims["at_hash"].encode("utf-8"), expected.encode("ascii")
        ):
            raise OIDCError("access token hash mismatch")
    sub = claims["sub"]
    if not isinstance(sub, str) or not sub or len(sub.encode("utf-8")) > 255:
        raise OIDCError("invalid subject")
    return sub


def authenticate(cfg: Settings, code: str, txn: dict) -> str:
    with http_client() as client:
        meta = discover(client, cfg)
        tokens = exchange(client, cfg, meta, code, txn)
        # Fresh JWKS for each callback supports key rotation; no header-supplied URLs.
        jwks = _json(client.get(meta["jwks_uri"], headers={"Accept": "application/json"}))
        return validate_id_token(cfg, tokens, jwks, txn["nonce"])


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise OIDCError("duplicate binding key")
        result[key] = value
    return result


def resolve_binding(cfg: Settings, sub: str) -> str:
    # Read on each login so operator updates do not need process-local cache refresh.
    with cfg.bindings.open("rb") as handle:
        raw = handle.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise OIDCError("oversized bindings")
    mapping = json.loads(raw, object_pairs_hook=_unique_pairs)
    if not isinstance(mapping, dict) or mapping.get("issuer") != cfg.issuer:
        raise OIDCError("binding issuer mismatch")
    subjects = mapping.get("subjects")
    if not isinstance(subjects, dict) or sub not in subjects:
        raise OIDCError("unprovisioned subject")
    # Reject ambiguous operator mappings. No email fallback or JIT provisioning.
    account_ids = [str(UUID(value)) for value in subjects.values()]
    if len(set(account_ids)) != len(account_ids):
        raise OIDCError("duplicate account binding")
    return str(UUID(subjects[sub]))
