"""Protocol-level negative tests; run in target Dify API environment."""
import base64
import json
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

sys.path.insert(0, str(Path(__file__).resolve().parent / "overlay" / "api"))
from libs.ce_oidc import (  # noqa: E402
    OIDCError, Settings, authenticate, authorization_url, discover, exchange,
    new_transaction, pkce_challenge, resolve_binding, state_key, validate_id_token,
)


def b64int(n):
    return base64.urlsafe_b64encode(
        n.to_bytes((n.bit_length() + 7) // 8, "big")
    ).rstrip(b"=").decode()


class OIDCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        numbers = cls.key.public_key().public_numbers()
        cls.jwks = {"keys": [{"kty": "RSA", "kid": "test", "use": "sig",
                             "alg": "RS256", "n": b64int(numbers.n), "e": b64int(numbers.e)}]}
        cls.cfg = Settings("https://idp.example/realm", "dify", "test-secret",
                           "https://dify.example/console/api/ce-oidc/callback",
                           "https://dify.example", Path("/nonexistent"),
                           "00000000-0000-0000-0000-000000000001")
        cls.meta = {
            "issuer": cls.cfg.issuer,
            "authorization_endpoint": "https://idp.example/auth",
            "token_endpoint": "https://idp.example/token",
            "jwks_uri": "https://idp.example/keys",
            "code_challenge_methods_supported": ["S256"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        }

    def token(self, *, key=None, headers=None, **changes):
        now = int(time.time())
        payload = {"iss": self.cfg.issuer, "sub": "subject-1",
                   "aud": self.cfg.client_id, "iat": now, "exp": now + 120, "nonce": "expected"}
        payload.update(changes)
        return {"id_token": jwt.encode(payload, key or self.key, algorithm="RS256",
                                       headers=headers or {"kid": "test"})}

    def test_pkce_rfc7636(self):
        self.assertEqual(
            pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
        )

    def test_transaction_browser_binding(self):
        state, browser, txn = new_transaction()
        other, _, _ = new_transaction()
        self.assertNotEqual(state_key(state, browser), state_key(state, other))
        self.assertNotEqual(txn["nonce"], txn["verifier"])
        with self.assertRaises(OIDCError):
            state_key("invalid", browser)

    def test_authorize_params(self):
        state, browser, txn = new_transaction()
        params = parse_qs(urlsplit(authorization_url(self.cfg, self.meta, state, txn)).query)
        self.assertEqual(params["state"], [state])
        self.assertEqual(params["nonce"], [txn["nonce"]])
        self.assertEqual(params["code_challenge"], [pkce_challenge(txn["verifier"])])
        self.assertEqual(params["code_challenge_method"], ["S256"])
        self.assertNotIn("client_secret", params)

    def test_valid_signed_token(self):
        self.assertEqual(validate_id_token(self.cfg, self.token(), self.jwks, "expected"), "subject-1")

    def test_invalid_claims(self):
        for changes in [
            {"iss": "https://evil.example"}, {"aud": "other"},
            {"aud": ["dify", "other"]}, {"azp": "other"},
            {"exp": int(time.time()) - 120}, {"iat": int(time.time()) + 120},
            {"nonce": "wrong"}, {"sub": ""}, {"at_hash": "wrong"},
        ]:
            with self.subTest(changes=changes), self.assertRaises((OIDCError, jwt.PyJWTError)):
                validate_id_token(self.cfg, self.token(**changes), self.jwks, "expected")

    def test_signature_forgery(self):
        attacker = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with self.assertRaises(jwt.PyJWTError):
            validate_id_token(self.cfg, self.token(key=attacker), self.jwks, "expected")

    def test_hs256_rejected(self):
        token = jwt.encode({"sub": "subject-1"}, "a" * 32, algorithm="HS256", headers={"kid": "test"})
        with self.assertRaises(OIDCError):
            validate_id_token(self.cfg, {"id_token": token}, self.jwks, "expected")

    def test_unknown_and_duplicate_kid(self):
        with self.assertRaises(OIDCError):
            validate_id_token(self.cfg, self.token(headers={"kid": "unknown"}), self.jwks, "expected")
        with self.assertRaises(OIDCError):
            validate_id_token(self.cfg, self.token(), {"keys": self.jwks["keys"] * 2}, "expected")

    def test_discovery_rejects_issuer_origin_pkce_downgrade(self):
        for changes in [
            {"issuer": "https://other.example"}, {"jwks_uri": "https://other.example/keys"},
            {"token_endpoint": "http://idp.example/token"},
            {"code_challenge_methods_supported": ["plain"]},
        ]:
            with self.subTest(changes=changes):
                with httpx.Client(transport=httpx.MockTransport(
                    lambda req: httpx.Response(200, json={**self.meta, **changes})
                )) as client, self.assertRaises(OIDCError):
                    discover(client, self.cfg)

    def test_token_exchange(self):
        def handler(req):
            data = parse_qs(req.content.decode())
            self.assertEqual(data["code_verifier"], ["verifier"])
            self.assertEqual(data["redirect_uri"], [self.cfg.redirect_uri])
            self.assertEqual(data["grant_type"], ["authorization_code"])
            self.assertTrue(req.headers["authorization"].startswith("Basic "))
            self.assertNotIn("client_secret", data)
            return httpx.Response(200, json=self.token())
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertIn("id_token", exchange(client, self.cfg, self.meta, "code", {"verifier": "verifier"}))

    def test_explicit_binding_only(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bindings.json"
            cfg = replace(self.cfg, bindings=path)
            account_id = str(uuid4())
            path.write_text(json.dumps({"issuer": cfg.issuer, "subjects": {"subject-1": account_id}}))
            self.assertEqual(resolve_binding(cfg, "subject-1"), account_id)
            with self.assertRaises(OIDCError):
                resolve_binding(cfg, "not-provisioned")
            path.write_text(json.dumps({"issuer": cfg.issuer, "subjects": {
                "subject-1": account_id, "subject-2": account_id
            }}))
            with self.assertRaises(OIDCError):
                resolve_binding(cfg, "subject-1")
            path.write_text('{"issuer":"https://other.example","subjects":{}}')
            with self.assertRaises(OIDCError):
                resolve_binding(cfg, "subject-1")


if __name__ == "__main__":
    unittest.main()
