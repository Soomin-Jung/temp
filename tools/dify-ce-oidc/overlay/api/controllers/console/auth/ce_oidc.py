"""Console-only CE OIDC endpoints. Existing Dify session issuance is reused."""
import json
import logging
import os

from flask import redirect, request, make_response
from flask_restx import Resource
from sqlalchemy import select
from sqlalchemy.orm import Session

from configs import dify_config
from controllers.console import console_ns
from controllers.console.wraps import setup_required
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from libs.ce_oidc import (
    OIDCError, Settings, TTL, authenticate, authorization_url, discover,
    http_client, new_transaction, resolve_binding, state_key,
)
from libs.helper import extract_remote_ip
from libs.token import (
    set_access_token_to_cookie, set_csrf_token_to_cookie,
    set_refresh_token_to_cookie,
)
from models.account import Account, AccountStatus, Tenant, TenantAccountJoin, TenantStatus
from services.account_service import AccountService

logger = logging.getLogger(__name__)
COOKIE = "__Host-ce-oidc-transaction"


def enabled() -> bool:
    return (
        dify_config.DEPLOYMENT_EDITION.value == "COMMUNITY"
        and os.getenv("CE_OIDC_ENABLED", "false").lower() == "true"
    )


def settings() -> Settings:
    return Settings.from_env(dify_config.CONSOLE_API_URL, dify_config.CONSOLE_WEB_URL)


def protect(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def fail(status=401):
    response = make_response({"message": "OIDC sign-in failed. Contact your administrator."}, status)
    response.delete_cookie(COOKIE, path="/", secure=True, httponly=True, samesite="Lax")
    return protect(response)


def login_existing(cfg: Settings, account_id: str):
    # Only active, explicitly pre-bound accounts in the approved normal workspace.
    # Do not create accounts, activate pending invitations, or change roles.
    with Session(db.engine) as session:
        account = session.get(Account, account_id)
        if account is None or account.status != AccountStatus.ACTIVE:
            raise OIDCError("inactive or missing account")
        memberships = session.scalars(
            select(TenantAccountJoin)
            .join(Tenant, Tenant.id == TenantAccountJoin.tenant_id)
            .where(TenantAccountJoin.account_id == account.id,
                   Tenant.status == TenantStatus.NORMAL)
        ).all()
        if len(memberships) != 1 or str(memberships[0].tenant_id) != cfg.tenant_id:
            raise OIDCError("account is not in the approved single workspace")
        return AccountService.login(account, session=session, ip_address=extract_remote_ip(request))


@console_ns.route("/ce-oidc/config")
class CEOIDCConfig(Resource):
    @setup_required
    def get(self):
        ready = enabled()
        if ready:
            try:
                settings()
            except (ValueError, KeyError, OIDCError):
                ready = False
        return protect(make_response({"enabled": ready}))


@console_ns.route("/ce-oidc/login")
class CEOIDCLogin(Resource):
    @setup_required
    def get(self):
        if not enabled():
            return fail(404)
        try:
            cfg = settings()
            state, browser, txn = new_transaction()
            with http_client() as client:
                meta = discover(client, cfg)
            target = authorization_url(cfg, meta, state, txn)
            # Save the issuer/client/redirect configuration with the transaction.
            txn.update(issuer=cfg.issuer, client_id=cfg.client_id, redirect_uri=cfg.redirect_uri)
            if not redis_client.set(state_key(state, browser), json.dumps(txn), ex=TTL, nx=True):
                raise OIDCError("transaction conflict")
            response = redirect(target)
            response.set_cookie(COOKIE, browser, max_age=TTL, secure=True, httponly=True,
                                samesite="Lax", path="/")
            return protect(response)
        except Exception as exc:
            # Never log raw exceptions, tokens, callback query or provider body.
            logger.warning("CE OIDC start failed: %s", type(exc).__name__)
            return fail()


@console_ns.route("/ce-oidc/callback")
class CEOIDCCallback(Resource):
    @setup_required
    def get(self):
        if not enabled():
            return fail(404)
        try:
            cfg = settings()
            states = request.args.getlist("state")
            browsers = request.cookies.getlist(COOKIE)
            if len(states) != 1 or len(browsers) != 1:
                raise OIDCError("missing or duplicate transaction")
            # GETDEL is atomic (Redis >= 6.2). Browser binding prevents login CSRF.
            raw = redis_client.getdel(state_key(states[0], browsers[0]))
            if raw is None:
                raise OIDCError("expired or consumed transaction")
            txn = json.loads(raw)
            if (txn["issuer"], txn["client_id"], txn["redirect_uri"]) != (
                cfg.issuer, cfg.client_id, cfg.redirect_uri
            ):
                raise OIDCError("configuration changed")
            callback_issuers = request.args.getlist("iss")
            if callback_issuers and callback_issuers != [cfg.issuer]:
                raise OIDCError("callback issuer mismatch")
            codes = request.args.getlist("code")
            if "error" in request.args or len(codes) != 1:
                raise OIDCError("authorization rejected")
            sub = authenticate(cfg, codes[0], txn)
            account_id = resolve_binding(cfg, sub)
            tokens = login_existing(cfg, account_id)
            # Fixed configured destination; no caller-controlled redirect URL.
            response = redirect(cfg.console_url + "/signin")
            set_access_token_to_cookie(request, response, tokens.access_token)
            set_refresh_token_to_cookie(request, response, tokens.refresh_token)
            set_csrf_token_to_cookie(request, response, tokens.csrf_token)
            response.delete_cookie(COOKIE, path="/", secure=True, httponly=True, samesite="Lax")
            return protect(response)
        except Exception as exc:
            logger.warning("CE OIDC callback failed: %s", type(exc).__name__)
            return fail()
