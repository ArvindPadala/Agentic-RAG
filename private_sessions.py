"""Encrypted HttpOnly guest sessions and small same-origin Gradio auth routes."""

import json
import time
from urllib.parse import urlsplit

from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from supabase_backend import AuthenticationError, SupabaseError


COOKIE_NAME = "agenticrag_private_session"


class PrivateSessions:
    def __init__(self, backend, cookie_key, app_url, site_key):
        parsed = urlsplit(app_url)
        if (parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/") or not parsed.hostname
                or not (parsed.scheme == "https" or
                        (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}))):
            raise ValueError("APP_PUBLIC_URL must be HTTPS (or HTTP loopback for development)")
        if not isinstance(site_key, str) or not site_key.strip():
            raise ValueError("TURNSTILE_SITE_KEY is required for private demo sessions")
        self.backend = backend
        self.fernet = Fernet(cookie_key)
        self.origin = app_url.rstrip("/")
        self.secure = parsed.scheme == "https"
        self.site_key = site_key

    def _check_origin(self, request):
        if request is None or request.headers.get("origin") != self.origin:
            raise AuthenticationError("Private-session requests must come from this application.")

    def _token(self, request):
        cookie = request.cookies.get(COOKIE_NAME) if request is not None else None
        if not cookie:
            return None
        self._check_origin(request)
        try:
            if len(cookie) > 4096:
                raise ValueError
            payload = json.loads(self.fernet.decrypt(cookie.encode("ascii")))
            expires = payload["expires_at"]
            token = payload["access_token"]
            if (isinstance(expires, bool) or not isinstance(expires, (int, float))
                    or expires <= time.time() or not isinstance(token, str) or not token):
                raise ValueError
            return token
        except (InvalidToken, UnicodeError, KeyError, ValueError, TypeError):
            raise AuthenticationError("Your private session expired. Start a new session.") from None

    def load_for_request(self, request, *, required=False):
        token = self._token(request)
        if token is None:
            if required:
                raise AuthenticationError("Start a private demo session before saving memory.")
            return None
        return self.backend.load_memory(token)

    def _cookie(self, token, expires):
        value = self.fernet.encrypt(json.dumps({
            "access_token": token, "expires_at": expires,
        }).encode("utf-8")).decode("ascii")
        if len(value) > 3800:
            raise SupabaseError("Private session exceeds the browser cookie limit.")
        return value

    def routes(self):
        router = APIRouter()

        @router.post("/private-session/start")
        async def start(request: Request):
            try:
                self._check_origin(request)
                body = bytearray()
                async for part in request.stream():
                    body.extend(part)
                    if len(body) > 8192:
                        raise HTTPException(413, "Session request is too large.")
                try:
                    captcha = json.loads(body).get("captcha_token")
                except (ValueError, AttributeError):
                    raise HTTPException(400, "Invalid session request.") from None
                # Reuse a valid private identity instead of creating abandoned
                # accounts on repeated clicks. Closed/expired sessions need a
                # new identity and CAPTCHA. Ordinary public chat creates none.
                try:
                    existing = self._token(request)
                    if existing:
                        await run_in_threadpool(self.backend.load_memory, existing)
                        return JSONResponse({"status": "ready"}, headers={"Cache-Control": "no-store"})
                except AuthenticationError:
                    pass
                token, expires = await run_in_threadpool(self.backend.start_guest, captcha)
                response = JSONResponse({"status": "ready"}, headers={"Cache-Control": "no-store"})
                response.set_cookie(
                    COOKIE_NAME, self._cookie(token, expires),
                    max_age=max(1, int(expires - time.time())),
                    httponly=True, secure=self.secure, samesite="lax", path="/",
                )
                return response
            except AuthenticationError as error:
                raise HTTPException(401, str(error)) from None
            except SupabaseError as error:
                raise HTTPException(503, str(error)) from None

        @router.post("/private-session/forget")
        async def forget(request: Request):
            # A cross-origin caller must not even clear the browser cookie.
            try:
                self._check_origin(request)
            except AuthenticationError:
                raise HTTPException(403, "Private-session requests must come from this application.") from None
            try:
                token = self._token(request)
            except AuthenticationError as error:
                # An expired token must not trap a browser in private mode.
                # Forget expired cookies locally; deletion needs a valid token.
                response = JSONResponse({"status": "expired", "detail": str(error)},
                                        headers={"Cache-Control": "no-store"})
                response.delete_cookie(COOKIE_NAME, httponly=True, secure=self.secure,
                                       samesite="lax", path="/")
                return response
            try:
                if token:
                    await run_in_threadpool(self.backend.forget_memory, token)
                response = JSONResponse({"status": "forgotten"}, headers={"Cache-Control": "no-store"})
                response.delete_cookie(COOKIE_NAME, httponly=True, secure=self.secure,
                                       samesite="lax", path="/")
                return response
            except SupabaseError as error:
                raise HTTPException(503, str(error)) from None

        @router.post("/private-session/end")
        async def end(request: Request):
            # Leaving private mode must remain possible during a storage
            # outage. This clears local credentials, not remote personal data.
            try:
                self._check_origin(request)
            except AuthenticationError:
                raise HTTPException(403, "Private-session requests must come from this application.") from None
            response = JSONResponse({"status": "public", "memory_deleted": False},
                                    headers={"Cache-Control": "no-store"})
            response.delete_cookie(COOKIE_NAME, httponly=True, secure=self.secure,
                                   samesite="lax", path="/")
            return response

        return router.routes

    def start_js(self):
        # Only the public CAPTCHA site key is embedded. Auth JWTs live inside
        # encrypted HttpOnly cookies and are never returned to browser JS.
        return """async () => {
            if (window.self !== window.top) {
                throw new Error('Open the direct app URL in a new tab to start a private session.');
            }
            if (!window.turnstile) {
                await new Promise((resolve, reject) => {
                    const script = document.createElement('script');
                    script.src = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
                    script.onload = resolve;
                    script.onerror = () => reject(new Error('Verification could not load. Please retry.'));
                    document.head.appendChild(script);
                });
            }
            const target = document.createElement('div');
            target.style.cssText = 'position:fixed;top:20%;left:50%;transform:translateX(-50%);z-index:99999;background:white;padding:20px';
            document.body.appendChild(target);
            let widget;
            try {
                const captcha = await new Promise((resolve, reject) => {
                    widget = window.turnstile.render(target, {
                        sitekey: SITE_KEY, callback: resolve,
                        'error-callback': () => reject(new Error('Verification failed. Please retry.')),
                        'expired-callback': () => reject(new Error('Verification expired. Please retry.'))
                    });
                });
                const result = await fetch('/private-session/start', {
                    method: 'POST', credentials: 'same-origin',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({captcha_token: captcha})
                });
                if (!result.ok) throw new Error('Private session could not start. Check configuration or retry later.');
            } finally {
                if (widget !== undefined) window.turnstile.remove(widget);
                target.remove();
            }
        }""".replace("SITE_KEY", json.dumps(self.site_key))

    @staticmethod
    def forget_js():
        return """async () => {
            const result = await fetch('/private-session/forget', {
                method: 'POST', credentials: 'same-origin',
                headers: {'Content-Type': 'application/json'}, body: '{}'
            });
            if (!result.ok) throw new Error('Memory could not be deleted. Please retry.');
            const data = await result.json();
            if (data.status === 'expired') {
                window.alert('The local session was cleared, but expired private memory needs retention cleanup.');
            }
        }"""

    @staticmethod
    def end_js():
        return """async () => {
            const result = await fetch('/private-session/end', {
                method: 'POST', credentials: 'same-origin',
                headers: {'Content-Type': 'application/json'}, body: '{}'
            });
            if (!result.ok) throw new Error('Private session could not end. Please retry.');
        }"""
