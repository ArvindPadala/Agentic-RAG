# Supabase setup: optional private demo sessions

The `AgenticRAG` Supabase project can provide private demo memory without forcing
visitors to register. Ordinary public chat remains the default and makes no
Supabase requests. A visitor opts in with **Start private demo session** and a
verification challenge. Supabase creates an anonymous authenticated identity.

## What this increment provides

- Memory is stored under the identity verified by Supabase Auth, with row-level
  security (RLS) and optimistic revision checks.
- An encrypted HttpOnly cookie holds the short-lived access token. Tokens are
  not sent to the LLM, returned in UI state, or exposed to browser JavaScript.
- Guest cookies last at most one hour and are not refreshed. With a stable
  cookie key, the browser can reload and recover saved memory within that period,
  including after an app restart. This is **temporary demo memory**, not a
  permanent account or cross-device recovery.
- Guests can scrub their own saved memory and end an active session. A closed
  marker prevents replayed credentials or in-flight saves from recreating it.
  Expired sessions can only clear the local cookie; orphaned records need retention
  cleanup. No automatic cleanup is claimed.
- **Return to public chat** clears the local credential and chat context without
  promising remote deletion. This works even when Supabase is unavailable. Use
  **Forget private memory** while the session is valid if you want to scrub its
  stored payload. Without an account, discarded credentials cannot be recovered
  through the app.
- Uploads remain blocked publicly until owner-scoped documents and ingestion
  are implemented. Starting a guest session does not grant corpus administration.

## 1. Apply the memory migration

In the project's **SQL Editor**, run the contents of:

`supabase/migrations/202610070001_user_memory.sql`

It creates only the memory table, ownership policies, and revision-checked save
function. It never imports the existing shared `memory.json`. Apply this migration
once; the table creation intentionally fails if an incompatible table already
exists instead of silently accepting an unknown schema.

The runtime uses a **publishable key** (or legacy `anon` key) plus the current
user's JWT. Do not configure a secret/service-role key or a database password.

## 2. Enable anonymous authentication and CAPTCHA

1. In Supabase **Authentication → Sign In / Providers**, enable anonymous sign-ins.
2. Create a free Cloudflare Turnstile widget. Allow your direct `.hf.space`
   hostname, and `localhost`/`127.0.0.1` if testing locally.
3. In Supabase **Authentication → Bot and Abuse Protection**, enable CAPTCHA,
   select Turnstile, and enter its **secret key**. CAPTCHA verification is
   delegated to Supabase Auth; this project setting is required.
4. Copy the Turnstile **site key** for the application configuration below.
5. Keep anonymous-sign-in rate limits enabled. Supabase documents a default of
   30 requests/hour per IP. Calls originate from the app server, so this can
   become a shared limit for visitors. Do not inflate the quota to hide an abuse
   or capacity problem.

Anonymous Auth users consume database space and Auth usage. The `anon` API role
is not an anonymous Auth user: only a verified Auth identity gets a private row.

## 3. Configure the application

Set these in your local environment or Hugging Face Space Settings. Keep secrets
out of Git and chat. Merely creating the Supabase project does not configure the
application, apply its migration, or deploy these changes.

| Setting | Value | HF storage |
|---|---|---|
| `APP_MODE` | `public_demo` | Variable |
| `SUPABASE_ENABLED` | `true` once setup is complete | Variable |
| `SUPABASE_URL` | Project URL, e.g. `https://<project-ref>.supabase.co` | Variable |
| `SUPABASE_PUBLISHABLE_KEY` | Publishable key from the project API settings | Secret (public-scope credential) |
| `SESSION_COOKIE_KEY` | Stable Fernet encryption key | Secret |
| `APP_PUBLIC_URL` | Direct app origin, e.g. `https://arvindpadala-agentic-document-rag.hf.space` | Variable |
| `TURNSTILE_SITE_KEY` | Turnstile site key | Variable |

Generate the cookie key locally and copy the result directly into HF Secrets:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Do not send the generated key here. A key change invalidates existing cookies.
The Turnstile secret belongs in Supabase's CAPTCHA configuration; only the site
key is used by this application.

`APP_PUBLIC_URL` must be an HTTPS **origin**, without a path/query. HTTP is allowed
only for localhost/127.0.0.1 development. Cookie mutations and private callbacks
require that exact Origin. Visitors must open the direct app URL in a top-level
tab: embedded HF pages can block the SameSite/third-party cookies needed for
private sessions. Public tryouts remain available in the embedded view.

Keep `SUPABASE_ENABLED=false` while configuration is incomplete. The public
read-only demo still works; incomplete enabled configuration fails startup.

## 4. Verify before public activation

- Start two separate browser profiles/incognito sessions and create private
  sessions. Save different synthetic facts. Neither session should load the
  other's memory.
- Reload the same browser while its cookie is valid: saved memory should return.
- Clear chat: conversation history resets, but only that identity's memory remains.
- Forget memory: verify that identity's personal payload is empty, its closed
  marker is set, and the local cookie clears. The marker cannot be removed by a
  user-level request. It is retained until anonymous-identity cleanup.
- Try a failed CAPTCHA, expired cookie, paused project, and concurrent saves.
  Failures must not fall back to shared memory or claim a successful save.
- Verify a failed CAPTCHA is actually rejected by the hosted Auth endpoint.
  Merely supplying the site key does not enable CAPTCHA on the project.
- Confirm the approved public corpus is provisioned. These changes do not copy
  existing private files or migrate the index to Supabase.

Python unit/in-process HTTP tests mock Supabase and Cloudflare. They do not prove
your project's Auth settings, live RLS grants, CAPTCHA, HF cookies, or browser
integration work. Those checks require this configuration and live verification.

## Retention and free-tier limits

Supabase does not automatically clean up anonymous accounts. For this dedicated
project, periodically review old anonymous identities and delete those that are
no longer needed. `user_memories` references `auth.users` with `ON DELETE CASCADE`,
so deleting an identity also removes its memory and closed marker. No
administrative key is kept in the running app for cleanup.

For an operator-reviewed cleanup in the dedicated project, the official docs
provide this example (it deletes accounts, not merely chat history):

```sql
delete from auth.users
where is_anonymous is true
  and created_at < now() - interval '30 days';
```

Choose and disclose the retention period before activation. Consider scheduling
cleanup as the next operational increment rather than relying indefinitely on
manual maintenance. Free projects can pause after inactivity; private-memory
outages must be visible while ordinary public chat remains independent.

Optional permanent account linking can be added later for recoverable/cross-device
memory. Anonymous memory must not be blindly reassigned when linking an existing
account. Linking/signup is never required for a basic public tryout.

## SQL regression checks

`supabase/tests/user_memory_rls.test.sql` contains 28 pgTAP assertions. Run it
against a disposable local Supabase database with the CLI, after applying the
migration. The file creates synthetic users inside a transaction and rolls back.

For plain Postgres tests, `tests/fixtures/postgres_auth_fixture.sql` supplies a
minimal Auth UID/role contract and requires pgTAP. It is test scaffolding, not a
Supabase migration and must never be applied to the hosted project.

With a running local Docker daemon, `make test-db` builds the small Postgres/pgTAP
test image, starts a disposable container with no network, published ports, or
host mounts, applies the fixture and migration, checks the TAP output, and
removes the container.
The initial image build downloads public dependencies. This verifies PostgreSQL
policies and the RPC, not the hosted Supabase Auth or PostgREST configuration.

## Official references

- https://supabase.com/docs/guides/auth/auth-anonymous
- https://supabase.com/docs/guides/auth/auth-captcha
- https://developers.cloudflare.com/turnstile/plans/
- https://supabase.com/docs/guides/database/postgres/row-level-security
