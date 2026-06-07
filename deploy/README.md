# Phase 1 — Internal Deployment Skeleton

**Status:** starter skeletons to validate, **not** production-final. Schemas for
Authelia/LiteLLM/Caddy drift between versions — pin versions and validate each
config against the version you deploy. Every secret here is a placeholder.

## Decisions locked
- **Access:** internal-only via UniFi **WireGuard**. **No public ingress yet.**
- **IdP:** **Authelia** (forward-auth + MFA on the browser path).
- **Gateway:** **LiteLLM OSS** (per-device keys, budgets, model groups) — no
  enterprise features.
- **Proxy:** **Caddy** (DNS-01 wildcard certs).
- **Public ingress (Tier C):** deferred until the internal stack passes the
  **pen-test gate** in this README.

## Topology (Phase 1)
```
Your device (WireGuard on)
   └─► Caddy :443  (bound to LAN/WG interface only — NOT port-forwarded)
         ├─ api.lan.example.com   → LiteLLM (bearer key, no forward-auth)   ← native/mobile/IDE/CLI
         ├─ chat.lan.example.com  → Authelia forward-auth (OIDC+MFA) → Open WebUI  ← browser/PWA
         ├─ auth.lan.example.com  → Authelia portal
         └─ litellm.lan.example.com → Authelia (admin group) → LiteLLM UI
   LiteLLM ──► engine upstreams (separate boxes, VLAN 30): GPU box / CPU box
```
Engines are **external upstreams** (configured by address), not in this compose,
because they live on their own hosts/VLAN. A commented local engine is included
for single-box smoke testing.

## Bring-up order
1. Create the WireGuard server on the UniFi gateway; enrol your devices.
2. Add split-horizon DNS: `*.lan.example.com` → the Docker host's LAN IP.
3. Fill `.env` from `.env.example` (domain, DNS API token, generated secrets).
4. `docker compose up -d postgres redis` → wait healthy.
5. `docker compose up -d litellm authelia caddy` (and `open-webui` if used).
6. Generate secrets:
   - `openssl rand -hex 32` for each Authelia secret and the LiteLLM master key.
   - Authelia user password hash: `docker run authelia/authelia:latest authelia crypto hash generate argon2`.
7. Smoke test over WireGuard:
   - `curl https://api.lan.example.com/v1/models -H "Authorization: Bearer $KEY"`
   - Stream test (verify no buffering): `curl -N .../v1/chat/completions -d '{"model":"...","stream":true,...}'`
   - Browser → `chat.lan.example.com` → Authelia login + MFA → Open WebUI.
8. In LiteLLM UI (`litellm.lan.example.com`, VPN-only): create one **virtual key
   per device**, assign model-access groups + budgets + RPM/TPM.

## Pen-test gate — must pass BEFORE enabling public ingress
Treat this as the go/no-go checklist for Tier C. Internal-first means you can run
these from inside the WireGuard network without exposing anything.

**Network / exposure**
- [ ] From the APP VLAN, port-scan the INFERENCE VLAN — only LiteLLM→engine port
      reachable; engine has **no** path from anywhere but LiteLLM.
- [ ] Engine has **no inbound from internet** and **egress denied** (except
      temporary HF pulls). Confirm with a deny-by-default rule + log.
- [ ] Caddy :443 is bound to the LAN/WG interface; **nothing** is port-forwarded
      on the UniFi gateway yet (verify from an external host: all closed).
- [ ] Admin surfaces (LiteLLM UI, Authelia admin) unreachable except over WG.

**AuthN / AuthZ**
- [ ] API path rejects missing/invalid/expired bearer keys (401/403).
- [ ] **Key revocation works**: revoke a device key → its requests fail immediately.
- [ ] Per-model access enforced: a key limited to model A cannot call model B.
- [ ] **Budget + RPM/TPM enforced**: exhaust a key's budget → blocked; hammer it →
      rate-limited.
- [ ] Browser path **requires MFA**; cannot reach Open WebUI with only first factor.
- [ ] No auth bypass via header injection (`Remote-User`/`X-Forwarded-*` spoofing
      must be stripped/overwritten by Caddy, not trusted from the client).
- [ ] Authelia session: no fixation, secure+httponly cookies, sane expiry.

**Transport / hardening**
- [ ] `testssl.sh` against each host: TLS 1.2+ only, strong ciphers, HSTS, valid
      wildcard cert.
- [ ] Streaming works through the proxy (no buffering stall) — `flush_interval -1`.
- [ ] Request size limits + timeouts set (long enough for streaming, bounded).

**Secrets / supply chain / containers**
- [ ] No secrets in images, compose, or git (`git log -p` / image history clean).
- [ ] Containers run non-root, drop caps, read-only FS where feasible; engine
      container has **no secrets** and **RO model mounts only**.
- [ ] Image digests pinned; base images patched.
- [ ] Postgres backup taken **and a restore tested**.

**Abuse / DoS posture (preview of public exposure)**
- [ ] CrowdSec/fail2ban configured against Caddy + Authelia logs (can stay in
      log-only mode internally; switch to enforce before public).
- [ ] LiteLLM budgets cap worst-case spend/compute from a compromised key.

## Enabling public ingress later (the toggle, not a redesign)
When the gate passes:
1. UniFi: port-forward **443 only** → Caddy; enable Threat Management + GeoIP.
2. Caddy: add the public hostnames (`api.example.com`, `chat.example.com`) — same
   blocks, public DNS names. Keep `*.lan.*` for internal.
3. Switch CrowdSec to **enforce**; tighten rate limits on the public API host.
4. Consider requiring WireGuard **or** client-cert for the public `/api` host if
   you want native clients private but the web UI public.
5. Re-run the pen-test gate from **outside** the network.

## Files
- `compose.yaml` — APP-tier services (Caddy, Authelia, LiteLLM, Postgres, Redis,
  Open WebUI). Engines referenced as external upstreams.
- `.env.example` — copy to `.env`, fill in.
- `caddy/Caddyfile` — routing + TLS + forward-auth + streaming-safe proxy.
- `caddy/Dockerfile` — Caddy with the DNS-01 provider plugin.
- `authelia/configuration.yml` — forward-auth, MFA, access rules, Redis sessions.
- `authelia/users.yml` — example user/groups (file backend).
- `litellm/config.yaml` — model aliases → engine upstreams, master key, DB.
