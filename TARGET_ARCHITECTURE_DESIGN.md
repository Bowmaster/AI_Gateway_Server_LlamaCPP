# Target Architecture Design — Self-Hosted LLM Gateway (Secure External Access)

**Companion to** `GATEWAY_REFACTOR_ANALYSIS.md`. This document turns the
discovery findings into a concrete, deployable design for a self-hosted,
externally-reachable local-LLM stack on the user's existing infrastructure.

**Given infrastructure (verified from conversation):**
- Owned domain name + static public IP.
- Ubiquiti gateway/firewall (UDM-class) with built-in IDS/IPS and native
  WireGuard VPN server.
- Two inference substrates: a CPU/RAM-abundant dual-Xeon box (no GPU) and a
  modest GPU box (RTX 5070 Ti 16GB, Blackwell).

**Stack decisions locked in earlier:**
- Engine: **llama.cpp family** (raw `llama-server`, or `ollama` for model
  orchestration). MoE models on the CPU box; quantized 7B–14B on the GPU box.
- **Client-side tools** (each client advertises its own tool set; the server
  runs no tool loop). → no bespoke agent container.
- **LiteLLM OSS** (no enterprise): per-device virtual keys, budgets, model
  access groups, RPM/TPM rate limits. *Enterprise-only features (UI SSO,
  JWT-auth mode, audit logs, Prometheus) are explicitly NOT used* — their job
  is delegated to Authelia/Authentik (free) and the proxy.
- Reverse proxy: **Caddy** (recommended for simplicity) or Traefik.
- Forward-auth IdP: **Authelia** (recommended, lightweight) or Authentik (if a
  full OIDC/SAML provider is wanted).

---

## 1. Access Model — decide this first

Three tiers, not mutually exclusive. **Recommended: start at Tier A, add Tier C
only if you must share with non-VPN users.**

| Tier | Mechanism | Public attack surface | Best for |
|---|---|---|---|
| **A. VPN-first (recommended)** | UniFi native **WireGuard** server; clients connect, then reach internal service IPs/DNS | **None** (no inbound ports) | Your own phone/laptop/desktop |
| **B. Tailscale overlay** | Tailscale/Headscale mesh; MagicDNS to services | None (relay/DERP) | Multi-site, easy device enrolment |
| **C. Public ingress** | Port-forward **443 only** → Caddy in a DMZ VLAN | Caddy is internet-facing (must harden) | Sharing with people who can't VPN |

**Why VPN-first wins here:** native mobile chat clients (iOS/Android) and IDE
clients only speak bearer auth — they can't do an OIDC browser redirect. Over
WireGuard they just hit `https://api.lan.example.com/v1` with a per-device key,
and there is *nothing* exposed to the internet to attack. You already own the
gateway that does this.

**If you enable Tier C (public):** do not port-forward 80 (use DNS-01 for
certs), and layer: UniFi Threat Management (IDS/IPS) + GeoIP filtering →
CrowdSec on the proxy → Authelia forward-auth + MFA on browser paths →
LiteLLM keys + rate limits on API paths.

---

## 2. Network Topology (UniFi)

Segment with VLANs and default-deny inter-VLAN rules:

```
                    Internet ──► UniFi Gateway (firewall, IDS/IPS, WireGuard server)
                                   │
          ┌────────────────────────┼─────────────────────────┐
          │ (Tier C only:          │ WireGuard tunnel          │
          │  PF 443 → Caddy)       │ (Tier A: your devices)    │
          ▼                        ▼                           │
   VLAN 10 — EDGE/DMZ        VLAN 20 — APP SERVICES       VLAN 30 — INFERENCE
   ┌──────────────┐         ┌──────────────────────┐     ┌────────────────────┐
   │ Caddy/Traefik│────────►│ LiteLLM (OSS)        │────►│ llama-server / ollama│
   │ (TLS, routing)│        │ Authelia / Authentik │     │  (GPU box)           │
   │  + CrowdSec   │        │ Open WebUI / LibreChat│    │ llama-server / ollama│
   └──────────────┘         │ Postgres, Redis       │    │  (CPU box, MoE)      │
                            └──────────────────────┘     │  models RO-mounted   │
                                                          └────────────────────┘
```

**Firewall rules (default deny, allow only):**
- EDGE → APP: 443/internal ports to Caddy's upstreams only.
- APP → INFERENCE: only LiteLLM may reach the engine port(s).
- INFERENCE → Internet: **deny**, except a temporary allow to HuggingFace during
  model pulls, then re-deny (or pre-stage models offline). The engine holds your
  models and should be the most isolated tier.
- Management/admin endpoints (Authelia admin, LiteLLM UI): reachable **only over
  WireGuard**, never via public ingress.
- **Split-horizon DNS** on the UniFi controller: internal clients resolve
  `*.lan.example.com` to internal IPs (avoids NAT hairpin); public names (Tier C)
  resolve to the static IP.

---

## 3. TLS / Certificates

- Use **ACME (Let's Encrypt) with the DNS-01 challenge** via your DNS provider's
  API token. DNS-01 means **no inbound port 80**, works behind VPN/tunnel, and
  supports a **wildcard** `*.lan.example.com` (+ `*.example.com` for Tier C).
- Caddy/Traefik automate issuance + renewal. One wildcard cert covers every
  subdomain so adding a service needs no new cert.
- Enforce TLS 1.2+ (prefer 1.3), HSTS, modern ciphers, security headers.
- **Streaming gotcha:** SSE/token streaming requires the proxy to disable
  response buffering and use long timeouts. In Caddy set `flush_interval -1` on
  the reverse_proxy and generous `read/write` timeouts; the app already emits
  `X-Accel-Buffering: no`. Get this wrong and streaming "hangs" until completion.

---

## 4. Service Inventory (rootless Docker Compose)

| Service | Role | Backing deps | Exposure |
|---|---|---|---|
| `caddy` (or `traefik`) | TLS termination, routing, ACME DNS-01, forward-auth caller | — | Entry point (VPN or PF 443) |
| `authelia` (or `authentik-server`+`worker`) | OIDC/forward-auth IdP, **MFA (TOTP/WebAuthn/passkeys)**, sessions | Redis (sessions), Postgres or SQLite | Behind Caddy |
| `litellm` | OpenAI `/v1` gateway: **per-device virtual keys, budgets, model groups, RPM/TPM** | Postgres (keys/spend), Redis (optional cache/limits) | Behind Caddy; only it reaches engines |
| `postgres` | State for LiteLLM (+ Authelia/Authentik) — separate DBs | — | Internal only |
| `redis` | Sessions (Authelia) + optional LiteLLM cache/rate-limit | — | Internal only |
| `engine-gpu` | `llama-server`/`ollama` on the GPU box | models RO mount | Internal (VLAN 30) only |
| `engine-cpu` | `llama-server`/`ollama` on the Xeon box (MoE) | models RO mount | Internal (VLAN 30) only |
| `llama-swap` *(opt)* | Model orchestration if using raw llama.cpp (skip if ollama) | — | Internal |
| `open-webui` (or `librechat`) *(opt)* | Browser/PWA chat UI | own DB/volume | Behind Caddy + forward-auth |
| `crowdsec` + bouncer *(Tier C)* | IDS/IPS on proxy + Authelia logs | — | — |
| `monitoring` *(opt)* | Uptime Kuma (health) + Grafana/Loki (logs); Prometheus optional | — | VPN-only |
| `backup` | `pg_dump` + restic/borg to offsite (B2/S3/NAS) | — | — |

> Keep engines on their physical boxes; Compose can be split per-host (or use a
> small orchestrator). The APP tier can co-locate; engines stay on VLAN 30.

---

## 5. Authentication & Authorization Architecture

**Two ingress paths, two credential classes — this is the core of the design:**

### Path 1 — Browser / Web UI (`chat.lan.example.com`)
```
Client (browser/PWA)
  └─► Caddy ──(forward_auth)──► Authelia  ── OIDC login + MFA (TOTP/WebAuthn) ──► session cookie
        └─► Open WebUI / LibreChat ──► (uses its own LiteLLM key server-side) ──► LiteLLM ──► engine
```
- Authelia enforces identity + **MFA** + group policy ("ai-users" group).
- The web UI holds a **single LiteLLM service key** server-side; per-human
  identity/MFA is handled by Authelia at the edge. (If you want per-human
  budgets in the UI too, Open WebUI can map users → distinct LiteLLM keys.)

### Path 2 — Native / Mobile / IDE / CLI (`api.lan.example.com/v1`)
```
Client (Chatbox/Enchanted/Continue/Aider/script)
  └─► Caddy (TLS, coarse rate-limit/CrowdSec) ──► LiteLLM
        └─ validates per-device Bearer virtual key
        └─ enforces: allowed models, daily/monthly budget, RPM/TPM
        └─► engine (only LiteLLM may reach VLAN 30)
```
- **One revocable bearer key per device.** Lose a phone → revoke that key only.
- **No forward-auth on this path** — native/mobile/IDE clients can't do the OIDC
  redirect; they only attach a bearer header. Gating `/v1` with OIDC would lock
  them out.
- Optional belt-and-suspenders: Caddy adds IP rate-limit / CrowdSec; on Tier C,
  also require the WireGuard network or a client-cert for the API host.

### Authorization matrix (who enforces what)
| Control | Enforced by |
|---|---|
| Human identity, MFA, SSO, sessions | Authelia/Authentik (free) + Caddy forward_auth |
| Reachability / coarse rate-limit / IP & geo | Caddy + UniFi firewall/IPS |
| **Per-device API keys (issue/revoke)** | LiteLLM OSS |
| **Per-model access** (which keys → which models) | LiteLLM model access groups |
| **Budgets / spend caps, RPM/TPM** | LiteLLM OSS |
| Engine isolation (no direct reach) | VLAN 30 firewall rules |

**Admin surfaces** (LiteLLM UI, Authelia/Authentik admin, Grafana) → **WireGuard-only**, plus their own login + MFA. LiteLLM OSS UI is protected by its master key/UI creds — do **not** expose it publicly; VPN-gate it.

---

## 6. Supporting Infrastructure — the "do it right" checklist

These are the base pieces that turn a working stack into a defensible one:

**Identity & secrets**
- Secrets **out of git and out of compose**: Docker secrets or SOPS+age (commit
  encrypted, decrypt at deploy). Never bake keys into images.
- DNS provider API token (for DNS-01) stored as a secret, scoped to the zone.
- Rotate the LiteLLM master key and DB creds on a schedule.

**Data & durability**
- Postgres for LiteLLM (keys/spend) and Authelia/Authentik — **separate
  databases**, daily `pg_dump`, shipped via **restic/borg** to offsite (B2/S3 or
  a NAS). **Test restores**, not just backups.
- Back up: proxy config, Authelia config + user DB, LiteLLM config, the model
  registry/aliases, and (encrypted) secrets. Models themselves are
  re-downloadable, so back up the *list*, not the GGUFs.

**Edge hardening / abuse control**
- UniFi: enable **Threat Management (IDS/IPS)**, GeoIP filtering, and
  per-VLAN default-deny.
- **CrowdSec** (or fail2ban) watching Caddy + Authelia logs → auto-ban scanners
  (Tier C especially).
- LiteLLM **budgets + TPM caps double as abuse/cost guards** even internally — a
  runaway client or compromised key is bounded.
- Request size limits + sane timeouts at the proxy (but long enough for
  streaming, see §3).

**Observability**
- **Uptime Kuma** for simple health/alerting (engine `/health`, LiteLLM, Authelia).
- Centralized logs (Loki/Grafana or just `docker logs` shipping) for the proxy,
  Authelia auth events, and LiteLLM request logs (OSS gives request/spend logs;
  Prometheus metrics are enterprise — Uptime Kuma + logs cover the homelab need).
- Alert on: auth failures spike, budget exhaustion, engine down, cert near expiry.

**Lifecycle**
- **Pin image digests**; update via Renovate (review) or Watchtower (auto, riskier).
- OS unattended security updates on the hosts.
- Rootless Docker + non-root users in containers; read-only root FS where
  possible; drop capabilities; the **engine container gets no secrets and no
  sensitive mounts** (models RO only).

---

## 7. End-to-End Request Flows

**Mobile app, over WireGuard (Tier A):**
```
Phone (WG on) → api.lan.example.com:443 → Caddy(TLS) → LiteLLM(key check,
   budget, model=qwen2.5-coder) → engine-gpu /v1/chat/completions → stream back
   → tool_calls (if any) executed ON THE PHONE → follow-up request → final answer
```

**Browser, public (Tier C):**
```
Laptop → chat.example.com:443 (PF) → Caddy → forward_auth → Authelia (OIDC+MFA)
   → cookie → Open WebUI → (service key) → LiteLLM → engine-cpu (MoE) → stream
```

**Coding agent (Continue/Aider) at the desk:**
```
IDE → api.lan.example.com/v1 (its own device key, its own tool set) → LiteLLM
   → engine-gpu → returns tool_calls → IDE runs the tools locally → loop
```

---

## 8. Phased Rollout

1. **Core path, no auth frills:** Caddy (DNS-01 wildcard) → LiteLLM (one key) →
   one engine. Reach it over WireGuard. Prove streaming + a client works.
2. **Keys & limits:** LiteLLM Postgres, per-device virtual keys, model access
   groups, budgets, RPM/TPM. Issue a key per device.
3. **Second engine + orchestration:** add the CPU/MoE engine; ollama or
   llama-swap for name-based model selection across both.
4. **Web UI + forward-auth:** Open WebUI/LibreChat behind Authelia (OIDC + MFA).
5. **Harden + observe:** CrowdSec, UniFi IPS/GeoIP, backups + restore test,
   Uptime Kuma/alerts, secrets to SOPS/Docker secrets.
6. **(Optional) public ingress (Tier C):** port-forward 443, tighten everything,
   only after 1–5 are solid.

---

## 9. Open Decisions

1. **Proxy:** Caddy (simplest, recommended) vs Traefik (label-driven dynamic
   routing + CrowdSec bouncer plugin). 
2. **IdP:** Authelia (light, file-config, recommended) vs Authentik (full
   OIDC/SAML provider if you'll federate multiple apps/SSO).
3. **Engine form:** raw `llama-server` + llama-swap (max CPU tuning control) vs
   `ollama` (native model orchestration, simpler) — benchmark on the Xeon box.
4. **Web UI:** Open WebUI (most popular, RBAC, PWA) vs LibreChat (multi-provider,
   built-in OIDC) — or none if you only use native/IDE clients.
5. **Public access at all?** If WireGuard covers your users, skip Tier C and its
   entire hardening burden.
6. **Per-human vs per-device budgeting** in the web UI (single service key vs
   user→key mapping in Open WebUI).

---

## 10. Minimal vs Full — what you can drop

- **Truly minimal (solo):** Caddy + one engine, WireGuard access, single static
  key. No LiteLLM, no Authelia, no Postgres. Three moving parts.
- **This design (multi-device, budgeted, web + native, MFA on browser):** Caddy +
  Authelia + LiteLLM(OSS) + Postgres/Redis + engines + UI. ~Eight services, all
  free/self-hosted, no LiteLLM enterprise.
- The jump between them is entirely about **multi-user keys/budgets + a browser
  SSO surface**. Everything in this design is justified by those two needs.
