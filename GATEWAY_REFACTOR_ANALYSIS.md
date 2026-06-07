# Discovery & Architecture Analysis — Local LLM Gateway Refactor

**Scope:** Read-only discovery pass against the current repo, mapped against the
proposed containerized / off-the-shelf-tooling target architecture.
**Verification basis:** Claims below are sourced from the actual files
(`ai_server.py` 1877 LOC, `llama_manager.py` 572, `server_config.py` 954,
`tools.py` 1636, `hardware_detector.py` 861 — ~5,900 LOC of Python total).
Where I'm inferring rather than certain (especially about the *llama-server
build's* runtime capabilities, which are not knowable from this repo), it is
flagged explicitly.

> **Note on the docs:** `CLAUDE.md`/`README.md` are out of date relative to the
> code (e.g. they describe `ai_server.py` as 27KB with 15 tools and an
> "OpenAI-compatible API"; the file is ~74KB/1877 LOC with 17 tools, and the
> gateway's *own* edge is **not** OpenAI-compatible — see §4). I trusted the
> source, not the prose.

---

## 1. Executive Summary

**Are you reinventing the wheel? Yes — substantially, in three of the four
buckets.** The headline:

- **TLS / routing:** Not reinvented — it's simply *absent*. The gateway serves
  plain HTTP on `0.0.0.0:8080` with a comment pointing at Windows Firewall.
  Pure greenfield for an off-the-shelf reverse proxy. Nothing to migrate.
- **AuthN/AuthZ:** Reinvented, minimally. A single static bearer key in ASGI
  middleware. The one thing it does right (constant-time compare) is the *only*
  thing worth preserving conceptually; LiteLLM replaces the implementation
  wholesale and adds everything it lacks (per-user keys, budgets, rate limits,
  rotation, logging).
- **OpenAI-protocol proxying:** Reinvented *badly* — and this is the important
  finding. The gateway does **not** expose the OpenAI protocol at its edge. It
  exposes a **bespoke `/chat` protocol** with server-global conversation state
  and a model that is **not** selectable per request. It proxies *internally* to
  llama-server's real `/v1/chat/completions`. So you have custom code sitting on
  top of an already-OpenAI-compatible server, hiding it behind a non-standard
  contract. LiteLLM does this layer for free and does it correctly.
- **Genuinely bespoke logic:** One real thing — the **server-side tool-calling
  agent loop** (17 tools, human-in-the-loop approval, web-content sanitization).
  This is the actual custom value and the *only* part LiteLLM does not provide.

**Recommendation: LiteLLM-centric hybrid.**

```
Internet ─► Caddy/Traefik (TLS, routing)
              └─► [Authelia/Authentik forward-auth]  (only for human/UI paths)
                    └─► LiteLLM proxy  (OpenAI protocol, API keys, budgets, rate limits, logging)
                          └─► llama-swap  (model name → load/unload orchestration)
                                └─► llama-server container(s)  (CPU inference, models RO-mounted)

         [thin custom "agent/tools" service]  ◄── optional, OpenAI-compatible, sits beside or behind LiteLLM
```

Adopt LiteLLM + a reverse proxy + llama-swap. Collapse ~5,900 LOC down to a
**thin custom service** that contains *only* the tool-execution agent loop and
its approval/sanitization logic — and even that is optional depending on whether
you want server-side tools at all. Everything else (process management, model
file resolution, idle unload, hardware tuning, context/summarization, auth, TLS)
is either commodity or should not live in the gateway at all.

The biggest structural realization: **subprocess management is the load-bearing
assumption of the entire current design, and it is exactly what the
container split deletes.** `restart()`, `idle_unload()`, `/model/switch`, and
crash diagnostics all assume the gateway is the *parent process* of
llama-server on the *same host*. In a two-container split the gateway cannot
`Popen`/`kill` a process in another container. That single change invalidates
`llama_manager.py` almost entirely — which is good, because that's the code
llama-swap exists to replace.

---

## 2. Inventory: What the Gateway Actually Does

Verified endpoint surface (`ai_server.py`):

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Model state, context/token usage, idle state, crash diagnostics |
| `/models` | GET | Lists registry models (custom schema, not OpenAI `/v1/models`) |
| `/model/switch` | POST | **Restarts the llama-server subprocess** with a new model |
| `/chat` | POST | Bespoke chat + server-side tool loop (non-streaming) |
| `/chat/approve` | POST | Resume a paused tool loop after human approval |
| `/chat/stream` | POST | SSE streaming; hybrid buffered-then-replayed when tools on |
| `/command` | POST | Mutate runtime state: `system`, `layers`, `mem`, `idle`, `tools` |
| `/hardware` | GET | Detected hardware profile + active llama.cpp flags |
| `/hardware/redetect` | POST | Re-run hardware detection |
| `/shutdown` | POST | Kill the server |

Responsibilities bundled into this one process:

1. **HTTP edge** — plain HTTP, no TLS, no CORS, binds all interfaces.
2. **AuthN** — optional single static bearer key (§3).
3. **Bespoke chat protocol** — custom request/response models; server-global
   conversation history; per-request `model` is **not** supported.
4. **Context management** — tiktoken counting, threshold-triggered LLM
   summarization of old turns, dynamic `max_tokens`, hard-limit rejection.
5. **Server-side tool agent loop** — 17 tools executed *in the gateway process*,
   max-5 iterations, approval gate, `tools_used` provenance.
6. **Subprocess lifecycle** — spawn/health-poll/kill llama-server, process-tree
   teardown, crash capture, idle unload + auto-reload.
7. **Model registry & file resolution** — maps model keys to local paths or HF
   repos; touches the filesystem to decide source.
8. **Hardware detection & llama.cpp flag tuning** — NUMA, batch sizes, KV-cache
   quant, flash-attn, speculative decoding, mlock, etc.
9. **HF download orchestration** — via llama-server's `-hf` flag.

Buckets 1, 2, 3, 6 are commodity or container-platform concerns. Bucket 5 is the
real bespoke asset. Buckets 4, 7, 8, 9 are ambiguous/relocatable (detailed
below).

---

## 3. Auth Implementation

**What's implemented** (`ai_server.py:63-90`, `server_config.py:41`):

- **Form:** Single static **API key as a bearer token**, via ASGI
  `BaseHTTPMiddleware`. Key comes from the `API_KEY` env var; if unset/empty,
  **authentication is entirely disabled (open access).**
- **Storage/comparison:** Compared with `secrets.compare_digest()` —
  **constant-time, correct.** Credential is an env var, not hardcoded. Good.
- **Coverage:** Applies to all paths *except* `/docs`, `/openapi.json`,
  `/redoc`, which are unauthenticated.

**Security-relevant gaps:**

- **Single shared secret.** No per-user/per-client keys, no scoping, no
  rotation, no revocation, no expiry. One leaked key = full compromise, with no
  way to revoke without rotating for everyone.
- **No authorization.** Any valid key can hit `/model/switch`, `/command`
  (mutates system prompt, idle timeout, toggles tools), `/shutdown`, and the
  filesystem-touching tools. No notion of roles or least privilege.
- **No rate limiting, no budgets, no usage accounting.**
- **No replay protection / nonce** — irrelevant for a static bearer scheme, but
  worth noting there's no request signing.
- **No TLS** — the bearer token crosses the wire in plaintext. On a LAN this is
  "trust the network," which the Windows-Firewall note effectively concedes.
- **Default-open.** Forgetting to set `API_KEY` silently disables auth with no
  warning at the edge (only an info log when it *is* enabled).
- **Single global session.** Conversation state, pending tool approvals, and
  summaries are process-global (`ServerState`), so the auth model couldn't
  support multiple distinct users even if keys were per-user — two callers would
  stomp each other's history and approvals.

**How much would LiteLLM / Authelia replace?**

- **LiteLLM proxy replaces ~100% of this**, and is a strict superset: virtual
  API keys per user/team, budgets, rate limits, request logging, key rotation,
  spend tracking. The bearer-key middleware should be **deleted**, not ported.
- **Authelia/Authentik** is the right tool *only* for human/browser access paths
  (a UI, dashboards) via forward-auth at the reverse proxy — OIDC/MFA/session
  cookies. It is **not** needed for machine-to-machine API traffic, which
  LiteLLM keys handle. Don't layer both on the programmatic path.

---

## 4. OpenAI Protocol Surface

**Finding: the gateway is not an OpenAI-compatible edge.** It is a custom API
that *consumes* an OpenAI-compatible upstream.

- **Edge endpoints are bespoke:** `/chat`, `/chat/stream`, `/chat/approve` with
  hand-rolled Pydantic models (`ChatRequest`/`ChatResponse`). There is **no**
  `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, or OpenAI-shaped
  `/v1/models` exposed to clients. `/models` returns a custom schema.
- **`model` is not a request field.** `ChatRequest` has no `model`. The active
  model is **server-global state** chosen out-of-band via `/model/switch`. This
  is the single largest divergence from the OpenAI contract, where `model` is
  per-request and routing is stateless. Any drop-in OpenAI client breaks here.
- **Response shape is custom:** `{response, tokens_input, tokens_generated,
  tokens_per_second, device, tools_used}` — not OpenAI `choices`/`usage`.
- **Streaming is doubly non-standard:**
  - With tools **off**, it forwards llama-server's real SSE deltas but appends a
    custom `{"type":"stream_end", ...}` event and its own `[DONE]`
    (`ai_server.py:1645-1660`).
  - With tools **on**, it runs the tool loop **non-streaming**, then **fakes**
    streaming by emitting the buffered answer **character-by-character** with
    artificial sleeps (`ai_server.py:1591-1602`). This is cosmetic streaming,
    not real token streaming, and it re-encodes deltas in a custom chunk shape.
- **Internally**, llama-server's `/v1/chat/completions` *is* used faithfully
  (`call_llama_server`, `call_llama_server_streaming`), including `tools`,
  `tool_choice:"auto"`, and `stream_options.include_usage`.

**Where a drop-in proxy breaks / needs glue:**

- Existing clients (e.g. the `ai_client.py` referenced in docs, now in a
  separate repo) depend on the **bespoke** schema and on `/command`,
  `/model/switch`, `/chat/approve`. Putting LiteLLM at the edge changes the
  contract: `model` becomes per-request, responses become OpenAI-shaped,
  streaming becomes real OpenAI SSE. **Clients must be updated** — this is a
  breaking change, but a beneficial one (it makes the standard ecosystem work).
- **Server-side tool execution has no OpenAI-protocol home.** OpenAI's contract
  (and LiteLLM's proxy) expects the *client* to execute tools and send results
  back. The gateway's model of executing tools *itself, on the server* is
  outside the protocol. This is the part that genuinely cannot be "dropped in"
  to LiteLLM and must live in the thin custom layer (§6).
- **Embeddings/completions:** not currently exposed; if needed later, LiteLLM
  gives them for free against a llama-server that supports them.

**Verdict:** LiteLLM does OpenAI proxying correctly and for free. The current
custom protocol is a **liability to retire**, not an asset to preserve — except
for the tool-loop semantics, which are protocol-external anyway.

---

## 5. Model Switching — The Crux

### How it works today (verified)

1. **Selection is out-of-band, not per-request.** Client calls
   `POST /model/switch {"model_key": "..."}` (`ai_server.py:945`). The chat
   requests themselves never name a model.
2. **Key → source resolution touches the filesystem** (`server_config.py`
   `get_model_source`, 721-785): returns `("local", path)`,
   `("local_finetuned", abspath)`, or `("huggingface", "repo:quant")`. It calls
   `os.path.exists()` to decide, so **the gateway must see the model files**.
3. **Switching = full subprocess restart** (`llama_manager.restart`, 416-437):
   `stop()` (kill process tree) → `sleep(1)` → `start()` with a freshly built
   `llama-server` command line: `--model <path>` or `-hf <repo>`, plus
   `--ctx-size`, `--n-gpu-layers`, threads, NUMA, batch sizes, KV-cache quant,
   flash-attn, speculative-decoding, mlock, etc.
4. **Per-model config exists** and is real: effective context =
   `min(model native context_length, hardware ctx limit)` (`ai_server.py:987`);
   per-model system-prompt overrides (`MODEL_SPECIFIC_PROMPTS`); sampling
   defaults are global, not per-model.
5. **Warmup/preload:** no explicit warmup request; relies on `--no-mmap`/`mlock`
   flags. Readiness is health-poll only (`_wait_for_ready`, up to 60s local /
   300s HF).
6. **Unload/lifecycle:** Python idle loop kills the subprocess after
   `IDLE_TIMEOUT_SECONDS` (default 900s) and **auto-reloads on the next request**
   (cold start). Switching **resets conversation history and summary**. There is
   also a `--sleep-idle-seconds` pass-through (`llama_manager.py:149`),
   suggesting the build *may* support llama-server's native idle sleep/wake.

### Why this fights the target architecture

The gateway's switching model is **filesystem + process-control**. The target
explicitly wants the gateway to hold **zero model-path knowledge** and treat
switching as a **network/control-plane** action. Concretely, the split breaks:

- `restart()` / `idle_unload()` / `reload_after_idle()` — gateway can't manage a
  process in another container.
- `get_model_source()` filesystem probing — gateway shouldn't mount models.
- Crash diagnostics via `process.poll()` + stderr capture — no child process.
- Hardware detection driving the launch command — that's the inference
  container's concern now.

So this code isn't "ported," it's **deleted and replaced by an orchestrator.**

### Fit assessment of the three options

**(a) llama-server's own runtime load/unload API — VERIFY, DON'T ASSUME.**
This repo cannot tell you whether your build supports runtime model swapping.
Two signals suggest it might: the README mentions
"`Model switching without restart (via /v1/models/load)`", and the config plumbs
`--sleep-idle-seconds`. **But the actual code never calls any load/unload
endpoint** — it always does a full process restart. So the README claim is
aspirational/unverified in practice. **What you must check on your build:**
whether `/v1/models`, a model load/unload route, router mode, or
`--model-alias`/multi-model serving exists in *your* compiled `llama-server`,
and whether `--sleep-idle-seconds` actually sleeps/wakes the weights. If yes,
some orchestration can collapse into llama-server itself. *Inference, not
certainty — flagged for you to confirm.*

**(b) llama-swap as orchestrator — RECOMMENDED.** This is the cleanest fit for
your stated goals. llama-swap sits in front of one or more llama-server
processes, keyed by **model name**, and transparently starts/stops/swaps the
right backend on demand (with configurable TTL/keep-alive). It turns "switch
model" into "send a request naming the model" — exactly the per-request,
control-plane behavior you want — and it absorbs the idle-unload logic you
hand-rolled. The gateway then holds **only names**, never paths. The model
files (and the launch flags from §8) become llama-swap's config, mounted RO into
that side of the split.

**(c) Container-per-model — viable but RAM-expensive on CPU.** Cleanest
isolation, but on a CPU-only headless box, keeping N models resident means N ×
(model RAM) simultaneously — your DL380 has 256GB, so a couple of 7B–31B models
could co-reside, but the 80B/large-MoE entries in the registry make this
spendy. If you keep them stopped to save RAM, you've reinvented orchestration —
so pair it with llama-swap or compose profiles anyway.

### The CPU-inference reality (your constraint)

Cold loads are disk→RAM bound and **slow** on CPU. Any swap-heavy design pays
first-request latency on every cold model. Implications:

- Prefer **one default model kept warm** (long/zero TTL) and treat alternates as
  on-demand with an accepted cold-start penalty.
- `mlock` + `--no-mmap` (already plumbed) help keep the hot model resident but
  cost RAM and make swaps *more* expensive.
- Multi-resident = RAM-expensive; on 256GB you have room for a small warm set,
  not the whole registry.
- **Recommended:** llama-swap with a warm default + a small, explicitly-chosen
  on-demand set, model files RO-mounted to the inference side only. Verify (a)
  first — if native sleep/wake works well on your build, you may not even need
  per-model *processes*, just sleep/wake on one server.

---

## 6. The 4-Bucket Triage

| # | Functionality (where) | Bucket | Classification & tool |
|---|---|---|---|
| 1 | Plain HTTP edge, binds 0.0.0.0:8080, Firewall note | TLS/Routing | **OFF-THE-SHELF** — Caddy/Traefik (auto-TLS, routing). Nothing exists to keep. |
| 2 | No CORS handling | TLS/Routing | **OFF-THE-SHELF** — reverse proxy / LiteLLM CORS config. |
| 3 | Static bearer-key middleware (`ai_server.py:63-90`) | AuthN/AuthZ | **OFF-THE-SHELF** — LiteLLM virtual keys (superset). Delete the middleware. |
| 4 | Human/UI auth (none today) | AuthN/AuthZ | **OFF-THE-SHELF** — Authelia/Authentik forward-auth, *only* on browser paths. |
| 5 | Rate limits / budgets / usage logging (none today) | AuthN/AuthZ | **OFF-THE-SHELF** — LiteLLM built-in. |
| 6 | `/chat` bespoke protocol + responses | OpenAI proxying | **OFF-THE-SHELF (retire)** — LiteLLM exposes correct OpenAI `/v1/*`. |
| 7 | Real token streaming (tools-off path) | OpenAI proxying | **OFF-THE-SHELF** — LiteLLM streams OpenAI SSE natively. |
| 8 | Faked char-by-char streaming (tools-on path) | OpenAI proxying | **GENUINELY BESPOKE — but DROP IT.** Cosmetic; not worth carrying. |
| 9 | `/models` list, registry metadata | OpenAI proxying / bespoke | **AMBIGUOUS** — LiteLLM `/v1/models` covers names; rich metadata (vram, usage, recommended) is custom UX value if you keep a UI. |
| 10 | Per-request `model` selection | OpenAI proxying | **OFF-THE-SHELF (gained, not lost)** — LiteLLM+llama-swap make this work; today it's missing. |
| 11 | Subprocess spawn/kill/health/restart (`llama_manager.py`) | bespoke infra | **OFF-THE-SHELF (replace)** — llama-swap / container runtime. Deleted by the split. |
| 12 | `/model/switch` = restart | Model switching | **OFF-THE-SHELF** — llama-swap keys by name; verify native load/unload (§5a). |
| 13 | Idle unload + auto-reload (`idle_check_loop`, `idle_unload`) | bespoke infra | **OFF-THE-SHELF** — llama-swap TTL or `--sleep-idle-seconds`. |
| 14 | Crash capture / diagnostics | bespoke infra | **OFF-THE-SHELF** — container restart policy + healthchecks + logs. |
| 15 | Model key→path resolution (`get_model_source`) | Coupling | **OFF-THE-SHELF (relocate)** — becomes llama-swap config; gateway holds names only. |
| 16 | HF download via `-hf` | bespoke infra | **OFF-THE-SHELF** — llama-server/llama-swap launch config on inference side. |
| 17 | Hardware detection + llama.cpp flag tuning (`hardware_detector.py`, `server_config.py`) | bespoke infra | **AMBIGUOUS** — valuable ops tooling, but belongs to the **inference container's** launch config, not the gateway. Keep as a deploy-time helper, move out of the request path. |
| 18 | Context mgmt: tiktoken counting, dynamic max_tokens, hard-limit reject | bespoke logic | **AMBIGUOUS** — real, but cl100k_base is wrong for Qwen/Gemma/Llama; partly redundant with llama-server's own context handling. Keep a thin version *only if* you keep server-side stateful chat. |
| 19 | Conversation summarization (`check_and_summarize_if_needed`) | bespoke logic | **AMBIGUOUS** — genuinely custom, but global-singleton (not multi-user safe) and arguably client responsibility. Decision needed. |
| 20 | Server-side tool agent loop, 17 tools (`tools.py`), max-iterations | bespoke logic | **GENUINELY BESPOKE — KEEP (thin).** Not commodity; LiteLLM does not execute tools server-side. The core asset. |
| 21 | Human-in-the-loop tool **approval** (`/chat/approve`, pending state) | bespoke logic | **GENUINELY BESPOKE — KEEP (thin).** Real safety feature, no off-the-shelf equivalent. Must be made concurrency-safe. |
| 22 | Web-content sanitization / prompt-injection filtering (`sanitize_web_content`) | bespoke logic | **GENUINELY BESPOKE — KEEP.** Defensive logic tied to the web tools; keep with the tool layer. |
| 23 | Protected-path guards, symlink-resolution for file tools | bespoke logic / security | **GENUINELY BESPOKE — KEEP, but reframe.** In containers the real control is *not mounting* sensitive paths into the tool sandbox; keep the guard as defense-in-depth. |
| 24 | `/command` runtime mutation (system prompt, idle, tools, layers) | bespoke control | **AMBIGUOUS** — convenience control-plane; some of it (idle, layers) becomes llama-swap/inference config; system-prompt/tools toggles belong to the thin layer if it persists. |

---

## 7. Coupling & Substrate Blockers

Concrete things that break (or must move) when containerized:

- **Subprocess parentage (hard blocker).** The entire `LlamaServerManager`
  assumes it `Popen`s and `psutil`-kills llama-server on the same host. Invalid
  across a container boundary. → llama-swap / runtime owns lifecycle.
- **Filesystem model resolution.** `get_model_source` / `model_exists` /
  `get_model_path` probe local paths. The gateway must **stop** mounting models.
  → names only; paths live on the inference side.
- **Hardcoded absolute Windows path.** `MODELS["my-finetuned"]["local_path"] =
  "C:\\Users\\blutterb\\Documents\\Github\\Bowmaster\\...\\output_3b_test_q4_k_m.gguf"`
  (`server_config.py:428`) — non-portable, leaks a username, dead on Linux.
- **Tools execute on the gateway host.** Every file/dir/network tool in
  `tools.py` runs in-process against the gateway's own filesystem and network
  (`_normalize_path`, `read_file`, `write_file`, `delete_file`, `web_search`,
  …). In a container these silently change meaning (they act on the *container's*
  FS). This is both a coupling issue and a security boundary that must be
  designed deliberately (dedicated sandbox container, explicit mounts).
- **Hardware detection in the request/host path.** `hardware_detector.py` +
  `.hardware_profile.json` assume the gateway runs where the inference hardware
  is. Wrong layer post-split.
- **Global single-session state.** `ServerState` (history, pending approvals,
  summary, `is_generating` lock) is process-global — blocks multi-user,
  blocks horizontal scaling, and makes the approval flow unsafe under
  concurrency.
- **Edge binding & no TLS.** `HOST=0.0.0.0`, `PORT=8080`, plain HTTP. Fine
  behind a reverse proxy; not fine as the public edge.
- **Windows-isms.** `CREATE_NEW_PROCESS_GROUP`, `.exe` fallbacks, Firewall docs
  — all assume the original Windows dev box; need to be neutral for a
  rootless-Docker Linux target. (`setup.ps1` vs `setup.sh` both exist.)

Substrate flexibility (container *or* direct install) is achievable: the cleanest
seam is "gateway/thin-layer speaks OpenAI to *some* base URL," which works
whether that URL is a llama-swap container or a locally-installed binary.

---

## 8. The "Thin Custom Layer" — What to Keep, What It Must Not Absorb

**Keep (this is the bespoke core):**

1. **Server-side tool-calling agent loop** — the OpenAI tool round-trip executed
   *on the server*: detect `tool_calls`, dispatch to the whitelisted registry
   (`execute_tool`), feed results back, bounded by `MAX_TOOL_ITERATIONS`.
2. **Human-in-the-loop approval** — pause on sensitive tools, return a pending
   set, resume on decision. (Must be reworked to be **per-session/keyed**, not
   global.)
3. **The tool implementations + their safety rails** — protected-path guard,
   symlink resolution, web-content sanitization. These are real and not
   commodity.

**Shape of the thin layer:** a small **OpenAI-compatible** service that LiteLLM
forwards to (or that calls LiteLLM as its upstream). It speaks `/v1/chat/
completions`, owns *only* the tool loop + approval + sanitization, and treats
the model as **a name passed through**. Ideally stateless per request (history
supplied by the caller), so it scales and survives restarts.

**What it must NOT absorb (explicitly out of scope for the thin layer):**

- ❌ TLS, routing, CORS → reverse proxy.
- ❌ API keys, budgets, rate limits, usage logging → LiteLLM.
- ❌ Process/model lifecycle, idle unload, swapping, HF download → llama-swap /
  inference container.
- ❌ Model-file paths, hardware detection, llama.cpp launch flags → inference
  side config.
- ❌ Faked char-by-char streaming → drop; rely on real OpenAI SSE.
- ⚠️ Stateful conversation history & summarization → **only if** you decide the
  server should own sessions (§9). Default to client-owned history and keep the
  thin layer stateless.

If you decide you *don't* want server-side tools at all (i.e. clients do their
own tool calls, the standard OpenAI pattern), then **the thin layer disappears
entirely** and you're left with pure off-the-shelf stack. That's a legitimate
endpoint of this analysis worth weighing.

---

## 9. Open Questions / Human Decisions Needed (before any code)

1. **Server-side tools: keep or push to clients?** This is the pivotal decision.
   Keeping them = a thin custom service exists. Dropping them = 100%
   off-the-shelf. Everything else hinges on this.
2. **Stateful chat: server-owned or client-owned history?** Today it's
   server-global and not multi-user safe. Going stateless (client sends history)
   unlocks LiteLLM-native scaling and deletes the summarization subsystem.
3. **llama-server build capabilities (VERIFY, see §5a).** Does *your* compiled
   binary support runtime model load/unload (`/v1/models` load route, router
   mode, aliases) and does `--sleep-idle-seconds` truly sleep/wake weights? The
   answer decides whether you need llama-swap, container-per-model, or just one
   server with native swap.
4. **Orchestrator choice:** llama-swap (recommended) vs container-per-model vs
   native llama-server multi-model — pending #3 and your RAM/latency tolerance.
5. **Warm set & cold-start budget:** which model stays hot, how many alternates
   are on-demand, acceptable first-request latency on CPU?
6. **Human access path:** is there (or will there be) a browser UI needing
   Authelia/Authentik forward-auth, or is this purely machine-to-machine (keys
   only)?
7. **Client migration:** existing clients use the bespoke `/chat`,
   `/model/switch`, `/chat/approve`. Are you willing to migrate them to standard
   OpenAI + per-request `model`? (Recommended, but it's a breaking change.)
8. **Tokenizer accuracy:** if any context logic survives, replace cl100k_base
   with per-model tokenization or lean on llama-server's own counts.
9. **Where do tools physically run** post-split (their own sandbox container?
   what's mounted?) — this is a security-design decision, not just plumbing.

---

## 10. Security Findings (worth addressing regardless of the refactor)

1. **Default-open auth.** Unset `API_KEY` ⇒ no auth, no loud warning at the edge.
   Fail closed, or at minimum log a prominent warning. (`server_config.py:41`,
   `ai_server.py:63`)
2. **Single shared static key, no rotation/scoping/expiry.** One leak = total,
   unrevocable-without-rotation compromise. → per-user keys (LiteLLM).
3. **No TLS.** Bearer token + all traffic in plaintext. → terminate TLS at the
   proxy.
4. **Unauthenticated `/docs`, `/openapi.json`, `/redoc`** expose the full API
   shape pre-auth. Minor, but tighten in prod. (`ai_server.py:73`)
5. **Powerful endpoints behind the same flat key:** `/shutdown` (DoS),
   `/command` (mutates system prompt, toggles tools, changes idle), and
   `/model/switch` are all reachable by any valid key with no authorization
   tier.
6. **Server-side tools execute on the host with the server's privileges.** File
   read/write/delete/move/copy and outbound web fetches run as the gateway
   process. The protected-path list + symlink resolution help, but the real
   control is sandboxing/least-privilege. In rootless Docker, **do not mount
   sensitive paths into the tool sandbox**; keep the path guard as
   defense-in-depth. (`tools.py` throughout)
7. **SSRF surface in web tools.** `read_webpage`/`web_search` fetch
   arbitrary/model-chosen URLs from inside your network. On a private host this
   can reach internal services. Approval mode mitigates (both are in
   `TOOLS_REQUIRING_APPROVAL`), and sanitization helps against prompt injection,
   but egress should be restricted at the network layer.
8. **Hardcoded absolute path leaking a username** in the registry
   (`server_config.py:428`) — remove; it's also dead config on the target host.
9. **Global approval state is races-prone.** `pending_tool_calls` /
   `pending_messages` are process-global; concurrent callers could approve/deny
   across each other's contexts. Make session-scoped before exposing to >1 user.
10. **Committed `.claude/settings.local.json`** embeds the developer's absolute
    Windows paths/username and broad `Bash(python:*)` allows — housekeeping, not
    a live server vuln, but it leaks environment detail into the repo.

---

## 11. Bottom Line

- You are reinventing **TLS/routing** (by omission), **auth**, and **OpenAI
  proxying** (by wrapping an already-compatible server in a non-standard
  protocol). Hand all three to **reverse proxy + LiteLLM**.
- **Model switching** is your filesystem/subprocess design fighting the
  container split. Replace it with **llama-swap keyed by model name** (after
  verifying what your llama-server build can do natively), so the gateway holds
  **names, not paths**.
- The **only genuinely bespoke, keep-worthy asset** is the **server-side
  tool-calling agent loop with approval + web sanitization**. Preserve it as a
  small, stateless, OpenAI-compatible **thin layer** — and recognize that if you
  decide tools belong on the client, even that disappears and you land on a
  fully off-the-shelf stack.
- Your stated goals (names not paths, control-plane switching, off-the-shelf
  TLS/auth, thin custom remainder) are **well-aligned with what the code
  actually is** — the refactor mostly means *deleting* infrastructure code that
  the container platform and LiteLLM/llama-swap already provide, not rebuilding
  it.
