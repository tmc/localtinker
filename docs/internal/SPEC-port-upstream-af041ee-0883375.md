# Spec: Port upstream Tinker SDK changes (`af041ee..0883375`) into localtinker

**Audience:** session `0EDD` (localtinker parity), cc `F8D1`.
**Upstream range:** `af041ee..origin/main` = `0883375` (v0.18.2 → v0.22.6).
**Reference:** full change summary at
`~/tmp/tinker-sdk/CHANGES-af041ee..b1e4ee3.md` (covers the whole range despite the
filename). Read it first — this spec assumes that context and only adds the
localtinker-specific porting plan.

---

## 0. Orientation — localtinker is Go, client + server

localtinker is a **Go reimplementation of both sides** of Tinker:

- **Client SDK:** `tinker/` (`client.go`, `trainer.go`, `sampler.go`,
  `checkpoint.go`, `types.go`).
- **Local server:** `internal/tinkerhttp` (routes), `internal/tinkercoord`
  (coordinator + `ClientConfig`), `internal/tinkertrain`, `internal/tinkerproto`,
  `internal/tinkerweb`, etc.

So "port the changes" means, per item, deciding whether the change is **client
behavior**, **server/route behavior**, **a shared wire-format/type**, or **all
three**, and landing it in the right Go package. Upstream is the Python SDK only
(client) — but several upstream features (new routes, new config flags, proto
fwd/bwd) require **matching server support** in localtinker to actually exercise.

**Ground rule:** keep parity at the *wire/behavior* level, not line-by-line. Go
idiom over Python transcription. Do not add a pyqwest/zstd/Pydantic equivalent
just because Python has one — see per-item notes. No `go.sum` hand-edits
(per repo guidelines).

---

## 1. Priorities

**Tier 1 — wire-format & config parity (do first):**
- `ClientConfigResponse` new flags (§2)
- Proto forward/backward request+response path (§3)
- `save_state`/`save_weights_for_sampler` `overwrite` field + drop 409-as-success (§4)

**Tier 2 — new endpoints & client surface:**
- `/api/v1/audit` + `AuditLogResponse` (§5)
- `assign_session_project` (`PUT /sessions/{id}/project`) (§5)
- `list_training_runs(project_id=...)` filter (§5)
- Checkpoint archive URL 302→200-JSON dual contract (§6)

**Tier 3 — robustness & ergonomics:**
- JWT on-demand refresh (§7)
- Billing-pause (402) handling (§8)
- Session-less REST client / `_skip_session` (§9)
- `TINKER_PROJECT_ID` env fallback (§9)
- Optional stuck-detection, sample-no-retries, sample-max-concurrent (§10)

**Skip / no-op in Go (document why):**
- **pyqwest transport** — Python-specific httpx workaround. Go's `net/http` is the
  transport. Map `use_pyqwest_transport` to a **no-op accepted-and-ignored** config
  field for wire parity only.
- **Pydantic→dataclass migration** — Python-internal. Go already uses structs.
  The *behavioral* parts that matter (tensor coercion, sparse CSR) are covered in §3.
- **zstd dependency bump / publish-pypi token hygiene** — Python packaging; N/A.
  (If localtinker adds zstd request decompression server-side, see §3.)

---

## 2. ClientConfig flags (Tier 1)

**Files:** `internal/tinkercoord/coordinator.go` (`ClientConfig` struct + the
`ClientConfig()` method), `internal/tinkerhttp/routes.go` (`ConfigResponse` +
`clientConfig` handler).

Upstream added these to `ClientConfigResponse` (defaults in parens):

| Upstream flag | Default | localtinker action |
|---|---|---|
| `parallel_fwdbwd_chunks` | `True` (flipped) | flip existing `ParallelFWDBWDChunks` default → true |
| `proto_write_fwdbwd` | `False` | add `ProtoWriteFWDBWD bool` |
| `proto_compress_fwdbwd` | `False` | add `ProtoCompressFWDBWD bool` |
| `fwd_via_fwdbwd` | `False` | add `FwdViaFWDBWD bool` |
| `sample_no_retries` | `False` | add `SampleNoRetries bool` |
| `sample_enable_stuck_detection` | `True` | add `SampleEnableStuckDetection bool` |
| `sample_max_concurrent_requests` | `2000` | add `SampleMaxConcurrentRequests int` |
| `billing_exception_max_pause_duration_sec` | `3600` | add `BillingExceptionMaxPauseDurationSec int` |
| `use_pyqwest_transport` | `True` | add field for wire parity, **client ignores it** |

Wire the new fields through `ConfigResponse` JSON (snake_case tags matching the
table) and the `clientConfig` handler. Existing fields
(`credential_default_source`, `sample_dispatch_bytes_semaphore_size`,
`inflight_response_bytes_semaphore_size`) are already present — keep.

**Client:** `tinker/client.go` should fetch `/client/config` once and stash the
resolved config (it likely already does via the coordinator path) and consult it
for the flags below. Add a Go `ClientConfig`-equivalent on the client side if not
present.

---

## 3. Proto forward/backward path (Tier 1, largest item)

Upstream now can serialize `ForwardBackwardRequest` as protobuf and deserialize
`ForwardBackwardOutput` from protobuf, gated by `proto_write_fwdbwd` /
`proto_compress_fwdbwd` / `fwd_via_fwdbwd`.

**New proto messages** (`internal/tinkerproto/tinkerv1` — extend the existing
proto, regenerate): `DType` enum, `SparseCsr`, `Tensor` (dense|sparse_csr oneof),
`BatchedTensor` (data/offsets/dtype/trailing_shape), `ArrayRecord`
(type_tag/fields/num_datums), `ForwardBackwardOutput`, `ForwardBackwardRequest`
(model_id/seq_id/data/loss_fn/loss_fn_config/**forward_only**).

**Client write path** (`tinker/trainer.go` + a new `internal/tinkerproto` or
`internal/tinkerhttp` request encoder mirroring `proto/request_conv.py`):
- When `ProtoWriteFWDBWD`: encode the request to proto bytes, POST with
  `Content-Type: application/x-protobuf`. Set `forward_only` on the message.
- When `ProtoCompressFWDBWD`: zstd-compress the body, set
  `Content-Encoding: zstd`. Go: `github.com/klauspost/compress/zstd` (already a
  common dep; check `go.mod`). Run compression off the hot path if it matters.
- Else: existing JSON path.
- `forward_only` is **honored only on the proto path**; JSON path ignores it
  (server hardcodes false). When `FwdViaFWDBWD && ProtoWriteFWDBWD`,
  `Trainer.Forward` routes through `/forward_backward` with `forward_only=true`
  instead of `/forward`.

**Server read path** (`internal/tinkerhttp/routes.go` `forwardBackward`):
- Detect `Content-Type: application/x-protobuf`; if `Content-Encoding: zstd`,
  decompress first (mirror upstream's ASGI middleware). Decode the proto request.
- Currently `forwardBackward` reads JSON (`ForwardBackwardInput`). Add a proto
  branch. Note localtinker's `sparse_test.go` already exercises sparse fwd/bwd
  input — reuse that codec.

**Server write path:** emit `ForwardBackwardOutput` proto when the client sends
`Accept: application/x-protobuf` (the retrieve_future response). Mirror
`response_conv.deserialize_forward_backward_output`: `BatchedTensor` per-datum
slicing, bf16→float32 / int32→int64 widening, `num_datums` authority.

**Tensor coercion (behavioral, port regardless of proto):** upstream `Datum`
now coerces `loss_fn_inputs` (torch/numpy/list) → `TensorData` with a per-field
dtype map (`target_tokens→int64`; `weights/advantages/logprobs/clip_*→float32`)
and CSR-eligibility for `target_tokens`/`weights`. localtinker's `types.go`
`Datum`/`TensorData` should enforce the same dtype defaults and reject ragged
lists. Check `internal/tinkerhttp/sparse_test.go` for the existing CSR shape.

**`PROTO_SUPPORTED_TYPES`** gate: only `SampleResponse` + `ForwardBackwardOutput`
send `Accept: application/x-protobuf`. Mirror on the client.

---

## 4. `overwrite` on save + remove 409-as-success (Tier 1)

**Wire/type:** `SaveWeightsRequest` gains `overwrite bool` (default false). Add to
localtinker's save-weights request type (`types.go` / `internal/tinkertrain`) and
the `saveWeights` / `saveWeightsForSampler` route handlers.

**Client (`tinker/trainer.go`):**
- `SaveState` / `SaveWeightsForSampler` gain an `overwrite` option/param.
- **Remove** any "treat 409 Conflict as success" logic (upstream deleted it).
  With `overwrite`, a name collision is handled server-side. If localtinker
  doesn't yet have the 409-hack, just add `overwrite` and ensure the server honors
  it (overwrite existing checkpoint vs. 409 when false + exists).

---

## 5. New REST endpoints & client methods (Tier 2)

### 5a. Audit log
- **Route:** `GET /api/v1/audit?event_type=all|checkpoints&day=YYYY-MM-DD`
  (`mux.HandleFunc` + handler in `routes.go`). Window: midnight–midnight UTC.
  Requires admin role upstream — localtinker can stub authz but should keep the
  shape.
- **Types:** `AuditLogResponse{ entries []AuditLogEntry }`,
  `AuditLogEntry{ timestamp, event, model_id?, tinker_path?, purpose? }` in
  `types.go` (server) + client decode.
- **Client:** `RestClient.GetAuditLog(eventType, day)` equivalent in `tinker/`.

### 5b. assign_session_project
- **Route:** `PUT /api/v1/sessions/{session_id}/project` body `{project_id}`.
  Note current mux has `GET /api/v1/sessions/` (`sessionPath`) but no PUT — add a
  `PUT /api/v1/sessions/` registration and route on method+path.
- Moves a session (and its runs/samplers) into a project; clearing unsupported.
- **Client:** `RestClient.AssignSessionProject(sessionID, projectID)`.

### 5c. list_training_runs project filter
- **Route:** `GET /api/v1/training_runs` — accept optional `project_id` query
  param, filter results (`trainingRuns` handler).
- **Client:** add `projectID` option to the list call.

---

## 6. Checkpoint archive URL: 302 → 200-JSON dual contract (Tier 2)

`tinker/checkpoint.go` (client) + the `training_runs/.../archive` server route.

- **Client:** accept **both** a `200` JSON `CheckpointArchiveUrlResponse`
  (`{url, expires}`) and the legacy `302` redirect (Location → url, `Expires`
  header → expiry parsed via RFC-1123, tz-aware). `Accept` header
  `application/gzip` → `application/json`. Retry `503` with backoff.
- **Server:** localtinker may emit either; prefer adding the 200-JSON response so
  the client's new primary path is exercised.
- Use a dedicated connection pool concept only if localtinker has one
  (`CHECKPOINT_ARCHIVE_URL` pool is a Python httpx detail — likely N/A in Go).

---

## 7. JWT on-demand refresh (Tier 3)

If localtinker's client has JWT auth (the server's `authToken` returns empty `""`
today and `ClientConfig.UseJWT=false`, so this is likely dormant):
- Port the *behavior* if/when JWT is enabled: `GetToken()` refreshes on demand
  when ≤60s of runway remains, guarded by a mutex with double-check; background
  refresher shares the mutex and backs off 60s after failure.
- **If JWT is not wired in localtinker, file as a deferred note** — don't build
  the machinery speculatively. Just ensure the server `authToken` contract and
  `use_jwt` flag round-trip.

---

## 8. Billing-pause (402) handling (Tier 3)

Client retry loop: on HTTP `402`, sleep-and-retry silently while inside the
`BillingExceptionMaxPauseDurationSec` window (default 1h); log at most once/60s;
reset incident after 5 min quiet; give up + surface fatal after the window.
- localtinker client retry lives in the coordinator/http client layer — add a
  402 branch there. The sampling path treats a billing pause like 429 backpressure
  (no-op return, retried by the outer loop).
- Server side: localtinker can emit 402 in tests to exercise it; not required.

---

## 9. Session-less REST client + project env fallback (Tier 3)

- **`_skip_session` equivalent:** a REST-only client constructed under a *weights
  access token* (different org) must **not** create a session (would land in that
  org's possibly-read-only Default project). In Go: add a `skipSession` option to
  the client constructor that bypasses session creation, heartbeat, and telemetry;
  `SessionID()` errors/panics if called. Used by the
  `CreateRestClientWithWeightsAccessToken` equivalent.
- **`TINKER_PROJECT_ID` fallback:** when `project_id` is empty, read
  `TINKER_PROJECT_ID` env. Add to the client/`ServiceClient` constructor.

---

## 10. Sampling/retry config (Tier 3)

`tinker/sampler.go` + retry config:
- `sample_enable_stuck_detection=false` → disable the progress-timeout "stuck"
  check on the sampling retry handler.
- `sample_no_retries=true` → `Sample()` bypasses the retry handler entirely.
- `sample_max_concurrent_requests` (default 2000) → **always** set the sampling
  retry handler's max-connections/semaphore from config, overriding any
  caller-provided value.

---

## 11. Tests & verification

Mirror upstream's new tests as Go tests where they map to ported behavior:
- proto request/response round-trip (`test_proto_request_conv.py`,
  `test_proto_response_conv.py`) → Go encode/decode tests in `internal/tinkerproto`
  or `internal/tinkerhttp` (extend `sparse_test.go`).
- tensor coercion (`test_tensor_data.py`) → `tinker` package test on `Datum`/`TensorData`.
- checkpoint archive dual-contract (`test_checkpoint_archive_url.py`) → client test
  hitting both 200-JSON and 302 fakes.
- `list_training_runs` project filter, `service_client` skip-session/env-fallback.

**Done = ** `go build ./...` + `go test ./...` green, and the new config flags +
routes are reachable end-to-end against the embedded server. Update
`docs/` parity notes (that's DEA1/F8D1's lane — coordinate).

---

## 12. Sequencing suggestion for 0EDD

1. §2 config flags (unblocks everything else; small, mechanical).
2. §4 `overwrite` (small, isolated).
3. §5 endpoints (audit, assign-project, project filter) — additive routes.
4. §6 checkpoint archive dual-contract.
5. §3 proto fwd/bwd (largest; do after the proto messages are regenerated).
6. §8–§10 robustness flags.
7. §7 JWT — likely defer unless JWT is live.

Land each as an atomic commit (golang-project style messages). Verify with the
localtinker server before moving on. Flag anything where upstream behavior can't
be reproduced because localtinker's server doesn't model it (e.g. RBAC, billing,
multi-org tokens) — stub the contract, note the gap in `docs/`.
