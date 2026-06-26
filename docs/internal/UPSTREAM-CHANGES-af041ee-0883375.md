# Tinker Python SDK — Changes in `af041ee..origin/main` (`0883375`)

**Version:** `0.18.2` → `0.22.6`
**Sync window:** upstream `98f0c42…` (2026‑04‑22) → `62d249f…` (2026‑06‑26)
**Scope:** ~50 files, ~3,500 insertions / ~700 deletions.

> Originally summarized `af041ee..b1e4ee3` (v0.22.3); extended to the latest
> `origin/main` (`0883375`, v0.22.6) — the three newest commits are folded into
> §14 below and the flag table at the end.

These commits are all "Sync contents" snapshots from the upstream (private) tinker
repo plus a merged PR for on‑demand JWT refresh. The net effect is several large
features layered onto the request/response plumbing, almost all gated behind
server‑resolved feature flags so they can be rolled out (and rolled back) without
an SDK release.

---

## 1. New HTTP transport: pyqwest (reqwest/hyper) — flag‑gated

A new optional async HTTP backend sits underneath httpx.

- **Dependencies added** (`pyproject.toml`): `pyqwest>=0.4.1`, `zstandard>=0.22.0`.
- `_base_client.py` adds `_default_pyqwest_transport()`, building an
  `AsyncPyqwestTransport(transport=pyqwest.HTTPTransport())`. The default async
  client is constructed on top of it when `use_pyqwest=True`.
- Gated by **`ClientConfigResponse.use_pyqwest_transport`** (default `True`). The
  server can flip every client back to httpx's native transport without a release.
- **Kill‑switch safety:** the one‑off `/api/v1/client/config` fetch and the
  `CHECKPOINT_ARCHIVE_URL` pool are pinned to httpx's default transport
  (`use_pyqwest_transport=False`), so the flag itself is always reachable even if
  pyqwest is broken.

The resolved `ClientConfigResponse` is now threaded down into every `AsyncTinker`
via a new private `_client_config` constructor arg (set by `InternalClientHolder`).

---

## 2. Proto wire path for forward/backward — flag‑gated

The SDK can now serialize `ForwardBackwardRequest` as protobuf and deserialize
`ForwardBackwardOutput` from protobuf, instead of JSON.

**New / changed config flags** (`ClientConfigResponse`):

| Flag | Default | Effect |
|------|---------|--------|
| `proto_write_fwdbwd` | `False` | POST fwd/bwd as proto bytes (`Content-Type: application/x-protobuf`); falls back to JSON otherwise. |
| `proto_compress_fwdbwd` | `False` | zstd‑compress the proto body (`Content-Encoding: zstd`); real payloads compress >10×. Proto path only. |
| `fwd_via_fwdbwd` | `False` | Route `TrainingClient.forward()` through `/forward_backward` with `forward_only=True` on the proto, instead of `/forward`. |
| `parallel_fwdbwd_chunks` | `True` *(was `False`)* | Default flipped on — fwd/bwd chunks now fire concurrently by default. |

**New modules / additions:**

- `proto/request_conv.py` (**new**) — `forward_backward_request_to_proto()` plus
  helpers to encode `TensorData` (dense + sparse CSR), `EncodedTextChunk`, and
  `ImageChunk` into the public proto. Public dtypes collapse to `{float32, int64}`.
- `proto/response_conv.py` — adds `deserialize_forward_backward_output()` and
  `_decode_batched_tensor_to_per_datum_arrays()`. Handles `BatchedTensor`
  slicing, `bfloat16→float32` and `int32→int64` widening, and multi‑chunk
  `ArrayRecord` responses. Prefers `ArrayRecord.num_datums` (survives server‑side
  field stripping like `drop_fwdbwd_logprobs`). `ForwardBackwardOutput` is added to
  `PROTO_SUPPORTED_TYPES` (controls the `Accept: application/x-protobuf` header).
- `proto/tinker_public_pb2.py` / `.pyi` — regenerated. New messages: `DType`
  enum, `SparseCsr`, `Tensor`, `BatchedTensor`, `ArrayRecord`,
  `ForwardBackwardOutput`, `ForwardBackwardRequest` (with `forward_only`,
  `loss_fn_config`, etc.).
- `resources/training.py` — `forward_backward()` gains a `forward_only` arg and
  the proto/zstd encoding branch; both `forward()` and `forward_backward()` JSON
  paths now dump via `to_pydantic_request()` (see §4). zstd compression runs in a
  worker thread (`asyncio.to_thread`).

---

## 3. Public types migrated from Pydantic to frozen dataclasses

`TensorData`, `Datum`, `ForwardBackwardInput`, `ForwardBackwardOutput` (and the
`forward_request` location) moved from thin Pydantic re‑exports to real
`@dataclass(frozen=True)` public types. The Pydantic versions now live under
`types/_pydantic_types/` and are used **only** at the JSON wire boundary.

- **`TensorData`** (`types/tensor_data.py`) — now a numpy‑backed frozen dataclass
  with: `from_numpy`/`from_torch`/`from_torch_sparse`, `to_numpy`/`to_torch`,
  `data`/`tolist` accessors, sparse‑CSR support (`sparse_crow_indices`,
  `sparse_col_indices`), and bf16→float32 widening. `from_torch_sparse` auto‑detects
  2‑D sparsity and only uses CSR when it actually saves space.
- **`Datum`** (`types/datum.py`) — frozen dataclass that coerces `loss_fn_inputs`
  values (torch tensors / numpy / Python lists) into `TensorData` in
  `__post_init__`. Per‑field dtype map (`target_tokens→int64`, `weights/advantages/
  logprobs/clip_*→float32`); `target_tokens` and `weights` are CSR‑eligible. Ragged
  nested lists raise a clear error.
- **`ForwardBackwardOutput`** — now documents the MoE routing metrics
  (`e_frac_with_tokens:mean`, `e_max_violation:*`, etc.).

---

## 4. Pydantic conversion layer reworked (`lib/_pydantic_conv.py`)

Now **bidirectional** rather than read‑only:

- **Read path** (unchanged shape): server JSON → Pydantic `model_validate` →
  registered converter → public dataclass. Adds a converter for
  `ForwardBackwardOutput`.
- **Write path** (**new**): `to_pydantic_request()` / `to_pydantic_input()` convert
  a public request dataclass back into its Pydantic mirror so callers can
  `model_dump(mode="json")` and emit the legacy JSON wire shape. Used by
  `resources/training.py`.

---

## 5. New REST endpoints (`lib/public_interfaces/rest_client.py`)

- **`get_audit_log(event_type="all"|"checkpoints", day=None)`** + async variant —
  `GET /api/v1/audit`. Requires the `tinker-admin` RBAC role. Returns the new
  `AuditLogResponse` / `AuditLogEntry` types (`types/audit_log_*.py`). Window is
  midnight‑to‑midnight UTC for the given day (default today).
- **`assign_session_project(session_id, project_id)`** + async variant —
  `PUT /api/v1/sessions/{id}/project`. Moves a session (and its runs/samplers) into
  a project. Clearing the project is not supported.
- `get_checkpoint_archive_url` now uses the dedicated `CHECKPOINT_ARCHIVE_URL`
  connection pool and routes through `execute_with_retries`.

---

## 6. Checkpoint archive URL: 302 → 200‑JSON migration (`resources/weights.py`)

`get_checkpoint_archive_url` now accepts **both** contracts:

- New: a `200` JSON `CheckpointArchiveUrlResponse` (parsed directly).
- Legacy: a `302` redirect (Location header → URL; `Expires` header → expiry).
- `Accept` header changed `application/gzip` → `application/json`.
- `Expires` parsing replaced with `email.utils.parsedate_to_datetime` (RFC‑compliant,
  tz‑aware, more lenient).

---

## 7. JWT auth: on‑demand refresh (`lib/_jwt_auth.py`, PR #39)

Adds a safety net so a delayed/failed background refresh can't leak a stale JWT
(the server rejects stale JWTs with `401 Invalid JWT`, which is **not** retried).

- `get_token()` now refreshes on demand if the cached token has
  ≤ `_REFRESH_ON_DEMAND_SECS` (60s) of runway, guarded by an `asyncio.Lock` with
  double‑checked locking so concurrent callers don't fire duplicate `/auth/token`
  requests.
- The background `_refresh_loop` shares the same lock, and now applies an explicit
  `_RETRY_DELAY_SECS` (60s) backoff after a failed refresh (the old `max(60, …)`
  floor on `delay` was removed, so without this it would tight‑loop on a stale token).
- New helper `_seconds_until_expiry()`.
- New test file `lib/_jwt_auth_test.py` (142 lines).

---

## 8. Billing‑pause handling (`lib/internal_client_holder.py`, et al.)

New `_should_pause_on_billing(status_code, detail)` on `InternalClientHolder`:

- On HTTP `402`, sleep‑and‑retry silently (no telemetry spam) while inside the
  max‑pause window; logs a `WARNING` at most once per 60s.
- After `billing_exception_max_pause_duration_sec` (default 1h) it gives up and
  falls through to the normal fatal dispatch. Incidents reset after 5min quiet.
- Wired into `execute_with_retries`, the `_APIFuture` retry loop, and
  `SamplingClient._send_asample_request` (which treats a billing pause like a 429
  backpressure no‑op).

---

## 9. `_APIFuture.result_async` refactor (`lib/api_future_impl.py`)

The monolithic retrieve‑result loop was decomposed into typed outcomes and small
handlers — same behavior, far more legible and testable:

- **Outcome types:** `_SuccessProto`, `_SuccessJson`, `_TryAgain`, `_MetadataOnly`,
  `_Failed`; transport errors normalized into `_TransportError` with a
  `_TransportErrorKind` (RETRY / RETRY_WITH_BACKOFF / RETRY_IF_BUDGET /
  RETRYABLE_EXCEPTION / FATAL).
- Loop state hoisted into `_LoopState`. New methods: `_fetch_via_rest`,
  `_handle_transport_error`, `_handle_outcome`, `_check_timeout`.
- HTTP status → action mapping centralized (`_rest_status_error_to_transport_error`):
  408→retry (surfaces queue_state), 410→retryable, 5xx→retry, 429→backoff, bare
  400→retry up to 3× (proxy‑injected), else fatal. FATAL/400 events now carry
  request/response headers + body for post‑mortems.

---

## 10. Stuck‑detection made optional (`lib/retry_handler.py`)

- `RetryConfig.enable_stuck_detection` (default `True`) gates the
  `progress_timeout` "Requests appear to be stuck" check. When false, requests can
  block indefinitely.
- `SamplingClient` disables it when the server sets
  `sample_enable_stuck_detection=False`.

---

## 11. `TrainingClient` cleanups (`lib/public_interfaces/training_client.py`)

- **`save_state` / `save_weights_for_sampler` gain `overwrite`** (also a new
  `SaveWeightsRequest.overwrite` field). The old "treat 409 Conflict as success"
  hack was **removed** in favor of explicit `overwrite`.
- `forward` / `forward_backward` share a new `_run_fwd_bwd(..., forward_only=)`
  implementation; `_send_single_*` helpers inlined.
- **Parallel‑chunk submission ordering:** when sending >1 chunk in parallel, the
  first chunk (lowest seq_id) is now submitted *last*, so by the time it lands the
  rest are already queued and the server can pick up the whole batch together.
- `load_state*` / `save_weights_for_sampler*` restructured to return `APIFuture`
  from a single `_*_impl`; `save_weights_and_get_sampling_client_submit` removed.
- New flag `sample_no_retries` (`ClientConfigResponse`): `SamplingClient.sample()`
  can bypass the retry handler entirely.
- New tokenizer mapping for `moonshotai/Kimi-K2.6`.

---

## 12. Telemetry decorator hygiene

Many `@capture_exceptions(fatal=True)` decorators were moved off the public
sync/async wrapper methods and onto the inner `_*_async` coroutines across
`service_client.py`, `sampling_client.py`, and `training_client.py` (so the
`TelemetryProvider` is discovered via the coroutine's closure, not the wrapper).
`ServiceClient` now passes `_strict_response_validation=True` through kwargs.

New connection‑pool type: `ClientConnectionPoolType.CHECKPOINT_ARCHIVE_URL`.

---

## 13. Misc

- **CLI:** `tinker checkpoint list` command renamed internally
  (`list` → `list_checkpoints`, `@cli.command(name="list")`) to stop shadowing the
  builtin `list`.

## 14. Latest commits (`b1e4ee3..0883375`, v0.22.3 → v0.22.6)

- **Session-less REST clients** (`internal_client_holder.py`, `service_client.py`):
  new `_skip_session` mode on `InternalClientHolder`. `_session_id` may now be
  `None`; heartbeat/telemetry are skipped. `create_rest_client_with_weights_access_token`
  passes `_skip_session=True` so a weights-info lookup under a *different org's*
  token doesn't try to create a session in that org's (possibly read-only) Default
  project. `get_session_id()` now asserts non-None.
- **`TINKER_PROJECT_ID` env fallback** (`service_client.py`): `ServiceClient(project_id=None)`
  falls back to the `TINKER_PROJECT_ID` environment variable.
- **`list_training_runs(project_id=...)`** (`rest_client.py`): optional filter to
  scope results to a single project (`project_id` query param).
- **`sample_max_concurrent_requests`** (new `ClientConfigResponse` flag, default
  `2000`): server-controlled cap on in-flight sampling requests. Always applied as
  the sampling `RetryConfig.max_connections`, overriding any caller-provided value.
- **Security/build:** `scripts/publish-pypi` no longer uses `set -x` and passes the
  PyPI token via `UV_PUBLISH_TOKEN` env (never in argv/logs). `zstandard` bumped
  `>=0.22.0` → `>=0.24.0`.
- New tests: `tests/test_list_training_runs.py`, `tests/test_service_client.py`.

## Tests

New / updated test coverage (~1,060 lines added):

- `tests/test_proto_request_conv.py` (378), `tests/test_proto_response_conv.py` (213)
- `tests/test_pydantic_conv.py` (66), `tests/test_tensor_data.py` (31)
- `tests/test_checkpoint_archive_url.py` (157), `tests/test_checkpoint_delete.py` (32)
- `lib/_jwt_auth_test.py` (142, new), `lib/service_client_test.py` (+28),
  `lib/internal_client_holder_test.py` (±18)
- `tests/conftest.py` trimmed (removed httpx fixture scaffolding, −68).

---

## Server‑flag summary (all in `ClientConfigResponse`)

| Flag | Default | Area |
|------|---------|------|
| `use_pyqwest_transport` | `True` | HTTP transport (§1) |
| `proto_write_fwdbwd` | `False` | proto fwd/bwd (§2) |
| `proto_compress_fwdbwd` | `False` | zstd compression (§2) |
| `fwd_via_fwdbwd` | `False` | forward via /forward_backward (§2) |
| `parallel_fwdbwd_chunks` | `True` *(flipped)* | parallel chunks (§2) |
| `sample_no_retries` | `False` | sampling retries (§11) |
| `sample_enable_stuck_detection` | `True` | stuck detection (§10) |
| `sample_max_concurrent_requests` | `2000` | sampling concurrency cap (§14) |
| `billing_exception_max_pause_duration_sec` | `3600` | billing pause (§8) |
