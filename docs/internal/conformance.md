# localtinker Conformance Report

This report tracks the Python SDK surface that localtinker intentionally
supports and the hosted service comparison gaps that remain.

## Local SDK Coverage

The script tests in `cmd/localtinker/testdata` run against the real Python
Tinker SDK when `TINKER_SDK_DIR` is available.

| Area | Evidence | Status |
| --- | --- | --- |
| Handshake | `sdk_handshake.txt` creates a `ServiceClient`, reads server capabilities, and verifies empty REST run and checkpoint listings. | Covered |
| Sessions | `internal/tinkerhttp/routes_test.go` covers SDK REST session listing and lookup shapes. | Covered locally |
| Futures | `sdk_handshake.txt`, `sdk_async_errors.txt`, `internal/tinkercoord/coordinator_test.go`, and `internal/tinkerhttp/routes_test.go` verify missing future errors, future id echoing, queue state, cancellation, and user-error futures. | Covered locally |
| Training | `sdk_smoke.txt` creates a LoRA training client, calls `get_info`, `forward`, `forward_backward`, and `optim_step`, and checks loss decreases. | Covered |
| Metrics | `sdk_smoke.txt` checks `loss:mean` and that `optimizer_backend:mlx` is reported by an optimizer step. | Covered |
| Cross-entropy | `internal/tinkertrain/crossentropy_test.go` covers dense target tensors, weighted dense loss, and returned per-token logprobs. | Covered locally |
| Checkpoints | `sdk_smoke.txt` saves, loads, loads optimizer state, lists, archives, publishes, unpublishes, sets TTL, and deletes checkpoints. It also opens the generated archive and checks expected files. | Covered locally |
| Archives | `internal/tinkerhttp/routes_test.go` covers HTTP archive download URLs, local archive expiration, owner metadata headers, and proxied host headers. | Covered locally |
| Sampler | `sdk_smoke.txt` saves sampler weights, creates a sampling client, and samples with logprobs, prompt logprobs, seed, temperature, top-p, top-k, and stop settings. `sdk_sampling.txt` and `internal/tinkertrain/sample_test.go` cover integer stops, tokenizer-backed string stops, generated token logprobs, prompt logprobs, and top-k prompt logprob shapes. | Covered locally |
| Malformed inputs | `sdk_malformed_inputs.txt` and `sdk_async_errors.txt` verify malformed training, sparse `TensorData`, and async request errors. | Covered |

Run the local suite with:

```
go test ./cmd/localtinker -run TestPythonSDKScript -count=1
```

Run all Go coverage with:

```
go test ./...
```

## Publicization Release Gate

Do not publicize localtinker until each gate below has current evidence from
the commit being announced.

| Gate | Command or artifact | Pass condition |
| --- | --- | --- |
| Clean tree | `git status --short` | No unrelated source, generated cache, model, binary, or secret changes. |
| Unit and route coverage | `MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib LOCALTINKER_QWEN3_8B_MLX_BASE=mlx-community/Qwen3-0.6B-bf16 GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off go test ./...`; latest recorded pass: `773a4d9`. | All packages pass from a clean checkout with the intended module graph. |
| Python SDK smoke | `MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib LOCALTINKER_QWEN3_8B_MLX_BASE=mlx-community/Qwen3-0.6B-bf16 GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off go test ./cmd/localtinker -run TestPythonSDKScript -count=1`; latest recorded pass through `go test ./...`: `773a4d9`. | All `cmd/localtinker/testdata/sdk_*.txt` scripts pass against a real Tinker SDK checkout. |
| Local runner override | Manual flow below | A normal SDK job runs with only endpoint and credential environment overrides. |
| Hosted comparison | JSONL artifact below | Hosted and local response keys, metric names, checkpoint metadata shape, and sampler output shapes are compared. |
| Public caveats | Known Differences below | Every unsupported hosted feature is either not advertised or documented as a caveat. |

Use isolated caches when running gates on a shared machine:

```
MLX_LIB_PATH=/path/to/mlx/lib GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off go test ./...
```

On this machine, the verified local MLX library directory is
`/Users/tmc/ml-explore/mlx-go/mlxc/lib`. The
`/Volumes/tmc/go/src/github.com/tmc/mlx-go-libs/dist/darwin-arm64` copy
contains `mlx.metallib.gz`; it loads `libmlxc.dylib` but does not satisfy the
default uncompressed metallib lookup used by the current tests.

## Local Runner Override Flow

This is the exact local override shape the SDK smoke test uses. It must keep
ordinary SDK code pointed at localtinker without changing imports or hosted SDK
call sites.

1. Build and start the local server:

```
go build -o /tmp/localtinker ./cmd/localtinker
/tmp/localtinker serve -addr 127.0.0.1:8080 -home /tmp/localtinker-home
```

2. Point the Python SDK at the local endpoint:

```
export TINKER_SDK_DIR=$HOME/go/src/github.com/thinking-machines-lab/tinker
export PYTHONPATH=$TINKER_SDK_DIR/src
export TINKER_API_KEY=tml-local-test
export TINKER_BASE_URL=http://127.0.0.1:8080
```

If the SDK virtualenv is not at `$TINKER_SDK_DIR/.venv/bin/python`, set:

```
export LOCALTINKER_SDK_PYTHON=/path/to/python
```

3. Run the SDK smoke suite:

```
go test ./cmd/localtinker -run TestPythonSDKScript -count=1
```

4. Run a normal SDK job without changing code:

```
${LOCALTINKER_SDK_PYTHON:-$TINKER_SDK_DIR/.venv/bin/python} \
  ./cmd/localtinker/examples/tinker_job.py --preset short
```

## Hosted Comparison

Hosted comparison artifacts under `hosted comparison fixtures` record
paired SDK runs against hosted Tinker and localtinker. Before treating a new SDK
surface as hosted-compatible, run the same minimal SDK job against both backends
and record the fields below. The credentialed hosted follow-up procedure is in
`docs/internal/hosted-probes.md`.

- training run ID
- future request IDs and terminal response shapes
- loss and optimizer metric names
- checkpoint path, listing metadata, archive metadata, and download behavior
- sampler response fields for the same prompt, seed, logprob flags, and
  sampling parameters

Record each run as JSONL so shape comparisons can be reviewed without replaying
the job. Each line should include a stable comparison ID, a case ID shared by
the hosted and local observations for the same SDK action, and an event payload:

```
{"comparison_id":"YYYYMMDD-<short-commit>","case_id":"train-short","backend":"hosted|local","event":"meta|capabilities|future|metrics|checkpoint|sampler","payload":{}}
```

Required events:

- `meta`: local commit, SDK commit, model, runner machine, Python executable,
  `TINKER_BASE_URL` class (`hosted` or `local`, not the secret URL), and
  checkpoint root.
- `capabilities`: supported model name, tokenizer ID, and advertised feature
  names.
- `future`: operation name, request ID, future ID, terminal category, terminal
  state, and top-level response keys.
- `metrics`: operation name and sorted metric keys; keep numeric values for
  loss and optimizer metrics.
- `checkpoint`: tinker path, checkpoint type, owner, visibility, expiration,
  archive URL scheme, archive metadata headers, and downloaded file names.
- `sampler`: prompt tokens, generated tokens, stop reason, sequence logprob
  shape, prompt logprob shape, seed, temperature, top-p, top-k, and stop input.

The comparison passes only when required response keys and advertised feature
flags match the supported local surface. Numeric values may differ, but metric
names and tensor/logprob shapes should match unless the difference is recorded
below.

Store the artifact under `hosted comparison fixtures` using the pattern:

```
YYYYMMDD-<short-commit>[-<case>]-hosted-local.jsonl
```

Each artifact should name the local commit, SDK commit, model, runner machine,
Python executable, `TINKER_BASE_URL` class (`hosted` or `local`, not the secret
URL), and checkpoint root. Do not record API keys, signed archive URLs, local
home directories, or downloaded model paths.

## Parity Fixes (2026-06-06 SDK audit)

A multi-agent audit mapped the upstream SDK HTTP contract
(`src/tinker/lib/public_interfaces/*`) against the local routes, then verified
each candidate against the actual code on both sides: 25 candidates
investigated, 6 confirmed, 19 cleared as non-gaps. The four confirmed major
gaps are fixed and pinned:

- `load_weights` read the wrong field. The SDK posts both `load_state` and
  `load_state_with_optimizer` to `/api/v1/load_weights` distinguished by the
  body key `optimizer` (`load_weights_request.py:18`); the handler decoded only
  `optimizer_state`, so `optimizer:true` was silently dropped and optimizer
  state was never restored. The handler now reads `optimizer` (with
  `optimizer_state` kept as a fallback alias) via `loadWeightsRequest`. Pinned
  by `internal/tinkerhttp.TestLoadWeightsOptimizerFlag`.
- `ttl_seconds` on `save_weights`/`save_weights_for_sampler`
  (`save_weights_request.py:20`) was not decoded, so a save-time TTL was lost.
  The save handlers now decode `ttl_seconds`, share `ttlSecondsToDuration` with
  the dedicated `/ttl` route, and thread it into `SetCheckpointTTL`. Pinned by
  `internal/tinkerhttp.TestSaveTTL` and
  `TestSaveWeightsRejectsNegativeTTL`.
- `GET /api/v1/samplers/{id}` returned a hardcoded `Qwen/Qwen3-8B` base model
  and ignored the session id, producing the wrong tokenizer for non-default
  base models. It now resolves the recorded session via
  `Coordinator.SamplerInfo` / `Manager.SamplerInfo` and returns 404 for unknown
  ids. Pinned by `internal/tinkertrain.TestSamplerInfo` and the updated
  `internal/tinkerhttp.TestSamplerRESTRoute`.
- `AdamParams.beta1/beta2/eps` were ignored. The training path used the
  dependency's `TrainingMode="separate"` factory, which hardcodes
  `beta2=0.999, eps=1e-8` and drops weight decay; the SDK defaults are
  `beta2=0.95, eps=1e-12`. `trainDenseStep` now builds the AdamW step directly
  with `training.NewCompiledAdamW`, applying SDK defaults via
  `AdamParams.withDefaults`. Pinned by
  `internal/tinkertrain.TestAdamParamsWithDefaults`.

Two cosmetic items remain intentionally unaddressed (functionally inert for the
local single-tenant runtime): the optional `weights_access_token` field on
`load_weights` (no access-control subsystem exists) and server-side
`tinker://` prefix validation on `create_sampling_session` (the SDK validates
this client-side).

## Parity Fixes (SDK window af041ee..b1e4ee3)

A follow-up audit of the upstream `af041ee..b1e4ee3` SDK window confirmed one
server-surface gap and cleared the rest:

- `save_weights`/`save_weights_for_sampler` now honor the SDK's explicit
  `overwrite` bool (`save_weights_request.py`, default false). With
  `overwrite:false` a named save against an existing checkpoint fails as a
  terminal in-band `FutureUserError` (`checkpoint already exists: <name>`),
  surfaced as HTTP 200 + `{error, category:"user"}` through `retrieve_future`;
  it is deliberately not an HTTP 409, since the SDK's `execute_with_retries`
  treats 409 as retryable and the SDK has removed its old "treat 409 as
  success" hack. With `overwrite:true` the existing checkpoint dir is removed
  before the save so stale files cannot leak. Sampler ephemeral saves
  (`path==""`) are exempt. `seq_id` is accepted and ignored for
  forward-compatibility. Pinned by
  `internal/tinkercoord.TestSaveWeightsDuplicateIsUserError`,
  `TestSaveWeightsForSamplerDuplicateIsUserError`,
  `TestSaveWeightsForSamplerEphemeralIgnoresOverwrite`,
  `internal/tinkertrain.TestManagerCheckpointExistsAndRemove`,
  `TestManagerCheckpointExistsCleansName`, and
  `internal/tinkerhttp.TestSaveWeightsHonorsOverwrite`.

Out of scope for this window: audit-log and `assign_session_project` (no
local equivalent and intentionally not surfaced); sparse CSR rehydration was
already implemented in the prior audit; the remaining SDK changes are
client-only.

## Total parity sweep (2026-06-06, upstream b1e4ee3 / v0.22.3)

A full route-by-route parity sweep of localtinker against the current upstream
Tinker SDK (`b1e4ee3`, v0.22.3) confirmed all endpoint groups at parity for the
single-tenant local surface, and closed one remaining response-fidelity gap:

- `weights_info` now reflects each model's real LoRA training configuration.
  `CreateModel` records `train_mlp`/`train_attn`/`train_unembed` from the SDK's
  `lora_config` (`service_client.py:125-131`, defaulting to `LoraConfig`'s
  all-true values when absent), and the `weights_info` handler echoes the stored
  flags instead of the previous hardcoded `train_unembed=false,
  train_mlp=true, train_attn=true`. The SDK reads these back during
  `create_training_client_from_state` (`service_client.py:280-284`) to recreate a
  training client, so resume workflows that used non-default flags now round-trip
  correctly. Unknown paths still fall back to the SDK's `LoraConfig` defaults.
  Pinned by `internal/tinkerhttp.TestWeightsInfoReflectsTrainingConfig` and
  `internal/tinkercoord.TestBoolFromMap`.

Deliberately out of scope (return a local error for unsupported hosted features,
never a silent stub): audit-log/RBAC, `assign_session_project` (multi-tenant),
`weights_access_token`, hosted signed-URL emulation, cross-owner auth, JWT
(disabled), billing 402, telemetry internals. Client-only and therefore no
server change: pyqwest transport, proto/zstd forward-backward (flag default-off;
localtinker advertises nothing so the SDK uses JSON), retry-handler/`_APIFuture`
internals, stuck-detection, and the CLI. Numeric MLX-vs-hosted differences remain
expected and documented.

## Ecosystem Parity (2026-06-08, unmodified tinker-cookbook recipes)

Wire parity asks whether localtinker speaks the SDK's protocol; ecosystem
parity asks the harder question: does an *unmodified* `tinker-cookbook` recipe
run against it end to end? `internal/tinkerhttp.TestCookbookRecipeScript`
(`cmd/localtinker/cookbook_script_test.go`) answers it by booting a localtinker
server and running real cookbook recipes (`testdata/recipe_*.txt`) through
rsc.io/script against the cached `mlx-community/Qwen3-0.6B-bf16` weights. Each
recipe runs on its own server because localtinker serves MLX operations one at a
time; recipes that download datasets, need an external judge/teacher API, or
require a cloud sandbox cannot run offline in CI and are skipped with a reason
rather than silently dropped.

Recipes that pass unmodified (model targeted as `Qwen/Qwen3-8B`, which the
server maps to the local 0.6B MLX checkpoint):

- **chat_sl** (supervised fine-tuning) — one LoRA step over an inline JSONL
  conversation dataset, then a resume from the saved checkpoint. Asserts
  `train_mean_nll` and the `state_path` checkpoint are written, and the resume
  reloads optimizer state and logs the resume line. `testdata/recipe_chat_sl.txt`.
- **math_rl** (GRPO) — one step on the synthetic offline `arithmetic`
  environment: samples rollouts, computes group rewards, trains. Asserts
  `reward/total`. `testdata/recipe_math_rl.txt`.
- **preference / shorter** (DPO-family) — pairwise-preference RL where the
  comparison dataset is an in-process dummy and the preference signal is response
  length, with no judge or teacher model. Asserts `reward/total`.
  `testdata/recipe_preference.txt`.

Real server gap found and fixed by running these recipes:

- **Optimizer-resume response type** — the chat_sl resume failed because the
  coordinator tagged the optimizer-state load's future result with
  `type="load_state_with_optimizer"`, but the SDK retrieves it into
  `LoadWeightsResponse`, whose type is `Literal["load_weights"]`, so pydantic
  rejected it. The optimizer load is distinguished by the `optimizer_state`
  flag, not the type; the coordinator now always reports `load_weights`
  (`internal/tinkercoord.LoadWeightsWithOptimizer`). This unblocked every
  cookbook recipe that resumes from a saved state. The existing route tests only
  covered the error paths, so the success-path type was never asserted before;
  the chat_sl recipe is now its regression test.

Skipped, with reason (cannot run unmodified offline in CI — none is a localtinker
server gap):

- **preference / dpo, rlhf** — pull HuggingFace datasets (`Anthropic/hh-rlhf`,
  `nvidia/HelpSteer3`, `argilla/ultrafeedback`) over the network; fail under the
  harness's `HF_HUB_OFFLINE=1`. Only `shorter` is offline.
- **distillation / sdft** — has an offline local-Arrow dataset branch, but its
  final-step evaluator hits a cookbook-internal bug
  (`tinker_cookbook/rl/rollouts.py:264` calls `.cleanup()` on a `list`), which
  fires whenever the 0.9/0.1 split leaves a non-empty test set — i.e. always.
  The bug is in the cookbook's RL test-set evaluator, not localtinker (the server
  served every sampling request up to that point). No CLI flag disables the
  split or the eval, so it cannot run unmodified.
- **evaluation** — no `recipes/evaluation/` family; the only eval entrypoint
  (`tinker_cookbook/eval/run_inspect_evals.py`) imports `inspect_ai` (absent from
  the venv), and the benchmark path downloads HF datasets with no synthetic
  fallback. It also emits no `metrics.jsonl` to assert against.
- **tool_use** — no `recipes/tool_use/` training entrypoint; `search_tool`
  requires a HuggingFace dataset download, a running Chroma vector-DB server, and
  the Google Gemini embedding API, and `harbor_rl` requires a Modal cloud Docker
  sandbox and an externally populated task cache.

## Can We Publicize?

Current answer: not yet for a broad public launch. It is close enough for a
limited public beta only if the caveats below are stated plainly.

| Area | Status | Publicization decision |
| --- | --- | --- |
| SDK route surface | Meaningful local coverage exists for sessions, futures, training, checkpoints, sampler flows, malformed inputs, and REST listings. | Beta-ready with the covered surface named. |
| Hosted comparison | `hosted comparison fixture` records hosted-vs-local checkpoint, optimizer-resume, metrics, and sampler shape evidence. `hosted comparison fixture` records custom-loss shape evidence. `hosted comparison fixture` records dense cross-entropy shape and loss evidence. The `20260511-0480f94-*` artifacts add current local policy-loss, cancel-future, queue-backpressure, and archive authorization evidence. The `20260511-55ffaf5-*` hosted artifacts replace several credential blockers with live hosted observations. `hosted comparison fixture` records hosted two-future scheduler timing. `hosted comparison fixture` records hosted archive denial for a synthetic invalid credential. | Covered for recorded shapes; numeric differences and policy-loss behavior remain caveats. |
| Checkpoint downloads | Local HTTP archive URLs and metadata are covered. `hosted comparison fixture` records the current local route shape. `hosted comparison fixture` records hosted owner signed-URL shape for sampler weights. `hosted comparison fixture` records hosted `401` invalid-credential denial for the same archive URL surface. Hosted cross-owner authorization still needs a valid second principal. | Disclose for beta; blocker for hosted-compatible wording. |
| Futures and queueing | Local queue, cancellation, panic containment, unfinished-queue recovery, and local queue backpressure timing are covered. `hosted comparison fixture` records the local cancel route and SDK retrieve-only surface. `hosted comparison fixture` records hosted raw `/api/v1/cancel_future` returning 404. `hosted comparison fixture` records the local one-slot queue case. `hosted comparison fixture` records hosted accepting two concurrent futures and returning `queue_state:"active"` for both. `hosted comparison fixture` records local scheduler retry, cancel, lease, and dashboard-node evidence. | Beta-ready with queue-state caveat. |
| Cross-entropy | Dense tensors, CSR `target_tokens`/`weights` rehydration, unsupported sparse tensor rejection, invalid weights, and logprobs are covered locally. `hosted comparison fixture` records matching hosted/local per-token logprob shapes and a forward loss mean difference. `hosted comparison fixture` records hosted arbitrary fractional dense weights succeeding with `loss:sum` and per-token outputs. | Beta-ready with numeric and metric-name caveats. |
| Custom losses | `hosted comparison fixture` records hosted and local `forward_backward_custom` success and `custom_loss:mean` metric shape evidence. | Beta-ready with numeric caveats. |
| Sampling | Generated logprobs, prompt logprobs, deterministic seed flow, string stops, and top-k prompt logprob shapes are covered locally and in hosted comparison rows. `hosted comparison fixture` records live hosted seeded samples for fixed token prompts; `hosted comparison fixture` records the paired local Qwen/Qwen3-8B run and matching sequence/logprob shapes. | Beta-ready with numeric/distribution caveats. |
| Packaging | Clean-checkout `MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib LOCALTINKER_QWEN3_8B_MLX_BASE=mlx-community/Qwen3-0.6B-bf16 GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off go test ./...`; latest recorded pass: `773a4d9`. | Beta-ready; rerun before any release commit. |
| Secrets and artifacts | Hosted comparison JSONL artifacts use scrubbed runner metadata (`python`, `local-runner`). No keys, binaries, downloaded weights, generated caches, or private model paths should be staged. | Beta-ready; keep scanning before release. |

## Known Differences

Each bullet cites the JSONL row(s) that back it, or states that no hosted
probe is on record. Evidence reviewed 2026-05-11.

- Futures have local queue, running, cancellation, and result-byte accounting.
  `hosted comparison fixture`
  records the local `/api/v1/cancel_future` route, queued-future cancellation,
  and current SDK evidence that generated futures are retrieve-only with no
  client cancel API.
  `hosted comparison fixture`
  records the local one-slot backpressure case: one future is running, the next
  remains queued until the first releases its slot, dashboard queue counts show
  one running and one queued, and FIFO dispatch is preserved. Hosted raw
  `POST /api/v1/cancel_future` returns 404 for `id`, `request_id`, and
  `future_id` request shapes in
  `hosted comparison fixture`.
  `hosted comparison fixture`
  records the hosted two-future timing fixture: both raw `forward_backward`
  submissions returned 202 with distinct request IDs, and metadata-only
  `retrieve_future` returned 408 `try_again` with `queue_state:"active"` for
  both futures across the probe window. Hosted did not expose the local
  one-running/one-queued state in this recorded fixture.
- Checkpoint archive URLs are local HTTP download URLs, not hosted signed
  download URLs. Hosted rows record `scheme=https`, `has_query=true`,
  `content_disposition_present=false`; local rows record `scheme=http`,
  `content_disposition_present=true` and an extra `checkpoint.json` file
  (`20260505-951b2dc` rows 11 and 23; `20260505-a995c00` rows 11 and 24).
- Checkpoint ownership is recorded as `owner=null`, `public=false` on both
  sides in the recorded archive_download events
  (`20260505-951b2dc` rows 11/23, `20260505-a995c00` rows 11/24).
  Local archive responses now reflect coordinator state on the
  `X-Tinker-Archive-Owner` and `X-Tinker-Archive-Visibility` headers
  (`private` by default, `public` after publish, back to `private` after
  unpublish); pinned by
  `internal/tinkerhttp.TestCheckpointArchiveAuthorization` and recorded
  in `hosted comparison fixture`
  and refreshed at `0480f94` in
  `hosted comparison fixture`.
  Hosted owner-side archive shape for sampler weights is recorded in
  `hosted comparison fixture`:
  hosted returns an HTTPS object-store signed URL ending in `archive.tar` with
  six `X-Goog-*` query keys and an `expires` timestamp. Hosted invalid-token
  denial is recorded in
  `hosted comparison fixture`:
  the same archive URL surface returns `401` with `detail:"Unable to validate
  credential"` for a synthetic invalid credential. A valid second hosted
  principal is still required for true cross-owner private denial.
  Hosted-style signed URL emulation and cross-owner authorization remain
  intentionally out of scope for the local coordinator.
- Dense cross-entropy per-token logprob shapes match the recorded hosted
  comparison, but forward loss means differ. Paired evidence: shape `[4]` on
  both sides and `absolute_difference=0.5989780426025391`
  (`20260505-497eb1c` row 11, hosted source `computed_from_logprobs`,
  local source `metrics.loss:mean`).
- Policy losses `importance_sampling`, `ppo`, `cispo`, and `dro` execute
  locally with the public Tinker loss formulas and return per-token
  `logprobs`; policy loss metrics use `loss:sum`. They are intentionally not
  advertised in `get_server_capabilities` or `tinker.Client.Capabilities`
  because hosted rejects the recorded SDK-shaped fixture before metrics. PPO
  and CISPO default to clip thresholds `0.8` and `1.2`; DRO requires an
  explicit `loss_fn_config["beta"]` because no hosted default is documented.
  Pinned by
  `internal/tinkertrain.TestDensePolicyLossesReturnWeightedSumAndLogprobs`.
  `hosted comparison fixture`
  records local `forward` and `forward_backward` rows for all four policy
  losses. Live hosted evidence in
  `hosted comparison fixture`
  shows the same SDK-shaped TensorData fixture failing for all four policy
  losses with `RequestFailedError`, category `unknown`, and message signature
  `could_not_convert_loss_function_inputs_to_array_record`. Local policy-loss
  execution remains available for experiments, but the advertised hosted-
  compatible loss list is limited to `cross_entropy`.
- CSR sparse `target_tokens` and `weights` are accepted and rehydrated to
  dense tensors in both the direct Go API and HTTP path. Pinned by
  `internal/tinkertrain.TestDatumTargetsRehydratesSparse`,
  `internal/tinkertrain.TestDatumWeightsRehydratesSparse`, and
  `internal/tinkerhttp.TestNormalizeTensorDataRehydratesNamedSparse`.
  Sparse tensors for unsupported names (for example `advantages` or
  policy-loss `logprobs`) are rejected before MLX execution. Local-only
  contract; no hosted probe recorded.
- Arbitrary non-prefix fractional dense weights (e.g. `[0.25, 1, 0, 0.75]`
  or `[1, 0, 0.3, 0, 0.7, 1]`) are accepted by `newDenseBatch` and
  `denseCrossEntropy` returns the weighted mean
  `(sum w_i * -logp_i) / sum w_i`. Pinned by
  `internal/tinkertrain.TestDenseCrossEntropyFractionalWeights` and
  recorded in
  `hosted comparison fixture`.
  Live hosted evidence in
  `hosted comparison fixture`
  shows the `[0.25, 1, 0, 0.75]` dense-weight fixture succeeds for both
  `forward` and `forward_backward`; hosted reports `loss:sum` and returns
  `elementwise_loss` plus `logprobs`, while local cross-entropy reports
  `loss:mean`.
- Multimodal model input chunks (`image` and `image_asset_pointer`) are
  parsed and token-counted at the SDK boundary, then refused at the MLX
  executor with a typed error that names the missing capability: `image`
  chunks fail with "image chunks require a vision backend, which the local
  MLX runtime does not provide"; `image_asset_pointer` chunks fail with
  "image_asset_pointer chunks require a local image asset store, which is
  not configured". `image.data` must begin with the PNG (`89 50 4E 47 …`)
  or JPEG (`FF D8 FF`) magic prefix matching the declared `format`;
  arbitrary nonempty bytes are rejected at parse. Pinned by
  `internal/tinkertrain.TestImageChunkHeaderValidation`,
  `TestMultimodalChunkParsesAndCounts`, `TestMultimodalExecutionRejected`,
  and the HTTP forward-rejection table in
  `internal/tinkerhttp.TestConformanceMalformedTrainingInputsReturnUserErrors`.
  Local-only contract; no hosted probe recorded.
- The image asset store boundary is exposed as the
  `tinkertrain.ImageAssetResolver` extension point. A fresh `Manager`
  installs `DefaultImageAssetResolver`, which refuses with the typed
  sentinel `ErrImageAssetStoreNotConfigured` so callers can detect the
  missing-store boundary via `errors.Is` instead of string matching.
  Embedders may plug in a real store (or
  `tinkertrain.NewMapImageAssetResolver` for tests) via
  `Manager.SetImageAssetResolver`. `tinkertrain.ResolveImageAssetPointer`
  resolves a pointer chunk through the resolver and revalidates the
  resulting bytes through `ValidateImageChunk`, so the magic-byte
  contract is the single source of truth for both inline and resolved
  image chunks. The MLX executor still refuses any multimodal chunk
  regardless of resolver — pinned by
  `TestExecutorRefusesImageAssetPointerEvenWithResolver`.
- HTTP runtime users wire a resolver into the live server programmatically:
  build a `tinkertrain.Manager`, call `SetImageAssetResolver`, and pass
  the Manager as `tinkercoord.Config.Train` to `tinkercoord.New` before
  constructing `tinkerhttp.New`. `Manager.ImageAssetResolver` retrieves
  the live resolver out-of-band, since the HTTP handlers refuse
  multimodal execution before the resolver would run. `cmd/localtinker`
  intentionally exposes no flag for this — image assets are an
  embedder-supplied capability, not a server-config knob. Pinned by
  `internal/tinkerhttp.TestImageAssetResolverReachableThroughHTTPConfig`
  and `TestImageAssetResolverDefaultRefusalThroughHTTPConfig`.
- Hosted optimizer-resume response shape is recorded
  (`tinker_path_kind=weights`, `path_prefix_ok=true`,
  `response_path_matches=true`, `optimizer_state=null` on both sides:
  `20260505-951b2dc` rows 10/22, `20260505-a995c00` rows 10/23), but exact
  optimizer internals and numeric continuation are not asserted as equivalent.
  The current paired evidence records the same cross-entropy
  `forward_backward`, `optim_step`, TTL-compatible `save_state`, and
  `create_training_client_from_state_with_optimizer` shape on hosted and local:
  `hosted comparison fixture`
  and
  `hosted comparison fixture`.
  Hosted `optim_step` metrics arrive empty
  (`20260505-951b2dc` row 7, `20260505-a995c00` row 7, and
  `hosted comparison fixture`)
  while local emits
  `loss:mean`, `optimizer_backend:mlx`, `optimizer_step:unique`
  (`20260505-951b2dc` row 19, `20260505-a995c00` row 20, and
  `hosted comparison fixture`).
  Exact optimizer numeric equivalence is therefore not observable from hosted
  `optim_step` responses; the remaining difference is metric surface, not a
  local implementation gap.
- Hosted and local numeric results are not expected to match exactly. The
  CE forward loss mean differs by ~0.599 (`20260505-497eb1c` row 11) and the
  sampler returns different generated tokens for the same seed/temperature/
  top-p/top-k input — hosted `[15136, 1]` vs local `[4, 4]`
  (`20260505-a995c00` rows 13 and 26). Hosted seeded sample rows for the
  current SDK are recorded in
  `hosted comparison fixture`;
  the paired local Qwen/Qwen3-8B run is recorded in
  `hosted comparison fixture`.
  The paired artifacts match sequence counts, generated-token counts,
  generated-logprob counts, prompt-logprob counts, and top-k prompt-logprob
  presence for the fixed prompt/seed cases; generated token IDs remain
  distribution evidence, not an equality assertion.
