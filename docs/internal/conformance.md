# localtinker Conformance Report

This report tracks the Python SDK surface that localtinker intentionally
supports and the hosted-comparison gaps that remain.

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
| Unit and route coverage | `MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off go test ./...`; latest recorded pass: `65f4c6e`. | All packages pass from a clean checkout with the intended module graph. |
| Python SDK smoke | `MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off go test ./cmd/localtinker -run TestPythonSDKScript -count=1`; latest recorded pass: `65f4c6e`. | All `cmd/localtinker/testdata/sdk_*.txt` scripts pass against a real Tinker SDK checkout. |
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

Hosted comparison artifacts under `docs/internal/hosted-comparison/` record
paired SDK runs against hosted Tinker and localtinker. Before treating a new SDK
surface as hosted-compatible, run the same minimal SDK job against both backends
and record:

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

Store the artifact under `docs/internal/hosted-comparison/` using the pattern:

```
YYYYMMDD-<short-commit>[-<case>]-hosted-local.jsonl
```

Each artifact should name the local commit, SDK commit, model, runner machine,
Python executable, `TINKER_BASE_URL` class (`hosted` or `local`, not the secret
URL), and checkpoint root. Do not record API keys, signed archive URLs, local
home directories, or downloaded model paths.

## Can We Publicize?

Current answer: not yet for a broad public launch. It is close enough for a
limited public beta only if the caveats below are stated plainly.

| Area | Status | Publicization decision |
| --- | --- | --- |
| SDK route surface | Meaningful local coverage exists for sessions, futures, training, checkpoints, sampler flows, malformed inputs, and REST listings. | Beta-ready with the covered surface named. |
| Hosted comparison | `docs/internal/hosted-comparison/20260505-a995c00-hosted-local.jsonl` records hosted-vs-local checkpoint, optimizer-resume, metrics, and sampler shape evidence. `docs/internal/hosted-comparison/20260505-ecc480f-custom-loss-hosted-local.jsonl` records custom-loss shape evidence. `docs/internal/hosted-comparison/20260505-497eb1c-ce-hosted-local.jsonl` records dense cross-entropy shape and loss evidence. | Covered for recorded shapes; numeric and authorization differences remain caveats. |
| Checkpoint downloads | Local HTTP archive URLs and metadata are covered; hosted signed URL and authorization behavior are not matched. | Disclose for beta; blocker for hosted-compatible wording. |
| Futures and queueing | Local queue, cancellation, panic containment, and unfinished-queue recovery are covered. Hosted cancellation could not be compared: the generated hosted SDK exposes retrieve-only futures and the probed hosted cancel route returned 404. Hosted timing/backpressure is also not compared. | Disclose for beta. |
| Cross-entropy | Dense tensors, invalid weights, sparse tensor rejection, and logprobs are covered locally. `docs/internal/hosted-comparison/20260505-497eb1c-ce-hosted-local.jsonl` records matching hosted/local per-token logprob shapes and a forward loss mean difference. | Beta-ready with numeric caveats. |
| Custom losses | `docs/internal/hosted-comparison/20260505-ecc480f-custom-loss-hosted-local.jsonl` records hosted and local `forward_backward_custom` success and `custom_loss:mean` metric shape evidence. | Beta-ready with numeric caveats. |
| Sampling | Generated logprobs, prompt logprobs, deterministic seed flow, string stops, and top-k prompt logprob shapes are covered locally and in hosted comparison rows. | Beta-ready with numeric/distribution caveats. |
| Packaging | Clean-checkout `MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off go test ./...`; latest recorded pass: `65f4c6e`. | Beta-ready; rerun before any release commit. |
| Secrets and artifacts | Hosted comparison JSONL artifacts use scrubbed runner metadata (`python`, `local-runner`). No keys, binaries, downloaded weights, generated caches, or private model paths should be staged. | Beta-ready; keep scanning before release. |

## Known Differences

Each bullet cites the JSONL row(s) that back it, or states that no hosted
probe is on record. Evidence reviewed 2026-05-07.

- Futures have local queue, running, cancellation, and result-byte accounting.
  Hosted cancellation is not exposed by the generated SDK and the probed
  hosted cancel route returned 404. No hosted probe of cancellation or queue
  backpressure is recorded in `docs/internal/hosted-comparison/`; the four
  JSONL artifacts only record successful `forward`, `forward_backward`,
  `optim_step`, `save_state`, `load_state_with_optimizer`, and
  `forward_backward_custom` futures (see e.g. `20260505-951b2dc` rows 3–9 and
  `20260505-a995c00` rows 3–9). Hosted scheduler timing and backpressure have
  not been compared.
- Checkpoint archive URLs are local HTTP download URLs, not hosted signed
  download URLs. Hosted rows record `scheme=https`, `has_query=true`,
  `content_disposition_present=false`; local rows record `scheme=http`,
  `content_disposition_present=true` and an extra `checkpoint.json` file
  (`20260505-951b2dc` rows 11 and 23; `20260505-a995c00` rows 11 and 24).
- Checkpoint ownership is recorded as `owner=null`, `public=false` on both
  sides in the recorded archive_download events
  (`20260505-951b2dc` rows 11/23, `20260505-a995c00` rows 11/24). No hosted
  probe of authorization-failure or cross-owner access is recorded.
- Dense cross-entropy per-token logprob shapes match the recorded hosted
  comparison, but forward loss means differ. Paired evidence: shape `[4]` on
  both sides and `absolute_difference=0.5989780426025391`
  (`20260505-497eb1c` row 11, hosted source `computed_from_logprobs`,
  local source `metrics.loss:mean`).
- Sparse `TensorData` inputs are rejected; only dense tensor inputs are
  supported. Local-only contract; no hosted probe recorded.
- Multimodal model input chunks (`image` and `image_asset_pointer`) are
  rejected instead of being silently ignored; image tensor processing is not
  implemented. Local-only contract; no hosted probe recorded.
- Hosted optimizer-resume response shape is recorded
  (`tinker_path_kind=weights`, `path_prefix_ok=true`,
  `response_path_matches=true`, `optimizer_state=null` on both sides:
  `20260505-951b2dc` rows 10/22, `20260505-a995c00` rows 10/23), but exact
  optimizer internals and numeric continuation are not asserted as
  equivalent. Hosted `optim_step` metrics arrive empty
  (`20260505-951b2dc` row 7, `20260505-a995c00` row 7) while local emits
  `loss:mean`, `optimizer_backend:mlx`, `optimizer_step:unique`
  (`20260505-951b2dc` row 19, `20260505-a995c00` row 20).
- Hosted and local numeric results are not expected to match exactly. The
  CE forward loss mean differs by ~0.599 (`20260505-497eb1c` row 11) and the
  sampler returns different generated tokens for the same seed/temperature/
  top-p/top-k input — hosted `[15136, 1]` vs local `[4, 4]`
  (`20260505-a995c00` rows 13 and 26). No systematic sampler-distribution
  comparison is recorded.
