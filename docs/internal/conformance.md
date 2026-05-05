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
| Sampler | `sdk_smoke.txt` saves sampler weights, creates a sampling client, and samples with logprobs, prompt logprobs, seed, temperature, top-p, top-k, and stop settings. `internal/tinkertrain/sample_test.go` covers integer stops, tokenizer-backed string stops, generated token logprobs, and prompt logprobs. | Covered locally |
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
| Unit and route coverage | `GOWORK=off go test ./...` | All packages pass from a clean checkout with the intended module graph. |
| Python SDK smoke | `go test ./cmd/localtinker -run TestPythonSDKScript -count=1` | All `cmd/localtinker/testdata/sdk_*.txt` scripts pass against a real Tinker SDK checkout. |
| Local runner override | Manual flow below | A normal SDK job runs with only endpoint and credential environment overrides. |
| Hosted comparison | JSONL artifact below | Hosted and local response keys, metric names, checkpoint metadata shape, and sampler output shapes are compared. |
| Public caveats | Known Differences below | Every unsupported hosted feature is either not advertised or documented as a caveat. |

Use isolated caches when running gates on a shared machine:

```
GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off go test ./...
```

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

No hosted comparison artifact is checked in yet. Before treating localtinker as
hosted-compatible, run the same minimal SDK job against hosted Tinker and
localtinker and record:

- training run ID
- future request IDs and terminal response shapes
- loss and optimizer metric names
- checkpoint path, listing metadata, archive metadata, and download behavior
- sampler response fields for the same prompt, seed, logprob flags, and
  sampling parameters

Record each run as JSONL so shape comparisons can be reviewed without replaying
the job. Each line should include:

```
{"backend":"hosted|local","event":"capabilities|future|metrics|checkpoint|sampler","payload":{}}
```

Required events:

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
YYYYMMDD-<short-commit>-hosted-local.jsonl
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
| Hosted comparison | No hosted-vs-local JSONL artifact is checked in. | Launch blocker for broad public claims. |
| Checkpoint downloads | Local HTTP archive URLs and metadata are covered; hosted signed URL and authorization behavior are not matched. | Disclose for beta; blocker for hosted-compatible wording. |
| Futures and queueing | Local queue, cancellation, panic containment, and unfinished-queue recovery are covered; hosted timing/backpressure is not compared. | Disclose for beta. |
| Cross-entropy | Dense tensors, invalid weights, sparse tensor rejection, and logprobs are covered locally; hosted numeric parity is not recorded. | Disclose for beta. |
| Custom losses | `ForwardBackwardCustom` is not implemented. | Do not advertise custom loss execution. |
| Sampling | Generated logprobs, prompt logprobs, deterministic seed flow, and string stops are covered locally; top-k prompt logprobs are not implemented. | Disclose unsupported top-k prompt logprobs. |
| Packaging | Clean-checkout `GOWORK=off go test ./...` must pass with the intended MLX module graph and native libraries. | Launch blocker until freshly proven. |
| Secrets and artifacts | No keys, binaries, downloaded weights, generated caches, or private model paths should be staged. | Hard launch gate. |

## Known Differences

- Futures have local queue, running, cancellation, and result-byte accounting,
  but hosted queue timing and backpressure have not been compared.
- Checkpoint archive URLs are local HTTP download URLs, not hosted signed
  download URLs.
- Checkpoint ownership is recorded as `local`; hosted authorization behavior
  has not been compared.
- Dense cross-entropy and per-token logprobs are local MLX behavior; hosted
  numeric parity has not been recorded.
- Sparse `TensorData` inputs are rejected; only dense tensor inputs are
  supported.
- `ForwardBackwardCustom` is not implemented.
- Top-k prompt logprobs are not implemented.
- Optimizer state is stored in local checkpoints, but hosted resume parity has
  not been recorded.
- Hosted and local numeric results are not expected to match exactly.
