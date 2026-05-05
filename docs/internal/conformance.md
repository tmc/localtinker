# localtinker Conformance Report

This report tracks the Python SDK surface that localtinker intentionally
supports and the hosted-comparison gaps that remain.

## Local SDK Coverage

The script tests in `cmd/localtinker/testdata` run against the real Python
Tinker SDK when `TINKER_SDK_DIR` is available.

| Area | Evidence | Status |
| --- | --- | --- |
| Handshake | `sdk_handshake.txt` creates a `ServiceClient`, reads server capabilities, and verifies empty REST run and checkpoint listings. | Covered |
| Futures | `sdk_handshake.txt` and `sdk_async_errors.txt` verify missing future errors, future id echoing, and user-error futures. | Covered |
| Training | `sdk_smoke.txt` creates a LoRA training client, calls `get_info`, `forward`, `forward_backward`, and `optim_step`, and checks loss decreases. | Covered |
| Metrics | `sdk_smoke.txt` checks `loss:mean` and that `optimizer_backend:mlx` is reported by an optimizer step. | Covered |
| Checkpoints | `sdk_smoke.txt` saves, loads, lists, archives, publishes, unpublishes, sets TTL, and deletes checkpoints. It also opens the generated archive and checks expected files. | Covered locally |
| Sampler | `sdk_smoke.txt` saves sampler weights, creates a sampling client, and samples one token. | Covered locally |
| Malformed inputs | `sdk_malformed_inputs.txt` and `sdk_async_errors.txt` verify malformed training and async request errors. | Covered |

Run the local suite with:

```
go test ./cmd/localtinker -run TestPythonSDKScript -count=1
```

Run all Go coverage with:

```
go test ./...
```

## Hosted Comparison

No hosted comparison artifact is checked in yet. Before treating localtinker as
hosted-compatible, run the same minimal SDK job against hosted Tinker and
localtinker and record:

- training run ID
- future request IDs and terminal response shapes
- loss and optimizer metric names
- checkpoint path, listing metadata, archive metadata, and download behavior
- sampler response fields for the same prompt, seed, and sampling parameters

## Known Differences

- Futures complete synchronously today; queue state is only a compatibility
  response for pending records.
- Checkpoint archive URLs are local `file://` URLs, not hosted signed download
  URLs.
- Checkpoint ownership is recorded as `local` and is not an authorization
  boundary.
- Cross-entropy still uses the local MLX shifted-token path internally.
- String stops, generated token logprobs, prompt logprobs, and top-k prompt
  logprobs are not implemented.
- Hosted and local numeric results are not expected to match exactly.
