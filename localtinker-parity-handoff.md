# LocalTinker SDK Parity Handoff

Date: 2026-05-11
Repo: `/Volumes/tmc/go/src/github.com/tmc/localtinker`
Current local HEAD: run `git rev-parse HEAD`
Last NotebookLM-audited local HEAD: `23cf8a8f91760ef0328de7bda48c671e66b537cf`
Push: not run
Notebook ID: `a912d601-badc-409b-bbdb-daf9316b843b`

## Current Verdict

NotebookLM was refreshed after the archive invalid-token evidence, determinism
loss-surface correction, and handoff refresh. It reported no remaining local
implementation or documentation gaps for the documented beta surface at
`23cf8a8`.
After a hosted API key was supplied, a follow-up probe pass recorded live hosted
rows for policy losses, fractional dense weights, sampler output, optimizer
metrics/resume shape, owner-side archive signed URLs, and raw cancel-future
route shape. No key or signed URL value was written to artifacts.

Synced notebook sources:

- `repo: localtinker` -> `9218ab3c-4a9c-44f9-ac38-d678479bade4`
- `localtinker-sdk-parity-status.md` -> `7254d15f-68ed-4b97-8f54-cacf187266ff`
- `repo: tinker sdk` -> `7c9465b0-2583-48ba-a288-e7ab5ff8e3b2`

The latest NotebookLM chat (`5e9f8b73-a826-4f7c-884f-fb885db0eed9`) answered
that no open local gaps remain, all known built-in loss methods execute
locally, only `cross_entropy` is intentionally advertised as hosted-compatible,
and the archive invalid-token artifact is not valid second-principal
cross-owner denial evidence.

Closed locally:

- Parse/multimodal boundary. Image and `image_asset_pointer` chunks parse and
  validate, then fail with typed MLX-executor errors because no local vision
  backend or production image asset store is configured.
- CSR sparse `target_tokens` and `weights`. Both the direct Go API and HTTP
  path rehydrate them through `tinkertrain.RehydrateCSR`; unsupported sparse
  tensor names are rejected before MLX execution.
- Stop shape validation. Invalid object, mixed, nested, negative, and
  fractional stop token shapes are rejected at the parse boundary.
- Policy loss execution. `cross_entropy`, `importance_sampling`, `ppo`,
  `cispo`, and `dro` execute locally; policy losses report `loss:sum`.
  Only `cross_entropy` is advertised as hosted-compatible because hosted
  rejects the recorded SDK-shaped policy-loss fixture before metrics.
- Determinism docs now describe the current local loss-function surface and
  cite `internal/tinkertrain/mlx.go` for accepted loss names, cross-entropy
  reduction, policy loss inputs, and loss config parsing.
- Optimizer metrics/resume comparison. Hosted and local artifacts now cover the
  same cross-entropy `forward_backward`, `optim_step`, TTL-compatible
  `save_state`, and `create_training_client_from_state_with_optimizer` flow.
  Hosted `optim_step` metrics are empty; local reports `loss:mean`,
  `optimizer_backend:mlx`, and `optimizer_step:unique`.
- Image asset extension point. `tinkertrain.ImageAssetResolver` is reachable
  through programmatic HTTP wiring.
- Local queue, cancel, checkpoint archive metadata, and archive visibility
  behavior are covered by focused tests and JSONL artifacts.

Open hosted/comparison gaps:

- Hosted private cross-owner archive denial, requiring a valid second
  principal. Hosted invalid-token archive denial is recorded as an
  authentication negative-control, not cross-owner evidence.

## Evidence Artifacts

Current in-repo artifacts:

- `docs/internal/hosted-comparison/20260511-0480f94-policy-losses-hosted-local.jsonl`
- `docs/internal/hosted-comparison/20260511-0480f94-cancel-future-local.jsonl`
- `docs/internal/hosted-comparison/20260511-0480f94-queue-backpressure-local.jsonl`
- `docs/internal/hosted-comparison/20260511-f06603b-queue-backpressure-hosted.jsonl`
- `docs/internal/hosted-comparison/20260511-0480f94-archive-auth-signed-url-local.jsonl`
- `docs/internal/hosted-comparison/20260508-e51c8f6-archive-visibility-local.jsonl`
- `docs/internal/hosted-comparison/20260508-e51c8f6-fractional-weights-local.jsonl`
- `docs/internal/hosted-comparison/20260511-55ffaf5-policy-losses-hosted.jsonl`
- `docs/internal/hosted-comparison/20260511-55ffaf5-fractional-weights-hosted.jsonl`
- `docs/internal/hosted-comparison/20260511-55ffaf5-sampler-distribution-hosted.jsonl`
- `docs/internal/hosted-comparison/20260511-b1f9f9c-sampler-distribution-local.jsonl`
- `docs/internal/hosted-comparison/20260511-55ffaf5-optimizer-metrics-hosted.jsonl`
- `docs/internal/hosted-comparison/20260511-50b2ee8-optimizer-metrics-local.jsonl`
- `docs/internal/hosted-comparison/20260511-55ffaf5-archive-auth-signed-url-hosted.jsonl`
- `docs/internal/hosted-comparison/20260511-28826b2-archive-invalid-token-hosted.jsonl`
- `docs/internal/hosted-comparison/20260511-55ffaf5-cancel-future-hosted.jsonl`

Coordinator handoffs:

- Final gap analysis: `/tmp/localtinker-final-gap-analysis-21D23B54.md`
- Hosted probe resume runbook:
  `docs/internal/hosted-probes.md`
- Preserved stale worktree patch:
  `/tmp/localtinker-wt-betadocs-dirty-20260511T051213Z.patch`

## Validation Already Run

```sh
jq -c . docs/internal/hosted-comparison/20260511-0480f94-*.jsonl
git diff --check
GOWORK=off go test ./internal/tinkercoord -run 'TestFutureQueueBoundsConcurrency|TestFutureQueueDispatchesFIFO' -count=1
GOWORK=off go test ./internal/tinkerhttp -run 'TestRetrieveFutureRoute|TestCancelFutureRoute' -count=1
MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off go test ./internal/tinkerhttp -run '^(TestCheckpointRoutes|TestCheckpointArchiveAuthorization|TestExpiredCheckpointIsHiddenAndArchiveGone)$' -count=1
MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib GOWORK=off go test ./cmd/localtinker -run 'TestPythonSDKScript/sdk_malformed_inputs' -count=1 -timeout=90s
GOWORK=off go test ./internal/tinkertrain ./internal/tinkerhttp -run 'TestDatumTargetsRehydratesSparse|TestDatumWeightsRehydratesSparse|TestNormalizeTensorDataRehydratesNamedSparse|TestDenseCrossEntropyFractionalWeights|TestDensePolicyLossesReturnWeightedSumAndLogprobs|TestMultimodalChunkParsesAndCounts|TestMultimodalExecutionRejected|TestImageChunkHeaderValidation' -count=1
jq -c . docs/internal/hosted-comparison/20260511-55ffaf5-optimizer-metrics-hosted.jsonl docs/internal/hosted-comparison/20260511-50b2ee8-optimizer-metrics-local.jsonl >/dev/null
MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib GOWORK=off go test ./internal/tinkertrain -run 'TestOptimizerStateRoundTrip|TestManagerLoadStateWithOptimizer|TestCheckpointMetadataJSON' -count=1
MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib GOWORK=off go test ./internal/tinkerhttp -run 'TestForwardBackwardAndOptimStepTune' -count=1 -timeout=3m
MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib GOWORK=off go test ./internal/tinkertrain ./internal/tinkerhttp ./internal/tinkercoord ./tinker -run 'TestDensePolicyLossesReturnWeightedSumAndLogprobs|TestDenseCrossEntropyReturnsWeightedLossAndLogprobs|TestTrainingInputValidation(AcceptsPolicyLossInputs|AcceptsPolicyLossConfig|RejectsPolicyLossConfig|RejectsPolicyLossInputs)$|TestCapabilitiesAdvertiseSamplerConformance|TestCapabilitiesReportHostedCompatibleLosses' -count=1
```

The broad `cmd/localtinker` package path can enter the known long-running
`sdk_custom_loss.py` smoke path. Use the focused SDK malformed-input gate above
for this parity slice unless the user explicitly asks for a full smoke run.

## Next Action

Remaining hosted work needs a valid second principal for cross-owner archive
denial; the invalid-token control should not be treated as a substitute for
that probe. `docs/internal/hosted-probes.md` now checks the primary credential,
optional `TINKER_BASE_URL`, and second-principal aliases in preflight so the
skip reason is visible before rerunning probes.

Do not print secret values. Keep commits local unless the user explicitly asks
to push.

## Worktree Cleanup

Main worktree was clean after the latest handoff refresh.

The stale worktree `/Volumes/tmc/go/src/github.com/tmc/localtinker-wt-betadocs`
was removed after preserving its dirty diff at
`/tmp/localtinker-wt-betadocs-dirty-20260511T051213Z.patch`.
The branch `parity-beta-docs` was deleted with `git branch -d`; it had no
commits ahead of `main`. The preserved diff is stale relative to `55ffaf5` and
should be treated only as recovery evidence.
