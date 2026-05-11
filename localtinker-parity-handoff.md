# LocalTinker SDK Parity Handoff

Date: 2026-05-11
Repo: `/Volumes/tmc/go/src/github.com/tmc/localtinker`
Current local HEAD: run `git rev-parse HEAD`
Last NotebookLM-audited local HEAD: `55ffaf57994d881cb5c284dbf951648381d1615c`
Push: not run
Notebook ID: `a912d601-badc-409b-bbdb-daf9316b843b`

## Current Verdict

NotebookLM was refreshed after the docs refresh and reported no remaining
local implementation or documentation gaps for the documented beta surface.
After a hosted API key was supplied, a follow-up probe pass recorded live hosted
rows for policy losses, fractional dense weights, sampler output, optimizer
metrics/resume shape, owner-side archive signed URLs, and raw cancel-future
route shape. No key or signed URL value was written to artifacts.

Synced notebook sources:

- `repo: localtinker` -> `34ba709e-2ed8-4b88-a786-41b6dcd3079f`
- `localtinker-sdk-parity-status.md` -> `67500cb9-176e-4909-8338-0b4b279fc93c`
- `repo: tinker sdk` -> `7c9465b0-2583-48ba-a288-e7ab5ff8e3b2`

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
- Image asset extension point. `tinkertrain.ImageAssetResolver` is reachable
  through programmatic HTTP wiring.
- Local queue, cancel, checkpoint archive metadata, and archive visibility
  behavior are covered by focused tests and JSONL artifacts.

Open hosted/comparison gaps:

- Hosted scheduler timing and operation backpressure.
- Hosted private cross-owner archive denial, requiring a second principal.
- Same-model local-vs-hosted sampler distribution comparison.
- Exact local-vs-hosted optimizer numeric equivalence after `optim_step`.
- Policy-loss capability semantics: local executes `importance_sampling`,
  `ppo`, `cispo`, and `dro`, but hosted at `55ffaf5` rejects the same
  SDK-shaped TensorData fixture before metrics with
  `could_not_convert_loss_function_inputs_to_array_record`.

## Evidence Artifacts

Current in-repo artifacts:

- `docs/internal/hosted-comparison/20260511-0480f94-policy-losses-hosted-local.jsonl`
- `docs/internal/hosted-comparison/20260511-0480f94-cancel-future-local.jsonl`
- `docs/internal/hosted-comparison/20260511-0480f94-queue-backpressure-local.jsonl`
- `docs/internal/hosted-comparison/20260511-0480f94-archive-auth-signed-url-local.jsonl`
- `docs/internal/hosted-comparison/20260508-e51c8f6-archive-visibility-local.jsonl`
- `docs/internal/hosted-comparison/20260508-e51c8f6-fractional-weights-local.jsonl`
- `docs/internal/hosted-comparison/20260511-55ffaf5-policy-losses-hosted.jsonl`
- `docs/internal/hosted-comparison/20260511-55ffaf5-fractional-weights-hosted.jsonl`
- `docs/internal/hosted-comparison/20260511-55ffaf5-sampler-distribution-hosted.jsonl`
- `docs/internal/hosted-comparison/20260511-55ffaf5-optimizer-metrics-hosted.jsonl`
- `docs/internal/hosted-comparison/20260511-55ffaf5-archive-auth-signed-url-hosted.jsonl`
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
```

The broad `cmd/localtinker` package path can enter the known long-running
`sdk_custom_loss.py` smoke path. Use the focused SDK malformed-input gate above
for this parity slice unless the user explicitly asks for a full smoke run.

## Next Action

Refresh NotebookLM from current `main` after committing this evidence, then ask
for a strict gap audit. Remaining hosted work needs either a second principal
for cross-owner archive denial or a paired local run for same-model sampler and
optimizer numeric comparisons.

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
