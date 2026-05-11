# LocalTinker SDK Parity Handoff

Date: 2026-05-11
Repo: `/Volumes/tmc/go/src/github.com/tmc/localtinker`
Current local HEAD: `dc2de0b179cf34d6b152c573d75709f3da4bfba0`
Push: not run
Notebook ID: `a912d601-badc-409b-bbdb-daf9316b843b`

## Current Verdict

NotebookLM was refreshed at HEAD `dc2de0b` and reported no remaining local
implementation gaps for the documented beta surface.

Synced notebook sources:

- `repo: localtinker` -> `8a6ab604-fb18-49d1-9ce1-9bcc3a050ea8`
- `localtinker-sdk-parity-status.md` -> `891817ed-dd77-4a56-b89c-7de6b342c098`
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

Open hosted-evidence gaps:

- Policy loss response shapes for hosted `importance_sampling`, `ppo`, `cispo`,
  and `dro`.
- Hosted cancel future behavior.
- Hosted scheduler timing and operation backpressure.
- Hosted checkpoint signed URL shape and private cross-owner archive denial.
- Hosted/local sampler distribution comparison.
- Hosted/local optimizer metrics and resume equivalence.

These are blocked in the current shell because no hosted credential source is
available:

- `TINKER_API_KEY` is absent.
- `TINKER_BASE_URL` is absent.
- `TINKER_CREDENTIAL_CMD` is absent.
- Candidate second-principal variables for archive auth are absent.
- Keychain and local password-manager checks did not find a usable credential.

## Evidence Artifacts

Current in-repo artifacts:

- `docs/internal/hosted-comparison/20260511-0480f94-policy-losses-hosted-local.jsonl`
- `docs/internal/hosted-comparison/20260511-0480f94-cancel-future-local.jsonl`
- `docs/internal/hosted-comparison/20260511-0480f94-queue-backpressure-local.jsonl`
- `docs/internal/hosted-comparison/20260511-0480f94-archive-auth-signed-url-local.jsonl`
- `docs/internal/hosted-comparison/20260508-e51c8f6-archive-visibility-local.jsonl`
- `docs/internal/hosted-comparison/20260508-e51c8f6-fractional-weights-local.jsonl`

Coordinator handoffs:

- Final gap analysis: `/tmp/localtinker-final-gap-analysis-21D23B54.md`
- Hosted probe resume runbook:
  `/tmp/localtinker-hosted-probe-resume-21D23B54.md`
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

## Next Action When Credentials Exist

Run `/tmp/localtinker-hosted-probe-resume-21D23B54.md`.

Do not print secret values. Record only scrubbed hosted metadata in JSONL. Keep
commits local unless the user explicitly asks to push.

## Worktree Cleanup

Main worktree is clean at `dc2de0b`.

One leftover worktree remains:

- `/Volumes/tmc/go/src/github.com/tmc/localtinker-wt-betadocs`
- branch: `parity-beta-docs`
- no commits ahead of `main`
- dirty files:
  - `docs/internal/conformance.md`
  - `docs/internal/roadmap.md`
  - `localtinker-parity-handoff.md`

The dirty diff was preserved at
`/tmp/localtinker-wt-betadocs-dirty-20260511T051213Z.patch`. The worktree was
not removed because deleting it would discard uncommitted worker-owned edits.
