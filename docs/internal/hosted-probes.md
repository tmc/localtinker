# Hosted Probe Runbook

This runbook resumes hosted evidence collection after real hosted inputs are
available. Do not print secret values or write them to artifacts.

Required inputs:

```sh
export TINKER_API_KEY=...
```

The current Python SDK defaults to
`https://tinker.thinkingmachines.dev/services/tinker-prod` when
`TINKER_BASE_URL` is unset. Set `TINKER_BASE_URL` only when probing a different
hosted deployment.

Checkpoint archive cross-owner denial also requires a second hosted principal:

```sh
export TINKER_API_KEY_2=...
```

Equivalent names such as `TINKER_SECOND_API_KEY`, `TINKER_ALT_API_KEY`, or
`TINKER_CROSS_OWNER_API_KEY` are acceptable if the runner normalizes them.

## Preflight

```sh
cd /Volumes/tmc/go/src/github.com/tmc/localtinker
git status --short --branch
git rev-parse HEAD
for v in TINKER_API_KEY TINKER_BASE_URL; do
	if [ -n "${!v:-}" ]; then echo "$v=present"; else echo "$v=missing"; fi
done
jq -c . docs/internal/hosted-comparison/*.jsonl >/dev/null
```

## Hosted Evidence And Regression Checks

| Gap | Artifact | Required hosted input |
| --- | --- | --- |
| Queue/backpressure regression check | `docs/internal/hosted-comparison/20260511-0480f94-queue-backpressure-local.jsonl` plus `docs/internal/hosted-comparison/20260511-f06603b-queue-backpressure-hosted.jsonl` | Reprobe only when hosted scheduler or future metadata semantics change |
| Checkpoint archive invalid-credential control | `docs/internal/hosted-comparison/20260511-28826b2-archive-invalid-token-hosted.jsonl` | Reprobe only when archive authorization error shape changes; this is not cross-owner evidence |
| Checkpoint archive cross-owner denial | `docs/internal/hosted-comparison/20260511-55ffaf5-archive-auth-signed-url-hosted.jsonl` plus future successor artifact | `TINKER_API_KEY`, valid second principal |
| Sampler distribution regression check | `docs/internal/hosted-comparison/20260511-55ffaf5-sampler-distribution-hosted.jsonl` plus `docs/internal/hosted-comparison/20260511-b1f9f9c-sampler-distribution-local.jsonl` | Reprobe only when sampler semantics or model mapping changes |
| Optimizer metric-surface regression check | `docs/internal/hosted-comparison/20260511-55ffaf5-optimizer-metrics-hosted.jsonl` plus `docs/internal/hosted-comparison/20260511-50b2ee8-optimizer-metrics-local.jsonl` | Reprobe only when optimizer response semantics change; hosted `optim_step` metrics are empty in the recorded fixture |
| Policy-loss capability regression check | `docs/internal/hosted-comparison/20260511-55ffaf5-policy-losses-hosted.jsonl` | Reprobe only if hosted starts accepting the recorded SDK-shaped TensorData fixture; local currently keeps policy-loss execution available but does not advertise it as hosted-compatible |

## Artifact Rules

- Record scrubbed metadata only: `backend`, `case_id`, `comparison_id`,
  `event`, and `payload`.
- Do not write secret URLs, API key values, home directories, downloaded model
  paths, or signed archive URLs.
- Record the hosted endpoint only as `base_url_class:"hosted"`.
- Record runner and Python paths using scrubbed labels such as
  `runner_machine:"local-runner"` and `python_executable:"python"`.
- If a hosted API surface is absent, record the real HTTP or SDK failure shape
  as evidence rather than leaving another blocker row.
- Validate each artifact with `jq -c . <artifact> >/dev/null`.
- Update `docs/internal/conformance.md` to replace the matching blocker text
  with the new hosted evidence row references.

## Local Verification

After recording hosted artifacts, rerun the focused local gates:

```sh
GOWORK=off go test ./internal/tinkercoord -run 'TestFutureQueueBoundsConcurrency|TestFutureQueueDispatchesFIFO' -count=1
GOWORK=off go test ./internal/tinkerhttp -run 'TestRetrieveFutureRoute|TestCancelFutureRoute' -count=1
MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib GOWORK=off go test ./internal/tinkertrain ./internal/tinkerhttp -run 'TestDensePolicyLossesReturnWeightedSumAndLogprobs|TestSampleDeterministic|TestCheckpointRoutes|TestCheckpointArchiveAuthorization|TestExpiredCheckpointIsHiddenAndArchiveGone' -count=1
MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib GOWORK=off go test ./cmd/localtinker -run 'TestPythonSDKScript/sdk_malformed_inputs' -count=1 -timeout=90s
```

## Notebook Check

Refresh NotebookLM and ask for a strict gap audit:

```sh
nlm source sync --name 'repo: localtinker' a912d601-badc-409b-bbdb-daf9316b843b .
nlm generate-chat --resolve-citations --citations tail \
	--source-match '^(repo: localtinker|localtinker-sdk-parity-status.md|repo: tinker sdk)$' \
	a912d601-badc-409b-bbdb-daf9316b843b \
	'Re-audit localtinker parity at current HEAD. Which gaps remain OPEN-local or OPEN-hosted-evidence?'
```

Do not mark parity complete until filesystem checks and NotebookLM agree that
no local gaps remain and no hosted-evidence gaps remain, or the user explicitly
accepts the hosted items as out of scope.
