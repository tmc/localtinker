# Hosted Probe Runbook

This runbook resumes hosted evidence collection after real hosted inputs are
available. Do not print secret values or write them to artifacts.

Required inputs:

```sh
export TINKER_API_KEY=...
export TINKER_BASE_URL=...
```

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

## Open Hosted Evidence

| Gap | Artifact | Required hosted input |
| --- | --- | --- |
| Fractional dense weights | `docs/internal/hosted-comparison/20260508-e51c8f6-fractional-weights-local.jsonl` or successor artifact | `TINKER_API_KEY`, `TINKER_BASE_URL` |
| Policy losses | `docs/internal/hosted-comparison/20260511-0480f94-policy-losses-hosted-local.jsonl` | `TINKER_API_KEY`, `TINKER_BASE_URL` |
| Cancel futures | `docs/internal/hosted-comparison/20260511-0480f94-cancel-future-local.jsonl` or successor artifact | `TINKER_API_KEY`, `TINKER_BASE_URL` |
| Queue/backpressure | `docs/internal/hosted-comparison/20260511-0480f94-queue-backpressure-local.jsonl` or successor artifact | `TINKER_API_KEY`, `TINKER_BASE_URL` |
| Checkpoint archive signed URL/auth | `docs/internal/hosted-comparison/20260511-0480f94-archive-auth-signed-url-local.jsonl` or successor artifact | `TINKER_API_KEY`, `TINKER_BASE_URL`, second principal |
| Sampler distribution | `docs/internal/hosted-comparison/YYYYMMDD-<short-head>-sampler-distribution-hosted-local.jsonl` | `TINKER_API_KEY`, `TINKER_BASE_URL` |
| Optimizer metrics | `docs/internal/hosted-comparison/YYYYMMDD-<short-head>-optimizer-metrics-hosted-local.jsonl` | `TINKER_API_KEY`, `TINKER_BASE_URL` |

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
