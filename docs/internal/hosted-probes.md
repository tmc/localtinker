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
jq -c . hosted comparison fixtures >/dev/null
```

## Open Hosted Evidence

| Gap | Artifact | Required hosted input |
| --- | --- | --- |
| Fractional dense weights | `hosted comparison fixture` or successor artifact | `TINKER_API_KEY`, `TINKER_BASE_URL` |
| Policy losses | `hosted comparison fixture` | `TINKER_API_KEY`, `TINKER_BASE_URL` |
| Cancel futures | `hosted comparison fixture` or successor artifact | `TINKER_API_KEY`, `TINKER_BASE_URL` |
| Queue/backpressure | `hosted comparison fixture` or successor artifact | `TINKER_API_KEY`, `TINKER_BASE_URL` |
| Checkpoint archive signed URL/auth | `hosted comparison fixture` or successor artifact | `TINKER_API_KEY`, `TINKER_BASE_URL`, second principal |
| Sampler distribution | `hosted comparison fixturesYYYYMMDD-<short-head>-sampler-distribution-hosted-local.jsonl` | `TINKER_API_KEY`, `TINKER_BASE_URL` |
| Optimizer metrics | `hosted comparison fixturesYYYYMMDD-<short-head>-optimizer-metrics-hosted-local.jsonl` | `TINKER_API_KEY`, `TINKER_BASE_URL` |

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

## Repository Check

Refresh repository audit and ask for a strict gap audit:

```sh
repository audit sync command
repository audit command
	--source-match '^(repo: localtinker|SDK parity status source|repo: tinker sdk)$' \
	repository-audit-id \
	'Re-audit localtinker parity at current HEAD. Which gaps remain OPEN-local or OPEN-hosted-evidence?'
```

Do not mark parity complete until filesystem checks and repository audit agree that
no local gaps remain and no hosted-evidence gaps remain, or the user explicitly
accepts the hosted items as out of scope.
