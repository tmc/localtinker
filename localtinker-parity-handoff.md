# LocalTinker SDK Parity Handoff

Date: 2026-05-05
Repo: `/Volumes/tmc/go/src/github.com/tmc/localtinker`
Current HEAD: see `git rev-parse --short HEAD`
Notebook ID: `a912d601-badc-409b-bbdb-daf9316b843b`

Do not push.

## Objective

Continue the loop:

1. Sync the current repo and a short status note into NotebookLM.
2. Ask NotebookLM for a strict `COMPLETE` / `NOT COMPLETE` parity verdict.
3. If `NOT COMPLETE`, verify the cited claim against the current repo and upstream SDK.
4. Implement or coordinate exactly the next focused action.
5. Run focused tests or artifacts as needed.
6. Commit clean atomic changes, no push.
7. Repeat until NotebookLM says `COMPLETE`.

## Hard Rules

- Do not push.
- Do not run `go clean -cache`, `chmod`, `rm`, or modify `/Users/tmc/go/pkg/mod/golang.org/toolchain` or the shared Go cache.
- Tests are now approved, but keep using isolated cache:
  `GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) ...`
- Broad gate has already passed once:
  `GOCACHE=/tmp/localtinker-gocache.AsjiEe GOWORK=off go test ./...`
- Do not stage binary files.
- Before staging/committing:
  `git status --short`
  `git diff --cached --name-status`
- Prefer scoped staging or `git commit --only`.
- Try `~/bin/git-auto-commit-message --auto` first. It has been failing because `ANTHROPIC_API_KEY` is unset; fallback manual commit messages have been used.
- Add git notes after commits with commit hash, date, agent, model, tty, pid, and `ITERM_SESSION_ID`.

## Current State

Current local tree:

- HEAD: see `git rev-parse --short HEAD`
- Check `git status --short` before acting.
- This handoff file is committed; update it only when the workflow materially changes.

Notebook sources before this handoff:

- `repo: localtinker` source: `7dd306c1-efac-4cd4-8d36-10fc3818fbb4`
- `localtinker-sdk-parity-status.md` source: `323aa0c2-76ba-4323-bafb-7307e86f249f`
- `repo: tinker sdk` source: `7c9465b0-2583-48ba-a288-e7ab5ff8e3b2`

Important: keep this file current if HEAD moves; resync NotebookLM before acting on any older verdict.

## Landed Parity Work

Recent relevant commits:

- `cbea8b0 docs: record broad packaging gate`
- `369a63f docs: record hosted cancel blocker`
- `46be27e docs: record dense ce parity`
- `5aa72d5 docs: add dense ce hosted comparison`
- `497eb1c tinkerhttp: add sampler rest lookup`
- `4fa0eac docs: record custom loss parity`
- `a2c0052 docs: add custom loss hosted comparison`
- `ecc480f tinkertrain: support sdk custom loss forward`
- `b4716ac docs: update sdk parity caveats`
- `dd9f001 docs: normalize sampler comparison shape`

Covered:

- Top-k prompt logprobs.
- Hosted/local sampler comparison shape normalization.
- Multimodal chunk rejection.
- Checkpoint metadata/archive shape evidence.
- Custom loss support and hosted/local artifact.
- SDK `get_sampler` REST lookup and fixture coverage.
- Dense CE hosted/local artifact and docs.
- Hosted cancel absence documented.
- Broad packaging gate passed and recorded in docs.
- Hosted-comparison metadata scrubbed.
- Current test evidence refreshed.

## Latest NotebookLM Verdict

The latest NotebookLM answer at `544d941` said `NOT COMPLETE` and proposed:

- Implement missing REST route mapping for `get_training_run_by_tinker_path`.
- Files suggested:
  - `cmd/localtinker/testdata/sdk_smoke.txt`
  - `internal/tinkerhttp/routes_test.go`
- Suggested focused test:
  `GOWORK=off go test ./internal/tinkerhttp -run TestGetTrainingRunByTinkerPath -count=1`

The SDK method parses the tinker path client-side and calls
`GET /api/v1/training_runs/{training_run_id}`. The current patch adds focused
coverage proving that path instead of adding a nonexistent SDK route.

Useful upstream SDK lookup:

```sh
rg -n "get_training_run_by_tinker_path|training_run_by_tinker_path|tinker_path" /Volumes/tmc/go/src/github.com/thinking-machines-lab/tinker/src/tinker
```

Useful local lookup:

```sh
rg -n "training_runs|tinker_path|get_training_run" internal/tinkerhttp internal/tinkercoord docs cmd/localtinker/testdata
```

## Notebook Sync Commands

Use current HEAD:

```sh
git status --short
git rev-parse --short HEAD
nlm source sync --force -n 'repo: localtinker' a912d601-badc-409b-bbdb-daf9316b843b .
```

Create a short status source at `/tmp/localtinker-sdk-parity-status.md`, then sync:

```sh
nlm source sync --force -n 'localtinker-sdk-parity-status.md' a912d601-badc-409b-bbdb-daf9316b843b /tmp/localtinker-sdk-parity-status.md
```

Ask:

```sh
nlm generate-chat a912d601-badc-409b-bbdb-daf9316b843b "STRICT MODE: Based only on current sources and latest status at HEAD $(git rev-parse --short HEAD), is LocalTinker now complete for publicizing as a local SDK-compatible beta with documented hosted-only differences? Answer COMPLETE or NOT COMPLETE. If NOT COMPLETE, give exactly one next concrete focused implementation/artifact/test action with files and commands. Do not repeat already-documented hosted cancel absence or already-passing broad packaging gate unless current sources prove the status is false."
```

## Test Commands Already Run

Broad gate passed:

```sh
GOCACHE=/tmp/localtinker-gocache.AsjiEe GOWORK=off go test ./...
```

Earlier equivalent command run by 3E3B:

```sh
GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off /Users/tmc/.local/homebrew/bin/go test ./...
```

Package results included:

- `cmd/localtinker` ok
- `cmd/localtinker-node` ok
- `cmd/localtinker-tray` ok
- `internal/tinkercoord` ok
- `internal/tinkerhttp` ok
- `internal/tinkertrain` ok
- `tinker` ok

## Lane Coordination

Session IDs:

- 3E3B101A: sampling/logprobs/current orchestrator
- A773080F: hosted credentials and hosted/local artifacts
- ACDCF211: conformance/docs
- BC1565BF: CE/tinkertrain/MLX
- FDF74F3B: scheduler replacement, but it has been unresponsive to `it2 send-text`
- 8860ECF4: node/runtime

Broadcast pattern:

```sh
it2 session send-text <SID> "/goal <goal text>"
it2 session send-text <SID> $'\r'
```

If expecting a reply:

```sh
MY_SID=$(it2 session current)
it2 session send-text <TARGET_SID> "Question.

RESPOND: it2 session send-text $MY_SID \"brief answer\"
FORMAT: brief text with files/commit hashes if relevant"
```

## Hosted Cancel Evidence

A773 could not produce the requested hosted/local queue-cancel artifact honestly:

- Hosted SDK/generated client exposes retrieve-only futures.
- Hosted `/api/v1/cancel_future` returned 404.
- Plausible alternate cancel routes returned 404.
- Hosted submitted request reached `complete_metadata`, not terminal `canceled`.
- Local cancellation evidence exists but is not equivalent hosted/local parity.

Evidence files:

- `/tmp/queue-cancel-hosted-46be27e.jsonl`
- `/tmp/queue-cancel-local-46be27e.jsonl`

This is documented in `docs/internal/conformance.md` by `369a63f`.

## Next Best Step

1. Sync NotebookLM at the current HEAD.
2. Ask strict verdict again.
3. If the current `get_training_run_by_tinker_path` patch is not committed yet,
   commit it cleanly, no push.
4. Run or verify:

```sh
GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off /Users/tmc/.local/homebrew/bin/go test ./internal/tinkerhttp -run TestGetTrainingRunByTinkerPath -count=1
GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off /Users/tmc/.local/homebrew/bin/go test ./cmd/localtinker -run 'TestPythonSDKScript/sdk_smoke' -count=1
```

5. Resync NotebookLM and repeat.
