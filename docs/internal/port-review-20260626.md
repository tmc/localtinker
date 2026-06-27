# Go-team review + history audit — upstream port af041ee..0883375 (2026-06-26)

Multi-lens go-team code review and commit-history audit of the 7-commit upstream
port on `port-upstream-0883375` (range `93bb27d..HEAD`). Every cited finding was
triaged against the working tree before action (filesystem is ground truth).

## Verdict

- **History: push-as-is.** Seven §-scoped commits, consistent package-scoped
  lowercase prefixes (`tinkercoord:`/`tinkerhttp:`/`tinkerproto:`/`docs:`),
  dependency-correct ordering (config → routes → contract → proto-schema →
  proto-decoder → docs). The §3 split (`ed8d398` generated schema, `42dba77`
  hand-written decoder) is bisect-safe — `ed8d398` full-tree-builds because
  `google.golang.org/protobuf` is already a direct dep at the base — and isolates
  the 926-line generated `.pb.go` blob from the reviewable decode logic. Keep it
  split. Docs-last (`05cbca1`) is correct: a meta-document about the whole window.
  The one blemish was a git-*note* typo ("§3 commit 1/3" vs "1/2"), fixed in place
  with `git notes edit` — no rebase.
- **Code: ready, after one fix.** Build + vet green; all pure-logic port tests
  pass. One real bug found and fixed (see F1).

## Findings actioned (commit `79b1ceb`)

- **F1 — zstd decompression bomb (High; api-design + smells + test-hygiene, all
  three independently).** `readMaybeCompressed` (`internal/tinkerhttp/proto.go`)
  read the decompressed zstd stream with an unbounded `io.ReadAll`. `MaxBytesReader`
  (`routes.go:71`) bounds only the *compressed* wire bytes, so a small in-limit
  zstd body could inflate without bound and OOM the single-tenant server.
  **Fixed:** decompressed read capped at `MaxRequestBytes` (the same budget that
  bounds every other route), oversized bodies rejected with 400.
- **F2 — proto route test required MLX weights it never exercised (Medium).**
  `TestForwardBackwardProtoRoute` called `create_model` with a real base model
  only to get a `model_id`, but the route enqueues a future against any non-empty
  `model_id` without loading the model, so the call was dead weight that 500'd in
  any weights-free env. **Fixed:** posts against a synthetic `model_id`, so the
  wire path runs green in CI with no weights. Added `TestForwardBackwardProtoZstdBomb`
  covering F1's cap.

After the fix the port's contribution to the suite is fully green weights-free.
The two remaining `tinkerhttp` failures (`TestForwardBackwardAndOptimStepTune`,
`TestCreateModelGetInfoAndUnload`) pre-date the port and need real Qwen weights —
out of scope, untouched.

## Deferred fast-follow polish (do not gate the push)

All triaged TRUE against the tree; left for a follow-up sitting because none is a
correctness blocker:

- **F3 — `buf.gen.yaml` generate-then-`git checkout` revert (High/Low, contested).**
  `public.proto` sets `go_package` to the mlx-go path while the module is
  localtinker, so `buf generate` reverts committed connect files to the wrong
  import path; recovery lives only in a YAML comment. Root-cause fix: set
  `go_package` to the localtinker path so generation is idempotent, or move
  buf+restore behind a `//go:generate` script.
- **F4 — `DTYPE_UNSPECIFIED` coerced to "float32" in `protoDTypeToString` but
  rejected by `bytesToFloat64s` (Medium).** Make the two agree — reject unspecified
  up front in `protoTensor` with one clear error.
- **F5 — unreferenced proto response messages (Low).** `BatchedTensor`,
  `ArrayRecord`, proto-`ForwardBackwardOutput` are generated but unused (response
  path is a known gap). Delete and re-add in the commit that implements the
  response path, or keep with the current honest comment.
- **F6 — `ParallelFWDBWDChunks: true` advertised while the coordinator serializes
  (`defaultMaxOperations = 1`) (Low).** Add a one-line comment at the return site
  noting it is advertised for client-wire parity while execution is serialized.
- **F7 — `seq_id` dropped on the proto path without the house-style
  "accepted and ignored" comment the JSON savers use (Low).** Add the matching
  one-line comment.
- Lower-tier: proto route never `defer r.Body.Close()` (the lone close-discipline
  exception); `auditLog` zero-limit defaults to 100 newest checkpoints; permissive
  `auditLog`/`assign_session_project` could route through one named
  `requireLocalAdmin` no-op; `writeUserError`/`writeError` split in the proto
  handler; `validateStopShape` carries `[]int`/`[][]int` arms unreachable from JSON.

## Patterns the panel said to keep

Decode-parity tests that reuse the JSON `normalizeAndValidateInput` rather than
forking validation; honest gap-naming docs (conformance.md / SPEC); the
Accept-negotiated 302↔200 checkpoint dual-contract pinned by tests; per-loss-fn
tensor validation with finite/shape/length checks before MLX; bounds-checked
`bytesToX` helpers with documented, unit-pinned bfloat16 widening.
