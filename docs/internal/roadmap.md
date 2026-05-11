# localtinker Roadmap

localtinker is a local Tinker-compatible coordinator and node runtime backed by
MLX. This file tracks the work needed to make it dependable for local training
and close enough to hosted Tinker that ordinary SDK workflows behave the same.

## Principles

- Keep public request handling typed and explicit.
- Reject malformed user input before it reaches MLX.
- Keep coordinator, node, artifact, and training concerns separate.
- Prefer small Go APIs with ordinary structs and errors.
- Prove compatibility with SDK-level tests.

## Current Shape

- `cmd/localtinker` serves the Python SDK HTTP API, Connect RPC API, and
  dashboard.
- `cmd/localtinker-node` provides node and artifact cache tooling.
- `cmd/localtinker-tray` provides the macOS menu bar monitor.
- `internal/tinkertrain` runs local MLX LoRA training and sampling.
- `internal/tinkerartifact`, `internal/tinkernode`, and `internal/tinkerproto`
  provide the node/cache substrate.
- The Connect RPC surface accepts node registration and heartbeats and can
  request node drain through admin state.
- The node watch stream validates watchers and emits drain commands for nodes
  placed into drain state.
- The artifact tracker records inventories and applies in-memory retention
  decisions for node-reported artifacts.
- Coordinator heartbeats return artifact prewarm roots for published manifests
  missing from a node inventory.
- Artifact prewarm roots are assigned to healthy nodes using active leases,
  queued operations, existing prewarm assignments, and deterministic tie-breaks.
- `localtinker-node run` prewarms missing artifacts from peers and reports the
  refreshed inventory.
- `localtinker-node run` watches coordinator commands and exits on drain
  directives.
- Watched node commands are acknowledged through the coordinator report stream
  and surfaced in node labels.
- `localtinker-node cache delete` removes installed artifacts and can report
  the updated inventory.
- Artifact transfer reports update visible node transfer state and completed
  artifact inventory.
- Node cache sync and prewarm report transfer start, failure, and completion
  when a node ID is available.
- Admin RPC exposes run summaries and run inspection payloads from coordinator
  state.
- Node report streams update visible node lifecycle and telemetry labels.
- Node operation start and terminal events update visible active/queued load.
- Checkpoint TTL metadata is enforced in listings and archive URL requests.
- Sampling accepts temperature, top-p, top-k, seed, max tokens, and integer stop
  tokens.
- `tinker` contains the experimental Go API.
- The HTTP API enforces the request byte limit advertised in client config.
- SDK conformance coverage includes handshake, empty REST listings, missing
  future errors, malformed training and async inputs, future id echoing, and a
  full MLX smoke workflow.
- SDK conformance coverage includes REST session listing and lookup routes.
- `docs/internal/conformance.md` records the local SDK conformance coverage and
  hosted-comparison gaps.
- Cross-entropy accepts dense target tensors and returns per-token logprobs in
  local MLX tests.
- Sampling returns generated token logprobs, prompt logprobs, and top-k prompt
  logprobs, and accepts tokenizer-backed string stops.
- Checkpoints include local optimizer state and archive metadata headers.

## 1. SDK Conformance

Goal: make the Python SDK see localtinker as a normal Tinker endpoint for the
supported surface.

- Expand the upstream Python SDK conformance suite beyond the current
  handshake, error, malformed-input, and MLX smoke coverage.
- Keep coverage for session creation, heartbeat, futures, model creation,
  `forward`, `forward_backward`, `optim_step`, save/load weights, sampler
  sessions, sampling, run listing, checkpoint listing, archive URL, publish,
  unpublish, TTL, and delete.
- Match hosted error response shapes, categories, and future id fields.
- Keep unsupported capabilities explicit in server capabilities.
- Expand malformed request fixtures as new supported routes are added.

## 2. Cross-Entropy Contract

Goal: keep the local `cross_entropy` tensor contract aligned with hosted
behavior.

- Keep coverage for 1D shape inference when `TensorData.shape` is omitted.
- Keep coverage for rectangular dense target tensors.
- Reject ragged tensors and shape/data mismatches at the HTTP boundary.
- Keep support for targets that are not just `model_input` shifted left by one.
- Keep support for arbitrary valid float weights.
- Validate target/weight shape compatibility.
- Return real per-token logprobs where the SDK expects them.

## 3. Futures and Scheduling

Goal: make coordinator scheduling match hosted behavior where advertised.

- Keep coverage for queued, running, complete, user_error,
  system_error, and canceled states.
- Keep coverage for bounded concurrency and operation byte accounting.
- Keep coverage for cancellation and lease timeout handling.
- Persist enough operation metadata to survive coordinator restarts.
- Match hosted `retrieve_future(..., allow_metadata_only=True)` behavior.
- Expose queue state in RPC and dashboard views.

## 4. Checkpoints and Artifacts

Goal: make checkpoints useful for both SDK workflows and node sync.

- Keep adapter weights, adapter config, optimizer state, and completion markers
  in a stable checkpoint layout.
- Serve checkpoint archive URLs with hosted-style expiration metadata.
- Keep tar archive downloads consumable by the Tinker CLI.
- Keep size, visibility, expiration, and owner metadata in checkpoint listings.
- Keep publish, unpublish, TTL, and delete stateful across coordinator restarts.
- Keep training checkpoints and sampler checkpoints distinct.
- Test download, extraction, load, and sampler creation end to end.

## 5. Optimizer State

Goal: keep training resume behavior compatible with hosted checkpoints.

- Keep optimizer state saved alongside LoRA weights.
- Keep `load_state_with_optimizer` support.
- Keep optimizer step counters.
- Test train, save, load, resume, and continued loss decrease.
- Determinism inputs (seed, model, adapter config, optimizer, data order,
  MLX backend) are documented in `docs/determinism.md`.

## 6. Sampling

Goal: make sampling responses match hosted behavior where advertised.

- Keep temperature, top-p, top-k, seed, max tokens, and integer stop tokens
  covered by deterministic sampler tests.
- Keep tokenizer-backed string stops covered.
- Keep generated token logprobs covered.
- Keep prompt logprobs covered.
- Keep top-k prompt logprobs covered while they remain advertised.
- Add deterministic sampler tests over a small cached model.

## 7. Node Runtime

Goal: make nodes useful local workers, not just cache/RPC scaffolding.

- Add coordinator registration and heartbeat loops.
- Advertise model, memory, disk, and backend capabilities.
- Assign work to nodes through leases.
- Extend load-aware assignment from artifact prewarm to operation leases.
- Extend node operation lifecycle events with leased work assignment and
  persisted terminal result handling.
- Wire artifact retention decisions into leased node commands.
- Extend node drain and health states with lease-aware draining and recovery.
- Test coordinator-node-node artifact sync.

## 8. Tray and Dashboard

Goal: make local operation visible without tailing logs.

- Show coordinator health, active runs, queued operations, node health, and
  recent errors using the HTTP dashboard and admin RPC surfaces.
- Link tray actions to dashboard pages.
- Show local checkpoint paths and archive availability.
- Keep tray polling cheap and robust when the coordinator is down.
- Add dashboard pages for runs, checkpoints, nodes, and artifacts.

## 9. Packaging

Goal: keep the project buildable from a clean checkout.

- Keep `go test ./...` passing with `GOWORK=off` against the upstream
  pseudo-versions of `mlx-go`, `mlx-go-lm`, and `modelir` (no `replace`
  directives in `go.mod` as of 2026-05-07).
- Add release build commands for coordinator, node, and tray.
- Document model cache, Python SDK, and credential setup.
- Avoid checking in binaries, generated caches, downloaded weights, or secrets.

## 10. Hosted Comparison

Goal: keep local behavior honest against hosted Tinker.

- Run a short hosted training job and record run ID, futures, losses, and
  checkpoint path.
- Run the same SDK script against localtinker.
- Compare response shapes, metric names, checkpoint metadata, and sampler
  behavior.
- Track differences in `docs/internal/conformance.md`.

## Known Gaps

- Hosted scheduler timing and operation-level backpressure are local-only
  evidence so far. Local queue/cancel behavior is pinned, and
  `docs/internal/hosted-comparison/20260511-55ffaf5-cancel-future-hosted.jsonl`
  records hosted raw `/api/v1/cancel_future` returning 404 for the request
  shapes localtinker accepts.
- Policy losses `importance_sampling`, `ppo`, `cispo`, and `dro` execute
  locally and are covered by local JSONL rows. Live hosted evidence in
  `docs/internal/hosted-comparison/20260511-55ffaf5-policy-losses-hosted.jsonl`
  shows the same SDK-shaped TensorData fixture fails before metrics with
  `could_not_convert_loss_function_inputs_to_array_record`.
- Arbitrary non-prefix fractional dense weights execute locally and on hosted.
  `docs/internal/hosted-comparison/20260511-55ffaf5-fractional-weights-hosted.jsonl`
  records hosted success for `[0.25, 1, 0, 0.75]`; hosted reports `loss:sum`,
  while local cross-entropy reports `loss:mean`.
- Checkpoint archive URLs are local HTTP download URLs. Local owner,
  visibility, expiration, and private/public state are pinned. Hosted owner
  signed URL shape is recorded in
  `docs/internal/hosted-comparison/20260511-55ffaf5-archive-auth-signed-url-hosted.jsonl`,
  but cross-owner authorization evidence still requires a second principal
  token.
- Hosted numerics and local MLX numerics will differ. Hosted sampler rows and
  hosted optimizer metrics/resume shape are now recorded at `55ffaf5`, but
  same-model local sampler distribution and exact optimizer numeric
  equivalence remain comparison gaps, not local implementation gaps.

## Next Milestones

1. Compare hosted/local sampler distributions across fixed prompts, seeds,
   temperature, top-p, and top-k.
2. Compare hosted/local optimizer metrics and resume behavior after
   `optim_step`.
3. Probe hosted private cross-owner archive denial with a second hosted
   principal.
4. Decide whether local policy-loss capability advertising should remain
   broader than the hosted behavior recorded at `55ffaf5`.
5. MLX library setup (`MLX_LIB_PATH`) for clean checkouts is documented in
   `docs/mlx-setup.md` and referenced from the README.
6. Deterministic sampler tests over a small cached model: covered by
   `TestSampleDeterministicSmallCachedModel`,
   `TestSampleDeterministicRepeats`, and `TestSampleDeterministicPrefix`
   in `internal/tinkertrain/sample_test.go`. They skip cleanly when
   `LOCALTINKER_SMALL_MODEL` (default `mlx-community/Qwen3-0.6B-4bit`)
   is not cached.
