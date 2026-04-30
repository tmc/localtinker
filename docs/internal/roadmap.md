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

## 1. SDK Conformance

Goal: make the Python SDK see localtinker as a normal Tinker endpoint for the
supported surface.

- Add a conformance suite driven by the upstream Python SDK.
- Cover session creation, heartbeat, futures, model creation, `forward`,
  `forward_backward`, `optim_step`, save/load weights, sampler sessions,
  sampling, run listing, checkpoint listing, archive URL, publish, unpublish,
  TTL, and delete.
- Match hosted error response shapes and categories.
- Keep unsupported capabilities explicit in server capabilities.
- Add malformed `loss_fn_inputs` fixtures.

## 2. Cross-Entropy Contract

Goal: implement the real `cross_entropy` tensor contract instead of relying on
the current shifted-token shortcut.

- Infer a 1D shape for flattened `TensorData` when `shape` is omitted.
- Accept rectangular dense target tensors.
- Reject ragged tensors and shape/data mismatches at the HTTP boundary.
- Support target tensors that are not just `model_input` shifted left by one.
- Support arbitrary valid float weights.
- Validate target/weight shape compatibility.
- Return real per-token logprobs where the SDK expects them.

## 3. Futures and Scheduling

Goal: replace mostly synchronous execution with a real coordinator scheduler.

- Add an operation queue with queued, running, complete, user_error,
  system_error, and canceled states.
- Add bounded concurrency and operation byte accounting.
- Add cancellation and lease timeout handling.
- Persist enough operation metadata to survive coordinator restarts.
- Match hosted `retrieve_future(..., allow_metadata_only=True)` behavior.
- Expose queue state in RPC and dashboard views.

## 4. Checkpoints and Artifacts

Goal: make checkpoints useful for both SDK workflows and node sync.

- Store adapter weights, adapter config, optimizer state, and completion markers
  in a stable checkpoint layout.
- Serve checkpoint archive URLs with hosted-style expiration metadata.
- Keep tar archive downloads consumable by the Tinker CLI.
- Keep size, visibility, expiration, and owner metadata in checkpoint listings.
- Keep publish, unpublish, TTL, and delete stateful across coordinator restarts.
- Keep training checkpoints and sampler checkpoints distinct.
- Test download, extraction, load, and sampler creation end to end.

## 5. Optimizer State

Goal: support real training resume.

- Save optimizer state alongside LoRA weights.
- Load optimizer state for `load_state_with_optimizer`.
- Track optimizer step counters.
- Test train, save, load, resume, and continued loss decrease.
- Document determinism inputs: seed, model, adapter config, optimizer, data
  order, and MLX backend.

## 6. Sampling

Goal: make sampling responses match hosted behavior where advertised.

- Keep temperature, top-p, top-k, seed, max tokens, and integer stop tokens
  covered by deterministic sampler tests.
- Add tokenizer-backed string stops.
- Return generated token logprobs.
- Return prompt logprobs when requested.
- Add top-k prompt logprobs only when implemented and advertised.
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

- Remove temporary relative `replace` directives once upstream `mlx-go`,
  `mlx-go-lm`, and `modelir` module versions are consumable directly.
- Keep `go test ./...` passing with `GOWORK=off`.
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
- Track differences in a conformance report.

## Known Gaps

- No true async scheduler or operation-level backpressure yet.
- `cross_entropy` still uses a shifted-token MLX training path.
- Arbitrary non-prefix fractional weights are not fully supported.
- Prompt logprobs and top-k prompt logprobs are not implemented.
- String stops need tokenizer-backed sampling.
- Optimizer state is not persisted.
- Checkpoint archive URLs point at local tar files; hosted download and
  ownership enforcement remain incomplete.
- The MLX dependency graph still needs temporary sibling-checkout replaces.
- Hosted numerics and local MLX numerics will differ.

## Next Milestones

1. Add SDK conformance tests for all currently supported routes.
2. Replace shifted-token-only loss handling with dense CE tensors.
3. Add hosted-style checkpoint download ownership and retention enforcement.
4. Add operation queue state and asynchronous futures.
5. Remove temporary MLX/module replaces and document MLX library setup.
6. Add tokenizer-backed string stops and generated-token logprobs.
