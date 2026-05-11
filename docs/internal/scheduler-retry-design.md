# Scheduler and Retry Design

This document sketches the next scheduler for LocalTinker jobs when multiple
worker nodes can cycle in and out. It is grounded in the current coordinator:
`tinkercoord.Coordinator` owns sessions, models, futures, leases, and the local
FIFO queue; `tinkerrpc.Server` owns mesh node watches, operation leases, drain
commands, and node load; `tinkerdb.Future` is the durable SDK-facing record;
`tinkernodecap.Capabilities` already describes node resources and model support.

## Goals

- Keep accepted futures durable across coordinator and node restarts.
- Dispatch only to nodes that are live, compatible, and not overloaded.
- Retry transient node loss without duplicating non-idempotent work.
- Preserve the SDK-facing future contract: queued/running responses become
  `try_again`, completed futures can return metadata before payload, and
  cancellation wins over late worker completion.
- Make every scheduler decision visible in dashboard state and JSONL evidence.

Non-goals for the first slice:

- Cross-cluster consensus.
- Exactly-once execution across arbitrary side effects.
- Changing Python SDK request/response shapes.

## Current Baseline

The current coordinator stores futures with state, metadata, request bytes,
lease ID, and lease expiration. It dispatches SDK-submitted futures to
in-process goroutines through a bounded semaphore. `RetrieveFuture` maps queued
and running states to `try_again`, expires a stale running lease as
`system_error`, and preserves `canceled` against late completion. On restart,
queued and running futures are converted to `system_error`.

The mesh path already has a separate scheduler in `internal/tinkerrpc`. It
tracks `operationState`, streams `NodeCommand` values through `Watch`, expires
operation leases back to its own queue, skips draining nodes, and emits
`RevokeLease` for node drain. That queue does not currently map back to
`tinkerdb.Future`, so SDK futures and mesh node operations have separate
ownership models.

The robust scheduler should keep the SDK public states but merge these two
execution paths: `tinkerdb.Future` is the durable job, and `tinkerrpc` becomes
one worker transport that claims and reports attempts against that job.

## Data Model

Extend `tinkerdb` with durable node and attempt records:

```go
type Node struct {
	ID              string
	SessionID       string
	LastSeenAt      time.Time
	State           string // live, draining, dead
	Capabilities    json.RawMessage
	MaxConcurrency  int
	Running         int
	StartedAt       time.Time
	DrainingSince   time.Time
}

type Future struct {
	// existing fields...
	Attempt         int
	MaxAttempts     int
	AssignedNodeID  string
	NextRunAt       time.Time
	LastAttemptAt   time.Time
	LastHeartbeatAt time.Time
	RetryReason     string
	IdempotencyKey   string
	Requirements    json.RawMessage
	Priority        int
}

type FutureAttempt struct {
	FutureID      string
	Attempt      int
	NodeID       string
	LeaseID      string
	State        string // running, lost, user_error, system_error, complete
	StartedAt    time.Time
	FinishedAt   time.Time
	Error         json.RawMessage
}
```

Keep `Future` as the SDK-facing summary. `FutureAttempt` is internal evidence
for retries and debugging. `Requirements` should be derived from current future
metadata (`type`, `model_id`, `loss_fn`, sampling vs training) plus model
resource needs. Store it as JSON first; introduce typed fields only when the
scheduler actually branches on them.

Store additions should be narrow:

- `PutNode`, `GetNode`, `ListNodes`.
- `PutFutureAttempt`, `ListFutureAttempts(futureID)`.
- A compare-and-swap style claim operation:
  `ClaimNextFuture(ctx, nodeID, now, leaseTimeout) (Future, bool, error)`.

The claim operation is the key boundary. It must atomically select one eligible
future and move it from `queued` to `running` with a new lease ID, node ID,
attempt number, and lease expiry.

## Node Lifecycle

Nodes register through a new internal route or coordinator method:

```go
RegisterNode(sessionID string, caps tinkernodecap.Capabilities) (Node, error)
NodeHeartbeat(nodeID string, running []RunningFuture) (Node, error)
DrainNode(nodeID string) error
```

Use `session_heartbeat` for SDK client liveness as-is. Do not overload it for
worker liveness; workers need capability and assignment state. A node heartbeat
should update `LastSeenAt`, capacity, thermal label, and optionally renew leases
for the futures it is still running.

Node states:

- `live`: eligible for new claims.
- `draining`: heartbeats are accepted and leases can finish, but no new claims.
- `dead`: heartbeat timeout exceeded; running leases become retry candidates.

The coordinator should mark nodes dead in a periodic reconciliation loop and on
read paths that already inspect futures. Avoid relying on one background goroutine
for correctness; every claim should also ignore stale nodes.

## Scheduling Algorithm

At each claim:

1. Load live nodes and compute available slots:
   `MaxConcurrency - Running`.
2. Select queued futures where `NextRunAt <= now`, state is `queued`, and
   requirements match the node.
3. Sort by priority, then `CreatedAt`, then future ID for deterministic FIFO.
4. Atomically claim one future for the node.
5. Return the future payload to the worker, not just the ID.

Eligibility rules:

- Model operation must match a node model or a node that can load that model.
- Training operations require `Features.LoRA` and optimizer support when needed.
- Sampling requires `Features.Sampling`.
- Request bytes must fit node memory/disk headroom policy.
- Thermal `throttled` nodes are excluded; `warm` nodes get lower priority.

For the current local-only implementation, the first slice can keep one embedded
node with `tinkernodecap.Probe` and route existing in-process execution through
the same claim/complete path. That proves the model before adding remote worker
HTTP.

## Retry Policy

Classify failures before deciding retry:

- User errors are terminal: validation failures, unsupported model/input,
  missing checkpoint, malformed tensor, and explicit cancel.
- System errors are retryable when tied to node loss, lease expiry, process
  crash, transient store errors, or worker transport failure.
- System errors are terminal after `MaxAttempts`.

Suggested defaults:

- `MaxAttempts`: 3 for training/sampling operations.
- Initial backoff: 1 second.
- Backoff multiplier: 2.
- Jitter: +/-20 percent.
- Cap: 30 seconds.

State transitions:

```text
queued -> running -> complete
queued -> canceled
running -> canceled
running -> queued      // retryable failure, attempts remaining
running -> system_error // attempts exhausted or non-retryable system error
running -> user_error
```

When retrying, keep the same future ID. Increment `Attempt`, clear
`AssignedNodeID` and `LeaseID`, set `NextRunAt`, and append a `FutureAttempt`
row with the failure reason. This preserves the SDK polling contract and avoids
forcing clients to chase replacement future IDs.

Do not retry operations after the point where non-idempotent side effects may
have become externally visible unless the operation has an idempotency key:

- `forward` and `forward_backward`: retryable.
- `sample`: retryable if seeded and no streaming response was delivered.
- `optim_step`: retryable only with a model-step idempotency key.
- `save_weights` and `save_weights_for_sampler`: retryable only when the path is
  stable and write completion can be checked.
- `load_weights`: retryable after verifying current model state is unchanged.

For the first implementation, mark `optim_step`, save, and load as terminal on
worker loss unless they are executed by the embedded local node. Add idempotency
keys before enabling remote retries for those operations.

## Lease Handling

A lease protects ownership of a running future, not completion authority. A
worker completion must include `future_id`, `attempt`, `lease_id`, and `node_id`.
The coordinator accepts completion only if all four still match the stored
future and the state is `running`.

Lease renewal:

- Workers renew leases through node heartbeat.
- Renewal is bounded by operation type; long operations may renew repeatedly.
- If renewal fails because the future was canceled or reassigned, the worker
  must stop and discard its result.

Lease expiry:

- If the assigned node is stale and the future is retryable, requeue.
- If attempts are exhausted or operation is non-idempotent, finish as
  `system_error` with a retry summary.
- `RetrieveFuture` may trigger this reconciliation for a single future, but the
  scheduler loop should also sweep expired leases.

## Worker Surface

Keep SDK routes stable. Prefer the existing Connect RPC node transport for
worker execution:

- `tinkerv1.CoordinatorService/Watch`
- `tinkerv1.CoordinatorService/Report`
- `tinkerv1.AdminService/DrainNode`
- `tinkerv1.NodeService/Cancel`

If an HTTP worker surface is still needed later, add it as an adapter over the
same durable claim and completion methods rather than as a separate scheduler.
Possible routes would live under `/api/v1/nodes`:

- `POST /api/v1/nodes/register`
- `POST /api/v1/nodes/heartbeat`
- `POST /api/v1/nodes/claim`
- `POST /api/v1/nodes/complete`
- `POST /api/v1/nodes/fail`
- `POST /api/v1/nodes/drain`

These routes would not be Python SDK compatibility surfaces. They can use local
auth or loopback-only defaults initially.

The existing `/api/v1/retrieve_future` and `/api/v1/cancel_future` behavior
should not change. Cancellation should update the durable future; node completion
with an old lease should be ignored.

Cancellation also needs an active worker signal. Today `CancelFuture` only
updates `tinkerdb.Future`. Once a remote node owns the lease, the coordinator
must notify the worker transport that the lease is revoked. For the Connect RPC
path, `tinkerrpc` should observe the canceled future and send `RevokeLease`
through the node's `Watch` stream; the node runtime can then call its existing
cancel path. The database state remains authoritative, so a missed revoke is
still safe: late completion is rejected by lease validation, and the worker
will stop on the next heartbeat or lease renewal failure.

## Observability

Extend `DashboardFuture` with:

- `attempt`
- `max_attempts`
- `assigned_node_id`
- `next_run_at`
- `retry_reason`
- `last_heartbeat_at`

Extend `QueueState` with:

- `retrying`
- `draining_nodes`
- `dead_nodes`
- `lease_expired`

Add dashboard node rows from `tinkerdb.Node`:

- node ID and session ID.
- state.
- last seen.
- running count / max concurrency.
- models and thermal label.

The dashboard must not show two competing node inventories. The web dashboard
combines `tinkercoord.DashboardSnapshot` with `tinkerrpc.Snapshot`, but node
identity, state, last seen time, capacity, and labels come from durable
`tinkerdb.Node` rows through `coord.nodes`. `tinkerrpc.Snapshot` is only for
ephemeral transport details such as active watch streams and pending command
counts. Do not render both snapshots as separate node lists; reconcile by node
ID when transport details are shown.

Every terminal failure after retries should include a compact error payload:

```json
{
  "code": "system_error",
  "message": "operation lease expired after retries",
  "attempts": 3,
  "last_node_id": "node_...",
  "retry_reason": "node_dead"
}
```

## Invariants

- A future has at most one current running lease.
- A completion is accepted only for the current attempt and lease.
- Canceled futures never transition to complete or retry.
- User errors are never retried.
- Retry keeps the same future ID.
- No live node receives work after entering `draining`.
- Stale running futures are reconciled on scheduler sweep and on retrieval.
- The public SDK state machine remains backward compatible.

## Staged Implementation

1. **Durable scheduler core**
   - Add node and attempt records to `tinkerdb`.
   - Add atomic claim helpers for JSON and memory stores.
   - Add tests for claim ordering, one-lease-per-future, stale lease requeue,
     attempts exhausted, and cancellation winning over late completion.

2. **Unify mesh operations with futures**
   - Make `tinkerrpc.operationState` carry `FutureID`, `Attempt`, and current
     `LeaseID`.
   - On `Watch`, claim queued futures from `tinkerdb` instead of a separate
     operation-only queue when the operation originates from the SDK.
   - On `Report`, complete or fail the current future attempt with the existing
     lease validation.
   - On node watch disconnect or drain, revoke or requeue the node's current
     future leases according to retry policy.
   - When `CancelFuture` changes a leased future to `canceled`, notify
     `tinkerrpc` so it emits `RevokeLease` on the owning node's `Watch` stream.
   - Keep `tinkerdb.Node` as the dashboard source of truth and merge any
     `tinkerrpc.Snapshot` transport data by node ID.

3. **Embedded local node**
   - Replace the current in-memory queue slice with a scheduler loop that
     registers one embedded node and claims work from the store.
   - Preserve `MaxOperations` as embedded node `MaxConcurrency`.
   - Keep all execution in-process.
   - Existing queue tests should still pass with minimal expectation changes.

4. **Retry classification**
   - Add operation metadata for idempotency and retryability.
   - Use `ErrorInfo.retryable` from node failures when available.
   - Retry `forward`, `forward_backward`, and seeded `sample`.
   - Keep optimizer/save/load conservative until idempotency keys are present.

5. **Optional worker HTTP routes**
   - Add internal node register/heartbeat/claim/complete/fail/drain routes only
     if Connect RPC is not sufficient for a worker deployment.
   - Add loopback-only or shared-secret protection before enabling non-local
     workers.
   - Add route tests for stale lease completion rejection and drain behavior.

6. **Remote node process**
   - Build a small worker that probes `tinkernodecap`, registers, claims, runs,
     renews leases, and completes/fails attempts.
   - Start with one model and one operation family before broadening.

7. **Operational evidence**
   - Add JSONL artifacts for node loss, retry success, retry exhaustion,
     draining, and late completion ignored.
   - Local retry, lease, cancel, and dashboard-node evidence is recorded in
     `docs/internal/hosted-comparison/20260511-ba8a1e5-scheduler-retry-local.jsonl`.
   - Refresh NotebookLM and re-audit the scheduler invariants before advertising
     robust multi-node execution.

## First Patch Recommendation

Start with stage 1 tests and store API, not new HTTP worker routes. The
smallest useful patch is a scheduler helper inside `internal/tinkercoord` that
can claim and reconcile futures against the existing JSON store. Once that is
green, wire `internal/tinkerrpc` operation leasing to those durable future
claims and only then route the current local goroutine executor through the
same path. That keeps the public SDK contract stable while making node churn a
property of the core scheduler instead of an add-on.
