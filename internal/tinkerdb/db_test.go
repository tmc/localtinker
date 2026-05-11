package tinkerdb

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"
)

func TestCheckpointMetadataPersists(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	expires := time.Date(2026, 4, 30, 12, 0, 0, 0, time.UTC)

	store, err := OpenJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.PutCheckpoint(context.Background(), Checkpoint{
		Path:      "tinker://model-a/weights/ckpt",
		Public:    true,
		Owner:     "local",
		ExpiresAt: &expires,
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	store, err = OpenJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := store.GetCheckpoint(context.Background(), "tinker://model-a/weights/ckpt")
	if err != nil {
		t.Fatal(err)
	}
	if !got.Public || got.Owner != "local" || got.ExpiresAt == nil || !got.ExpiresAt.Equal(expires) {
		t.Fatalf("checkpoint = %#v", got)
	}
}

func TestNodeMetadataPersists(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	started := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	seen := started.Add(time.Second)
	draining := started.Add(2 * time.Second)
	caps := json.RawMessage(`{"models":[{"name":"Qwen/Qwen3-8B"}],"max_concurrency":2}`)

	store, err := OpenJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.PutNode(context.Background(), Node{
		ID:             "node-a",
		SessionID:      "sess-a",
		State:          "draining",
		Capabilities:   caps,
		MaxConcurrency: 2,
		Running:        1,
		StartedAt:      started,
		LastSeenAt:     seen,
		DrainingSince:  draining,
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	store, err = OpenJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := store.GetNode(context.Background(), "node-a")
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != "node-a" || got.SessionID != "sess-a" || got.State != "draining" {
		t.Fatalf("node identity = %#v", got)
	}
	if got.MaxConcurrency != 2 || got.Running != 1 {
		t.Fatalf("node load = %#v", got)
	}
	if !got.StartedAt.Equal(started) || !got.LastSeenAt.Equal(seen) || !got.DrainingSince.Equal(draining) {
		t.Fatalf("node times = %#v", got)
	}
	if !sameJSON(got.Capabilities, caps) {
		t.Fatalf("capabilities = %s, want %s", got.Capabilities, caps)
	}
	nodes, err := store.ListNodes(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes) != 1 {
		t.Fatalf("nodes = %d, want 1", len(nodes))
	}
}

func TestFutureRetryMetadataPersists(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	created := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	next := created.Add(5 * time.Second)
	attempt := created.Add(time.Second)
	heartbeat := created.Add(2 * time.Second)
	reqs := json.RawMessage(`{"operation":"forward","model_id":"model-a"}`)

	store, err := OpenJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.PutFuture(context.Background(), Future{
		ID:              "fut-a",
		State:           "queued",
		CreatedAt:       created,
		Attempt:         2,
		MaxAttempts:     3,
		AssignedNodeID:  "node-a",
		NextRunAt:       next,
		LastAttemptAt:   attempt,
		LastHeartbeatAt: heartbeat,
		RetryReason:     "node_dead",
		IdempotencyKey:  "step-1",
		Requirements:    reqs,
		Priority:        7,
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	store, err = OpenJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := store.GetFuture(context.Background(), "fut-a")
	if err != nil {
		t.Fatal(err)
	}
	if got.Attempt != 2 || got.MaxAttempts != 3 || got.AssignedNodeID != "node-a" {
		t.Fatalf("future retry counters = %#v", got)
	}
	if !got.NextRunAt.Equal(next) || !got.LastAttemptAt.Equal(attempt) || !got.LastHeartbeatAt.Equal(heartbeat) {
		t.Fatalf("future retry times = %#v", got)
	}
	if got.RetryReason != "node_dead" || got.IdempotencyKey != "step-1" || got.Priority != 7 {
		t.Fatalf("future retry metadata = %#v", got)
	}
	if !sameJSON(got.Requirements, reqs) {
		t.Fatalf("requirements = %s, want %s", got.Requirements, reqs)
	}
}

func TestClaimNextFutureOrdersAndAssigns(t *testing.T) {
	store := OpenMemory()
	now := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	for _, future := range []Future{
		{ID: "fut-low", State: "queued", CreatedAt: now.Add(-3 * time.Second), Priority: 1},
		{ID: "fut-later", State: "queued", CreatedAt: now.Add(-2 * time.Second), Priority: 10},
		{ID: "fut-first", State: "queued", CreatedAt: now.Add(-time.Second), Priority: 10},
	} {
		if err := store.PutFuture(context.Background(), future); err != nil {
			t.Fatal(err)
		}
	}

	got, ok, err := store.ClaimNextFuture(context.Background(), "node-a", now, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("ClaimNextFuture found no future")
	}
	if got.ID != "fut-later" {
		t.Fatalf("claimed %q, want fut-later", got.ID)
	}
	if got.State != "running" || got.AssignedNodeID != "node-a" || got.Attempt != 1 {
		t.Fatalf("claimed future = %#v", got)
	}
	if got.LeaseID != "fut-later-attempt-1" {
		t.Fatalf("LeaseID = %q", got.LeaseID)
	}
	if !got.StartedAt.Equal(now) || !got.LastAttemptAt.Equal(now) || !got.LastHeartbeatAt.Equal(now) {
		t.Fatalf("claim times = %#v", got)
	}
	if !got.LeaseExpiresAt.Equal(now.Add(time.Minute)) {
		t.Fatalf("LeaseExpiresAt = %s, want %s", got.LeaseExpiresAt, now.Add(time.Minute))
	}
}

func TestClaimNextFutureSkipsDelayedAndRunning(t *testing.T) {
	store := OpenMemory()
	now := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	for _, future := range []Future{
		{ID: "fut-running", State: "running", CreatedAt: now.Add(-3 * time.Second), Priority: 100},
		{ID: "fut-delayed", State: "queued", CreatedAt: now.Add(-2 * time.Second), NextRunAt: now.Add(time.Second), Priority: 100},
		{ID: "fut-ready", State: "queued", CreatedAt: now.Add(-time.Second), Priority: 1},
	} {
		if err := store.PutFuture(context.Background(), future); err != nil {
			t.Fatal(err)
		}
	}

	got, ok, err := store.ClaimNextFuture(context.Background(), "node-a", now, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("ClaimNextFuture found no future")
	}
	if got.ID != "fut-ready" {
		t.Fatalf("claimed %q, want fut-ready", got.ID)
	}
}

func TestClaimNextFutureNoEligibleFuture(t *testing.T) {
	store := OpenMemory()
	now := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	if err := store.PutFuture(context.Background(), Future{
		ID:        "fut-delayed",
		State:     "queued",
		CreatedAt: now,
		NextRunAt: now.Add(time.Second),
	}); err != nil {
		t.Fatal(err)
	}

	_, ok, err := store.ClaimNextFuture(context.Background(), "node-a", now, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("ClaimNextFuture found delayed future")
	}
}

func TestFutureAttemptsPersistAndSort(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	started := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	finished := started.Add(time.Second)
	errPayload := json.RawMessage(`{"code":"system_error","message":"node died"}`)

	store, err := OpenJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, attempt := range []FutureAttempt{
		{FutureID: "fut-b", Attempt: 1, State: "running"},
		{
			FutureID:   "fut-a",
			Attempt:    2,
			NodeID:     "node-b",
			LeaseID:    "lease-b",
			State:      "complete",
			StartedAt:  started.Add(2 * time.Second),
			FinishedAt: finished.Add(2 * time.Second),
		},
		{
			FutureID:   "fut-a",
			Attempt:    1,
			NodeID:     "node-a",
			LeaseID:    "lease-a",
			State:      "lost",
			StartedAt:  started,
			FinishedAt: finished,
			Error:      errPayload,
		},
	} {
		if err := store.PutFutureAttempt(context.Background(), attempt); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	store, err = OpenJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	attempts, err := store.ListFutureAttempts(context.Background(), "fut-a")
	if err != nil {
		t.Fatal(err)
	}
	if len(attempts) != 2 {
		t.Fatalf("attempts = %d, want 2", len(attempts))
	}
	if attempts[0].Attempt != 1 || attempts[1].Attempt != 2 {
		t.Fatalf("attempt order = %#v", attempts)
	}
	if attempts[0].NodeID != "node-a" || attempts[0].LeaseID != "lease-a" || attempts[0].State != "lost" {
		t.Fatalf("attempt[0] = %#v", attempts[0])
	}
	if !attempts[0].StartedAt.Equal(started) || !attempts[0].FinishedAt.Equal(finished) {
		t.Fatalf("attempt[0] times = %#v", attempts[0])
	}
	if !sameJSON(attempts[0].Error, errPayload) {
		t.Fatalf("attempt[0] error = %s, want %s", attempts[0].Error, errPayload)
	}

	all, err := store.ListFutureAttempts(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 3 || all[0].FutureID != "fut-a" || all[2].FutureID != "fut-b" {
		t.Fatalf("all attempts = %#v", all)
	}
}

func TestRequeueFutureRecordsLostAttempt(t *testing.T) {
	store := OpenMemory()
	started := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	now := started.Add(time.Second)
	next := started.Add(5 * time.Second)
	if err := store.PutFuture(context.Background(), Future{
		ID:              "fut-a",
		State:           "running",
		StartedAt:       started,
		Attempt:         2,
		AssignedNodeID:  "node-a",
		LeaseID:         "lease-a",
		LeaseExpiresAt:  started.Add(time.Minute),
		LastHeartbeatAt: started,
	}); err != nil {
		t.Fatal(err)
	}

	got, ok, err := store.RequeueFuture(context.Background(), "fut-a", "lease-a", "node_dead", next, now)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("RequeueFuture did not requeue")
	}
	if got.State != "queued" || got.AssignedNodeID != "" || got.LeaseID != "" {
		t.Fatalf("future ownership = %#v", got)
	}
	if got.Attempt != 2 || got.RetryReason != "node_dead" {
		t.Fatalf("future retry = %#v", got)
	}
	if !got.NextRunAt.Equal(next) || !got.StartedAt.IsZero() || !got.LeaseExpiresAt.IsZero() || !got.LastHeartbeatAt.IsZero() {
		t.Fatalf("future times = %#v", got)
	}

	attempts, err := store.ListFutureAttempts(context.Background(), "fut-a")
	if err != nil {
		t.Fatal(err)
	}
	if len(attempts) != 1 {
		t.Fatalf("attempts = %d, want 1", len(attempts))
	}
	attempt := attempts[0]
	if attempt.Attempt != 2 || attempt.NodeID != "node-a" || attempt.LeaseID != "lease-a" || attempt.State != "lost" {
		t.Fatalf("attempt = %#v", attempt)
	}
	if !attempt.StartedAt.Equal(started) || !attempt.FinishedAt.Equal(now) {
		t.Fatalf("attempt times = %#v", attempt)
	}
	var payload struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(attempt.Error, &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Code != "system_error" || payload.Message != "node_dead" {
		t.Fatalf("attempt error = %#v", payload)
	}
}

func TestRenewFutureLeaseExtendsRunningLease(t *testing.T) {
	store := OpenMemory()
	started := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	now := started.Add(10 * time.Second)
	if err := store.PutFuture(context.Background(), Future{
		ID:              "fut-a",
		State:           "running",
		LeaseID:         "lease-a",
		LeaseExpiresAt:  started.Add(time.Minute),
		LastHeartbeatAt: started,
	}); err != nil {
		t.Fatal(err)
	}

	got, ok, err := store.RenewFutureLease(context.Background(), "fut-a", "lease-a", now, 2*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("RenewFutureLease did not renew")
	}
	if !got.LastHeartbeatAt.Equal(now) {
		t.Fatalf("LastHeartbeatAt = %s, want %s", got.LastHeartbeatAt, now)
	}
	if !got.LeaseExpiresAt.Equal(now.Add(2 * time.Minute)) {
		t.Fatalf("LeaseExpiresAt = %s, want %s", got.LeaseExpiresAt, now.Add(2*time.Minute))
	}
}

func TestRenewFutureLeaseRejectsStaleLease(t *testing.T) {
	store := OpenMemory()
	started := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	if err := store.PutFuture(context.Background(), Future{
		ID:              "fut-a",
		State:           "running",
		LeaseID:         "lease-a",
		LeaseExpiresAt:  started.Add(time.Minute),
		LastHeartbeatAt: started,
	}); err != nil {
		t.Fatal(err)
	}

	got, ok, err := store.RenewFutureLease(context.Background(), "fut-a", "lease-old", started.Add(time.Second), 2*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("RenewFutureLease accepted stale lease")
	}
	if !got.LeaseExpiresAt.Equal(started.Add(time.Minute)) || !got.LastHeartbeatAt.Equal(started) {
		t.Fatalf("future changed = %#v", got)
	}
}

func TestRequeueFutureRejectsStaleLease(t *testing.T) {
	store := OpenMemory()
	started := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	if err := store.PutFuture(context.Background(), Future{
		ID:      "fut-a",
		State:   "running",
		Attempt: 1,
		LeaseID: "lease-a",
	}); err != nil {
		t.Fatal(err)
	}

	got, ok, err := store.RequeueFuture(context.Background(), "fut-a", "lease-old", "node_dead", time.Time{}, started)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("RequeueFuture accepted stale lease")
	}
	if got.State != "running" || got.LeaseID != "lease-a" {
		t.Fatalf("future changed = %#v", got)
	}
	attempts, err := store.ListFutureAttempts(context.Background(), "fut-a")
	if err != nil {
		t.Fatal(err)
	}
	if len(attempts) != 0 {
		t.Fatalf("attempts = %d, want 0", len(attempts))
	}
}

func TestFinishFutureRecordsAttempt(t *testing.T) {
	store := OpenMemory()
	started := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	now := started.Add(time.Second)
	result := json.RawMessage(`{"ok":true}`)
	if err := store.PutFuture(context.Background(), Future{
		ID:             "fut-a",
		State:          "running",
		StartedAt:      started,
		Attempt:        3,
		AssignedNodeID: "node-a",
		LeaseID:        "lease-a",
	}); err != nil {
		t.Fatal(err)
	}

	got, ok, err := store.FinishFuture(context.Background(), "fut-a", "lease-a", "complete", result, nil, now)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("FinishFuture did not finish")
	}
	if got.State != "complete" || !got.CompletedAt.Equal(now) || got.AssignedNodeID != "" || got.LeaseID != "" {
		t.Fatalf("future = %#v", got)
	}
	if got.ResultBytes != int64(len(result)) || !sameJSON(got.Result, result) {
		t.Fatalf("future result = %s bytes=%d", got.Result, got.ResultBytes)
	}
	attempts, err := store.ListFutureAttempts(context.Background(), "fut-a")
	if err != nil {
		t.Fatal(err)
	}
	if len(attempts) != 1 {
		t.Fatalf("attempts = %d, want 1", len(attempts))
	}
	attempt := attempts[0]
	if attempt.Attempt != 3 || attempt.NodeID != "node-a" || attempt.LeaseID != "lease-a" || attempt.State != "complete" {
		t.Fatalf("attempt = %#v", attempt)
	}
	if !attempt.StartedAt.Equal(started) || !attempt.FinishedAt.Equal(now) {
		t.Fatalf("attempt times = %#v", attempt)
	}
	if !sameJSON(attempt.Result, result) {
		t.Fatalf("attempt result = %s, want %s", attempt.Result, result)
	}
}

func TestFinishFutureRejectsStaleLease(t *testing.T) {
	store := OpenMemory()
	now := time.Date(2026, 5, 11, 12, 0, 0, 0, time.UTC)
	if err := store.PutFuture(context.Background(), Future{
		ID:      "fut-a",
		State:   "running",
		Attempt: 1,
		LeaseID: "lease-a",
	}); err != nil {
		t.Fatal(err)
	}

	got, ok, err := store.FinishFuture(context.Background(), "fut-a", "lease-old", "complete", json.RawMessage(`{"ok":true}`), nil, now)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("FinishFuture accepted stale lease")
	}
	if got.State != "running" || got.LeaseID != "lease-a" || len(got.Result) != 0 {
		t.Fatalf("future changed = %#v", got)
	}
	attempts, err := store.ListFutureAttempts(context.Background(), "fut-a")
	if err != nil {
		t.Fatal(err)
	}
	if len(attempts) != 0 {
		t.Fatalf("attempts = %d, want 0", len(attempts))
	}
}

func sameJSON(a, b json.RawMessage) bool {
	var av, bv any
	if err := json.Unmarshal(a, &av); err != nil {
		return false
	}
	if err := json.Unmarshal(b, &bv); err != nil {
		return false
	}
	return jsonEqual(av, bv)
}

func jsonEqual(a, b any) bool {
	ab, err := json.Marshal(a)
	if err != nil {
		return false
	}
	bb, err := json.Marshal(b)
	if err != nil {
		return false
	}
	return string(ab) == string(bb)
}
