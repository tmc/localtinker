package tinkercoord

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/tmc/localtinker/internal/tinkerdb"
)

func TestSessionHeartbeat(t *testing.T) {
	now := time.Date(2026, 4, 29, 12, 0, 0, 0, time.UTC)
	c, err := New(Config{
		Store: tinkerdb.OpenMemory(),
		Now: func() time.Time {
			return now
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	session, err := c.CreateSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	now = now.Add(time.Second)
	session, err = c.Heartbeat(context.Background(), session.ID)
	if err != nil {
		t.Fatal(err)
	}
	if session.HeartbeatN != 1 {
		t.Fatalf("HeartbeatN = %d, want 1", session.HeartbeatN)
	}
	if !session.LastSeenAt.Equal(now) {
		t.Fatalf("LastSeenAt = %s, want %s", session.LastSeenAt, now)
	}
}

func TestCapabilitiesAdvertiseSamplerConformance(t *testing.T) {
	c, err := New(Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	caps := c.Capabilities(context.Background())
	if len(caps.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(caps.Models))
	}
	supported := caps.Models[0].Supported
	for _, feature := range []string{
		"sample",
		"sample_generated_logprobs",
		"sample_prompt_logprobs",
		"sample_string_stops",
	} {
		if !slices.Contains(supported, feature) {
			t.Fatalf("supported = %v, missing %q", supported, feature)
		}
	}
	if slices.Contains(supported, "top_k_prompt_logprobs") {
		t.Fatalf("supported = %v, must not advertise top_k_prompt_logprobs", supported)
	}
}

func TestRetrieveFutureMetadataThenComplete(t *testing.T) {
	c, err := New(Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	future, err := c.CompleteFuture(context.Background(),
		map[string]any{"ok": true},
		map[string]any{"bytes": 1},
	)
	if err != nil {
		t.Fatal(err)
	}

	got, err := c.RetrieveFuture(context.Background(), future.ID, true)
	if err != nil {
		t.Fatal(err)
	}
	if got.State != FutureCompleteMetadata {
		t.Fatalf("first state = %q, want %q", got.State, FutureCompleteMetadata)
	}
	if len(got.Metadata) == 0 {
		t.Fatal("first retrieval returned no metadata")
	}

	got, err = c.RetrieveFuture(context.Background(), future.ID, true)
	if err != nil {
		t.Fatal(err)
	}
	if got.State != FutureComplete {
		t.Fatalf("second state = %q, want %q", got.State, FutureComplete)
	}
	if len(got.Result) == 0 {
		t.Fatal("second retrieval returned no result")
	}
}

func TestQueuedFutureLifecycleAndDashboardQueue(t *testing.T) {
	start := time.Date(2026, 5, 4, 12, 0, 0, 0, time.UTC)
	now := start
	c, err := New(Config{
		Store: tinkerdb.OpenMemory(),
		Now: func() time.Time {
			return now
		},
		LeaseTimeout: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}

	future, err := c.EnqueueFuture(context.Background(),
		map[string]any{"type": "forward", "model_id": "model-a"},
		17,
		func(context.Context) (any, error) {
			now = now.Add(time.Second)
			return map[string]any{"ok": true}, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if future.State != FutureQueued {
		t.Fatalf("initial state = %q, want %q", future.State, FutureQueued)
	}
	eventuallyFutureState(t, c, future.ID, FutureComplete)

	snap, err := c.DashboardSnapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if snap.Queue.Complete != 1 || snap.Queue.ResultBytes == 0 {
		t.Fatalf("queue = %#v, want one complete future with result bytes", snap.Queue)
	}
	got := snap.Futures[0]
	if got.Operation != "forward" || got.ModelID != "model-a" || got.RequestBytes != 17 {
		t.Fatalf("future dashboard detail = %#v", got)
	}
}

func TestCancelQueuedFuture(t *testing.T) {
	c, err := New(Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	future, err := c.EnqueueFuture(context.Background(),
		map[string]any{"type": "forward", "model_id": "model-a"},
		1,
		func(context.Context) (any, error) {
			time.Sleep(10 * time.Millisecond)
			return map[string]any{"ok": true}, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	canceled, err := c.CancelFuture(context.Background(), future.ID)
	if err != nil {
		t.Fatal(err)
	}
	if canceled.State != FutureCanceled {
		t.Fatalf("state = %q, want %q", canceled.State, FutureCanceled)
	}
	got, err := c.RetrieveFuture(context.Background(), future.ID, false)
	if err != nil {
		t.Fatal(err)
	}
	if got.State != FutureCanceled {
		t.Fatalf("retrieved state = %q, want %q", got.State, FutureCanceled)
	}
}

func TestRunningFutureLeaseExpiry(t *testing.T) {
	now := time.Date(2026, 5, 4, 12, 0, 0, 0, time.UTC)
	store := tinkerdb.OpenMemory()
	c, err := New(Config{
		Store: store,
		Now: func() time.Time {
			return now
		},
		LeaseTimeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	future := tinkerdb.Future{
		ID:             "fut-running",
		State:          FutureRunning,
		CreatedAt:      now.Add(-2 * time.Second),
		StartedAt:      now.Add(-2 * time.Second),
		LeaseExpiresAt: now.Add(-time.Second),
	}
	if err := store.PutFuture(context.Background(), future); err != nil {
		t.Fatal(err)
	}
	got, err := c.RetrieveFuture(context.Background(), future.ID, false)
	if err != nil {
		t.Fatal(err)
	}
	if got.State != FutureSystemError {
		t.Fatalf("state = %q, want %q", got.State, FutureSystemError)
	}
}

func TestDashboardSnapshotFutureDetails(t *testing.T) {
	c, err := New(Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	future, err := c.CompleteFuture(context.Background(),
		map[string]any{
			"metrics": map[string]float64{
				"loss:mean":    1.25,
				"tokens:sum":   4,
				"examples:sum": 1,
			},
		},
		map[string]any{
			"type":     "forward_backward",
			"model_id": "model-a",
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if future.ID == "" {
		t.Fatal("empty future id")
	}

	snap, err := c.DashboardSnapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Futures) != 1 {
		t.Fatalf("futures = %d, want 1", len(snap.Futures))
	}
	got := snap.Futures[0]
	if got.Operation != "forward_backward" || got.ModelID != "model-a" {
		t.Fatalf("future detail = %#v", got)
	}
	if got.Metrics["loss:mean"] != 1.25 || got.Metrics["tokens:sum"] != 4 {
		t.Fatalf("metrics = %#v", got.Metrics)
	}
}

func eventuallyFutureState(t *testing.T, c *Coordinator, id, want string) Future {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		got, err := c.RetrieveFuture(context.Background(), id, false)
		if err != nil {
			t.Fatal(err)
		}
		if got.State == want {
			return got
		}
		if time.Now().After(deadline) {
			t.Fatalf("future %s state = %q, want %q", id, got.State, want)
		}
		time.Sleep(time.Millisecond)
	}
}
