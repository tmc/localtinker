package tinkercoord

import (
	"context"
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
