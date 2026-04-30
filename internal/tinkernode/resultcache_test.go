package tinkernode

import (
	"context"
	"errors"
	"testing"
)

func TestMemoryResultCacheKeepsFirstResult(t *testing.T) {
	ctx := context.Background()
	c := NewMemoryResultCache()

	first := OperationResult{
		OperationID: "op-1",
		LeaseID:     "lease-1",
		Kind:        OperationOptimStep,
		State:       ResultSucceeded,
		Payload:     []byte("first"),
		Metadata:    map[string]string{"checkpoint": "a"},
	}
	if err := c.Store(ctx, first); err != nil {
		t.Fatal(err)
	}
	if err := c.Store(ctx, OperationResult{OperationID: "op-1", Payload: []byte("second")}); err != nil {
		t.Fatal(err)
	}
	got, ok, err := c.Get(ctx, "op-1")
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("missing cached result")
	}
	if string(got.Payload) != "first" {
		t.Fatalf("payload = %q, want first", got.Payload)
	}

	got.Payload[0] = 'x'
	got.Metadata["checkpoint"] = "changed"
	again, ok, err := c.Get(ctx, "op-1")
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("missing cached result")
	}
	if string(again.Payload) != "first" || again.Metadata["checkpoint"] != "a" {
		t.Fatalf("cache result was mutated: %+v", again)
	}
}

func TestMemoryResultCacheAcknowledge(t *testing.T) {
	ctx := context.Background()
	c := NewMemoryResultCache()
	if err := c.Store(ctx, OperationResult{OperationID: "op-1"}); err != nil {
		t.Fatal(err)
	}
	if err := c.Acknowledge(ctx, "op-1"); err != nil {
		t.Fatal(err)
	}
	_, ok, err := c.Get(ctx, "op-1")
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("acknowledged result remains cached")
	}
	if err := c.Acknowledge(ctx, "op-1"); !errors.Is(err, ErrResultNotFound) {
		t.Fatalf("Acknowledge missing result err = %v, want ErrResultNotFound", err)
	}
}
