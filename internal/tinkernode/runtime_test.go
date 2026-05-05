package tinkernode

import (
	"context"
	"errors"
	"testing"
	"time"

	"connectrpc.com/connect"

	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
)

func TestRuntimeRunCachesResult(t *testing.T) {
	cache := NewMemoryResultCache()
	r := NewRuntime("node-1", WithResultCache(cache))
	resp, err := r.Run(context.Background(), connect.NewRequest(&tinkerv1.NodeCommand{
		OperationId: "op-1",
		LeaseId:     "lease-1",
		Kind:        string(OperationOptimStep),
		PayloadJson: []byte(`{"ok":true}`),
	}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Msg.GetOperationId() != "op-1" || resp.Msg.GetKind() != string(OperationOptimStep) {
		t.Fatalf("Run result = %+v, want op-1 optim_step", resp.Msg)
	}
	got, ok, err := cache.Get(context.Background(), "op-1")
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("result was not cached")
	}
	if got.State != ResultSucceeded || string(got.Payload) != `{"ok":true}` {
		t.Fatalf("cached result = %+v", got)
	}
}

func TestRuntimeCancelActiveOperation(t *testing.T) {
	entered := make(chan struct{})
	r := NewRuntime("node-1", WithOperationRunner(func(ctx context.Context, cmd *tinkerv1.NodeCommand) OperationResult {
		close(entered)
		<-ctx.Done()
		return canceledResult(cmd, ctx.Err())
	}))

	done := make(chan error, 1)
	go func() {
		_, err := r.Run(context.Background(), connect.NewRequest(&tinkerv1.NodeCommand{
			OperationId: "op-1",
			LeaseId:     "lease-1",
			Kind:        string(OperationForwardBackward),
		}))
		done <- err
	}()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("runner did not start")
	}

	cancel, err := r.Cancel(context.Background(), connect.NewRequest(&tinkerv1.CancelRequest{OperationId: "op-1"}))
	if err != nil {
		t.Fatal(err)
	}
	if !cancel.Msg.GetCanceled() {
		t.Fatal("Cancel returned canceled=false")
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("Run did not finish after Cancel")
	}
	got, ok, err := r.cache.Get(context.Background(), "op-1")
	if err != nil {
		t.Fatal(err)
	}
	if !ok || got.State != ResultCanceled {
		t.Fatalf("cached result = %+v ok=%v, want canceled", got, ok)
	}
}

func TestRuntimeDrainRefusesNewWork(t *testing.T) {
	r := NewRuntime("node-1")
	if _, err := r.Drain(context.Background(), connect.NewRequest(&tinkerv1.DrainRequest{Reason: "test"})); err != nil {
		t.Fatal(err)
	}
	health, err := r.Health(context.Background(), connect.NewRequest(&tinkerv1.NodeHealthRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if health.Msg.GetState() != NodeStateDrained {
		t.Fatalf("state = %q, want %q", health.Msg.GetState(), NodeStateDrained)
	}
	_, err = r.Run(context.Background(), connect.NewRequest(&tinkerv1.NodeCommand{OperationId: "op-1"}))
	if connect.CodeOf(err) != connect.CodeFailedPrecondition {
		t.Fatalf("Run while drained err = %v, want failed_precondition", err)
	}
}

func TestRuntimeHealthReportsActiveLeasesWhileDraining(t *testing.T) {
	release := make(chan struct{})
	started := make(chan struct{})
	r := NewRuntime("node-1", WithOperationRunner(func(ctx context.Context, cmd *tinkerv1.NodeCommand) OperationResult {
		close(started)
		select {
		case <-release:
			return OperationResult{State: ResultSucceeded}
		case <-ctx.Done():
			return canceledResult(cmd, ctx.Err())
		}
	}))
	go func() {
		_, _ = r.Run(context.Background(), connect.NewRequest(&tinkerv1.NodeCommand{OperationId: "op-1"}))
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("runner did not start")
	}
	if _, err := r.Drain(context.Background(), connect.NewRequest(&tinkerv1.DrainRequest{})); err != nil {
		t.Fatal(err)
	}
	health, err := r.Health(context.Background(), connect.NewRequest(&tinkerv1.NodeHealthRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if health.Msg.GetState() != NodeStateDraining || health.Msg.GetLoad().GetActiveLeases() != 1 {
		t.Fatalf("health = %+v, want draining with 1 active lease", health.Msg)
	}
	close(release)
	for i := 0; i < 100; i++ {
		health, err = r.Health(context.Background(), connect.NewRequest(&tinkerv1.NodeHealthRequest{}))
		if err != nil {
			t.Fatal(err)
		}
		if health.Msg.GetState() == NodeStateDrained {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("runtime did not transition to drained")
}

func TestRuntimeRunRequiresOperationID(t *testing.T) {
	r := NewRuntime("node-1")
	_, err := r.Run(context.Background(), connect.NewRequest(&tinkerv1.NodeCommand{}))
	if connect.CodeOf(err) != connect.CodeInvalidArgument || !errors.Is(err, ErrNoOperationID) {
		t.Fatalf("Run missing operation ID err = %v", err)
	}
}
