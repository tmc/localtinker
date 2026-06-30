package tinkermeshiroh_test

import (
	"context"
	"testing"
	"time"

	"github.com/tmc/localtinker/internal/tinkerid"
	"github.com/tmc/localtinker/internal/tinkermesh"
	"github.com/tmc/localtinker/internal/tinkermeshiroh"
)

// TestHeartbeatLoopback brings up a coordinator-side and a node-side transport
// on the same topics and confirms a node's signed heartbeat is received and
// verified by the coordinator — the core node->coordinator liveness path.
func TestHeartbeatLoopback(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()

	routine, critical := tinkermeshiroh.DeriveTopics("run-test-1")

	coordKey, err := tinkerid.LoadOrCreate(t.TempDir())
	if err != nil {
		t.Fatalf("coord key: %v", err)
	}
	nodeKey, err := tinkerid.LoadOrCreate(t.TempDir())
	if err != nil {
		t.Fatalf("node key: %v", err)
	}

	coord, err := tinkermeshiroh.Dial(ctx, tinkermeshiroh.Config{
		Key:           coordKey,
		CoordinatorID: coordKey.ID(),
		RoutineTopic:  routine,
		CriticalTopic: critical,
		BindAddr:      "127.0.0.1:0",
	})
	if err != nil {
		t.Fatalf("dial coord: %v", err)
	}
	defer coord.Close()

	node, err := tinkermeshiroh.Dial(ctx, tinkermeshiroh.Config{
		Key:           nodeKey,
		CoordinatorID: coordKey.ID(),
		RoutineTopic:  routine,
		CriticalTopic: critical,
		Bootstrap:     []string{coord.SeedAddr()},
		BindAddr:      "127.0.0.1:0",
	})
	if err != nil {
		t.Fatalf("dial node: %v", err)
	}
	defer node.Close()

	// Node publishes heartbeats until the coordinator receives one (gossip
	// delivery is eventual once the swarm connects).
	go func() {
		hb := tinkermesh.Heartbeat{
			UnixNano: time.Now().UnixNano(),
			Load:     tinkermesh.Load{ActiveLeases: 2, QueuedOperations: 1, MemoryAvailableBytes: 1 << 30},
		}
		for {
			select {
			case <-ctx.Done():
				return
			default:
			}
			_ = node.PublishHeartbeat(ctx, hb)
			time.Sleep(250 * time.Millisecond)
		}
	}()

	select {
	case hb := <-coord.Heartbeats():
		if hb.NodeID != nodeKey.ID() {
			t.Fatalf("heartbeat from %s, want node %s", hb.NodeID, nodeKey.ID())
		}
		if hb.Load.ActiveLeases != 2 {
			t.Fatalf("active leases = %d, want 2", hb.Load.ActiveLeases)
		}
		if err := tinkermesh.VerifyHeartbeat(hb); err != nil {
			t.Fatalf("received heartbeat not verifiable: %v", err)
		}
	case <-ctx.Done():
		t.Fatal("coordinator did not receive a heartbeat before timeout")
	}
}

func TestDeriveTopicsDistinctAndDeterministic(t *testing.T) {
	r1, c1 := tinkermeshiroh.DeriveTopics("run-a")
	r2, c2 := tinkermeshiroh.DeriveTopics("run-a")
	if r1 != r2 || c1 != c2 {
		t.Fatal("DeriveTopics not deterministic")
	}
	if r1 == c1 {
		t.Fatal("routine and critical topics collide")
	}
	r3, _ := tinkermeshiroh.DeriveTopics("run-b")
	if r1 == r3 {
		t.Fatal("different runs share a routine topic")
	}
}
