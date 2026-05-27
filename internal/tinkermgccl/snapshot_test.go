package tinkermgccl

import (
	"context"
	"testing"
	"time"

	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerdb"
	tinkerv1 "github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/reflect/protoreflect"
)

func TestFromDashboardBuildsPlannableSnapshot(t *testing.T) {
	now := time.Date(2026, 5, 19, 12, 0, 0, 0, time.UTC)
	caps := marshalProto(t, &tinkerv1.NodeCapabilities{
		Backends: []*tinkerv1.Backend{{Name: "metal", Device: "default", UnifiedMemory: true}},
		Memory:   &tinkerv1.MemoryInfo{TotalBytes: 16 << 30, AvailableBytes: 8 << 30},
		Models:   []*tinkerv1.NodeModel{{Name: "qwen", RootHash: "root", Dtype: "float16", CanTrain: true}},
		Features: &tinkerv1.NodeFeatures{Sampling: true},
	})
	load := marshalProto(t, &tinkerv1.NodeLoad{ActiveLeases: 1, QueuedOperations: 2})
	nodes := []tinkercoord.DashboardNode{
		{
			ID:             "node-b",
			State:          "healthy",
			Labels:         map[string]string{"rack": "r1"},
			Capabilities:   caps,
			Load:           load,
			MaxConcurrency: 2,
			Running:        1,
		},
		{
			ID:             "node-a",
			State:          "healthy",
			Labels:         map[string]string{"rack": "r1"},
			Capabilities:   caps,
			MaxConcurrency: 2,
		},
		{ID: "draining", State: "draining"},
	}
	snap, err := FromDashboard(nodes, Options{Epoch: "e1", Now: now})
	if err != nil {
		t.Fatal(err)
	}
	if nodes[0].ID != "node-b" {
		t.Fatalf("FromDashboard sorted caller slice in place")
	}
	if err := snap.Validate(); err != nil {
		t.Fatal(err)
	}
	if len(snap.Nodes) != 2 || snap.Nodes[0].ID != "node-a" || snap.Nodes[1].ID != "node-b" {
		t.Fatalf("nodes = %#v", snap.Nodes)
	}
	if got := snap.Nodes[1].Load.QueuedBatches; got != 2 {
		t.Fatalf("queued batches = %d, want 2", got)
	}
	if len(snap.Islands) != 1 || snap.Islands[0].ID != "localtinker-local" {
		t.Fatalf("islands = %#v", snap.Islands)
	}
	if got := snap.Islands[0].Capabilities.MaxRanks; got != 2 {
		t.Fatalf("island max ranks = %d, want 2", got)
	}
	if got := snap.Nodes[0].Backends[0].Capabilities.DeviceKinds; len(got) != 1 || got[0] != DeviceMetal {
		t.Fatalf("device kinds = %#v, want metal", got)
	}
	if len(snap.Links) != 1 || snap.Links[0].Kind != LinkLocalPeer {
		t.Fatalf("links = %#v", snap.Links)
	}
}

func TestSnapshotReadsCoordinatorNodes(t *testing.T) {
	now := time.Date(2026, 5, 19, 12, 0, 0, 0, time.UTC)
	coord, err := tinkercoord.New(tinkercoord.Config{
		Store: tinkerdb.OpenMemory(),
		Now: func() time.Time {
			return now
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := coord.RecordNode(context.Background(), tinkerdb.Node{
		ID:             "node-1",
		State:          "healthy",
		MaxConcurrency: 1,
		LastSeenAt:     now,
	}); err != nil {
		t.Fatal(err)
	}
	snap, err := Snapshot(context.Background(), coord, Options{Epoch: Epoch("e1"), Now: now})
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Nodes) != 2 {
		t.Fatalf("nodes = %d, want 2", len(snap.Nodes))
	}
	if err := snap.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestFromDashboardRejectsInvalidCapabilities(t *testing.T) {
	_, err := FromDashboard([]tinkercoord.DashboardNode{{
		ID:           "node-1",
		State:        "healthy",
		Capabilities: []byte("{"),
	}}, Options{Epoch: "e1", Now: time.Date(2026, 5, 19, 12, 0, 0, 0, time.UTC)})
	if err == nil {
		t.Fatal("FromDashboard accepted invalid capabilities")
	}
}

func marshalProto(t *testing.T, msg interface{ ProtoReflect() protoreflect.Message }) []byte {
	t.Helper()
	out, err := protojson.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	return out
}
