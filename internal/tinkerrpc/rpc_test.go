package tinkerrpc

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"connectrpc.com/connect"

	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerdb"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1/tinkerv1connect"
)

func TestRegisterNodeAndListNodes(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	adminClient := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)

	reg, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
		Name:   "Node A",
		Labels: map[string]string{"rack": "desk"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if got := reg.Msg.GetAssignedNodeId(); got != "node-a" {
		t.Fatalf("assigned node id = %q, want node-a", got)
	}
	if got := reg.Msg.GetConfig()["rpc_max_bytes"]; got != "134217728" {
		t.Fatalf("rpc_max_bytes = %q", got)
	}

	if _, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-a",
		Load:   &tinkerv1.NodeLoad{ActiveLeases: 1},
	})); err != nil {
		t.Fatal(err)
	}

	nodes, err := adminClient.ListNodes(context.Background(), connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes.Msg.GetNodes()) != 1 {
		t.Fatalf("nodes = %d, want 1", len(nodes.Msg.GetNodes()))
	}
	node := nodes.Msg.GetNodes()[0]
	if node.GetNodeId() != "node-a" || node.GetName() != "Node A" || node.GetLoad().GetActiveLeases() != 1 {
		t.Fatalf("node summary = %+v", node)
	}
}

func TestDrainNodeRequestsDrainOnHeartbeat(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	adminClient := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)

	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
		Name:   "Node A",
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-a",
		Load:   &tinkerv1.NodeLoad{ActiveLeases: 1},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := adminClient.DrainNode(context.Background(), connect.NewRequest(&tinkerv1.DrainNodeRequest{
		NodeId: "node-a",
		Reason: "test",
	})); err != nil {
		t.Fatal(err)
	}

	nodes, err := adminClient.ListNodes(context.Background(), connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if got := nodes.Msg.GetNodes()[0].GetState(); got != "draining" {
		t.Fatalf("state = %q, want draining", got)
	}

	hb, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-a",
		Load:   &tinkerv1.NodeLoad{ActiveLeases: 0},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if !hb.Msg.GetDrainRequested() {
		t.Fatal("drain_requested = false, want true")
	}

	nodes, err = adminClient.ListNodes(context.Background(), connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if got := nodes.Msg.GetNodes()[0].GetState(); got != "drained" {
		t.Fatalf("state = %q, want drained", got)
	}
}

func TestArtifactTrackerManifestInventoryAndPeers(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	tracker := tinkerv1connect.NewArtifactTrackerClient(server.Client(), server.URL)
	admin := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)

	manifest := &tinkerv1.Manifest{
		Kind:      "training_checkpoint",
		Storage:   "tinker",
		Name:      "ckpt",
		RootHash:  "abc123",
		ChunkSize: 4,
		Files: []*tinkerv1.ManifestFile{{
			Path:   "weights.bin",
			Size:   4,
			Sha256: "file",
			Chunks: []*tinkerv1.ChunkRef{{Index: 0, Size: 4, Sha256: "chunk"}},
		}},
	}
	pub, err := tracker.PublishManifest(context.Background(), connect.NewRequest(&tinkerv1.PublishManifestRequest{
		Manifest: manifest,
		Alias:    "latest",
	}))
	if err != nil {
		t.Fatal(err)
	}
	if pub.Msg.GetRootHash() != "abc123" {
		t.Fatalf("root hash = %q", pub.Msg.GetRootHash())
	}
	got, err := tracker.GetManifest(context.Background(), connect.NewRequest(&tinkerv1.GetManifestRequest{RootHashOrAlias: "latest"}))
	if err != nil {
		t.Fatal(err)
	}
	if got.Msg.GetManifest().GetRootHash() != "abc123" {
		t.Fatalf("manifest root = %q", got.Msg.GetManifest().GetRootHash())
	}

	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
		Labels: map[string]string{
			"artifact_peer_url": "http://127.0.0.1:9000",
			"rack":              "desk",
		},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.ReportInventory(context.Background(), connect.NewRequest(&tinkerv1.ReportInventoryRequest{
		NodeId: "node-a",
		Artifacts: []*tinkerv1.ArtifactInventory{{
			RootHash:     "abc123",
			State:        "complete",
			BytesPresent: 4,
		}},
	})); err != nil {
		t.Fatal(err)
	}
	peers, err := tracker.ListPeers(context.Background(), connect.NewRequest(&tinkerv1.ListPeersRequest{
		RootHash:        "abc123",
		PreferredLabels: []string{"rack=desk"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if len(peers.Msg.GetPeers()) != 1 {
		t.Fatalf("peers = %d, want 1", len(peers.Msg.GetPeers()))
	}
	if peers.Msg.GetPeers()[0].GetAddress() != "http://127.0.0.1:9000" {
		t.Fatalf("peer address = %q", peers.Msg.GetPeers()[0].GetAddress())
	}

	artifacts, err := admin.ListArtifacts(context.Background(), connect.NewRequest(&tinkerv1.ListArtifactsRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if len(artifacts.Msg.GetArtifacts()) != 1 || artifacts.Msg.GetArtifacts()[0].GetAlias() != "latest" {
		t.Fatalf("artifacts = %+v", artifacts.Msg.GetArtifacts())
	}
}
