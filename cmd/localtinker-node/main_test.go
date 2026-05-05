package main

import (
	"context"
	"errors"
	"flag"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"connectrpc.com/connect"

	"github.com/tmc/localtinker/internal/tinkerartifact"
	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerdb"
	"github.com/tmc/localtinker/internal/tinkernode"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1/tinkerv1connect"
	"github.com/tmc/localtinker/internal/tinkerrpc"
)

func TestRunHelp(t *testing.T) {
	for _, args := range [][]string{
		{"help"},
		{"--help"},
		{"run", "--help"},
		{"cache", "--help"},
		{"cache", "import", "--help"},
		{"cache", "sync", "--help"},
		{"cache", "delete", "--help"},
	} {
		if err := run(args); err != nil {
			t.Fatalf("run(%q) = %v, want nil", args, err)
		}
	}
}

func TestRunUsageErrors(t *testing.T) {
	if err := run(nil); !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("run(nil) = %v, want ErrHelp", err)
	}
	if err := run([]string{"bogus"}); err == nil {
		t.Fatal(`run(["bogus"]) = nil, want error`)
	}
	if err := run([]string{"cache"}); err == nil {
		t.Fatal(`run(["cache"]) = nil, want error`)
	}
}

func TestCacheDelete(t *testing.T) {
	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, "weights.bin"), []byte("delete"), 0o644); err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(t.TempDir(), "node")
	store, err := tinkerartifact.OpenStore(filepath.Join(root, "artifact-store"))
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := store.AddDirectory(context.Background(), src, tinkerartifact.ManifestOptions{
		Kind:      tinkerartifact.ArtifactTrainingCheckpoint,
		Storage:   tinkerartifact.StorageTinker,
		Name:      "delete",
		ChunkSize: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"cache", "delete", "-root", root, "-root-hash", manifest.RootHash}); err != nil {
		t.Fatal(err)
	}
	if store.Has(manifest.RootHash) {
		t.Fatal("artifact still installed")
	}
}

func TestHandleCommandAppliesArtifactRetention(t *testing.T) {
	store, roots := testStoreWithArtifacts(t, map[string]string{
		"root-a": "aaaa",
		"root-b": "bbbbbbbb",
	})
	if done, err := handleCommand(store, &tinkerv1.NodeCommand{
		CommandId: "retain-1",
		Directive: &tinkerv1.NodeCommand_ArtifactRetention{
			ArtifactRetention: &tinkerv1.ApplyArtifactRetention{
				ProtectedRootHashes: []string{roots["root-b"]},
				TargetFreeBytes:     4,
			},
		},
	}, func() {}); err != nil || done {
		t.Fatalf("handle retention: done=%v err=%v", done, err)
	}
	if store.Has(roots["root-a"]) {
		t.Fatal("root-a still installed")
	}
	if !store.Has(roots["root-b"]) {
		t.Fatal("root-b was deleted")
	}
}

func TestHandleCommandDeletesArtifacts(t *testing.T) {
	store, roots := testStoreWithArtifacts(t, map[string]string{
		"root-a": "aaaa",
		"root-b": "bbbb",
	})
	if done, err := handleCommand(store, &tinkerv1.NodeCommand{
		CommandId: "delete-1",
		Directive: &tinkerv1.NodeCommand_DeleteArtifact{
			DeleteArtifact: &tinkerv1.DeleteArtifact{
				RootHashes: []string{roots["root-a"]},
			},
		},
	}, func() {}); err != nil || done {
		t.Fatalf("handle delete: done=%v err=%v", done, err)
	}
	if store.Has(roots["root-a"]) {
		t.Fatal("root-a still installed")
	}
	if !store.Has(roots["root-b"]) {
		t.Fatal("root-b was deleted")
	}
}

func TestCacheImportPublishAndSync(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := tinkerrpc.New(coord)
	if err != nil {
		t.Fatal(err)
	}
	coordMux := http.NewServeMux()
	rpc.Register(coordMux)
	coordServer := httptest.NewServer(coordMux)
	defer coordServer.Close()

	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, "weights.bin"), []byte("mesh weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	sourceRoot := filepath.Join(t.TempDir(), "source")
	if err := run([]string{"cache", "import", "-root", sourceRoot, "-src", src, "-name", "latest", "-coordinator", coordServer.URL, "-chunk-size", "4"}); err != nil {
		t.Fatal(err)
	}
	sourceStore, err := tinkerartifact.OpenStore(filepath.Join(sourceRoot, "artifact-store"))
	if err != nil {
		t.Fatal(err)
	}
	refs, err := sourceStore.Inventory()
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) != 1 {
		t.Fatalf("refs = %d, want 1", len(refs))
	}

	peerMux := http.NewServeMux()
	tinkernode.RegisterArtifactPeer(peerMux, sourceStore)
	peerServer := httptest.NewServer(peerMux)
	defer peerServer.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(coordServer.Client(), coordServer.URL)
	tracker := tinkerv1connect.NewArtifactTrackerClient(coordServer.Client(), coordServer.URL)
	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "source",
		Labels: map[string]string{
			"artifact_peer_url": peerServer.URL,
		},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.ReportInventory(context.Background(), connect.NewRequest(&tinkerv1.ReportInventoryRequest{
		NodeId: "source",
		Artifacts: []*tinkerv1.ArtifactInventory{{
			RootHash:     refs[0].RootHash,
			State:        "complete",
			BytesPresent: 12,
		}},
	})); err != nil {
		t.Fatal(err)
	}

	targetRoot := filepath.Join(t.TempDir(), "target")
	if err := run([]string{"cache", "sync", "-root", targetRoot, "-coordinator", coordServer.URL, "-root-hash", "latest"}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(targetRoot, "artifact-store", "artifacts", "sha256", refs[0].RootHash, "files", "weights.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "mesh weights" {
		t.Fatalf("synced weights = %q", got)
	}
}

func testStoreWithArtifacts(t *testing.T, files map[string]string) (*tinkerartifact.Store, map[string]string) {
	t.Helper()
	store, err := tinkerartifact.OpenStore(filepath.Join(t.TempDir(), "artifact-store"))
	if err != nil {
		t.Fatal(err)
	}
	roots := make(map[string]string)
	for name, body := range files {
		src := t.TempDir()
		if err := os.WriteFile(filepath.Join(src, "weights.bin"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		m, err := store.AddDirectory(context.Background(), src, tinkerartifact.ManifestOptions{
			Kind:      tinkerartifact.ArtifactTrainingCheckpoint,
			Storage:   tinkerartifact.StorageTinker,
			Name:      name,
			ChunkSize: 4,
		})
		if err != nil {
			t.Fatal(err)
		}
		roots[name] = m.RootHash
	}
	return store, roots
}

func TestHandlePrewarmSyncsAndReportsInventory(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := tinkerrpc.New(coord)
	if err != nil {
		t.Fatal(err)
	}
	coordMux := http.NewServeMux()
	rpc.Register(coordMux)
	coordServer := httptest.NewServer(coordMux)
	defer coordServer.Close()

	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, "weights.bin"), []byte("prewarm weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	sourceStore, err := tinkerartifact.OpenStore(filepath.Join(t.TempDir(), "source", "artifact-store"))
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := sourceStore.AddDirectory(context.Background(), src, tinkerartifact.ManifestOptions{
		Kind:      tinkerartifact.ArtifactTrainingCheckpoint,
		Storage:   tinkerartifact.StorageTinker,
		Name:      "prewarm",
		ChunkSize: 4,
	})
	if err != nil {
		t.Fatal(err)
	}

	peerMux := http.NewServeMux()
	tinkernode.RegisterArtifactPeer(peerMux, sourceStore)
	peerServer := httptest.NewServer(peerMux)
	defer peerServer.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(coordServer.Client(), coordServer.URL)
	tracker := tinkerv1connect.NewArtifactTrackerClient(coordServer.Client(), coordServer.URL)
	admin := tinkerv1connect.NewTinkerAdminClient(coordServer.Client(), coordServer.URL)
	if _, err := tracker.PublishManifest(context.Background(), connect.NewRequest(&tinkerv1.PublishManifestRequest{
		Manifest: tinkerartifact.ToProto(manifest),
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "source",
		Labels: map[string]string{
			"artifact_peer_url": peerServer.URL,
		},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.ReportInventory(context.Background(), connect.NewRequest(&tinkerv1.ReportInventoryRequest{
		NodeId: "source",
		Artifacts: []*tinkerv1.ArtifactInventory{{
			RootHash:     manifest.RootHash,
			State:        "complete",
			BytesPresent: uint64(manifest.Size),
		}},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "target",
		Labels: map[string]string{
			"artifact_peer_url": "http://127.0.0.1:1",
		},
	})); err != nil {
		t.Fatal(err)
	}

	targetStore, err := tinkerartifact.OpenStore(filepath.Join(t.TempDir(), "target", "artifact-store"))
	if err != nil {
		t.Fatal(err)
	}
	if err := handlePrewarm(context.Background(), tracker, targetStore, "target", []string{manifest.RootHash}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(targetStore.Root, "artifacts", "sha256", manifest.RootHash, "files", "weights.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "prewarm weights" {
		t.Fatalf("prewarmed weights = %q", got)
	}

	peers, err := tracker.ListPeers(context.Background(), connect.NewRequest(&tinkerv1.ListPeersRequest{RootHash: manifest.RootHash}))
	if err != nil {
		t.Fatal(err)
	}
	if len(peers.Msg.GetPeers()) != 2 {
		t.Fatalf("peers = %d, want 2", len(peers.Msg.GetPeers()))
	}
	nodes, err := admin.ListNodes(context.Background(), connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	var labels map[string]string
	for _, node := range nodes.Msg.GetNodes() {
		if node.GetNodeId() == "target" {
			labels = node.GetLabels()
		}
	}
	if labels["last_transfer_root_hash"] != manifest.RootHash || labels["last_transfer_state"] != "complete" || labels["last_transfer_peer_node_id"] != "source" {
		t.Fatalf("target labels = %+v", labels)
	}
}

func TestCacheImportAndSyncHFHub(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := tinkerrpc.New(coord)
	if err != nil {
		t.Fatal(err)
	}
	coordMux := http.NewServeMux()
	rpc.Register(coordMux)
	coordServer := httptest.NewServer(coordMux)
	defer coordServer.Close()

	sourceHF := filepath.Join(t.TempDir(), "source-hf")
	t.Setenv("HUGGINGFACE_HUB_CACHE", sourceHF)
	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, "config.json"), []byte(`{"model_type":"smoke"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "model.safetensors"), []byte("hf mesh weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	sourceRoot := filepath.Join(t.TempDir(), "source")
	if err := run([]string{
		"cache", "import",
		"-root", sourceRoot,
		"-src", src,
		"-name", "Qwen/Qwen3-8B",
		"-kind", "base_model",
		"-storage", "hf_hub",
		"-repo-id", "Qwen/Qwen3-8B",
		"-repo-type", "model",
		"-revision", "main",
		"-commit-hash", "feedface",
		"-coordinator", coordServer.URL,
		"-alias", "qwen-smoke",
		"-chunk-size", "4",
	}); err != nil {
		t.Fatal(err)
	}
	sourceStore, err := tinkerartifact.OpenStore(filepath.Join(sourceRoot, "artifact-store"))
	if err != nil {
		t.Fatal(err)
	}
	refs, err := sourceStore.Inventory()
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) != 1 {
		t.Fatalf("refs = %d, want 1", len(refs))
	}

	peerMux := http.NewServeMux()
	tinkernode.RegisterArtifactPeer(peerMux, sourceStore)
	peerServer := httptest.NewServer(peerMux)
	defer peerServer.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(coordServer.Client(), coordServer.URL)
	tracker := tinkerv1connect.NewArtifactTrackerClient(coordServer.Client(), coordServer.URL)
	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "source",
		Labels: map[string]string{
			"artifact_peer_url": peerServer.URL,
		},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.ReportInventory(context.Background(), connect.NewRequest(&tinkerv1.ReportInventoryRequest{
		NodeId: "source",
		Artifacts: []*tinkerv1.ArtifactInventory{{
			RootHash: refs[0].RootHash,
			State:    "complete",
		}},
	})); err != nil {
		t.Fatal(err)
	}

	targetHF := filepath.Join(t.TempDir(), "target-hf")
	t.Setenv("HUGGINGFACE_HUB_CACHE", targetHF)
	targetRoot := filepath.Join(t.TempDir(), "target")
	if err := run([]string{"cache", "sync", "-root", targetRoot, "-coordinator", coordServer.URL, "-root-hash", "qwen-smoke"}); err != nil {
		t.Fatal(err)
	}
	paths, err := tinkerartifact.NewHFPaths(targetHF, "model", "Qwen/Qwen3-8B")
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(paths.Snapshots, "feedface", "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hf mesh weights" {
		t.Fatalf("synced HF data = %q", got)
	}
	if _, err := os.Stat(filepath.Join(targetRoot, "artifact-store", "artifacts", "sha256", refs[0].RootHash, "files")); !os.IsNotExist(err) {
		t.Fatalf("hf_hub files directory exists or stat failed: %v", err)
	}
}
