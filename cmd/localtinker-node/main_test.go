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
