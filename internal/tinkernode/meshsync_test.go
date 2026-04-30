package tinkernode

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"connectrpc.com/connect"

	"github.com/tmc/localtinker/internal/tinkerartifact"
)

func TestArtifactPeerSync(t *testing.T) {
	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, "weights.safetensors"), []byte("0123456789abcdef"), 0o644); err != nil {
		t.Fatal(err)
	}
	source, err := tinkerartifact.OpenStore(filepath.Join(t.TempDir(), "source"))
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := source.AddDirectory(context.Background(), src, tinkerartifact.ManifestOptions{
		Kind:      tinkerartifact.ArtifactTrainingCheckpoint,
		Storage:   tinkerartifact.StorageTinker,
		Name:      "step-1",
		Created:   time.Unix(1, 0).UTC(),
		ChunkSize: 5,
	})
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	RegisterArtifactPeer(mux, source, connect.WithReadMaxBytes(128<<20), connect.WithSendMaxBytes(128<<20))
	server := httptest.NewServer(mux)
	defer server.Close()

	target, err := tinkerartifact.OpenStore(filepath.Join(t.TempDir(), "target"))
	if err != nil {
		t.Fatal(err)
	}
	if err := SyncArtifact(context.Background(), target, manifest, []string{server.URL}); err != nil {
		t.Fatal(err)
	}
	if !target.Has(manifest.RootHash) {
		t.Fatal("target does not have synced artifact")
	}
	got, err := os.ReadFile(filepath.Join(target.Root, "artifacts", "sha256", manifest.RootHash, "files", "weights.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "0123456789abcdef" {
		t.Fatalf("synced data = %q", got)
	}
}
