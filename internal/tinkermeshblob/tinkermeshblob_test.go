package tinkermeshblob_test

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/go-iroh/netaddr"
	"github.com/tmc/localtinker/internal/tinkerartifact"
	"github.com/tmc/localtinker/internal/tinkermeshblob"
	"github.com/tmc/mlx-go-iroh/manifest"
)

// fakeHub serves a checkpoint's chunks from a blob store keyed by BLAKE3 hash —
// standing in for a peer or hub fetch without a network.
type fakeHub struct {
	store *blobStore
	calls int
}

type blobStore struct{ data map[[32]byte][]byte }

func (h *fakeHub) Fetch(_ context.Context, b manifest.Blob) ([]byte, error) {
	h.calls++
	d, ok := h.store.data[b.Hash]
	if !ok {
		return nil, os.ErrNotExist
	}
	return d, nil
}

// peerNone is a PeerFetcher that never has a peer, forcing the hub path.
type peerNone struct{}

func (peerNone) Fetch(context.Context, netaddr.EndpointAddr, manifest.Blob) ([]byte, error) {
	return nil, os.ErrNotExist
}

func TestCheckpointReplicationRoundTrip(t *testing.T) {
	ctx := context.Background()

	// Source: an artifact store holding a multi-chunk checkpoint.
	srcDir := t.TempDir()
	payload := bytes.Repeat([]byte("localtinker-checkpoint-bytes-"), 4096) // ~110 KiB
	if err := os.WriteFile(filepath.Join(srcDir, "weights.bin"), payload, 0o644); err != nil {
		t.Fatalf("write payload: %v", err)
	}
	srcStore, err := tinkerartifact.OpenStore(filepath.Join(t.TempDir(), "src-store"))
	if err != nil {
		t.Fatalf("open src store: %v", err)
	}
	m, err := srcStore.AddDirectory(ctx, srcDir, tinkerartifact.ManifestOptions{
		ID:        "ckpt-1",
		Kind:      tinkerartifact.ArtifactTrainingCheckpoint,
		Storage:   tinkerartifact.StorageTinker,
		Name:      "ckpt",
		Version:   "1",
		ChunkSize: 16 << 10, // 16 KiB -> several chunks
	})
	if err != nil {
		t.Fatalf("add directory: %v", err)
	}

	// Build the iroh manifest and a hub-backed blob source from the source store.
	im, err := tinkermeshblob.BuildManifest(srcStore, m.RootHash)
	if err != nil {
		t.Fatalf("build manifest: %v", err)
	}
	if len(im.Entries) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(im.Entries))
	}
	bs, err := tinkermeshblob.NewBlobStore(srcStore, m.RootHash)
	if err != nil {
		t.Fatalf("new blob store: %v", err)
	}
	hub := &fakeHub{store: &blobStore{data: map[[32]byte][]byte{}}}
	for _, e := range im.Entries {
		data, ok := bs.GetBlob(e.Blob.Hash)
		if !ok {
			t.Fatalf("blob store missing %x", e.Blob.Hash)
		}
		hub.store.data[e.Blob.Hash] = data
	}

	// Destination: a fresh store with no chunks. Pull via hub fallback and install.
	dstStore, err := tinkerartifact.OpenStore(filepath.Join(t.TempDir(), "dst-store"))
	if err != nil {
		t.Fatalf("open dst store: %v", err)
	}
	sync, err := manifest.NewSync(im)
	if err != nil {
		t.Fatalf("new sync: %v", err)
	}
	if err := sync.Pull(ctx, im, peerNone{}, hub); err != nil {
		t.Fatalf("pull: %v", err)
	}
	if !sync.UsedHub() {
		t.Fatal("expected hub fallback")
	}
	if err := tinkermeshblob.Install(ctx, dstStore, m, sync); err != nil {
		t.Fatalf("install: %v", err)
	}

	// The destination now has the checkpoint, reconstructed and verified.
	if !dstStore.Has(m.RootHash) {
		t.Fatal("destination store does not have the installed checkpoint")
	}
}
