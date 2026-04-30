package tinkerartifact

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestStoreAddDirectoryAndInstallFromChunks(t *testing.T) {
	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, "a.txt"), []byte("hello artifact mesh"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(src, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "nested", "b.bin"), []byte("0123456789"), 0o600); err != nil {
		t.Fatal(err)
	}

	source, err := OpenStore(filepath.Join(t.TempDir(), "source"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := source.AddDirectory(context.Background(), src, ManifestOptions{
		Kind:      ArtifactTrainingCheckpoint,
		Storage:   StorageTinker,
		Name:      "ckpt",
		Created:   time.Unix(1, 0).UTC(),
		ChunkSize: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.RootHash == "" {
		t.Fatal("empty root hash")
	}
	hashes, err := source.ChunkHashes(m.RootHash)
	if err != nil {
		t.Fatal(err)
	}
	if len(hashes) == 0 {
		t.Fatal("no chunks")
	}

	target, err := OpenStore(filepath.Join(t.TempDir(), "target"))
	if err != nil {
		t.Fatal(err)
	}
	for _, hash := range hashes {
		in, _, err := source.OpenChunk(m.RootHash, hash)
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(in.Name())
		_ = in.Close()
		if err != nil {
			t.Fatal(err)
		}
		if err := target.AddChunk(hash, data); err != nil {
			t.Fatal(err)
		}
	}
	if err := target.InstallFromChunks(context.Background(), m); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(target.Root, "artifacts", "sha256", m.RootHash, "files", "nested", "b.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "0123456789" {
		t.Fatalf("installed data = %q", got)
	}
}

func TestStoreDeleteRemovesArtifactButKeepsChunks(t *testing.T) {
	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, "weights.bin"), []byte("delete me"), 0o644); err != nil {
		t.Fatal(err)
	}
	store, err := OpenStore(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := store.AddDirectory(context.Background(), src, ManifestOptions{
		Kind:      ArtifactTrainingCheckpoint,
		Storage:   StorageTinker,
		Name:      "ckpt",
		ChunkSize: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	hashes, err := store.ChunkHashes(m.RootHash)
	if err != nil {
		t.Fatal(err)
	}
	if len(hashes) == 0 {
		t.Fatal("no chunks")
	}
	if err := store.Delete(m.RootHash); err != nil {
		t.Fatal(err)
	}
	if store.Has(m.RootHash) {
		t.Fatal("artifact still installed")
	}
	if _, err := store.Manifest(m.RootHash); !errors.Is(err, ErrNotFound) {
		t.Fatalf("manifest error = %v, want ErrNotFound", err)
	}
	if _, err := os.Stat(store.chunkPath(hashes[0])); err != nil {
		t.Fatal(err)
	}
	if err := store.Delete(m.RootHash); !errors.Is(err, ErrNotFound) {
		t.Fatalf("delete missing = %v, want ErrNotFound", err)
	}
}

func TestProtoRoundTripKeepsRootHash(t *testing.T) {
	m := Manifest{
		Kind:      ArtifactBaseModel,
		Storage:   StorageTinker,
		Name:      "model",
		Created:   time.Now(),
		Size:      123,
		ChunkSize: 4,
		Files: []ManifestFile{{
			Path:   "weights.bin",
			Size:   4,
			Mode:   0o644,
			SHA256: "file",
			Chunks: []ChunkRef{{Index: 0, Size: 4, SHA256: "chunk"}},
		}},
	}
	root, err := RootHash(m)
	if err != nil {
		t.Fatal(err)
	}
	m.RootHash = root
	got := FromProto(ToProto(m))
	if got.RootHash != root {
		t.Fatalf("root hash = %q, want %q", got.RootHash, root)
	}
	if err := verifyManifestIdentity(got); err != nil {
		t.Fatal(err)
	}
}

func TestHFHubInstallFromChunks(t *testing.T) {
	hfRoot := filepath.Join(t.TempDir(), "hf")
	t.Setenv("HUGGINGFACE_HUB_CACHE", hfRoot)

	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, "config.json"), []byte(`{"model_type":"smoke"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "model.safetensors"), []byte("hf weights"), 0o644); err != nil {
		t.Fatal(err)
	}

	source, err := OpenStore(filepath.Join(t.TempDir(), "source"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := source.InstallHFHubFromDirectory(context.Background(), src, ManifestOptions{
		Kind:      ArtifactBaseModel,
		Storage:   StorageHFHub,
		Name:      "Qwen/Qwen3-8B",
		ChunkSize: 4,
		Metadata: Metadata{
			RepoID:     "Qwen/Qwen3-8B",
			RepoType:   "model",
			Revision:   "main",
			CommitHash: "abcdef123456",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	paths, err := NewHFPaths(hfRoot, "model", "Qwen/Qwen3-8B")
	if err != nil {
		t.Fatal(err)
	}
	ref, err := os.ReadFile(filepath.Join(paths.Refs, "main"))
	if err != nil {
		t.Fatal(err)
	}
	if string(ref) != "abcdef123456" {
		t.Fatalf("ref = %q", ref)
	}
	snapshotFile := filepath.Join(paths.Snapshots, "abcdef123456", "model.safetensors")
	got, err := os.ReadFile(snapshotFile)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hf weights" {
		t.Fatalf("snapshot data = %q", got)
	}

	targetHF := filepath.Join(t.TempDir(), "target-hf")
	t.Setenv("HUGGINGFACE_HUB_CACHE", targetHF)
	target, err := OpenStore(filepath.Join(t.TempDir(), "target"))
	if err != nil {
		t.Fatal(err)
	}
	hashes, err := source.ChunkHashes(m.RootHash)
	if err != nil {
		t.Fatal(err)
	}
	for _, hash := range hashes {
		chunk, _, err := source.OpenChunk(m.RootHash, hash)
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(chunk.Name())
		_ = chunk.Close()
		if err != nil {
			t.Fatal(err)
		}
		if err := target.AddChunk(hash, data); err != nil {
			t.Fatal(err)
		}
	}
	if err := target.InstallFromChunks(context.Background(), m); err != nil {
		t.Fatal(err)
	}
	targetPaths, err := NewHFPaths(targetHF, "model", "Qwen/Qwen3-8B")
	if err != nil {
		t.Fatal(err)
	}
	got, err = os.ReadFile(filepath.Join(targetPaths.Snapshots, "abcdef123456", "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hf weights" {
		t.Fatalf("target snapshot data = %q", got)
	}
	if _, err := os.Stat(filepath.Join(target.Root, "artifacts", "sha256", m.RootHash, "files")); !os.IsNotExist(err) {
		t.Fatalf("hf_hub artifact files directory exists or stat failed: %v", err)
	}

	if err := os.Remove(filepath.Join(targetPaths.Snapshots, "abcdef123456", "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if target.Has(m.RootHash) {
		t.Fatal("target reports incomplete hf_hub artifact as installed")
	}
}
