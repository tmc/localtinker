package tinkerdb

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestCheckpointMetadataPersists(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	expires := time.Date(2026, 4, 30, 12, 0, 0, 0, time.UTC)

	store, err := OpenJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.PutCheckpoint(context.Background(), Checkpoint{
		Path:      "tinker://model-a/weights/ckpt",
		Public:    true,
		Owner:     "local",
		ExpiresAt: &expires,
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	store, err = OpenJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := store.GetCheckpoint(context.Background(), "tinker://model-a/weights/ckpt")
	if err != nil {
		t.Fatal(err)
	}
	if !got.Public || got.Owner != "local" || got.ExpiresAt == nil || !got.ExpiresAt.Equal(expires) {
		t.Fatalf("checkpoint = %#v", got)
	}
}
