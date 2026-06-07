package tinkercoord

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/localtinker/internal/tinkerdb"
)

// seedCheckpoint writes a minimal on-disk checkpoint dir so the coordinator's
// duplicate check (which stats the adapters file) observes an existing save.
func seedCheckpoint(t *testing.T, root, modelID, kind, name string) string {
	t.Helper()
	dir := filepath.Join(root, modelID, kind, name)
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "adapters.safetensors")
	if err := os.WriteFile(file, []byte("stale"), 0644); err != nil {
		t.Fatal(err)
	}
	return file
}

func userErrorMessage(t *testing.T, c *Coordinator, id string) string {
	t.Helper()
	got, err := c.RetrieveFuture(context.Background(), id, false)
	if err != nil {
		t.Fatal(err)
	}
	if got.State != FutureUserError {
		t.Fatalf("state = %q, want %q", got.State, FutureUserError)
	}
	return string(got.Error)
}

func TestSaveWeightsDuplicateIsUserError(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LOCALTINKER_CHECKPOINT_ROOT", root)
	stale := seedCheckpoint(t, root, "m", "weights", "ckpt")

	c, err := New(Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}

	// overwrite=false on an existing name fails as a terminal user error and
	// leaves the existing checkpoint untouched.
	fut, err := c.SaveWeights(context.Background(), "m", "ckpt", 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if msg := userErrorMessage(t, c, fut.ID); !strings.Contains(msg, "already exists") {
		t.Fatalf("error = %q, want it to mention already exists", msg)
	}
	if _, err := os.Stat(stale); err != nil {
		t.Fatalf("existing checkpoint removed on a rejected save: %v", err)
	}

	// overwrite=true removes the stale checkpoint before saving. The default
	// manager has no model, so the subsequent save reports a different user
	// error, but the stale files must already be gone.
	fut, err = c.SaveWeights(context.Background(), "m", "ckpt", 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if msg := userErrorMessage(t, c, fut.ID); strings.Contains(msg, "already exists") {
		t.Fatalf("overwrite=true still reported a duplicate: %q", msg)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("overwrite did not remove stale checkpoint: err=%v", err)
	}
}

func TestSaveWeightsForSamplerDuplicateIsUserError(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LOCALTINKER_CHECKPOINT_ROOT", root)
	stale := seedCheckpoint(t, root, "m", "sampler_weights", "ckpt")

	c, err := New(Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}

	// A named sampler save honors overwrite the same way save_weights does.
	fut, err := c.SaveWeightsForSampler(context.Background(), "m", "ckpt", 1, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if msg := userErrorMessage(t, c, fut.ID); !strings.Contains(msg, "already exists") {
		t.Fatalf("error = %q, want it to mention already exists", msg)
	}
	if _, err := os.Stat(stale); err != nil {
		t.Fatalf("existing sampler checkpoint removed on a rejected save: %v", err)
	}

	fut, err = c.SaveWeightsForSampler(context.Background(), "m", "ckpt", 1, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if msg := userErrorMessage(t, c, fut.ID); strings.Contains(msg, "already exists") {
		t.Fatalf("overwrite=true still reported a duplicate: %q", msg)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("overwrite did not remove stale sampler checkpoint: err=%v", err)
	}
}

// TestSaveWeightsForSamplerEphemeralIgnoresOverwrite proves that an empty path
// (ephemeral sampler save) is never treated as a duplicate even when a prior
// ephemeral name collides, because ephemeral names are session-unique.
func TestSaveWeightsForSamplerEphemeralIgnoresOverwrite(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LOCALTINKER_CHECKPOINT_ROOT", root)
	// Seed the dir an ephemeral save with seq 7 would synthesize.
	seedCheckpoint(t, root, "m", "sampler_weights", "ephemeral-7")

	c, err := New(Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	fut, err := c.SaveWeightsForSampler(context.Background(), "m", "", 7, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	// No model is registered, so the save still fails, but it must not fail with
	// the duplicate "already exists" error: ephemeral saves skip the check.
	if msg := userErrorMessage(t, c, fut.ID); strings.Contains(msg, "already exists") {
		t.Fatalf("ephemeral save reported a duplicate: %q", msg)
	}
}
