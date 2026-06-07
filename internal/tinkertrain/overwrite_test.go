package tinkertrain

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// writingTrainModel records save calls and writes a real checkpoint dir so the
// Manager existence helpers observe the same files saveAdapter would create.
type writingTrainModel struct {
	id    string
	saves int
}

func (*writingTrainModel) forwardBackward(context.Context, ForwardBackwardInput, bool) (ForwardBackwardOutput, error) {
	return ForwardBackwardOutput{}, nil
}

func (*writingTrainModel) optimStep(context.Context, AdamParams) (OptimStepOutput, error) {
	return OptimStepOutput{}, nil
}

func (m *writingTrainModel) saveState(_ context.Context, name string) (string, error) {
	return m.write(kindState, name)
}

func (m *writingTrainModel) saveForSampler(_ context.Context, name string) (string, error) {
	return m.write(kindSampler, name)
}

func (m *writingTrainModel) write(kind, name string) (string, error) {
	m.saves++
	path := checkpointTinkerPath(m.id, kind, name)
	parsed, err := ParseTinkerPath(path)
	if err != nil {
		return "", err
	}
	dir := checkpointDir(parsed)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(dir, "adapters.safetensors"), []byte("w"), 0644); err != nil {
		return "", err
	}
	return path, nil
}

func (*writingTrainModel) loadState(context.Context, string, bool) error { return nil }

func (*writingTrainModel) sample(context.Context, SampleRequest) (SampleOutput, error) {
	return SampleOutput{}, nil
}

func TestManagerCheckpointExistsAndRemove(t *testing.T) {
	t.Setenv("LOCALTINKER_CHECKPOINT_ROOT", t.TempDir())

	model := &writingTrainModel{id: "m"}
	m := &Manager{models: map[string]trainModel{"m": model}}
	ctx := context.Background()

	tests := []struct {
		name   string
		exists func(string) bool
		save   func(string) (string, error)
		remove func(string) error
	}{
		{
			name:   "state",
			exists: func(n string) bool { return m.StateCheckpointExists("m", n) },
			save:   func(n string) (string, error) { return m.SaveState(ctx, "m", n) },
			remove: func(n string) error { return m.RemoveStateCheckpoint("m", n) },
		},
		{
			name:   "sampler",
			exists: func(n string) bool { return m.SamplerCheckpointExists("m", n) },
			save:   func(n string) (string, error) { return m.SaveForSampler(ctx, "m", n) },
			remove: func(n string) error { return m.RemoveSamplerCheckpoint("m", n) },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const ckpt = "ckpt"
			if tt.exists(ckpt) {
				t.Fatal("checkpoint exists before any save")
			}
			if _, err := tt.save(ckpt); err != nil {
				t.Fatal(err)
			}
			if !tt.exists(ckpt) {
				t.Fatal("checkpoint missing after save")
			}
			if err := tt.remove(ckpt); err != nil {
				t.Fatal(err)
			}
			if tt.exists(ckpt) {
				t.Fatal("checkpoint exists after remove")
			}
		})
	}
}

// TestManagerCheckpointExistsCleansName proves the existence check matches the
// cleaned directory name saveAdapter writes, so a raw save_weights path and the
// duplicate check agree on what "already exists" means.
func TestManagerCheckpointExistsCleansName(t *testing.T) {
	t.Setenv("LOCALTINKER_CHECKPOINT_ROOT", t.TempDir())
	model := &writingTrainModel{id: "m"}
	m := &Manager{models: map[string]trainModel{"m": model}}
	const raw = "my ckpt"
	if _, err := m.SaveState(context.Background(), "m", raw); err != nil {
		t.Fatal(err)
	}
	if !m.StateCheckpointExists("m", raw) {
		t.Fatalf("StateCheckpointExists(%q) = false after save", raw)
	}
}
