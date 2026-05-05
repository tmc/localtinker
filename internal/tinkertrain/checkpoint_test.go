package tinkertrain

import (
	"archive/tar"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestCheckpointArchive(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LOCALTINKER_CHECKPOINT_ROOT", root)

	dir := filepath.Join(root, "model_a", "weights", "ckpt")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"adapters.safetensors":  "weights",
		"adapter_config.json":   "{}\n",
		checkpointMetadataFile:  `{"format":"localtinker.checkpoint","version":1,"has_optimizer":true}` + "\n",
		checkpointOptimizerFile: `{"format":"localtinker.optimizer","version":1,"optimizer":"adamw","step":7}` + "\n",
		checkpointCompleteFile:  "ok\n",
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0644); err != nil {
			t.Fatal(err)
		}
	}

	archive, err := CheckpointArchive("tinker://model_a/weights/ckpt")
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Ext(archive) != ".tar" {
		t.Fatalf("archive = %q, want .tar", archive)
	}

	f, err := os.Open(archive)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tr := tar.NewReader(f)
	seen := make(map[string]bool)
	for {
		hdr, err := tr.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatal(err)
		}
		seen[hdr.Name] = true
	}
	for name := range files {
		if !seen[name] {
			t.Fatalf("archive missing %q; saw %#v", name, seen)
		}
	}
}

func TestOptimizerStateRoundTrip(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LOCALTINKER_CHECKPOINT_ROOT", root)

	dir := filepath.Join(root, "model_a", "weights", "ckpt")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	state := optimizerState{
		Format:    "localtinker.optimizer",
		Version:   1,
		Optimizer: "adamw",
		Step:      12,
		LastAdam: AdamParams{
			LearningRate: 1e-4,
			Beta1:        0.9,
			Beta2:        0.95,
			Eps:          1e-12,
		},
	}
	if err := writeJSONFile(filepath.Join(dir, checkpointOptimizerFile), state); err != nil {
		t.Fatal(err)
	}
	got, err := readOptimizerState(filepath.Join(dir, checkpointOptimizerFile))
	if err != nil {
		t.Fatal(err)
	}
	if got.Step != state.Step || got.LastAdam.LearningRate != state.LastAdam.LearningRate {
		t.Fatalf("optimizer state = %#v, want %#v", got, state)
	}
}

func TestManagerLoadStateWithOptimizer(t *testing.T) {
	model := &fakeTrainModel{}
	m := &Manager{models: map[string]trainModel{"m": model}}
	if err := m.LoadStateWithOptimizer(context.Background(), "m", "tinker://m/weights/ckpt", true); err != nil {
		t.Fatal(err)
	}
	if !model.loadedOptimizer {
		t.Fatal("optimizer state was not requested")
	}
}

func TestCheckpointMetadataJSON(t *testing.T) {
	meta := checkpointMetadata{
		Format:       "localtinker.checkpoint",
		Version:      1,
		ModelID:      "model-a",
		BaseModel:    "Qwen/Qwen3-8B",
		Kind:         "weights",
		Name:         "ckpt",
		IsLoRA:       true,
		LoRARank:     8,
		TrainMLP:     true,
		TrainAttn:    true,
		HasOptimizer: true,
		Step:         3,
	}
	data, err := json.Marshal(meta)
	if err != nil {
		t.Fatal(err)
	}
	var got checkpointMetadata
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if !got.HasOptimizer || got.Step != 3 || got.Kind != "weights" {
		t.Fatalf("metadata = %#v", got)
	}
}

type fakeTrainModel struct {
	loadedOptimizer bool
}

func (*fakeTrainModel) forwardBackward(context.Context, ForwardBackwardInput, bool) (ForwardBackwardOutput, error) {
	return ForwardBackwardOutput{}, nil
}

func (*fakeTrainModel) optimStep(context.Context, AdamParams) (OptimStepOutput, error) {
	return OptimStepOutput{}, nil
}

func (*fakeTrainModel) saveState(context.Context, string) (string, error) {
	return "", nil
}

func (m *fakeTrainModel) loadState(_ context.Context, _ string, optimizer bool) error {
	m.loadedOptimizer = optimizer
	return nil
}

func (*fakeTrainModel) saveForSampler(context.Context, string) (string, error) {
	return "", nil
}

func (*fakeTrainModel) sample(context.Context, SampleRequest) (SampleOutput, error) {
	return SampleOutput{}, nil
}
