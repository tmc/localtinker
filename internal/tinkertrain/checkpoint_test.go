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
	seen := readArchive(t, f)
	for name := range files {
		if seen[name] != files[name] {
			t.Fatalf("archive missing %q; saw %#v", name, seen)
		}
	}
}

func TestSamplerCheckpointArchiveOmitsOptimizerState(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LOCALTINKER_CHECKPOINT_ROOT", root)

	dir := filepath.Join(root, "model_a", "sampler_weights", "ckpt")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"adapters.safetensors": "weights",
		"adapter_config.json":  "{}\n",
		checkpointMetadataFile: `{"format":"localtinker.checkpoint","version":1,"has_optimizer":false}` + "\n",
		checkpointCompleteFile: "ok\n",
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0644); err != nil {
			t.Fatal(err)
		}
	}

	archive, err := CheckpointArchive("tinker://model_a/sampler_weights/ckpt")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(archive)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	seen := readArchive(t, f)
	for name := range files {
		if seen[name] != files[name] {
			t.Fatalf("archive missing %q; saw %#v", name, seen)
		}
	}
	if _, ok := seen[checkpointOptimizerFile]; ok {
		t.Fatalf("sampler archive unexpectedly included %q", checkpointOptimizerFile)
	}
}

func TestOptimizerStateRoundTrip(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LOCALTINKER_CHECKPOINT_ROOT", root)

	dir := filepath.Join(root, "model_a", "weights", "ckpt")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	// localtinker currently persists scalar resume metadata for the optimizer.
	// It does not claim to serialize MLX Adam moment tensors.
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

func TestOptimizerStateRejectsInvalidState(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, checkpointOptimizerFile)
	if _, err := readOptimizerState(path); err == nil {
		t.Fatal("read missing optimizer state succeeded")
	}
	if err := os.WriteFile(path, []byte(`{"format":"other","version":1}`+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := readOptimizerState(path); err == nil {
		t.Fatal("read invalid optimizer state succeeded")
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
	if err := m.LoadState(context.Background(), "m", "tinker://m/weights/ckpt"); err != nil {
		t.Fatal(err)
	}
	if model.loadedOptimizer {
		t.Fatal("plain LoadState requested optimizer state")
	}
	model.loadErr = os.ErrNotExist
	if err := m.LoadStateWithOptimizer(context.Background(), "m", "tinker://m/weights/ckpt", true); err == nil {
		t.Fatal("LoadStateWithOptimizer swallowed load error")
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
	loadErr         error
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
	return m.loadErr
}

func (*fakeTrainModel) saveForSampler(context.Context, string) (string, error) {
	return "", nil
}

func (*fakeTrainModel) sample(context.Context, SampleRequest) (SampleOutput, error) {
	return SampleOutput{}, nil
}

func readArchive(t *testing.T, f *os.File) map[string]string {
	t.Helper()

	tr := tar.NewReader(f)
	seen := make(map[string]string)
	for {
		hdr, err := tr.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatal(err)
		}
		if hdr.Typeflag != tar.TypeReg {
			t.Fatalf("archive entry %q type = %d, want regular file", hdr.Name, hdr.Typeflag)
		}
		if filepath.Clean(hdr.Name) != hdr.Name || filepath.IsAbs(hdr.Name) || hdr.Name == "." || hdr.Name == ".." {
			t.Fatalf("archive entry has unsafe path %q", hdr.Name)
		}
		data, err := io.ReadAll(tr)
		if err != nil {
			t.Fatal(err)
		}
		seen[hdr.Name] = string(data)
	}
	return seen
}
