package tinkerartifact

import (
	"path/filepath"
	"testing"
)

func TestHFCacheRoot(t *testing.T) {
	t.Setenv("HUGGINGFACE_HUB_CACHE", "/tmp/hf-hub")
	t.Setenv("HF_HOME", "/tmp/hf-home")
	got, err := HFCacheRoot()
	if err != nil {
		t.Fatal(err)
	}
	if got != "/tmp/hf-hub" {
		t.Fatalf("HFCacheRoot() = %q, want /tmp/hf-hub", got)
	}

	t.Setenv("HUGGINGFACE_HUB_CACHE", "")
	got, err = HFCacheRoot()
	if err != nil {
		t.Fatal(err)
	}
	if got != filepath.Join("/tmp/hf-home", "hub") {
		t.Fatalf("HFCacheRoot() = %q, want HF_HOME hub", got)
	}
}

func TestHFRepoFolder(t *testing.T) {
	tests := []struct {
		name     string
		repoType string
		repoID   string
		want     string
	}{
		{"default model", "", "Qwen/Qwen3-8B", "models--Qwen--Qwen3-8B"},
		{"model", "model", "mlx-community/Llama-3", "models--mlx-community--Llama-3"},
		{"dataset", "dataset", "org/data", "datasets--org--data"},
		{"space", "space", "org/app", "spaces--org--app"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := HFRepoFolder(tt.repoType, tt.repoID)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("HFRepoFolder(%q, %q) = %q, want %q", tt.repoType, tt.repoID, got, tt.want)
			}
		})
	}
}

func TestHFRepoFolderRejectsTraversal(t *testing.T) {
	for _, repoID := range []string{"", "../x", "x/../y", "x//y", `x\y`} {
		if _, err := HFRepoFolder("model", repoID); err == nil {
			t.Fatalf("HFRepoFolder accepted %q", repoID)
		}
	}
}
