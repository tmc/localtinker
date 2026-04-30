package tinkertrain

import (
	"archive/tar"
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
		"adapters.safetensors": "weights",
		"adapter_config.json":  "{}\n",
		checkpointCompleteFile: "ok\n",
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
