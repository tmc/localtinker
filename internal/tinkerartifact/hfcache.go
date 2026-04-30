package tinkerartifact

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// HFCacheRoot returns the Hugging Face hub cache root.
//
// Resolution follows huggingface_hub: HUGGINGFACE_HUB_CACHE, then
// HF_HOME/hub, then ~/.cache/huggingface/hub.
func HFCacheRoot() (string, error) {
	if dir := os.Getenv("HUGGINGFACE_HUB_CACHE"); dir != "" {
		return filepath.Clean(dir), nil
	}
	if dir := os.Getenv("HF_HOME"); dir != "" {
		return filepath.Join(filepath.Clean(dir), "hub"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home directory: %w", err)
	}
	return filepath.Join(home, ".cache", "huggingface", "hub"), nil
}

// HFRepoFolder returns the cache folder name used by huggingface_hub.
func HFRepoFolder(repoType, repoID string) (string, error) {
	repoType = strings.TrimSpace(repoType)
	if repoType == "" {
		repoType = "model"
	}
	prefix, ok := hfRepoPrefixes[repoType]
	if !ok {
		return "", fmt.Errorf("unsupported repo type %q", repoType)
	}
	if err := validHFRepoID(repoID); err != nil {
		return "", err
	}
	return prefix + "--" + strings.ReplaceAll(repoID, "/", "--"), nil
}

// HFPaths contains derived paths for a Hugging Face cache repository.
type HFPaths struct {
	Root       string
	RepoID     string
	RepoType   string
	RepoFolder string
	RepoRoot   string
	Blobs      string
	Refs       string
	Snapshots  string
	Locks      string
}

// NewHFPaths derives cache paths for repoID below root.
func NewHFPaths(root, repoType, repoID string) (HFPaths, error) {
	folder, err := HFRepoFolder(repoType, repoID)
	if err != nil {
		return HFPaths{}, err
	}
	repoRoot := filepath.Join(root, folder)
	return HFPaths{
		Root:       root,
		RepoID:     repoID,
		RepoType:   repoType,
		RepoFolder: folder,
		RepoRoot:   repoRoot,
		Blobs:      filepath.Join(repoRoot, "blobs"),
		Refs:       filepath.Join(repoRoot, "refs"),
		Snapshots:  filepath.Join(repoRoot, "snapshots"),
		Locks:      filepath.Join(root, ".locks", folder),
	}, nil
}

func validHFRepoID(repoID string) error {
	if repoID == "" {
		return fmt.Errorf("empty repo id")
	}
	if strings.Contains(repoID, "\\") {
		return fmt.Errorf("invalid repo id %q", repoID)
	}
	for _, part := range strings.Split(repoID, "/") {
		if part == "" || part == "." || part == ".." {
			return fmt.Errorf("invalid repo id %q", repoID)
		}
	}
	return nil
}

var hfRepoPrefixes = map[string]string{
	"model":   "models",
	"models":  "models",
	"dataset": "datasets",
	"space":   "spaces",
}
