package tinkerartifact

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// InstallHFHubFromDirectory installs src into a Hugging Face cache snapshot and
// records the corresponding tinker manifest.
func (s *Store) InstallHFHubFromDirectory(ctx context.Context, src string, opts ManifestOptions) (Manifest, error) {
	opts.Storage = StorageHFHub
	m, err := BuildManifest(ctx, src, opts)
	if err != nil {
		return Manifest{}, err
	}
	if err := s.installManifestAndChunks(ctx, src, m); err != nil {
		return Manifest{}, err
	}
	if err := s.installHFHubFromSource(ctx, src, m); err != nil {
		return Manifest{}, err
	}
	return m, nil
}

func (s *Store) installHFHubFromChunks(ctx context.Context, m Manifest) error {
	if err := s.validateHFManifest(m); err != nil {
		return err
	}
	root, err := HFCacheRoot()
	if err != nil {
		return err
	}
	paths, err := NewHFPaths(root, m.Metadata.RepoType, m.Metadata.RepoID)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(paths.Blobs, 0o755); err != nil {
		return err
	}
	for _, f := range m.Files {
		if err := ctx.Err(); err != nil {
			return err
		}
		tmp := filepath.Join(s.Root, "partial", m.RootHash+".hf", filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(tmp), 0o755); err != nil {
			return err
		}
		if err := s.reconstructFile(tmp, f); err != nil {
			return err
		}
		if err := verifyFile(tmp, f.SHA256); err != nil {
			return err
		}
		if err := installHFBlob(paths, blobID(f), tmp); err != nil {
			return err
		}
	}
	if err := writeHFSnapshot(paths, m); err != nil {
		return err
	}
	if err := s.writeManifestOnly(m); err != nil {
		return err
	}
	_ = os.RemoveAll(filepath.Join(s.Root, "partial", m.RootHash+".hf"))
	return nil
}

func (s *Store) installHFHubFromSource(ctx context.Context, src string, m Manifest) error {
	if err := s.validateHFManifest(m); err != nil {
		return err
	}
	root, err := HFCacheRoot()
	if err != nil {
		return err
	}
	paths, err := NewHFPaths(root, m.Metadata.RepoType, m.Metadata.RepoID)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(paths.Blobs, 0o755); err != nil {
		return err
	}
	for _, f := range m.Files {
		if err := ctx.Err(); err != nil {
			return err
		}
		srcPath := filepath.Join(src, filepath.FromSlash(f.Path))
		if err := verifyFile(srcPath, f.SHA256); err != nil {
			return err
		}
		if err := installHFBlob(paths, blobID(f), srcPath); err != nil {
			return err
		}
	}
	return writeHFSnapshot(paths, m)
}

func (s *Store) validateHFManifest(m Manifest) error {
	if m.Storage != StorageHFHub {
		return fmt.Errorf("manifest storage %q is not hf_hub", m.Storage)
	}
	if m.Metadata.RepoID == "" {
		return fmt.Errorf("hf_hub manifest missing repo id")
	}
	if m.Metadata.CommitHash == "" {
		return fmt.Errorf("hf_hub manifest missing commit hash")
	}
	return verifyManifestIdentity(m)
}

func installHFBlob(paths HFPaths, blobID, src string) error {
	lock, err := LockFile(filepath.Join(paths.Locks, blobID+".lock"))
	if err != nil {
		return err
	}
	defer lock.Unlock()

	dst := filepath.Join(paths.Blobs, blobID)
	if _, err := os.Stat(dst); err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	tmp := dst + ".tmp"
	if err := copyFile(tmp, src, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		if errors.Is(err, os.ErrExist) {
			return nil
		}
		return err
	}
	return nil
}

func writeHFSnapshot(paths HFPaths, m Manifest) error {
	snapshot := filepath.Join(paths.Snapshots, m.Metadata.CommitHash)
	if err := os.MkdirAll(snapshot, 0o755); err != nil {
		return err
	}
	for _, f := range m.Files {
		link := filepath.Join(snapshot, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
			return err
		}
		_ = os.Remove(link)
		target := filepath.ToSlash(filepath.Join("..", "..", "blobs", blobID(f)))
		if err := os.Symlink(target, link); err != nil {
			blob := filepath.Join(paths.Blobs, blobID(f))
			if err := copyFile(link, blob, 0o644); err != nil {
				return err
			}
		}
	}
	if m.Metadata.Revision != "" {
		if err := os.MkdirAll(paths.Refs, 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(paths.Refs, m.Metadata.Revision), []byte(m.Metadata.CommitHash), 0o644); err != nil {
			return err
		}
	}
	return nil
}

func hfSnapshotComplete(m Manifest) error {
	if err := verifyManifestIdentity(m); err != nil {
		return err
	}
	root, err := HFCacheRoot()
	if err != nil {
		return err
	}
	paths, err := NewHFPaths(root, m.Metadata.RepoType, m.Metadata.RepoID)
	if err != nil {
		return err
	}
	if m.Metadata.CommitHash == "" {
		return fmt.Errorf("hf_hub manifest missing commit hash")
	}
	for _, f := range m.Files {
		path := filepath.Join(paths.Snapshots, m.Metadata.CommitHash, filepath.FromSlash(f.Path))
		if err := verifyFile(path, f.SHA256); err != nil {
			return err
		}
	}
	return nil
}

func blobID(f ManifestFile) string {
	if f.BlobID != "" {
		return f.BlobID
	}
	return f.SHA256
}
