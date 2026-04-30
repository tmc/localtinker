package tinkerartifact

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

var ErrNotFound = errors.New("artifact not found")

// ManifestOptions configures manifest construction for a local directory.
type ManifestOptions struct {
	ID        string
	Kind      ArtifactKind
	Storage   StorageKind
	Name      string
	Version   string
	Created   time.Time
	ChunkSize int64
	Metadata  Metadata
}

// Store keeps immutable artifacts and content-addressed chunks below Root.
type Store struct {
	Root string
}

// OpenStore opens or creates an artifact store below root.
func OpenStore(root string) (*Store, error) {
	if root == "" {
		return nil, fmt.Errorf("open artifact store: empty root")
	}
	s := &Store{Root: filepath.Clean(root)}
	for _, dir := range []string{s.artifactsDir(), s.chunksDir(), filepath.Join(s.Root, "partial")} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("open artifact store: %w", err)
		}
	}
	return s, nil
}

// AddDirectory imports src as an immutable artifact and returns its manifest.
func (s *Store) AddDirectory(ctx context.Context, src string, opts ManifestOptions) (Manifest, error) {
	manifest, err := BuildManifest(ctx, src, opts)
	if err != nil {
		return Manifest{}, err
	}
	if manifest.Storage == StorageHFHub {
		if err := s.installManifestAndChunks(ctx, src, manifest); err != nil {
			return Manifest{}, err
		}
		return manifest, nil
	}
	if err := s.installFiles(ctx, src, manifest); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}

// BuildManifest constructs a content-addressed manifest for src.
func BuildManifest(ctx context.Context, src string, opts ManifestOptions) (Manifest, error) {
	src = filepath.Clean(src)
	st, err := os.Stat(src)
	if err != nil {
		return Manifest{}, fmt.Errorf("build manifest: %w", err)
	}
	if !st.IsDir() {
		return Manifest{}, fmt.Errorf("build manifest: %s is not a directory", src)
	}
	chunkSize := opts.ChunkSize
	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}
	created := opts.Created
	if created.IsZero() {
		created = time.Now().UTC()
	}
	storage := opts.Storage
	if storage == "" {
		storage = StorageTinker
	}

	var files []ManifestFile
	err = filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		file, err := manifestFile(path, filepath.ToSlash(rel), info, chunkSize)
		if err != nil {
			return err
		}
		files = append(files, file)
		return nil
	})
	if err != nil {
		return Manifest{}, fmt.Errorf("build manifest: %w", err)
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })

	var size int64
	for _, f := range files {
		size += f.Size
	}
	m := Manifest{
		ID:        opts.ID,
		Kind:      opts.Kind,
		Storage:   storage,
		Name:      opts.Name,
		Version:   opts.Version,
		Created:   created,
		Size:      size,
		ChunkSize: chunkSize,
		Files:     files,
		Metadata:  opts.Metadata,
	}
	if m.Storage == StorageHFHub {
		for i := range m.Files {
			if m.Files[i].BlobID == "" {
				m.Files[i].BlobID = m.Files[i].SHA256
			}
		}
	}
	root, err := RootHash(m)
	if err != nil {
		return Manifest{}, err
	}
	m.RootHash = root
	return m, nil
}

// RootHash returns the canonical SHA-256 identity for m.
func RootHash(m Manifest) (string, error) {
	m.RootHash = ""
	m.Created = time.Time{}
	m.Size = 0
	data, err := json.Marshal(m)
	if err != nil {
		return "", fmt.Errorf("root hash: %w", err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// Manifest loads an installed manifest by root hash.
func (s *Store) Manifest(rootHash string) (Manifest, error) {
	var m Manifest
	data, err := os.ReadFile(s.manifestPath(rootHash))
	if errors.Is(err, os.ErrNotExist) {
		return Manifest{}, ErrNotFound
	}
	if err != nil {
		return Manifest{}, err
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return Manifest{}, err
	}
	return m, nil
}

// Has reports whether rootHash is fully installed.
func (s *Store) Has(rootHash string) bool {
	m, err := s.Manifest(rootHash)
	if err != nil {
		return false
	}
	if m.Storage == StorageHFHub {
		return hfSnapshotComplete(m) == nil
	}
	_, err = os.Stat(filepath.Join(s.artifactPath(rootHash), "files"))
	return err == nil
}

// Delete removes the installed artifact for rootHash.
//
// Delete leaves content-addressed chunks in place because chunks may be shared
// by other artifacts.
func (s *Store) Delete(rootHash string) error {
	rootHash = cleanHash(rootHash)
	if rootHash == "" {
		return fmt.Errorf("delete artifact: empty root hash")
	}
	path := s.artifactPath(rootHash)
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return ErrNotFound
	} else if err != nil {
		return err
	}
	return os.RemoveAll(path)
}

// Inventory returns installed artifacts.
func (s *Store) Inventory() ([]Ref, error) {
	roots, err := os.ReadDir(s.artifactsDir())
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []Ref
	for _, ent := range roots {
		if !ent.IsDir() {
			continue
		}
		m, err := s.Manifest(ent.Name())
		if err != nil {
			continue
		}
		if !s.Has(m.RootHash) {
			continue
		}
		out = append(out, Ref{
			ID:       m.ID,
			Kind:     m.Kind,
			Storage:  m.Storage,
			Name:     m.Name,
			Version:  m.Version,
			RootHash: m.RootHash,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].RootHash < out[j].RootHash })
	return out, nil
}

// ChunkHashes lists chunk hashes present for rootHash.
func (s *Store) ChunkHashes(rootHash string) ([]string, error) {
	m, err := s.Manifest(rootHash)
	if err != nil {
		return nil, err
	}
	seen := make(map[string]bool)
	for _, f := range m.Files {
		for _, c := range f.Chunks {
			if _, err := os.Stat(s.chunkPath(c.SHA256)); err == nil {
				seen[c.SHA256] = true
			}
		}
	}
	out := make([]string, 0, len(seen))
	for h := range seen {
		out = append(out, h)
	}
	sort.Strings(out)
	return out, nil
}

// OpenChunk opens a verified chunk for reading.
func (s *Store) OpenChunk(rootHash, chunkHash string) (*os.File, int64, error) {
	m, err := s.Manifest(rootHash)
	if err != nil {
		return nil, 0, err
	}
	var size int64 = -1
	for _, f := range m.Files {
		for _, c := range f.Chunks {
			if c.SHA256 == chunkHash {
				size = c.Size
				break
			}
		}
	}
	if size < 0 {
		return nil, 0, ErrNotFound
	}
	file, err := os.Open(s.chunkPath(chunkHash))
	if errors.Is(err, os.ErrNotExist) {
		return nil, 0, ErrNotFound
	}
	return file, size, err
}

// AddChunk verifies and stores chunk data in the content-addressed chunk store.
func (s *Store) AddChunk(hash string, data []byte) error {
	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != hash {
		return fmt.Errorf("chunk hash mismatch: got %s want %s", got, hash)
	}
	return s.writeChunk(hash, data)
}

// InstallFromChunks reconstructs and verifies an artifact from stored chunks.
func (s *Store) InstallFromChunks(ctx context.Context, m Manifest) error {
	if err := verifyManifestIdentity(m); err != nil {
		return err
	}
	if s.Has(m.RootHash) {
		return nil
	}
	if m.Storage == StorageHFHub {
		return s.installHFHubFromChunks(ctx, m)
	}
	tmp := filepath.Join(s.Root, "partial", m.RootHash+".tmp")
	_ = os.RemoveAll(tmp)
	if err := os.MkdirAll(filepath.Join(tmp, "files"), 0o755); err != nil {
		return err
	}
	for _, f := range m.Files {
		if err := ctx.Err(); err != nil {
			return err
		}
		dst := filepath.Join(tmp, "files", filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if err := s.reconstructFile(dst, f); err != nil {
			return err
		}
		if err := verifyFile(dst, f.SHA256); err != nil {
			return err
		}
		mode := fs.FileMode(f.Mode)
		if mode == 0 {
			mode = 0o644
		}
		if err := os.Chmod(dst, mode.Perm()); err != nil {
			return err
		}
	}
	data, err := json.MarshalIndent(m, "", "\t")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(tmp, "manifest.json"), data, 0o644); err != nil {
		return err
	}
	final := s.artifactPath(m.RootHash)
	if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
		return err
	}
	if err := os.Rename(tmp, final); err != nil {
		if errors.Is(err, os.ErrExist) {
			_ = os.RemoveAll(tmp)
			return nil
		}
		return err
	}
	return nil
}

func (s *Store) installFiles(ctx context.Context, src string, m Manifest) error {
	if err := verifyManifestIdentity(m); err != nil {
		return err
	}
	if s.Has(m.RootHash) {
		return nil
	}
	tmp := filepath.Join(s.Root, "partial", m.RootHash+".tmp")
	_ = os.RemoveAll(tmp)
	if err := os.MkdirAll(filepath.Join(tmp, "files"), 0o755); err != nil {
		return err
	}
	for _, f := range m.Files {
		if err := ctx.Err(); err != nil {
			return err
		}
		from := filepath.Join(src, filepath.FromSlash(f.Path))
		to := filepath.Join(tmp, "files", filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(to), 0o755); err != nil {
			return err
		}
		if err := copyFile(to, from, fs.FileMode(f.Mode).Perm()); err != nil {
			return err
		}
		if err := s.storeFileChunks(from, f.Chunks); err != nil {
			return err
		}
	}
	data, err := json.MarshalIndent(m, "", "\t")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(tmp, "manifest.json"), data, 0o644); err != nil {
		return err
	}
	final := s.artifactPath(m.RootHash)
	if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
		return err
	}
	return os.Rename(tmp, final)
}

func (s *Store) installManifestAndChunks(ctx context.Context, src string, m Manifest) error {
	if err := verifyManifestIdentity(m); err != nil {
		return err
	}
	if s.Has(m.RootHash) {
		return nil
	}
	for _, f := range m.Files {
		if err := ctx.Err(); err != nil {
			return err
		}
		from := filepath.Join(src, filepath.FromSlash(f.Path))
		if err := s.storeFileChunks(from, f.Chunks); err != nil {
			return err
		}
	}
	return s.writeManifestOnly(m)
}

func (s *Store) writeManifestOnly(m Manifest) error {
	if err := verifyManifestIdentity(m); err != nil {
		return err
	}
	tmp := filepath.Join(s.Root, "partial", m.RootHash+".manifest.tmp")
	_ = os.RemoveAll(tmp)
	if err := os.MkdirAll(tmp, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(m, "", "\t")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(tmp, "manifest.json"), data, 0o644); err != nil {
		return err
	}
	final := s.artifactPath(m.RootHash)
	if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
		return err
	}
	if err := os.Rename(tmp, final); err != nil {
		if errors.Is(err, os.ErrExist) {
			_ = os.RemoveAll(tmp)
			return nil
		}
		return err
	}
	return nil
}

func manifestFile(path, rel string, info fs.FileInfo, chunkSize int64) (ManifestFile, error) {
	if chunkSize <= 0 || chunkSize > int64(int(^uint(0)>>1)) {
		return ManifestFile{}, fmt.Errorf("invalid chunk size %d", chunkSize)
	}
	file, err := os.Open(path)
	if err != nil {
		return ManifestFile{}, err
	}
	defer file.Close()

	fileHash := sha256.New()
	buf := make([]byte, int(chunkSize))
	var chunks []ChunkRef
	var off int64
	for index := int64(0); ; index++ {
		n, err := io.ReadFull(file, buf)
		if errors.Is(err, io.EOF) {
			break
		}
		if errors.Is(err, io.ErrUnexpectedEOF) {
			err = nil
		}
		if err != nil {
			return ManifestFile{}, err
		}
		part := buf[:n]
		_, _ = fileHash.Write(part)
		sum := sha256.Sum256(part)
		chunks = append(chunks, ChunkRef{
			Index:  index,
			Offset: off,
			Size:   int64(n),
			SHA256: hex.EncodeToString(sum[:]),
		})
		off += int64(n)
		if n < len(buf) {
			break
		}
	}
	return ManifestFile{
		Path:   rel,
		Size:   info.Size(),
		Mode:   uint32(info.Mode().Perm()),
		SHA256: hex.EncodeToString(fileHash.Sum(nil)),
		Chunks: chunks,
	}, nil
}

func (s *Store) storeFileChunks(path string, chunks []ChunkRef) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	for _, c := range chunks {
		data := make([]byte, c.Size)
		if _, err := file.ReadAt(data, c.Offset); err != nil {
			return err
		}
		if err := s.writeChunk(c.SHA256, data); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) writeChunk(hash string, data []byte) error {
	path := s.chunkPath(hash)
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		if errors.Is(err, os.ErrExist) {
			return nil
		}
		return err
	}
	return nil
}

func (s *Store) reconstructFile(dst string, f ManifestFile) error {
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer out.Close()
	for _, c := range f.Chunks {
		in, err := os.Open(s.chunkPath(c.SHA256))
		if err != nil {
			return err
		}
		_, copyErr := io.Copy(out, in)
		closeErr := in.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}

func verifyManifestIdentity(m Manifest) error {
	if m.RootHash == "" {
		return fmt.Errorf("manifest root hash is empty")
	}
	got, err := RootHash(m)
	if err != nil {
		return err
	}
	if got != m.RootHash {
		return fmt.Errorf("manifest root hash mismatch: got %s want %s", got, m.RootHash)
	}
	return nil
}

func verifyFile(path, want string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	h := sha256.New()
	if _, err := io.Copy(h, file); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("file hash mismatch for %s: got %s want %s", path, got, want)
	}
	return nil
}

func copyFile(dst, src string, mode fs.FileMode) error {
	if mode == 0 {
		mode = 0o644
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, in)
	closeErr := out.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func (s *Store) artifactsDir() string {
	return filepath.Join(s.Root, "artifacts", "sha256")
}

func (s *Store) artifactPath(rootHash string) string {
	return filepath.Join(s.artifactsDir(), cleanHash(rootHash))
}

func (s *Store) manifestPath(rootHash string) string {
	return filepath.Join(s.artifactPath(rootHash), "manifest.json")
}

func (s *Store) chunksDir() string {
	return filepath.Join(s.Root, "chunks", "sha256")
}

func (s *Store) chunkPath(hash string) string {
	hash = cleanHash(hash)
	prefix := hash
	if len(prefix) > 2 {
		prefix = prefix[:2]
	}
	return filepath.Join(s.chunksDir(), prefix, hash)
}

func cleanHash(hash string) string {
	return strings.TrimSpace(strings.TrimPrefix(hash, "sha256:"))
}
