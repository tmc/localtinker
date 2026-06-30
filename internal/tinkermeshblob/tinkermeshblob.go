// Package tinkermeshblob replicates localtinker checkpoints node-to-node over
// iroh blobs, with a hub fallback. localtinker artifacts are already manifests
// of content-addressed chunks; this bridges them to mlx-go-iroh's BAO-verified
// blob transport and pull-with-hub-fallback, so a late-joining node fetches a
// checkpoint from peers — verified — and falls back to a central hub (the
// existing artifact HTTP endpoint, or any [manifest.HubRepo]) if no peer serves
// a chunk.
//
// Each artifact chunk crosses two content addresses: localtinker keys chunks by
// SHA-256 (the chunk's name on the mesh), while iroh blobs verify by BLAKE3 (the
// transport hash). A pulled chunk is BLAKE3-verified by the blob layer on
// arrival and SHA-256-verified again by the artifact store on install, so a
// corrupt chunk is caught either way.
package tinkermeshblob

import (
	"context"
	"fmt"
	"io"

	"github.com/tmc/localtinker/internal/tinkerartifact"
	iblob "github.com/tmc/mlx-go-iroh/blob"
	"github.com/tmc/mlx-go-iroh/manifest"
)

// chunkVersion is the manifest version for a checkpoint's chunk set. Chunks are
// immutable by content hash, so a single version suffices.
const chunkVersion = 1

// BuildManifest builds an iroh blob manifest from a localtinker artifact: one
// blob per chunk, named by the chunk's SHA-256 and addressed by its BLAKE3 hash.
// It reads each chunk from store to compute the BLAKE3 hash, so the returned
// manifest is dialable by a peer that has the chunk bytes.
func BuildManifest(store *tinkerartifact.Store, rootHash string) (manifest.Manifest, error) {
	hashes, err := store.ChunkHashes(rootHash)
	if err != nil {
		return manifest.Manifest{}, fmt.Errorf("build manifest: chunk hashes: %w", err)
	}
	entries := make([]manifest.Entry, 0, len(hashes))
	for _, sha := range hashes {
		data, err := readChunk(store, rootHash, sha)
		if err != nil {
			return manifest.Manifest{}, err
		}
		blob, err := manifest.NewBlob(sha, chunkVersion, data)
		if err != nil {
			return manifest.Manifest{}, fmt.Errorf("build manifest: blob %s: %w", sha, err)
		}
		entries = append(entries, manifest.Entry{Blob: blob})
	}
	return manifest.Manifest{Version: chunkVersion, Entries: entries}, nil
}

// NewBlobStore builds an in-memory blob store serving every chunk of a
// checkpoint by its BLAKE3 hash, so a node can serve the checkpoint to peers
// over the blobs ALPN. It reads the chunks from store once.
func NewBlobStore(store *tinkerartifact.Store, rootHash string) (*iblob.MemoryStore, error) {
	hashes, err := store.ChunkHashes(rootHash)
	if err != nil {
		return nil, fmt.Errorf("blob store: chunk hashes: %w", err)
	}
	bs := iblob.NewMemoryStore()
	for _, sha := range hashes {
		data, err := readChunk(store, rootHash, sha)
		if err != nil {
			return nil, err
		}
		bs.Put(data)
	}
	return bs, nil
}

// Install writes every loaded chunk from a completed [manifest.Sync] into the
// artifact store under its SHA-256 name (re-verified by the store), then
// reconstructs the artifact from those chunks. m is the localtinker manifest
// being installed; sync holds the pulled chunk bytes keyed by SHA-256 name.
func Install(ctx context.Context, store *tinkerartifact.Store, m tinkerartifact.Manifest, sync *manifest.Sync) error {
	if !sync.AllLoaded() {
		return fmt.Errorf("install: sync not complete")
	}
	for _, sha := range chunkNames(m) {
		data, ok := sync.Blob(sha)
		if !ok {
			return fmt.Errorf("install: missing chunk %s", sha)
		}
		if err := store.AddChunk(sha, data); err != nil {
			return fmt.Errorf("install: add chunk %s: %w", sha, err)
		}
	}
	if err := store.InstallFromChunks(ctx, m); err != nil {
		return fmt.Errorf("install: reconstruct: %w", err)
	}
	return nil
}

// chunkNames returns the distinct SHA-256 chunk names of m in a stable order.
func chunkNames(m tinkerartifact.Manifest) []string {
	seen := make(map[string]bool)
	var out []string
	for _, f := range m.Files {
		for _, c := range f.Chunks {
			if !seen[c.SHA256] {
				seen[c.SHA256] = true
				out = append(out, c.SHA256)
			}
		}
	}
	return out
}

func readChunk(store *tinkerartifact.Store, rootHash, sha string) ([]byte, error) {
	f, _, err := store.OpenChunk(rootHash, sha)
	if err != nil {
		return nil, fmt.Errorf("read chunk %s: %w", sha, err)
	}
	defer f.Close()
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read chunk %s: %w", sha, err)
	}
	return data, nil
}
