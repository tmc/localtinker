package tinkernode

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"

	"connectrpc.com/connect"

	"github.com/tmc/localtinker/internal/tinkerartifact"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1/tinkerv1connect"
)

// SyncArtifact fetches missing chunks from peerURLs and installs manifest.
func SyncArtifact(ctx context.Context, store *tinkerartifact.Store, manifest tinkerartifact.Manifest, peerURLs []string) error {
	if store.Has(manifest.RootHash) {
		return nil
	}
	if len(peerURLs) == 0 {
		return fmt.Errorf("sync artifact %s: no peers", manifest.RootHash)
	}
	needed := chunkSet(manifest)
	for _, peerURL := range peerURLs {
		if len(needed) == 0 {
			break
		}
		client := tinkerv1connect.NewArtifactPeerClient(
			http.DefaultClient,
			peerURL,
			connect.WithReadMaxBytes(128<<20),
			connect.WithSendMaxBytes(128<<20),
		)
		have, err := client.Have(ctx, connect.NewRequest(&tinkerv1.HaveRequest{RootHash: manifest.RootHash}))
		if err != nil {
			continue
		}
		for _, hash := range have.Msg.GetChunkHashes() {
			if !needed[hash] {
				continue
			}
			data, err := fetchChunk(ctx, client, manifest.RootHash, hash)
			if err != nil {
				continue
			}
			if err := store.AddChunk(hash, data); err != nil {
				return err
			}
			delete(needed, hash)
		}
	}
	if len(needed) != 0 {
		return fmt.Errorf("sync artifact %s: missing %d chunks", manifest.RootHash, len(needed))
	}
	if err := store.InstallFromChunks(ctx, manifest); err != nil {
		return fmt.Errorf("sync artifact %s: install: %w", manifest.RootHash, err)
	}
	return nil
}

func chunkSet(m tinkerartifact.Manifest) map[string]bool {
	out := make(map[string]bool)
	for _, f := range m.Files {
		for _, c := range f.Chunks {
			out[c.SHA256] = true
		}
	}
	return out
}

func fetchChunk(ctx context.Context, client tinkerv1connect.ArtifactPeerClient, root, hash string) ([]byte, error) {
	stream, err := client.FetchChunk(ctx, connect.NewRequest(&tinkerv1.FetchChunkRequest{
		RootHash:  root,
		ChunkHash: hash,
	}))
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	var total int64 = -1
	var off int64
	for stream.Receive() {
		frame := stream.Msg()
		if frame.GetRootHash() != root || frame.GetChunkHash() != hash {
			return nil, fmt.Errorf("unexpected chunk frame")
		}
		if frame.GetOffset() != off {
			return nil, fmt.Errorf("chunk frame offset %d, want %d", frame.GetOffset(), off)
		}
		if total < 0 {
			total = frame.GetTotalSize()
		}
		buf.Write(frame.GetData())
		off += int64(len(frame.GetData()))
	}
	if err := stream.Err(); err != nil {
		return nil, err
	}
	if total >= 0 && off != total {
		return nil, errors.New("short chunk stream")
	}
	return buf.Bytes(), nil
}
