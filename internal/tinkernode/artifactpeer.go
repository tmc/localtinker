package tinkernode

import (
	"context"
	"errors"
	"io"
	"net/http"

	"connectrpc.com/connect"

	"github.com/tmc/localtinker/internal/tinkerartifact"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1/tinkerv1connect"
)

const chunkFrameSize = 1 << 20

// ArtifactPeer serves local artifact chunks to trusted mesh peers.
type ArtifactPeer struct {
	store *tinkerartifact.Store
}

// NewArtifactPeer returns a Connect ArtifactPeer service backed by store.
func NewArtifactPeer(store *tinkerartifact.Store) *ArtifactPeer {
	return &ArtifactPeer{store: store}
}

// RegisterArtifactPeer registers the ArtifactPeer service on mux.
func RegisterArtifactPeer(mux *http.ServeMux, store *tinkerartifact.Store, opts ...connect.HandlerOption) {
	path, h := tinkerv1connect.NewArtifactPeerHandler(NewArtifactPeer(store), opts...)
	mux.Handle(path, h)
}

func (p *ArtifactPeer) Have(_ context.Context, req *connect.Request[tinkerv1.HaveRequest]) (*connect.Response[tinkerv1.HaveResponse], error) {
	hashes, err := p.store.ChunkHashes(req.Msg.GetRootHash())
	if errors.Is(err, tinkerartifact.ErrNotFound) {
		return nil, connect.NewError(connect.CodeNotFound, err)
	}
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	return connect.NewResponse(&tinkerv1.HaveResponse{ChunkHashes: hashes}), nil
}

func (p *ArtifactPeer) FetchChunk(_ context.Context, req *connect.Request[tinkerv1.FetchChunkRequest], stream *connect.ServerStream[tinkerv1.ChunkFrame]) error {
	root := req.Msg.GetRootHash()
	hash := req.Msg.GetChunkHash()
	file, size, err := p.store.OpenChunk(root, hash)
	if errors.Is(err, tinkerartifact.ErrNotFound) {
		return connect.NewError(connect.CodeNotFound, err)
	}
	if err != nil {
		return connect.NewError(connect.CodeInternal, err)
	}
	defer file.Close()

	buf := make([]byte, chunkFrameSize)
	var off int64
	for {
		n, err := file.Read(buf)
		if n > 0 {
			if sendErr := stream.Send(&tinkerv1.ChunkFrame{
				RootHash:  root,
				ChunkHash: hash,
				Offset:    off,
				TotalSize: size,
				Data:      append([]byte(nil), buf[:n]...),
			}); sendErr != nil {
				return sendErr
			}
			off += int64(n)
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return connect.NewError(connect.CodeInternal, err)
		}
	}
}
