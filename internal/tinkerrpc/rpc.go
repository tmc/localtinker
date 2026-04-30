package tinkerrpc

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"connectrpc.com/connect"
	"google.golang.org/protobuf/proto"

	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1/tinkerv1connect"
)

const RPCMaxBytes = 128 << 20

type Server struct {
	coord         *tinkercoord.Coordinator
	coordinatorID string

	mu    sync.Mutex
	nodes map[string]*nodeState

	manifests map[string]*tinkerv1.Manifest
	aliases   map[string]string
}

type nodeState struct {
	req        *tinkerv1.RegisterNodeRequest
	load       *tinkerv1.NodeLoad
	labels     map[string]string
	state      string
	lastSeenAt time.Time
	artifacts  map[string]*tinkerv1.ArtifactInventory
}

type Snapshot struct {
	CoordinatorID string             `json:"coordinator_id"`
	Nodes         []NodeSnapshot     `json:"nodes"`
	Artifacts     []ArtifactSnapshot `json:"artifacts"`
}

type NodeSnapshot struct {
	NodeID     string             `json:"node_id"`
	Name       string             `json:"name"`
	State      string             `json:"state"`
	Load       *tinkerv1.NodeLoad `json:"load,omitempty"`
	Labels     map[string]string  `json:"labels,omitempty"`
	LastSeenAt time.Time          `json:"last_seen_at"`
	Artifacts  int                `json:"artifacts"`
}

type ArtifactSnapshot struct {
	RootHash string `json:"root_hash"`
	Kind     string `json:"kind"`
	Storage  string `json:"storage"`
	Alias    string `json:"alias,omitempty"`
}

func New(coord *tinkercoord.Coordinator) (*Server, error) {
	if coord == nil {
		return nil, errors.New("nil coordinator")
	}
	id, err := newID("coord")
	if err != nil {
		return nil, err
	}
	return &Server{
		coord:         coord,
		coordinatorID: id,
		nodes:         make(map[string]*nodeState),
		manifests:     make(map[string]*tinkerv1.Manifest),
		aliases:       make(map[string]string),
	}, nil
}

func (s *Server) Snapshot() Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := Snapshot{CoordinatorID: s.coordinatorID}
	for id, node := range s.nodes {
		name := id
		if node.req != nil && node.req.GetName() != "" {
			name = node.req.GetName()
		}
		out.Nodes = append(out.Nodes, NodeSnapshot{
			NodeID:     id,
			Name:       name,
			State:      node.state,
			Load:       cloneLoad(node.load),
			Labels:     cloneMap(node.labels),
			LastSeenAt: node.lastSeenAt,
			Artifacts:  len(node.artifacts),
		})
	}
	sort.Slice(out.Nodes, func(i, j int) bool {
		return out.Nodes[i].NodeID < out.Nodes[j].NodeID
	})
	roots := make([]string, 0, len(s.manifests))
	for root := range s.manifests {
		roots = append(roots, root)
	}
	sort.Strings(roots)
	for _, root := range roots {
		m := s.manifests[root]
		out.Artifacts = append(out.Artifacts, ArtifactSnapshot{
			RootHash: m.GetRootHash(),
			Kind:     m.GetKind(),
			Storage:  m.GetStorage(),
			Alias:    aliasForRoot(s.aliases, root),
		})
	}
	return out
}

func (s *Server) Register(mux *http.ServeMux) {
	opts := []connect.HandlerOption{
		connect.WithReadMaxBytes(RPCMaxBytes),
		connect.WithSendMaxBytes(RPCMaxBytes),
	}

	path, h := tinkerv1connect.NewTinkerCoordinatorHandler(s, opts...)
	mux.Handle(path, h)
	path, h = tinkerv1connect.NewTinkerAdminHandler(s, opts...)
	mux.Handle(path, h)
	path, h = tinkerv1connect.NewArtifactTrackerHandler(s, opts...)
	mux.Handle(path, h)
}

func (s *Server) RegisterNode(_ context.Context, req *connect.Request[tinkerv1.RegisterNodeRequest]) (*connect.Response[tinkerv1.RegisterNodeResponse], error) {
	msg := req.Msg
	nodeID := msg.GetNodeId()
	if nodeID == "" {
		var err error
		nodeID, err = newID("node")
		if err != nil {
			return nil, connect.NewError(connect.CodeInternal, err)
		}
	}

	s.mu.Lock()
	s.nodes[nodeID] = &nodeState{
		req:        msg,
		labels:     cloneMap(msg.GetLabels()),
		state:      "healthy",
		lastSeenAt: time.Now().UTC(),
		artifacts:  make(map[string]*tinkerv1.ArtifactInventory),
	}
	s.mu.Unlock()

	return connect.NewResponse(&tinkerv1.RegisterNodeResponse{
		CoordinatorId:       s.coordinatorID,
		AssignedNodeId:      nodeID,
		HeartbeatIntervalMs: int64((5 * time.Second) / time.Millisecond),
		LeaseTimeoutMs:      int64((30 * time.Second) / time.Millisecond),
		Config: map[string]string{
			"rpc_max_bytes": "134217728",
		},
	}), nil
}

func (s *Server) Heartbeat(_ context.Context, req *connect.Request[tinkerv1.HeartbeatRequest]) (*connect.Response[tinkerv1.HeartbeatResponse], error) {
	msg := req.Msg
	nodeID := msg.GetNodeId()
	if nodeID == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing node_id"))
	}

	s.mu.Lock()
	node := s.nodes[nodeID]
	if node == nil {
		node = &nodeState{state: "healthy", artifacts: make(map[string]*tinkerv1.ArtifactInventory)}
		s.nodes[nodeID] = node
	}
	node.load = msg.GetLoad()
	node.lastSeenAt = time.Now().UTC()
	if len(msg.GetArtifacts()) > 0 {
		node.artifacts = inventoryMap(msg.GetArtifacts())
	}
	s.mu.Unlock()

	return connect.NewResponse(&tinkerv1.HeartbeatResponse{
		CoordinatorId: s.coordinatorID,
	}), nil
}

func (s *Server) Watch(ctx context.Context, _ *connect.Request[tinkerv1.WatchRequest], _ *connect.ServerStream[tinkerv1.NodeCommand]) error {
	<-ctx.Done()
	return ctx.Err()
}

func (s *Server) Report(_ context.Context, stream *connect.ClientStream[tinkerv1.NodeEvent]) (*connect.Response[tinkerv1.ReportResponse], error) {
	for stream.Receive() {
	}
	if err := stream.Err(); err != nil {
		return nil, err
	}
	return connect.NewResponse(&tinkerv1.ReportResponse{}), nil
}

func (s *Server) ListNodes(context.Context, *connect.Request[tinkerv1.ListNodesRequest]) (*connect.Response[tinkerv1.ListNodesResponse], error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	resp := &tinkerv1.ListNodesResponse{}
	for id, node := range s.nodes {
		name := id
		if node.req != nil && node.req.GetName() != "" {
			name = node.req.GetName()
		}
		resp.Nodes = append(resp.Nodes, &tinkerv1.NodeSummary{
			NodeId: id,
			Name:   name,
			State:  node.state,
			Load:   node.load,
			Labels: cloneMap(node.labels),
		})
	}
	return connect.NewResponse(resp), nil
}

func (s *Server) DrainNode(context.Context, *connect.Request[tinkerv1.DrainNodeRequest]) (*connect.Response[tinkerv1.DrainNodeResponse], error) {
	return connect.NewResponse(&tinkerv1.DrainNodeResponse{}), nil
}

func (s *Server) ListRuns(context.Context, *connect.Request[tinkerv1.ListRunsRequest]) (*connect.Response[tinkerv1.ListRunsResponse], error) {
	return connect.NewResponse(&tinkerv1.ListRunsResponse{}), nil
}

func (s *Server) InspectRun(context.Context, *connect.Request[tinkerv1.InspectRunRequest]) (*connect.Response[tinkerv1.InspectRunResponse], error) {
	return connect.NewResponse(&tinkerv1.InspectRunResponse{}), nil
}

func (s *Server) ListArtifacts(context.Context, *connect.Request[tinkerv1.ListArtifactsRequest]) (*connect.Response[tinkerv1.ListArtifactsResponse], error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	resp := &tinkerv1.ListArtifactsResponse{}
	roots := make([]string, 0, len(s.manifests))
	for root := range s.manifests {
		roots = append(roots, root)
	}
	sort.Strings(roots)
	for _, root := range roots {
		m := s.manifests[root]
		resp.Artifacts = append(resp.Artifacts, &tinkerv1.ArtifactRef{
			RootHash: m.GetRootHash(),
			Kind:     m.GetKind(),
			Storage:  m.GetStorage(),
			Alias:    aliasForRoot(s.aliases, root),
		})
	}
	return connect.NewResponse(resp), nil
}

func (s *Server) PublishManifest(_ context.Context, req *connect.Request[tinkerv1.PublishManifestRequest]) (*connect.Response[tinkerv1.PublishManifestResponse], error) {
	manifest := req.Msg.GetManifest()
	if manifest == nil || manifest.GetRootHash() == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing manifest root_hash"))
	}
	root := manifest.GetRootHash()
	s.mu.Lock()
	s.manifests[root] = proto.Clone(manifest).(*tinkerv1.Manifest)
	if alias := strings.TrimSpace(req.Msg.GetAlias()); alias != "" {
		s.aliases[alias] = root
	}
	s.mu.Unlock()
	return connect.NewResponse(&tinkerv1.PublishManifestResponse{
		RootHash: root,
	}), nil
}

func (s *Server) GetManifest(_ context.Context, req *connect.Request[tinkerv1.GetManifestRequest]) (*connect.Response[tinkerv1.GetManifestResponse], error) {
	key := strings.TrimSpace(req.Msg.GetRootHashOrAlias())
	if key == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing root_hash_or_alias"))
	}
	s.mu.Lock()
	root := key
	if aliasRoot := s.aliases[key]; aliasRoot != "" {
		root = aliasRoot
	}
	manifest := s.manifests[root]
	s.mu.Unlock()
	if manifest == nil {
		return nil, connect.NewError(connect.CodeNotFound, errors.New("manifest not found"))
	}
	return connect.NewResponse(&tinkerv1.GetManifestResponse{
		Manifest: proto.Clone(manifest).(*tinkerv1.Manifest),
	}), nil
}

func (s *Server) DeleteAlias(_ context.Context, req *connect.Request[tinkerv1.DeleteAliasRequest]) (*connect.Response[tinkerv1.DeleteAliasResponse], error) {
	s.mu.Lock()
	delete(s.aliases, req.Msg.GetAlias())
	s.mu.Unlock()
	return connect.NewResponse(&tinkerv1.DeleteAliasResponse{}), nil
}

func (s *Server) ReportInventory(_ context.Context, req *connect.Request[tinkerv1.ReportInventoryRequest]) (*connect.Response[tinkerv1.ReportInventoryResponse], error) {
	nodeID := req.Msg.GetNodeId()
	if nodeID == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing node_id"))
	}
	s.mu.Lock()
	node := s.nodes[nodeID]
	if node == nil {
		node = &nodeState{state: "healthy", artifacts: make(map[string]*tinkerv1.ArtifactInventory)}
		s.nodes[nodeID] = node
	}
	node.artifacts = inventoryMap(req.Msg.GetArtifacts())
	node.lastSeenAt = time.Now().UTC()
	s.mu.Unlock()
	return connect.NewResponse(&tinkerv1.ReportInventoryResponse{}), nil
}

func (s *Server) ApplyRetention(context.Context, *connect.Request[tinkerv1.ApplyRetentionRequest]) (*connect.Response[tinkerv1.ApplyRetentionResponse], error) {
	return connect.NewResponse(&tinkerv1.ApplyRetentionResponse{}), nil
}

func (s *Server) ListPeers(_ context.Context, req *connect.Request[tinkerv1.ListPeersRequest]) (*connect.Response[tinkerv1.ListPeersResponse], error) {
	root := req.Msg.GetRootHash()
	if root == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing root_hash"))
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	var peers []*tinkerv1.PeerInfo
	for id, node := range s.nodes {
		inv := node.artifacts[root]
		if inv == nil || inv.GetState() != "complete" {
			continue
		}
		if !labelsMatch(node.labels, req.Msg.GetPreferredLabels()) {
			continue
		}
		addr := peerAddress(node.labels)
		if addr == "" {
			continue
		}
		peers = append(peers, &tinkerv1.PeerInfo{
			NodeId:     id,
			Address:    addr,
			Transports: []string{"connect"},
			Labels:     cloneMap(node.labels),
		})
	}
	sort.Slice(peers, func(i, j int) bool { return peers[i].GetNodeId() < peers[j].GetNodeId() })
	return connect.NewResponse(&tinkerv1.ListPeersResponse{Peers: peers}), nil
}

func (s *Server) SignalPeer(_ context.Context, stream *connect.BidiStream[tinkerv1.SignalMessage, tinkerv1.SignalMessage]) error {
	for {
		msg, err := stream.Receive()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if err := stream.Send(msg); err != nil {
			return err
		}
	}
}

func (s *Server) ReportTransfer(context.Context, *connect.Request[tinkerv1.ReportTransferRequest]) (*connect.Response[tinkerv1.ReportTransferResponse], error) {
	return connect.NewResponse(&tinkerv1.ReportTransferResponse{}), nil
}

func cloneMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneLoad(in *tinkerv1.NodeLoad) *tinkerv1.NodeLoad {
	if in == nil {
		return nil
	}
	return proto.Clone(in).(*tinkerv1.NodeLoad)
}

func inventoryMap(in []*tinkerv1.ArtifactInventory) map[string]*tinkerv1.ArtifactInventory {
	out := make(map[string]*tinkerv1.ArtifactInventory)
	for _, inv := range in {
		if inv.GetRootHash() == "" {
			continue
		}
		out[inv.GetRootHash()] = proto.Clone(inv).(*tinkerv1.ArtifactInventory)
	}
	return out
}

func peerAddress(labels map[string]string) string {
	for _, key := range []string{"artifact_peer_url", "peer_url"} {
		if v := strings.TrimSpace(labels[key]); v != "" {
			return v
		}
	}
	return ""
}

func labelsMatch(labels map[string]string, want []string) bool {
	for _, expr := range want {
		if expr == "" {
			continue
		}
		k, v, ok := strings.Cut(expr, "=")
		if !ok {
			if labels[expr] == "" {
				return false
			}
			continue
		}
		if labels[strings.TrimSpace(k)] != strings.TrimSpace(v) {
			return false
		}
	}
	return true
}

func aliasForRoot(aliases map[string]string, root string) string {
	var out []string
	for alias, aliasRoot := range aliases {
		if aliasRoot == root {
			out = append(out, alias)
		}
	}
	sort.Strings(out)
	if len(out) == 0 {
		return ""
	}
	return out[0]
}

func newID(prefix string) (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return prefix + "_" + hex.EncodeToString(b[:]), nil
}
