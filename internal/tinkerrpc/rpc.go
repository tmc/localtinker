package tinkerrpc

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"connectrpc.com/connect"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerdb"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1/tinkerv1connect"
)

const RPCMaxBytes = 128 << 20

const (
	nodeHealthy  = "healthy"
	nodeDraining = "draining"
	nodeDrained  = "drained"

	operationQueued   = "queued"
	operationLeased   = "leased"
	operationRunning  = "running"
	operationComplete = "complete"
	operationFailed   = "failed"
	operationCanceled = "canceled"
)

type Server struct {
	coord         *tinkercoord.Coordinator
	coordinatorID string

	mu    sync.Mutex
	nodes map[string]*nodeState

	manifests          map[string]*tinkerv1.Manifest
	aliases            map[string]string
	prewarmAssignments map[string]string
	operations         map[string]*operationState
	operationQueue     []string
	leaseTimeout       time.Duration
	now                func() time.Time
	nextSeq            int64
}

type nodeState struct {
	req         *tinkerv1.RegisterNodeRequest
	load        *tinkerv1.NodeLoad
	labels      map[string]string
	state       string
	drainReason string
	lastSeenAt  time.Time
	artifacts   map[string]*tinkerv1.ArtifactInventory
}

type operationState struct {
	id            string
	kind          string
	payloadJSON   []byte
	model         *tinkerv1.ModelRef
	state         string
	nodeID        string
	commandID     string
	leaseID       string
	createdAt     time.Time
	leasedAt      time.Time
	startedAt     time.Time
	completedAt   time.Time
	deadline      time.Time
	ackPending    bool
	attempts      int
	lastErrorCode string
	lastError     string
	revokePending bool
	revokeReason  string
}

type Snapshot struct {
	CoordinatorID string              `json:"coordinator_id"`
	Nodes         []NodeSnapshot      `json:"nodes"`
	Artifacts     []ArtifactSnapshot  `json:"artifacts"`
	Operations    []OperationSnapshot `json:"operations"`
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

type OperationSnapshot struct {
	OperationID   string    `json:"operation_id"`
	Kind          string    `json:"kind"`
	State         string    `json:"state"`
	NodeID        string    `json:"node_id,omitempty"`
	LeaseID       string    `json:"lease_id,omitempty"`
	CreatedAt     time.Time `json:"created_at"`
	Deadline      time.Time `json:"deadline,omitempty"`
	Attempts      int       `json:"attempts"`
	AckPending    bool      `json:"ack_pending,omitempty"`
	LastErrorCode string    `json:"last_error_code,omitempty"`
	LastError     string    `json:"last_error,omitempty"`
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
		coord:              coord,
		coordinatorID:      id,
		nodes:              make(map[string]*nodeState),
		manifests:          make(map[string]*tinkerv1.Manifest),
		aliases:            make(map[string]string),
		prewarmAssignments: make(map[string]string),
		operations:         make(map[string]*operationState),
		leaseTimeout:       30 * time.Second,
		now:                time.Now,
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
	operationIDs := make([]string, 0, len(s.operations))
	for id := range s.operations {
		operationIDs = append(operationIDs, id)
	}
	sort.Strings(operationIDs)
	for _, id := range operationIDs {
		op := s.operations[id]
		out.Operations = append(out.Operations, OperationSnapshot{
			OperationID:   op.id,
			Kind:          op.kind,
			State:         op.state,
			NodeID:        op.nodeID,
			LeaseID:       op.leaseID,
			CreatedAt:     op.createdAt,
			Deadline:      op.deadline,
			Attempts:      op.attempts,
			AckPending:    op.ackPending,
			LastErrorCode: op.lastErrorCode,
			LastError:     op.lastError,
		})
	}
	return out
}

// EnqueueOperation queues a node operation for assignment by Watch.
func (s *Server) EnqueueOperation(kind string, payloadJSON []byte, model *tinkerv1.ModelRef) (string, error) {
	kind = strings.TrimSpace(kind)
	if kind == "" {
		return "", errors.New("missing operation kind")
	}
	id, err := newID("op")
	if err != nil {
		return "", err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var modelCopy *tinkerv1.ModelRef
	if model != nil {
		modelCopy = proto.Clone(model).(*tinkerv1.ModelRef)
	}
	op := &operationState{
		id:          id,
		kind:        kind,
		payloadJSON: append([]byte(nil), payloadJSON...),
		model:       modelCopy,
		state:       operationQueued,
		createdAt:   s.now().UTC(),
	}
	s.operations[id] = op
	s.operationQueue = append(s.operationQueue, id)
	metadata, err := json.Marshal(map[string]any{
		"type":   kind,
		"source": "tinkerrpc",
	})
	if err != nil {
		delete(s.operations, id)
		s.operationQueue = s.operationQueue[:len(s.operationQueue)-1]
		return "", err
	}
	if err := s.coord.PutFuture(context.Background(), tinkerdb.Future{
		ID:           id,
		State:        tinkercoord.FutureQueued,
		Metadata:     metadata,
		CreatedAt:    op.createdAt,
		Operation:    kind,
		RequestBytes: int64(len(payloadJSON)),
		MaxAttempts:  3,
	}); err != nil {
		delete(s.operations, id)
		s.operationQueue = s.operationQueue[:len(s.operationQueue)-1]
		return "", err
	}
	return id, nil
}

// CancelOperation asks the node holding opID's lease to revoke it.
func (s *Server) CancelOperation(opID, reason string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	op := s.operations[opID]
	if op == nil {
		return false, nil
	}
	switch op.state {
	case operationQueued:
		op.state = operationCanceled
		op.lastErrorCode = "canceled"
		op.lastError = reason
		if _, err := s.coord.CancelFuture(context.Background(), opID); err != nil {
			return false, err
		}
		return true, nil
	case operationLeased, operationRunning:
		if reason == "" {
			reason = "operation canceled"
		}
		errJSON, err := json.Marshal(map[string]any{"code": "canceled", "message": reason})
		if err != nil {
			return false, err
		}
		if _, ok, err := s.coord.FinishFutureLease(context.Background(), op.id, op.leaseID, tinkercoord.FutureCanceled, nil, errJSON, s.now().UTC()); err != nil {
			return false, err
		} else if !ok {
			return false, nil
		}
		op.state = operationCanceled
		op.lastErrorCode = "canceled"
		op.lastError = reason
		op.revokePending = true
		op.revokeReason = reason
		return true, nil
	default:
		return false, nil
	}
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

func (s *Server) RegisterNode(ctx context.Context, req *connect.Request[tinkerv1.RegisterNodeRequest]) (*connect.Response[tinkerv1.RegisterNodeResponse], error) {
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
		state:      nodeHealthy,
		lastSeenAt: time.Now().UTC(),
		artifacts:  make(map[string]*tinkerv1.ArtifactInventory),
	}
	s.mu.Unlock()
	if err := s.recordNode(ctx, nodeID, nodeHealthy, msg.GetName(), msg.GetLabels(), msg.GetCapabilities(), nil); err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}

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

func (s *Server) Heartbeat(ctx context.Context, req *connect.Request[tinkerv1.HeartbeatRequest]) (*connect.Response[tinkerv1.HeartbeatResponse], error) {
	msg := req.Msg
	nodeID := msg.GetNodeId()
	if nodeID == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing node_id"))
	}

	s.mu.Lock()
	node := s.nodes[nodeID]
	if node == nil {
		node = &nodeState{state: nodeHealthy, artifacts: make(map[string]*tinkerv1.ArtifactInventory)}
		s.nodes[nodeID] = node
	}
	node.load = msg.GetLoad()
	node.lastSeenAt = time.Now().UTC()
	if len(msg.GetArtifacts()) > 0 {
		node.artifacts = inventoryMap(msg.GetArtifacts())
	}
	drain := node.state == nodeDraining || node.state == nodeDrained
	if node.state == nodeDraining && msg.GetLoad().GetActiveLeases() == 0 {
		node.state = nodeDrained
	}
	state := node.state
	name := ""
	labels := cloneMap(node.labels)
	caps := (*tinkerv1.NodeCapabilities)(nil)
	if node.req != nil {
		name = node.req.GetName()
		caps = node.req.GetCapabilities()
	}
	prewarm := s.prewarmRootsLocked(nodeID)
	s.mu.Unlock()
	if err := s.recordNode(ctx, nodeID, state, name, labels, caps, msg.GetLoad()); err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	return connect.NewResponse(&tinkerv1.HeartbeatResponse{
		CoordinatorId:  s.coordinatorID,
		DrainRequested: drain,
		PrewarmRoots:   prewarm,
	}), nil
}

func (s *Server) recordNode(ctx context.Context, id, state, name string, labels map[string]string, caps *tinkerv1.NodeCapabilities, load *tinkerv1.NodeLoad) error {
	if state == "" {
		state = nodeHealthy
	}
	node := tinkerdb.Node{
		ID:         id,
		Name:       name,
		State:      state,
		Labels:     cloneMap(labels),
		Running:    int(load.GetActiveLeases()),
		LastSeenAt: s.now().UTC(),
	}
	if caps != nil {
		raw, err := protojson.Marshal(caps)
		if err != nil {
			return err
		}
		node.Capabilities = raw
		node.MaxConcurrency = int(caps.GetMaxConcurrency())
	}
	if load != nil {
		raw, err := protojson.Marshal(load)
		if err != nil {
			return err
		}
		node.Load = raw
	}
	return s.coord.RecordNode(ctx, node)
}

func (s *Server) prewarmRootsLocked(nodeID string) []string {
	counts := s.prewarmAssignmentCountsLocked()
	roots := make([]string, 0, len(s.manifests))
	for root := range s.manifests {
		roots = append(roots, root)
	}
	sort.Strings(roots)

	var out []string
	for _, root := range roots {
		assigned := s.prewarmAssignments[root]
		if assigned != "" && !s.prewarmAssignmentValidLocked(root, assigned) {
			delete(s.prewarmAssignments, root)
			assigned = ""
		}
		if assigned == "" {
			assigned = s.selectPrewarmNodeLocked(root, counts)
			if assigned != "" {
				s.prewarmAssignments[root] = assigned
				counts[assigned]++
			}
		}
		if assigned == nodeID {
			out = append(out, root)
		}
	}
	return out
}

func (s *Server) prewarmAssignmentCountsLocked() map[string]int {
	counts := make(map[string]int)
	for root, nodeID := range s.prewarmAssignments {
		if s.prewarmAssignmentValidLocked(root, nodeID) {
			counts[nodeID]++
		}
	}
	return counts
}

func (s *Server) prewarmAssignmentValidLocked(root, nodeID string) bool {
	node := s.nodes[nodeID]
	return s.nodeCanPrewarmLocked(node, root)
}

func (s *Server) selectPrewarmNodeLocked(root string, assigned map[string]int) string {
	var bestID string
	var bestNode *nodeState
	for id, node := range s.nodes {
		if !s.nodeCanPrewarmLocked(node, root) {
			continue
		}
		if bestID == "" || lessLoadedNode(id, node, assigned[id], bestID, bestNode, assigned[bestID]) {
			bestID, bestNode = id, node
		}
	}
	return bestID
}

func (s *Server) nodeCanPrewarmLocked(node *nodeState, root string) bool {
	if node == nil || node.state != nodeHealthy {
		return false
	}
	if inv := node.artifacts[root]; inv != nil && inv.GetState() == "complete" {
		return false
	}
	return true
}

func lessLoadedNode(id string, node *nodeState, assigned int, bestID string, best *nodeState, bestAssigned int) bool {
	load := nodeLoadScore(node) + assigned
	bestLoad := nodeLoadScore(best) + bestAssigned
	if load != bestLoad {
		return load < bestLoad
	}
	mem := nodeMemoryAvailable(node)
	bestMem := nodeMemoryAvailable(best)
	if mem != bestMem {
		return mem > bestMem
	}
	return id < bestID
}

func nodeLoadScore(node *nodeState) int {
	if node == nil || node.load == nil {
		return 0
	}
	return int(node.load.GetActiveLeases()) + int(node.load.GetQueuedOperations())
}

func nodeMemoryAvailable(node *nodeState) uint64 {
	if node == nil || node.load == nil {
		return 0
	}
	return node.load.GetMemoryAvailableBytes()
}

func (s *Server) Watch(ctx context.Context, req *connect.Request[tinkerv1.WatchRequest], stream *connect.ServerStream[tinkerv1.NodeCommand]) error {
	nodeID := strings.TrimSpace(req.Msg.GetNodeId())
	if nodeID == "" {
		return connect.NewError(connect.CodeInvalidArgument, errors.New("missing node_id"))
	}
	if coordID := strings.TrimSpace(req.Msg.GetCoordinatorId()); coordID != "" && coordID != s.coordinatorID {
		return connect.NewError(connect.CodeInvalidArgument, errors.New("coordinator_id mismatch"))
	}

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		cmd, err := s.watchCommand(nodeID)
		if err != nil {
			return err
		}
		if cmd != nil {
			return stream.Send(cmd)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func (s *Server) watchCommand(nodeID string) (*tinkerv1.NodeCommand, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.expireOperationLeasesLocked(); err != nil {
		return nil, err
	}
	node := s.nodes[nodeID]
	if node == nil {
		return nil, connect.NewError(connect.CodeNotFound, errors.New("unknown node"))
	}
	state := node.state
	reason := node.drainReason

	if state == nodeHealthy {
		if cmd := s.revokeOperationCommandLocked(nodeID); cmd != nil {
			return cmd, nil
		}
		if cmd := s.ackOperationCommandLocked(nodeID); cmd != nil {
			return cmd, nil
		}
		return s.runOperationCommandLocked(nodeID)
	}
	if reason == "" {
		reason = "admin requested drain"
	}
	now := s.now().UTC()
	return &tinkerv1.NodeCommand{
		CommandId:        "drain-" + nodeID,
		Kind:             "drain",
		SeqId:            s.nextSeqLocked(),
		DeadlineUnixNano: now.Add(30 * time.Second).UnixNano(),
		Directive: &tinkerv1.NodeCommand_Drain{
			Drain: &tinkerv1.DrainLease{Reason: reason, Checkpoint: true},
		},
	}, nil
}

func (s *Server) revokeOperationCommandLocked(nodeID string) *tinkerv1.NodeCommand {
	var selected *operationState
	for _, op := range s.operations {
		if op.nodeID != nodeID || !op.revokePending {
			continue
		}
		if selected == nil || op.id < selected.id {
			selected = op
		}
	}
	if selected == nil {
		return nil
	}
	selected.revokePending = false
	reason := selected.revokeReason
	if reason == "" {
		reason = "operation canceled"
	}
	now := s.now().UTC()
	seq := s.nextSeqLocked()
	return &tinkerv1.NodeCommand{
		CommandId:        "revoke-" + selected.id + "-" + strconv.FormatInt(seq, 10),
		LeaseId:          selected.leaseID,
		OperationId:      selected.id,
		Kind:             selected.kind,
		SeqId:            seq,
		DeadlineUnixNano: now.Add(30 * time.Second).UnixNano(),
		Directive: &tinkerv1.NodeCommand_Revoke{
			Revoke: &tinkerv1.RevokeLease{Reason: reason},
		},
	}
}

func (s *Server) ackOperationCommandLocked(nodeID string) *tinkerv1.NodeCommand {
	var operationIDs []string
	for _, op := range s.operations {
		if op.nodeID == nodeID && op.ackPending {
			operationIDs = append(operationIDs, op.id)
			op.ackPending = false
		}
	}
	if len(operationIDs) == 0 {
		return nil
	}
	sort.Strings(operationIDs)
	now := s.now().UTC()
	seq := s.nextSeqLocked()
	return &tinkerv1.NodeCommand{
		CommandId:        "ack-" + nodeID + "-" + strconv.FormatInt(seq, 10),
		Kind:             "ack_operation",
		SeqId:            seq,
		DeadlineUnixNano: now.Add(30 * time.Second).UnixNano(),
		Directive: &tinkerv1.NodeCommand_AckOperation{
			AckOperation: &tinkerv1.AcknowledgeOperation{OperationIds: operationIDs},
		},
	}
}

func (s *Server) runOperationCommandLocked(nodeID string) (*tinkerv1.NodeCommand, error) {
	if !s.nodeCanRunLocked(nodeID) || s.bestRunNodeLocked() != nodeID {
		return nil, nil
	}
	for len(s.operationQueue) > 0 {
		id := s.operationQueue[0]
		s.operationQueue = s.operationQueue[1:]
		op := s.operations[id]
		if op == nil || op.state != operationQueued {
			continue
		}
		future, ok, err := s.coord.ClaimFuture(context.Background(), nodeID, s.now().UTC(), s.leaseTimeout)
		if err != nil {
			return nil, connect.NewError(connect.CodeInternal, err)
		}
		if !ok || future.ID != op.id {
			if ok {
				s.operationQueue = append(s.operationQueue, op.id)
			}
			continue
		}
		commandID, err := newID("cmd")
		if err != nil {
			return nil, connect.NewError(connect.CodeInternal, err)
		}
		op.state = operationLeased
		op.nodeID = nodeID
		op.commandID = commandID
		op.leaseID = future.LeaseID
		op.leasedAt = future.StartedAt
		op.deadline = future.LeaseExpiresAt
		op.attempts = future.Attempt
		return &tinkerv1.NodeCommand{
			CommandId:        commandID,
			LeaseId:          future.LeaseID,
			OperationId:      op.id,
			Kind:             op.kind,
			SeqId:            s.nextSeqLocked(),
			DeadlineUnixNano: op.deadline.UnixNano(),
			PayloadJson:      append([]byte(nil), op.payloadJSON...),
			Model:            cloneModelRef(op.model),
			Directive: &tinkerv1.NodeCommand_Run{
				Run: &tinkerv1.RunOperation{},
			},
		}, nil
	}
	return nil, nil
}

func (s *Server) nodeCanRunLocked(nodeID string) bool {
	node := s.nodes[nodeID]
	return node != nil && node.state == nodeHealthy
}

func (s *Server) bestRunNodeLocked() string {
	counts := s.operationAssignmentCountsLocked()
	var bestID string
	var bestNode *nodeState
	for id, node := range s.nodes {
		if !s.nodeCanRunLocked(id) {
			continue
		}
		if bestID == "" || lessLoadedNode(id, node, counts[id], bestID, bestNode, counts[bestID]) {
			bestID, bestNode = id, node
		}
	}
	return bestID
}

func (s *Server) operationAssignmentCountsLocked() map[string]int {
	counts := make(map[string]int)
	for _, op := range s.operations {
		if op.nodeID == "" {
			continue
		}
		switch op.state {
		case operationLeased, operationRunning:
			counts[op.nodeID]++
		}
	}
	return counts
}

func (s *Server) expireOperationLeasesLocked() error {
	now := s.now().UTC()
	for _, op := range s.operations {
		switch op.state {
		case operationLeased, operationRunning:
		default:
			continue
		}
		if op.deadline.IsZero() || op.deadline.After(now) {
			continue
		}
		if _, ok, err := s.coord.RequeueFutureLease(context.Background(), op.id, op.leaseID, "operation lease expired", now, now); err != nil {
			return connect.NewError(connect.CodeInternal, err)
		} else if !ok {
			continue
		}
		op.state = operationQueued
		op.nodeID = ""
		op.commandID = ""
		op.leaseID = ""
		op.deadline = time.Time{}
		s.operationQueue = append(s.operationQueue, op.id)
	}
	return nil
}

func (s *Server) nextSeqLocked() int64 {
	s.nextSeq++
	return s.nextSeq
}

func (s *Server) Report(_ context.Context, stream *connect.ClientStream[tinkerv1.NodeEvent]) (*connect.Response[tinkerv1.ReportResponse], error) {
	for stream.Receive() {
		if err := s.applyNodeEvent(stream.Msg()); err != nil {
			return nil, err
		}
	}
	if err := stream.Err(); err != nil {
		return nil, err
	}
	return connect.NewResponse(&tinkerv1.ReportResponse{}), nil
}

func (s *Server) applyNodeEvent(event *tinkerv1.NodeEvent) error {
	nodeID := event.GetNodeId()
	if nodeID == "" {
		return connect.NewError(connect.CodeInvalidArgument, errors.New("missing node_id"))
	}
	seen := time.Now().UTC()
	if unix := event.GetUnixNano(); unix != 0 {
		seen = time.Unix(0, unix).UTC()
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	node := s.nodes[nodeID]
	if node == nil {
		node = &nodeState{state: nodeHealthy, artifacts: make(map[string]*tinkerv1.ArtifactInventory)}
		s.nodes[nodeID] = node
	}
	node.lastSeenAt = seen
	if lifecycle := event.GetLifecycle(); lifecycle != nil && lifecycle.GetEvent() != "" {
		node.state = lifecycle.GetEvent()
		for k, v := range lifecycle.GetMetadata() {
			if node.labels == nil {
				node.labels = make(map[string]string)
			}
			node.labels[k] = v
		}
	}
	if telemetry := event.GetTelemetry(); telemetry != nil {
		for k, v := range telemetry.GetLabels() {
			if node.labels == nil {
				node.labels = make(map[string]string)
			}
			node.labels[k] = v
		}
	}
	if event.GetAck() != nil {
		if node.labels == nil {
			node.labels = make(map[string]string)
		}
		if commandID := strings.TrimSpace(event.GetCommandId()); commandID != "" {
			node.labels["last_command_ack_id"] = commandID
		}
		if kind := strings.TrimSpace(event.GetKind()); kind != "" {
			node.labels["last_command_ack_kind"] = kind
		}
	}
	switch {
	case event.GetStarted() != nil:
		s.operationStartedLocked(event)
		load := nodeLoad(node)
		load.ActiveLeases++
		if load.QueuedOperations > 0 {
			load.QueuedOperations--
		}
	case event.GetCompleted() != nil || event.GetFailed() != nil:
		if err := s.operationTerminalLocked(event); err != nil {
			return err
		}
		load := nodeLoad(node)
		if load.ActiveLeases > 0 {
			load.ActiveLeases--
		}
	}
	return nil
}

func (s *Server) operationStartedLocked(event *tinkerv1.NodeEvent) {
	op := s.matchOperationLocked(event)
	if op == nil {
		return
	}
	op.state = operationRunning
	op.startedAt = s.eventTime(event)
}

func (s *Server) operationTerminalLocked(event *tinkerv1.NodeEvent) error {
	op := s.matchOperationLocked(event)
	if op == nil {
		return nil
	}
	op.completedAt = s.eventTime(event)
	op.deadline = time.Time{}
	op.ackPending = true
	if op.state == operationCanceled {
		op.state = operationFailed
		op.lastErrorCode = "canceled"
		if op.lastError == "" {
			op.lastError = "operation canceled"
		}
		return nil
	}
	if failed := event.GetFailed(); failed != nil {
		op.state = operationFailed
		if err := failed.GetError(); err != nil {
			op.lastErrorCode = err.GetCode()
			op.lastError = err.GetMessage()
		}
		errJSON, err := json.Marshal(map[string]any{"code": op.lastErrorCode, "message": op.lastError})
		if err != nil {
			return connect.NewError(connect.CodeInternal, err)
		}
		if _, ok, err := s.coord.FinishFutureLease(context.Background(), op.id, op.leaseID, tinkercoord.FutureSystemError, nil, errJSON, op.completedAt); err != nil {
			return connect.NewError(connect.CodeInternal, err)
		} else if !ok {
			return connect.NewError(connect.CodeFailedPrecondition, errors.New("stale operation lease"))
		}
		return nil
	}
	var result json.RawMessage
	if completed := event.GetCompleted(); completed != nil && completed.GetResult() != nil {
		if raw, err := protojson.Marshal(completed.GetResult()); err == nil {
			result = raw
		} else {
			return connect.NewError(connect.CodeInternal, err)
		}
	}
	if _, ok, err := s.coord.FinishFutureLease(context.Background(), op.id, op.leaseID, tinkercoord.FutureComplete, result, nil, op.completedAt); err != nil {
		return connect.NewError(connect.CodeInternal, err)
	} else if !ok {
		return connect.NewError(connect.CodeFailedPrecondition, errors.New("stale operation lease"))
	}
	op.state = operationComplete
	op.lastErrorCode = ""
	op.lastError = ""
	return nil
}

func (s *Server) matchOperationLocked(event *tinkerv1.NodeEvent) *operationState {
	op := s.operations[event.GetOperationId()]
	if op == nil || op.leaseID != event.GetLeaseId() || op.nodeID != event.GetNodeId() {
		return nil
	}
	return op
}

func (s *Server) eventTime(event *tinkerv1.NodeEvent) time.Time {
	if unix := event.GetUnixNano(); unix != 0 {
		return time.Unix(0, unix).UTC()
	}
	return s.now().UTC()
}

func nodeLoad(node *nodeState) *tinkerv1.NodeLoad {
	if node.load == nil {
		node.load = &tinkerv1.NodeLoad{}
	}
	return node.load
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
	sort.Slice(resp.Nodes, func(i, j int) bool {
		return resp.Nodes[i].GetNodeId() < resp.Nodes[j].GetNodeId()
	})
	return connect.NewResponse(resp), nil
}

func (s *Server) DrainNode(_ context.Context, req *connect.Request[tinkerv1.DrainNodeRequest]) (*connect.Response[tinkerv1.DrainNodeResponse], error) {
	nodeID := req.Msg.GetNodeId()
	if nodeID == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing node_id"))
	}
	s.mu.Lock()
	node := s.nodes[nodeID]
	if node == nil {
		s.mu.Unlock()
		return nil, connect.NewError(connect.CodeNotFound, errors.New("unknown node"))
	}
	node.drainReason = strings.TrimSpace(req.Msg.GetReason())
	if node.load != nil && node.load.GetActiveLeases() == 0 {
		node.state = nodeDrained
	} else {
		node.state = nodeDraining
	}
	s.mu.Unlock()
	return connect.NewResponse(&tinkerv1.DrainNodeResponse{}), nil
}

func (s *Server) ListRuns(ctx context.Context, _ *connect.Request[tinkerv1.ListRunsRequest]) (*connect.Response[tinkerv1.ListRunsResponse], error) {
	runs, err := s.coord.TrainingRuns(ctx, 10000, 0)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	resp := &tinkerv1.ListRunsResponse{}
	for _, run := range runs.TrainingRuns {
		resp.Runs = append(resp.Runs, runSummary(run))
	}
	return connect.NewResponse(resp), nil
}

func (s *Server) InspectRun(ctx context.Context, req *connect.Request[tinkerv1.InspectRunRequest]) (*connect.Response[tinkerv1.InspectRunResponse], error) {
	runID := strings.TrimSpace(req.Msg.GetRunId())
	if runID == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing run_id"))
	}
	runs, err := s.coord.TrainingRuns(ctx, 10000, 0)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	var run tinkercoord.TrainingRun
	found := false
	for _, candidate := range runs.TrainingRuns {
		if candidate.TrainingRunID == runID {
			run = candidate
			found = true
			break
		}
	}
	if !found {
		return nil, connect.NewError(connect.CodeNotFound, errors.New("run not found"))
	}
	checkpoints, err := s.coord.Checkpoints(ctx, runID, 10000, 0)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	runJSON, err := json.Marshal(struct {
		Run         tinkercoord.TrainingRun  `json:"run"`
		Checkpoints []tinkercoord.Checkpoint `json:"checkpoints,omitempty"`
	}{
		Run:         run,
		Checkpoints: checkpoints.Checkpoints,
	})
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	snapshot, err := s.coord.DashboardSnapshot(ctx)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	var events [][]byte
	for _, future := range snapshot.Futures {
		if future.ModelID != runID {
			continue
		}
		eventJSON, err := json.Marshal(future)
		if err != nil {
			return nil, connect.NewError(connect.CodeInternal, err)
		}
		events = append(events, eventJSON)
	}
	return connect.NewResponse(&tinkerv1.InspectRunResponse{
		RunJson:    runJSON,
		EventsJson: events,
	}), nil
}

func runSummary(run tinkercoord.TrainingRun) *tinkerv1.RunSummary {
	return &tinkerv1.RunSummary{
		RunId:             run.TrainingRunID,
		Name:              run.BaseModel,
		Algorithm:         runAlgorithm(run),
		State:             runState(run),
		CurrentCheckpoint: runCheckpointPath(run),
	}
}

func runAlgorithm(run tinkercoord.TrainingRun) string {
	if run.IsLoRA {
		return "lora"
	}
	return "full"
}

func runState(run tinkercoord.TrainingRun) string {
	if run.Corrupted {
		return "corrupted"
	}
	return "ready"
}

func runCheckpointPath(run tinkercoord.TrainingRun) string {
	if run.LastCheckpoint != nil {
		return run.LastCheckpoint.TinkerPath
	}
	if run.LastSamplerCheckpoint != nil {
		return run.LastSamplerCheckpoint.TinkerPath
	}
	return ""
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
	alias := strings.TrimSpace(req.Msg.GetAlias())
	if alias == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing alias"))
	}
	s.mu.Lock()
	delete(s.aliases, alias)
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
		node = &nodeState{state: nodeHealthy, artifacts: make(map[string]*tinkerv1.ArtifactInventory)}
		s.nodes[nodeID] = node
	}
	node.artifacts = inventoryMap(req.Msg.GetArtifacts())
	node.lastSeenAt = time.Now().UTC()
	s.mu.Unlock()
	return connect.NewResponse(&tinkerv1.ReportInventoryResponse{}), nil
}

func (s *Server) ApplyRetention(_ context.Context, req *connect.Request[tinkerv1.ApplyRetentionRequest]) (*connect.Response[tinkerv1.ApplyRetentionResponse], error) {
	nodeID := req.Msg.GetNodeId()
	if nodeID == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing node_id"))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	node := s.nodes[nodeID]
	if node == nil {
		return nil, connect.NewError(connect.CodeNotFound, errors.New("unknown node"))
	}
	if req.Msg.GetTargetFreeBytes() == 0 {
		return connect.NewResponse(&tinkerv1.ApplyRetentionResponse{}), nil
	}

	protected := make(map[string]bool)
	for _, root := range req.Msg.GetProtectedRootHashes() {
		if root != "" {
			protected[root] = true
		}
	}
	roots := make([]string, 0, len(node.artifacts))
	for root := range node.artifacts {
		if !protected[root] {
			roots = append(roots, root)
		}
	}
	sort.Strings(roots)

	resp := &tinkerv1.ApplyRetentionResponse{}
	for _, root := range roots {
		if resp.GetBytesDeleted() >= req.Msg.GetTargetFreeBytes() {
			break
		}
		inv := node.artifacts[root]
		delete(node.artifacts, root)
		resp.DeletedRootHashes = append(resp.DeletedRootHashes, root)
		resp.BytesDeleted += inv.GetBytesPresent()
	}
	return connect.NewResponse(resp), nil
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

func (s *Server) ReportTransfer(_ context.Context, req *connect.Request[tinkerv1.ReportTransferRequest]) (*connect.Response[tinkerv1.ReportTransferResponse], error) {
	msg := req.Msg
	nodeID := msg.GetNodeId()
	if nodeID == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing node_id"))
	}
	root := msg.GetRootHash()
	if root == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing root_hash"))
	}
	state := msg.GetState()
	if state == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, errors.New("missing state"))
	}

	s.mu.Lock()
	node := s.nodes[nodeID]
	if node == nil {
		node = &nodeState{state: nodeHealthy, artifacts: make(map[string]*tinkerv1.ArtifactInventory)}
		s.nodes[nodeID] = node
	}
	if node.labels == nil {
		node.labels = make(map[string]string)
	}
	node.labels["last_transfer_root_hash"] = root
	node.labels["last_transfer_state"] = state
	if peer := msg.GetPeerNodeId(); peer != "" {
		node.labels["last_transfer_peer_node_id"] = peer
	}
	if bytes := msg.GetBytes(); bytes > 0 {
		node.labels["last_transfer_bytes"] = strconv.FormatUint(bytes, 10)
	}
	if info := msg.GetError(); info != nil {
		if code := info.GetCode(); code != "" {
			node.labels["last_transfer_error_code"] = code
		}
		if message := info.GetMessage(); message != "" {
			node.labels["last_transfer_error_message"] = message
		}
	}
	if state == "complete" {
		node.artifacts[root] = &tinkerv1.ArtifactInventory{
			RootHash:     root,
			State:        "complete",
			BytesPresent: msg.GetBytes(),
		}
	}
	node.lastSeenAt = time.Now().UTC()
	s.mu.Unlock()
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

func cloneModelRef(in *tinkerv1.ModelRef) *tinkerv1.ModelRef {
	if in == nil {
		return nil
	}
	return proto.Clone(in).(*tinkerv1.ModelRef)
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
