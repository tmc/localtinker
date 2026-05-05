package tinkernode

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"time"

	"connectrpc.com/connect"

	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1/tinkerv1connect"
)

const (
	NodeStateHealthy  = "healthy"
	NodeStateDraining = "draining"
	NodeStateDrained  = "drained"
)

var ErrNoOperationID = errors.New("missing operation id")

// OperationRunner runs one node-local operation.
type OperationRunner func(context.Context, *tinkerv1.NodeCommand) OperationResult

// Runtime serves the node RPC API for local operation execution.
type Runtime struct {
	mu      sync.Mutex
	nodeID  string
	state   string
	active  map[string]context.CancelFunc
	cache   ResultCache
	runner  OperationRunner
	load    *tinkerv1.NodeLoad
	drained bool
}

// RuntimeOption configures a Runtime.
type RuntimeOption func(*Runtime)

// WithResultCache sets the cache used for terminal operation results.
func WithResultCache(cache ResultCache) RuntimeOption {
	return func(r *Runtime) {
		r.cache = cache
	}
}

// WithOperationRunner sets the operation runner.
func WithOperationRunner(runner OperationRunner) RuntimeOption {
	return func(r *Runtime) {
		r.runner = runner
	}
}

// NewRuntime returns a node Runtime.
func NewRuntime(nodeID string, opts ...RuntimeOption) *Runtime {
	r := &Runtime{
		nodeID: nodeID,
		state:  NodeStateHealthy,
		active: make(map[string]context.CancelFunc),
		cache:  NewMemoryResultCache(),
		load:   &tinkerv1.NodeLoad{},
	}
	for _, opt := range opts {
		opt(r)
	}
	if r.cache == nil {
		r.cache = NewMemoryResultCache()
	}
	return r
}

// RegisterRuntime registers the TinkerNode service on mux.
func RegisterRuntime(mux *http.ServeMux, runtime *Runtime, opts ...connect.HandlerOption) {
	path, h := tinkerv1connect.NewTinkerNodeHandler(runtime, opts...)
	mux.Handle(path, h)
}

// SetNodeID updates the node ID returned by Health.
func (r *Runtime) SetNodeID(nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.nodeID = nodeID
}

// SetBaseLoad sets health load fields not derived from active operations.
func (r *Runtime) SetBaseLoad(load *tinkerv1.NodeLoad) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.load = cloneNodeLoad(load)
}

// ActiveLeases returns the number of active local operations.
func (r *Runtime) ActiveLeases() int32 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return int32(len(r.active))
}

// Acknowledge evicts cached terminal results after coordinator acknowledgement.
func (r *Runtime) Acknowledge(ctx context.Context, operationIDs ...string) error {
	for _, operationID := range operationIDs {
		if operationID == "" {
			continue
		}
		if err := r.cache.Acknowledge(ctx, operationID); err != nil && !errors.Is(err, ErrResultNotFound) {
			return err
		}
	}
	return nil
}

func (r *Runtime) Run(ctx context.Context, req *connect.Request[tinkerv1.NodeCommand]) (*connect.Response[tinkerv1.OperationResult], error) {
	cmd := req.Msg
	operationID := cmd.GetOperationId()
	if operationID == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, ErrNoOperationID)
	}
	if got, ok, err := r.cache.Get(ctx, operationID); err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	} else if ok {
		return connect.NewResponse(protoResult(got)), nil
	}

	opCtx, cancel, err := r.start(operationID)
	if err != nil {
		return nil, err
	}
	defer r.finish(operationID)
	defer cancel()

	if deadline := cmd.GetDeadlineUnixNano(); deadline > 0 {
		var deadlineCancel context.CancelFunc
		opCtx, deadlineCancel = context.WithDeadline(opCtx, time.Unix(0, deadline))
		defer deadlineCancel()
	}
	opCtx = mergeContext(ctx, opCtx)

	result := r.run(opCtx, cmd)
	result.OperationID = operationID
	if result.LeaseID == "" {
		result.LeaseID = cmd.GetLeaseId()
	}
	if result.Kind == "" {
		result.Kind = OperationKind(cmd.GetKind())
	}
	if result.Created.IsZero() {
		result.Created = time.Now()
	}
	if result.State == "" {
		result.State = ResultSucceeded
	}
	if err := r.cache.Store(context.Background(), result); err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	return connect.NewResponse(protoResult(result)), nil
}

func (r *Runtime) Cancel(_ context.Context, req *connect.Request[tinkerv1.CancelRequest]) (*connect.Response[tinkerv1.CancelResponse], error) {
	operationID := req.Msg.GetOperationId()
	if operationID == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument, ErrNoOperationID)
	}
	r.mu.Lock()
	cancel := r.active[operationID]
	r.mu.Unlock()
	if cancel == nil {
		return connect.NewResponse(&tinkerv1.CancelResponse{}), nil
	}
	cancel()
	return connect.NewResponse(&tinkerv1.CancelResponse{Canceled: true}), nil
}

func (r *Runtime) Drain(context.Context, *connect.Request[tinkerv1.DrainRequest]) (*connect.Response[tinkerv1.DrainResponse], error) {
	r.mu.Lock()
	r.drained = true
	if len(r.active) == 0 {
		r.state = NodeStateDrained
	} else {
		r.state = NodeStateDraining
	}
	r.mu.Unlock()
	return connect.NewResponse(&tinkerv1.DrainResponse{}), nil
}

func (r *Runtime) Health(context.Context, *connect.Request[tinkerv1.NodeHealthRequest]) (*connect.Response[tinkerv1.NodeHealthResponse], error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return connect.NewResponse(&tinkerv1.NodeHealthResponse{
		NodeId: r.nodeID,
		State:  r.state,
		Load:   r.currentLoadLocked(),
	}), nil
}

func (r *Runtime) start(operationID string) (context.Context, context.CancelFunc, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	switch r.state {
	case NodeStateDraining, NodeStateDrained:
		return nil, nil, connect.NewError(connect.CodeFailedPrecondition, errors.New("node is draining"))
	}
	if _, ok := r.active[operationID]; ok {
		return nil, nil, connect.NewError(connect.CodeAlreadyExists, errors.New("operation already running"))
	}
	ctx, cancel := context.WithCancel(context.Background())
	r.active[operationID] = cancel
	return ctx, cancel, nil
}

func (r *Runtime) finish(operationID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.active, operationID)
	if r.drained && len(r.active) == 0 {
		r.state = NodeStateDrained
	}
}

func (r *Runtime) run(ctx context.Context, cmd *tinkerv1.NodeCommand) OperationResult {
	if r.runner != nil {
		return r.runner(ctx, cmd)
	}
	if err := ctx.Err(); err != nil {
		return canceledResult(cmd, err)
	}
	return OperationResult{
		OperationID: cmd.GetOperationId(),
		LeaseID:     cmd.GetLeaseId(),
		Kind:        OperationKind(cmd.GetKind()),
		State:       ResultSucceeded,
		Payload:     append([]byte(nil), cmd.GetPayloadJson()...),
	}
}

func (r *Runtime) currentLoadLocked() *tinkerv1.NodeLoad {
	load := cloneNodeLoad(r.load)
	load.ActiveLeases = int32(len(r.active))
	return load
}

func canceledResult(cmd *tinkerv1.NodeCommand, err error) OperationResult {
	return OperationResult{
		OperationID: cmd.GetOperationId(),
		LeaseID:     cmd.GetLeaseId(),
		Kind:        OperationKind(cmd.GetKind()),
		State:       ResultCanceled,
		Error:       err.Error(),
	}
}

func protoResult(r OperationResult) *tinkerv1.OperationResult {
	out := &tinkerv1.OperationResult{
		OperationId: r.OperationID,
		Kind:        string(r.Kind),
		PayloadJson: append([]byte(nil), r.Payload...),
	}
	if r.Error != "" {
		out.Error = &tinkerv1.ErrorInfo{
			Code:    string(r.State),
			Message: r.Error,
		}
	}
	return out
}

func mergeContext(a, b context.Context) context.Context {
	ctx, cancel := context.WithCancel(a)
	go func() {
		select {
		case <-b.Done():
			cancel()
		case <-ctx.Done():
		}
	}()
	return ctx
}

func cloneNodeLoad(load *tinkerv1.NodeLoad) *tinkerv1.NodeLoad {
	if load == nil {
		return &tinkerv1.NodeLoad{}
	}
	return &tinkerv1.NodeLoad{
		ActiveLeases:         load.GetActiveLeases(),
		QueuedOperations:     load.GetQueuedOperations(),
		MemoryAvailableBytes: load.GetMemoryAvailableBytes(),
		TemperatureCelsius:   load.GetTemperatureCelsius(),
	}
}
