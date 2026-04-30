package tinkernode

import (
	"context"
	"errors"
	"sync"
	"time"
)

// OperationKind identifies a node operation for idempotent result caching.
type OperationKind string

const (
	OperationForwardBackward       OperationKind = "forward_backward"
	OperationOptimStep             OperationKind = "optim_step"
	OperationSaveState             OperationKind = "save_state"
	OperationLoadState             OperationKind = "load_state"
	OperationSaveWeightsForSampler OperationKind = "save_weights_for_sampler"
	OperationUnload                OperationKind = "unload"
	OperationDrain                 OperationKind = "drain"
)

// IsMutating reports whether kind advances model or lease state.
func IsMutating(kind OperationKind) bool {
	switch kind {
	case OperationForwardBackward,
		OperationOptimStep,
		OperationSaveState,
		OperationLoadState,
		OperationSaveWeightsForSampler,
		OperationUnload,
		OperationDrain:
		return true
	default:
		return false
	}
}

// ResultState is the terminal state of a cached operation.
type ResultState string

const (
	ResultSucceeded ResultState = "succeeded"
	ResultFailed    ResultState = "failed"
	ResultCanceled  ResultState = "canceled"
)

var ErrResultNotFound = errors.New("result not found")

// OperationResult is the terminal result for a node-local operation.
type OperationResult struct {
	OperationID string
	LeaseID     string
	Kind        OperationKind
	State       ResultState
	Payload     []byte
	Error       string
	Created     time.Time
	Metadata    map[string]string
}

// ResultCache stores terminal operation results until coordinator ack.
type ResultCache interface {
	Get(context.Context, string) (OperationResult, bool, error)
	Store(context.Context, OperationResult) error
	Acknowledge(context.Context, string) error
}

// MemoryResultCache is an in-memory ResultCache implementation.
type MemoryResultCache struct {
	mu      sync.Mutex
	results map[string]OperationResult
}

// NewMemoryResultCache returns an empty in-memory result cache.
func NewMemoryResultCache() *MemoryResultCache {
	return &MemoryResultCache{results: make(map[string]OperationResult)}
}

// Get returns a terminal operation result by operation ID.
func (c *MemoryResultCache) Get(ctx context.Context, operationID string) (OperationResult, bool, error) {
	if err := ctx.Err(); err != nil {
		return OperationResult{}, false, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	r, ok := c.results[operationID]
	if !ok {
		return OperationResult{}, false, nil
	}
	return cloneResult(r), true, nil
}

// Store records a terminal result. Storing the same operation ID again keeps
// the original result.
func (c *MemoryResultCache) Store(ctx context.Context, r OperationResult) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if r.Created.IsZero() {
		r.Created = time.Now()
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.results == nil {
		c.results = make(map[string]OperationResult)
	}
	if _, ok := c.results[r.OperationID]; ok {
		return nil
	}
	c.results[r.OperationID] = cloneResult(r)
	return nil
}

// Acknowledge evicts a cached result after coordinator acknowledgement.
func (c *MemoryResultCache) Acknowledge(ctx context.Context, operationID string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.results[operationID]; !ok {
		return ErrResultNotFound
	}
	delete(c.results, operationID)
	return nil
}

func cloneResult(r OperationResult) OperationResult {
	if r.Payload != nil {
		r.Payload = append([]byte(nil), r.Payload...)
	}
	if r.Metadata != nil {
		m := make(map[string]string, len(r.Metadata))
		for k, v := range r.Metadata {
			m[k] = v
		}
		r.Metadata = m
	}
	return r
}
