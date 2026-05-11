// Package tinkerdb stores coordinator state for localtinker.
package tinkerdb

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

var ErrNotFound = errors.New("not found")

type Store interface {
	CreateSession(context.Context, Session) error
	GetSession(context.Context, string) (Session, error)
	ListSessions(context.Context) ([]Session, error)
	UpdateSessionHeartbeat(context.Context, string, time.Time) error
	PutModel(context.Context, Model) error
	GetModel(context.Context, string) (Model, error)
	ListModels(context.Context) ([]Model, error)
	DeleteModel(context.Context, string) error
	PutNode(context.Context, Node) error
	GetNode(context.Context, string) (Node, error)
	ListNodes(context.Context) ([]Node, error)
	PutFuture(context.Context, Future) error
	GetFuture(context.Context, string) (Future, error)
	ListFutures(context.Context) ([]Future, error)
	ClaimNextFuture(context.Context, string, time.Time, time.Duration) (Future, bool, error)
	RenewFutureLease(context.Context, string, string, time.Time, time.Duration) (Future, bool, error)
	RequeueFuture(context.Context, string, string, string, time.Time, time.Time) (Future, bool, error)
	FinishFuture(context.Context, string, string, string, json.RawMessage, json.RawMessage, time.Time) (Future, bool, error)
	PutFutureAttempt(context.Context, FutureAttempt) error
	ListFutureAttempts(context.Context, string) ([]FutureAttempt, error)
	PutCheckpoint(context.Context, Checkpoint) error
	GetCheckpoint(context.Context, string) (Checkpoint, error)
	ListCheckpoints(context.Context) ([]Checkpoint, error)
	DeleteCheckpoint(context.Context, string) error
	Close() error
}

type Session struct {
	ID         string    `json:"id"`
	CreatedAt  time.Time `json:"created_at"`
	LastSeenAt time.Time `json:"last_seen_at"`
	HeartbeatN int64     `json:"heartbeat_n"`
}

type Future struct {
	ID          string          `json:"id"`
	State       string          `json:"state"`
	Result      json.RawMessage `json:"result_json,omitempty"`
	Error       json.RawMessage `json:"error_json,omitempty"`
	Metadata    json.RawMessage `json:"metadata_json,omitempty"`
	CreatedAt   time.Time       `json:"created_at"`
	StartedAt   time.Time       `json:"started_at,omitempty"`
	CompletedAt time.Time       `json:"completed_at,omitempty"`

	MetadataDelivered bool            `json:"metadata_delivered,omitempty"`
	Operation         string          `json:"operation,omitempty"`
	ModelID           string          `json:"model_id,omitempty"`
	RequestBytes      int64           `json:"request_bytes,omitempty"`
	ResultBytes       int64           `json:"result_bytes,omitempty"`
	LeaseID           string          `json:"lease_id,omitempty"`
	LeaseExpiresAt    time.Time       `json:"lease_expires_at,omitempty"`
	Attempt           int             `json:"attempt,omitempty"`
	MaxAttempts       int             `json:"max_attempts,omitempty"`
	AssignedNodeID    string          `json:"assigned_node_id,omitempty"`
	NextRunAt         time.Time       `json:"next_run_at,omitempty"`
	LastAttemptAt     time.Time       `json:"last_attempt_at,omitempty"`
	LastHeartbeatAt   time.Time       `json:"last_heartbeat_at,omitempty"`
	RetryReason       string          `json:"retry_reason,omitempty"`
	IdempotencyKey    string          `json:"idempotency_key,omitempty"`
	Requirements      json.RawMessage `json:"requirements_json,omitempty"`
	Priority          int             `json:"priority,omitempty"`
}

type FutureAttempt struct {
	FutureID   string          `json:"future_id"`
	Attempt    int             `json:"attempt"`
	NodeID     string          `json:"node_id,omitempty"`
	LeaseID    string          `json:"lease_id,omitempty"`
	State      string          `json:"state"`
	StartedAt  time.Time       `json:"started_at,omitempty"`
	FinishedAt time.Time       `json:"finished_at,omitempty"`
	Result     json.RawMessage `json:"result_json,omitempty"`
	Error      json.RawMessage `json:"error_json,omitempty"`
}

type Model struct {
	ID          string    `json:"id"`
	SessionID   string    `json:"session_id"`
	BaseModel   string    `json:"base_model"`
	TokenizerID string    `json:"tokenizer_id"`
	IsLoRA      bool      `json:"is_lora"`
	LoRARank    int       `json:"lora_rank"`
	CreatedAt   time.Time `json:"created_at"`
}

type Node struct {
	ID             string            `json:"id"`
	Name           string            `json:"name,omitempty"`
	SessionID      string            `json:"session_id,omitempty"`
	State          string            `json:"state"`
	Labels         map[string]string `json:"labels,omitempty"`
	Capabilities   json.RawMessage   `json:"capabilities_json,omitempty"`
	Load           json.RawMessage   `json:"load_json,omitempty"`
	MaxConcurrency int               `json:"max_concurrency,omitempty"`
	Running        int               `json:"running,omitempty"`
	StartedAt      time.Time         `json:"started_at,omitempty"`
	LastSeenAt     time.Time         `json:"last_seen_at,omitempty"`
	DrainingSince  time.Time         `json:"draining_since,omitempty"`
}

type Checkpoint struct {
	Path      string     `json:"path"`
	Public    bool       `json:"public"`
	Owner     string     `json:"owner,omitempty"`
	ExpiresAt *time.Time `json:"expires_at,omitempty"`
}

type JSONStore struct {
	mu   sync.Mutex
	path string
	data diskState
}

type diskState struct {
	Sessions    map[string]Session       `json:"sessions"`
	Models      map[string]Model         `json:"models"`
	Nodes       map[string]Node          `json:"nodes"`
	Futures     map[string]Future        `json:"futures"`
	Attempts    map[string]FutureAttempt `json:"attempts"`
	Checkpoints map[string]Checkpoint    `json:"checkpoints"`
}

func OpenJSON(path string) (*JSONStore, error) {
	s := &JSONStore{path: path}
	if err := s.load(); err != nil {
		return nil, err
	}
	return s, nil
}

func OpenMemory() *JSONStore {
	return &JSONStore{
		data: diskState{
			Sessions:    make(map[string]Session),
			Models:      make(map[string]Model),
			Nodes:       make(map[string]Node),
			Futures:     make(map[string]Future),
			Attempts:    make(map[string]FutureAttempt),
			Checkpoints: make(map[string]Checkpoint),
		},
	}
}

func (s *JSONStore) CreateSession(_ context.Context, session Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.init()
	s.data.Sessions[session.ID] = session
	return s.saveLocked()
}

func (s *JSONStore) GetSession(_ context.Context, id string) (Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	session, ok := s.data.Sessions[id]
	if !ok {
		return Session{}, ErrNotFound
	}
	return session, nil
}

func (s *JSONStore) ListSessions(_ context.Context) ([]Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sessions := make([]Session, 0, len(s.data.Sessions))
	for _, session := range s.data.Sessions {
		sessions = append(sessions, session)
	}
	return sessions, nil
}

func (s *JSONStore) UpdateSessionHeartbeat(_ context.Context, id string, at time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	session, ok := s.data.Sessions[id]
	if !ok {
		return ErrNotFound
	}
	session.LastSeenAt = at
	session.HeartbeatN++
	s.data.Sessions[id] = session
	return s.saveLocked()
}

func (s *JSONStore) PutModel(_ context.Context, model Model) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.init()
	s.data.Models[model.ID] = model
	return s.saveLocked()
}

func (s *JSONStore) GetModel(_ context.Context, id string) (Model, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	model, ok := s.data.Models[id]
	if !ok {
		return Model{}, ErrNotFound
	}
	return model, nil
}

func (s *JSONStore) ListModels(_ context.Context) ([]Model, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	models := make([]Model, 0, len(s.data.Models))
	for _, model := range s.data.Models {
		models = append(models, model)
	}
	return models, nil
}

func (s *JSONStore) DeleteModel(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.data.Models[id]; !ok {
		return ErrNotFound
	}
	delete(s.data.Models, id)
	return s.saveLocked()
}

func (s *JSONStore) PutNode(_ context.Context, node Node) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.init()
	s.data.Nodes[node.ID] = node
	return s.saveLocked()
}

func (s *JSONStore) GetNode(_ context.Context, id string) (Node, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	node, ok := s.data.Nodes[id]
	if !ok {
		return Node{}, ErrNotFound
	}
	return node, nil
}

func (s *JSONStore) ListNodes(_ context.Context) ([]Node, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	nodes := make([]Node, 0, len(s.data.Nodes))
	for _, node := range s.data.Nodes {
		nodes = append(nodes, node)
	}
	return nodes, nil
}

func (s *JSONStore) PutFuture(_ context.Context, future Future) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.init()
	s.data.Futures[future.ID] = future
	return s.saveLocked()
}

func (s *JSONStore) GetFuture(_ context.Context, id string) (Future, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	future, ok := s.data.Futures[id]
	if !ok {
		return Future{}, ErrNotFound
	}
	return future, nil
}

func (s *JSONStore) ListFutures(_ context.Context) ([]Future, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	futures := make([]Future, 0, len(s.data.Futures))
	for _, future := range s.data.Futures {
		futures = append(futures, future)
	}
	return futures, nil
}

func (s *JSONStore) ClaimNextFuture(_ context.Context, nodeID string, now time.Time, leaseTimeout time.Duration) (Future, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.init()
	now = now.UTC()
	eligible := make([]Future, 0, len(s.data.Futures))
	for _, future := range s.data.Futures {
		if future.State != "queued" {
			continue
		}
		if !future.NextRunAt.IsZero() && future.NextRunAt.After(now) {
			continue
		}
		eligible = append(eligible, future)
	}
	if len(eligible) == 0 {
		return Future{}, false, nil
	}
	sort.Slice(eligible, func(i, j int) bool {
		a, b := eligible[i], eligible[j]
		if a.Priority != b.Priority {
			return a.Priority > b.Priority
		}
		if !a.CreatedAt.Equal(b.CreatedAt) {
			return a.CreatedAt.Before(b.CreatedAt)
		}
		return a.ID < b.ID
	})
	future := eligible[0]
	future.State = "running"
	future.StartedAt = now
	future.Attempt++
	future.AssignedNodeID = nodeID
	future.LastAttemptAt = now
	future.LastHeartbeatAt = now
	future.LeaseID = fmt.Sprintf("%s-attempt-%d", future.ID, future.Attempt)
	if leaseTimeout > 0 {
		future.LeaseExpiresAt = now.Add(leaseTimeout)
	} else {
		future.LeaseExpiresAt = time.Time{}
	}
	s.data.Futures[future.ID] = future
	if err := s.saveLocked(); err != nil {
		return Future{}, false, err
	}
	return future, true, nil
}

func (s *JSONStore) RenewFutureLease(_ context.Context, id, leaseID string, now time.Time, leaseTimeout time.Duration) (Future, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.init()
	future, ok := s.data.Futures[id]
	if !ok {
		return Future{}, false, ErrNotFound
	}
	if future.State != "running" || future.LeaseID != leaseID {
		return future, false, nil
	}
	now = now.UTC()
	future.LastHeartbeatAt = now
	if leaseTimeout > 0 {
		future.LeaseExpiresAt = now.Add(leaseTimeout)
	}
	s.data.Futures[future.ID] = future
	if err := s.saveLocked(); err != nil {
		return Future{}, false, err
	}
	return future, true, nil
}

func (s *JSONStore) RequeueFuture(_ context.Context, id, leaseID, reason string, nextRunAt, now time.Time) (Future, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.init()
	future, ok := s.data.Futures[id]
	if !ok {
		return Future{}, false, ErrNotFound
	}
	if future.State != "running" || future.LeaseID != leaseID {
		return future, false, nil
	}
	now = now.UTC()
	attempt := FutureAttempt{
		FutureID:   future.ID,
		Attempt:    future.Attempt,
		NodeID:     future.AssignedNodeID,
		LeaseID:    future.LeaseID,
		State:      "lost",
		StartedAt:  future.StartedAt,
		FinishedAt: now,
		Error: json.RawMessage(fmt.Sprintf(
			`{"code":"system_error","message":%q}`,
			reason,
		)),
	}
	future.State = "queued"
	future.StartedAt = time.Time{}
	future.AssignedNodeID = ""
	future.LeaseID = ""
	future.LeaseExpiresAt = time.Time{}
	future.LastHeartbeatAt = time.Time{}
	future.NextRunAt = nextRunAt.UTC()
	future.RetryReason = reason
	s.data.Futures[future.ID] = future
	s.data.Attempts[futureAttemptKey(attempt.FutureID, attempt.Attempt)] = attempt
	if err := s.saveLocked(); err != nil {
		return Future{}, false, err
	}
	return future, true, nil
}

func (s *JSONStore) FinishFuture(_ context.Context, id, leaseID, state string, result, errPayload json.RawMessage, now time.Time) (Future, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.init()
	future, ok := s.data.Futures[id]
	if !ok {
		return Future{}, false, ErrNotFound
	}
	if future.State != "running" || future.LeaseID != leaseID {
		return future, false, nil
	}
	now = now.UTC()
	attempt := FutureAttempt{
		FutureID:   future.ID,
		Attempt:    future.Attempt,
		NodeID:     future.AssignedNodeID,
		LeaseID:    future.LeaseID,
		State:      state,
		StartedAt:  future.StartedAt,
		FinishedAt: now,
		Result:     cloneRaw(result),
		Error:      cloneRaw(errPayload),
	}
	future.State = state
	future.CompletedAt = now
	future.AssignedNodeID = ""
	future.LeaseID = ""
	future.LeaseExpiresAt = time.Time{}
	future.LastHeartbeatAt = time.Time{}
	future.Result = cloneRaw(result)
	future.Error = cloneRaw(errPayload)
	future.ResultBytes = int64(len(future.Result))
	if future.ResultBytes == 0 {
		future.ResultBytes = int64(len(future.Error))
	}
	s.data.Futures[future.ID] = future
	s.data.Attempts[futureAttemptKey(attempt.FutureID, attempt.Attempt)] = attempt
	if err := s.saveLocked(); err != nil {
		return Future{}, false, err
	}
	return future, true, nil
}

func (s *JSONStore) PutFutureAttempt(_ context.Context, attempt FutureAttempt) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.init()
	s.data.Attempts[futureAttemptKey(attempt.FutureID, attempt.Attempt)] = attempt
	return s.saveLocked()
}

func (s *JSONStore) ListFutureAttempts(_ context.Context, futureID string) ([]FutureAttempt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	attempts := make([]FutureAttempt, 0, len(s.data.Attempts))
	for _, attempt := range s.data.Attempts {
		if futureID != "" && attempt.FutureID != futureID {
			continue
		}
		attempts = append(attempts, attempt)
	}
	sort.Slice(attempts, func(i, j int) bool {
		if attempts[i].FutureID != attempts[j].FutureID {
			return attempts[i].FutureID < attempts[j].FutureID
		}
		return attempts[i].Attempt < attempts[j].Attempt
	})
	return attempts, nil
}

func (s *JSONStore) PutCheckpoint(_ context.Context, checkpoint Checkpoint) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.init()
	s.data.Checkpoints[checkpoint.Path] = checkpoint
	return s.saveLocked()
}

func (s *JSONStore) GetCheckpoint(_ context.Context, path string) (Checkpoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	checkpoint, ok := s.data.Checkpoints[path]
	if !ok {
		return Checkpoint{}, ErrNotFound
	}
	return checkpoint, nil
}

func (s *JSONStore) ListCheckpoints(_ context.Context) ([]Checkpoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	checkpoints := make([]Checkpoint, 0, len(s.data.Checkpoints))
	for _, checkpoint := range s.data.Checkpoints {
		checkpoints = append(checkpoints, checkpoint)
	}
	return checkpoints, nil
}

func (s *JSONStore) DeleteCheckpoint(_ context.Context, path string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.data.Checkpoints, path)
	return s.saveLocked()
}

func (s *JSONStore) Close() error { return nil }

func (s *JSONStore) load() error {
	s.init()
	if s.path == "" {
		return nil
	}
	b, err := os.ReadFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read tinker db: %w", err)
	}
	if len(b) == 0 {
		return nil
	}
	if err := json.Unmarshal(b, &s.data); err != nil {
		return fmt.Errorf("decode tinker db: %w", err)
	}
	s.init()
	return nil
}

func (s *JSONStore) saveLocked() error {
	if s.path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return fmt.Errorf("create tinker db dir: %w", err)
	}
	b, err := json.MarshalIndent(s.data, "", "\t")
	if err != nil {
		return fmt.Errorf("encode tinker db: %w", err)
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return fmt.Errorf("write tinker db: %w", err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		return fmt.Errorf("replace tinker db: %w", err)
	}
	return nil
}

func (s *JSONStore) init() {
	if s.data.Sessions == nil {
		s.data.Sessions = make(map[string]Session)
	}
	if s.data.Models == nil {
		s.data.Models = make(map[string]Model)
	}
	if s.data.Nodes == nil {
		s.data.Nodes = make(map[string]Node)
	}
	if s.data.Futures == nil {
		s.data.Futures = make(map[string]Future)
	}
	if s.data.Attempts == nil {
		s.data.Attempts = make(map[string]FutureAttempt)
	}
	if s.data.Checkpoints == nil {
		s.data.Checkpoints = make(map[string]Checkpoint)
	}
}

func futureAttemptKey(futureID string, attempt int) string {
	return fmt.Sprintf("%s/%d", futureID, attempt)
}

func cloneRaw(raw json.RawMessage) json.RawMessage {
	if raw == nil {
		return nil
	}
	return append(json.RawMessage(nil), raw...)
}
