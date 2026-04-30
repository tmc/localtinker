// Package tinkerdb stores coordinator state for localtinker.
package tinkerdb

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
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
	PutFuture(context.Context, Future) error
	GetFuture(context.Context, string) (Future, error)
	ListFutures(context.Context) ([]Future, error)
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
	CompletedAt time.Time       `json:"completed_at,omitempty"`

	MetadataDelivered bool `json:"metadata_delivered,omitempty"`
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

type JSONStore struct {
	mu   sync.Mutex
	path string
	data diskState
}

type diskState struct {
	Sessions map[string]Session `json:"sessions"`
	Models   map[string]Model   `json:"models"`
	Futures  map[string]Future  `json:"futures"`
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
			Sessions: make(map[string]Session),
			Models:   make(map[string]Model),
			Futures:  make(map[string]Future),
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
	if s.data.Futures == nil {
		s.data.Futures = make(map[string]Future)
	}
}
