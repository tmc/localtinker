// Package tinkercoord coordinates localtinker sessions and futures.
package tinkercoord

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/tmc/localtinker/internal/tinkerdb"
	"github.com/tmc/localtinker/internal/tinkertrain"
)

const (
	FutureQueued           = "queued"
	FutureRunning          = "running"
	FutureComplete         = "complete"
	FutureCompleteMetadata = "complete_metadata"
	FutureTryAgain         = "try_again"
	FutureUserError        = "user_error"
	FutureSystemError      = "system_error"
	FutureCanceled         = "canceled"
)

type Coordinator struct {
	store tinkerdb.Store
	train *tinkertrain.Manager
	now   func() time.Time
}

type Config struct {
	Store tinkerdb.Store
	Train *tinkertrain.Manager
	Now   func() time.Time
}

type Session struct {
	ID         string    `json:"session_id"`
	CreatedAt  time.Time `json:"created_at"`
	LastSeenAt time.Time `json:"last_seen_at"`
	HeartbeatN int64     `json:"heartbeat_n"`
}

type ClientConfig struct {
	UseJWT               bool `json:"use_jwt"`
	ParallelFWDBWDChunks bool `json:"parallel_fwdbwd_chunks"`
	MaxRequestBytes      int  `json:"max_request_bytes"`
}

type Capability struct {
	ModelID       string   `json:"model_id"`
	ContextLength int      `json:"context_length"`
	TokenizerID   string   `json:"tokenizer_id"`
	Supported     []string `json:"supported,omitempty"`
}

type CreateModelRequest struct {
	SessionID  string         `json:"session_id"`
	BaseModel  string         `json:"base_model"`
	LoRAConfig map[string]any `json:"lora_config"`
}

type ModelInfo struct {
	ID          string
	BaseModel   string
	TokenizerID string
	IsLoRA      bool
	LoRARank    int
}

type ServerCapabilities struct {
	Models []Capability `json:"models"`
}

type Cursor struct {
	Offset     int `json:"offset"`
	Limit      int `json:"limit"`
	TotalCount int `json:"total_count"`
}

type TrainingRun struct {
	TrainingRunID         string            `json:"training_run_id"`
	BaseModel             string            `json:"base_model"`
	ModelOwner            string            `json:"model_owner"`
	IsLoRA                bool              `json:"is_lora"`
	Corrupted             bool              `json:"corrupted"`
	LoRARank              int               `json:"lora_rank,omitempty"`
	LastRequestTime       time.Time         `json:"last_request_time"`
	LastCheckpoint        *Checkpoint       `json:"last_checkpoint,omitempty"`
	LastSamplerCheckpoint *Checkpoint       `json:"last_sampler_checkpoint,omitempty"`
	UserMetadata          map[string]string `json:"user_metadata,omitempty"`
}

type TrainingRunsResponse struct {
	TrainingRuns []TrainingRun `json:"training_runs"`
	Cursor       Cursor        `json:"cursor"`
}

type Checkpoint struct {
	CheckpointID   string     `json:"checkpoint_id"`
	CheckpointType string     `json:"checkpoint_type"`
	Time           time.Time  `json:"time"`
	TinkerPath     string     `json:"tinker_path"`
	SizeBytes      *int64     `json:"size_bytes,omitempty"`
	Public         bool       `json:"public"`
	ExpiresAt      *time.Time `json:"expires_at,omitempty"`
}

type CheckpointsResponse struct {
	Checkpoints []Checkpoint `json:"checkpoints"`
	Cursor      *Cursor      `json:"cursor,omitempty"`
}

type Future struct {
	ID       string          `json:"future_id"`
	State    string          `json:"state"`
	Result   json.RawMessage `json:"result,omitempty"`
	Error    json.RawMessage `json:"error,omitempty"`
	Metadata json.RawMessage `json:"metadata,omitempty"`
}

type DashboardSnapshot struct {
	GeneratedAt  time.Time          `json:"generated_at"`
	ClientConfig ClientConfig       `json:"client_config"`
	Capabilities ServerCapabilities `json:"capabilities"`
	Sessions     []Session          `json:"sessions"`
	Models       []DashboardModel   `json:"models"`
	Futures      []DashboardFuture  `json:"futures"`
}

type DashboardModel struct {
	ID          string    `json:"id"`
	SessionID   string    `json:"session_id"`
	BaseModel   string    `json:"base_model"`
	TokenizerID string    `json:"tokenizer_id"`
	IsLoRA      bool      `json:"is_lora"`
	LoRARank    int       `json:"lora_rank"`
	CreatedAt   time.Time `json:"created_at"`
}

type DashboardFuture struct {
	ID          string    `json:"id"`
	State       string    `json:"state"`
	Operation   string    `json:"operation,omitempty"`
	ModelID     string    `json:"model_id,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	CompletedAt time.Time `json:"completed_at,omitempty"`
	ResultBytes int       `json:"result_bytes"`
	ErrorBytes  int       `json:"error_bytes"`
	Metrics     MetricMap `json:"metrics,omitempty"`
	Error       string    `json:"error,omitempty"`
}

type MetricMap map[string]float64

func New(cfg Config) (*Coordinator, error) {
	if cfg.Store == nil {
		return nil, errors.New("nil store")
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.Train == nil {
		cfg.Train = tinkertrain.NewManager()
	}
	return &Coordinator{store: cfg.Store, train: cfg.Train, now: cfg.Now}, nil
}

func (c *Coordinator) ClientConfig(_ context.Context) ClientConfig {
	return ClientConfig{
		UseJWT:               false,
		ParallelFWDBWDChunks: false,
		MaxRequestBytes:      128 << 20,
	}
}

func (c *Coordinator) CreateSession(ctx context.Context) (Session, error) {
	now := c.now().UTC()
	id, err := newID("sess")
	if err != nil {
		return Session{}, err
	}
	session := tinkerdb.Session{
		ID:         id,
		CreatedAt:  now,
		LastSeenAt: now,
	}
	if err := c.store.CreateSession(ctx, session); err != nil {
		return Session{}, err
	}
	return fromDBSession(session), nil
}

func (c *Coordinator) Heartbeat(ctx context.Context, sessionID string) (Session, error) {
	now := c.now().UTC()
	if err := c.store.UpdateSessionHeartbeat(ctx, sessionID, now); err != nil {
		return Session{}, err
	}
	session, err := c.store.GetSession(ctx, sessionID)
	if err != nil {
		return Session{}, err
	}
	return fromDBSession(session), nil
}

func (c *Coordinator) Capabilities(_ context.Context) ServerCapabilities {
	return ServerCapabilities{
		Models: []Capability{{
			ModelID:       "Qwen/Qwen3-8B",
			ContextLength: 32768,
			TokenizerID:   "Qwen/Qwen3-8B",
			Supported:     []string{"handshake", "future", "cross_entropy", "adamw", "save_weights_for_sampler", "sample"},
		}},
	}
}

func (c *Coordinator) DashboardSnapshot(ctx context.Context) (DashboardSnapshot, error) {
	sessions, err := c.store.ListSessions(ctx)
	if err != nil {
		return DashboardSnapshot{}, err
	}
	models, err := c.store.ListModels(ctx)
	if err != nil {
		return DashboardSnapshot{}, err
	}
	futures, err := c.store.ListFutures(ctx)
	if err != nil {
		return DashboardSnapshot{}, err
	}
	outSessions := make([]Session, 0, len(sessions))
	for _, session := range sessions {
		outSessions = append(outSessions, fromDBSession(session))
	}
	sort.Slice(outSessions, func(i, j int) bool {
		return outSessions[i].CreatedAt.After(outSessions[j].CreatedAt)
	})
	outModels := make([]DashboardModel, 0, len(models))
	for _, model := range models {
		outModels = append(outModels, DashboardModel{
			ID:          model.ID,
			SessionID:   model.SessionID,
			BaseModel:   model.BaseModel,
			TokenizerID: model.TokenizerID,
			IsLoRA:      model.IsLoRA,
			LoRARank:    model.LoRARank,
			CreatedAt:   model.CreatedAt,
		})
	}
	sort.Slice(outModels, func(i, j int) bool {
		return outModels[i].CreatedAt.After(outModels[j].CreatedAt)
	})
	outFutures := make([]DashboardFuture, 0, len(futures))
	for _, future := range futures {
		outFutures = append(outFutures, dashboardFuture(future))
	}
	sort.Slice(outFutures, func(i, j int) bool {
		return outFutures[i].CreatedAt.After(outFutures[j].CreatedAt)
	})
	return DashboardSnapshot{
		GeneratedAt:  c.now().UTC(),
		ClientConfig: c.ClientConfig(ctx),
		Capabilities: c.Capabilities(ctx),
		Sessions:     outSessions,
		Models:       outModels,
		Futures:      outFutures,
	}, nil
}

func (c *Coordinator) TrainingRuns(ctx context.Context, limit, offset int) (TrainingRunsResponse, error) {
	if limit <= 0 {
		limit = 20
	}
	if offset < 0 {
		offset = 0
	}
	models, err := c.store.ListModels(ctx)
	if err != nil {
		return TrainingRunsResponse{}, err
	}
	checkpoints, err := c.checkpointRecords(ctx)
	if err != nil {
		return TrainingRunsResponse{}, err
	}
	futures, err := c.store.ListFutures(ctx)
	if err != nil {
		return TrainingRunsResponse{}, err
	}
	last := make(map[string]time.Time)
	for _, future := range futures {
		var meta struct {
			ModelID string `json:"model_id"`
		}
		_ = json.Unmarshal(future.Metadata, &meta)
		if meta.ModelID == "" {
			continue
		}
		at := future.CompletedAt
		if at.IsZero() {
			at = future.CreatedAt
		}
		if at.After(last[meta.ModelID]) {
			last[meta.ModelID] = at
		}
	}
	runs := make([]TrainingRun, 0, len(models))
	for _, model := range models {
		run := TrainingRun{
			TrainingRunID:   model.ID,
			BaseModel:       model.BaseModel,
			ModelOwner:      "local",
			IsLoRA:          model.IsLoRA,
			Corrupted:       false,
			LoRARank:        model.LoRARank,
			LastRequestTime: model.CreatedAt,
		}
		if t := last[model.ID]; !t.IsZero() {
			run.LastRequestTime = t
		}
		for i := range checkpoints {
			ckpt := checkpoints[i]
			parsed, err := tinkertrain.ParseTinkerPath(ckpt.TinkerPath)
			if err != nil || parsed.ModelID != model.ID {
				continue
			}
			if ckpt.CheckpointType == "training" && (run.LastCheckpoint == nil || ckpt.Time.After(run.LastCheckpoint.Time)) {
				copy := ckpt
				run.LastCheckpoint = &copy
			}
			if ckpt.CheckpointType == "sampler" && (run.LastSamplerCheckpoint == nil || ckpt.Time.After(run.LastSamplerCheckpoint.Time)) {
				copy := ckpt
				run.LastSamplerCheckpoint = &copy
			}
		}
		runs = append(runs, run)
	}
	sort.Slice(runs, func(i, j int) bool {
		return runs[i].LastRequestTime.After(runs[j].LastRequestTime)
	})
	total := len(runs)
	if offset > total {
		offset = total
	}
	end := offset + limit
	if end > total {
		end = total
	}
	return TrainingRunsResponse{
		TrainingRuns: runs[offset:end],
		Cursor: Cursor{
			Offset:     offset,
			Limit:      limit,
			TotalCount: total,
		},
	}, nil
}

func (c *Coordinator) Checkpoints(ctx context.Context, trainingRunID string, limit, offset int) (CheckpointsResponse, error) {
	checkpoints, err := c.checkpointRecords(ctx)
	if err != nil {
		return CheckpointsResponse{}, err
	}
	filtered := checkpoints[:0]
	for _, checkpoint := range checkpoints {
		if trainingRunID != "" {
			parsed, err := tinkertrain.ParseTinkerPath(checkpoint.TinkerPath)
			if err != nil || parsed.ModelID != trainingRunID {
				continue
			}
		}
		filtered = append(filtered, checkpoint)
	}
	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].Time.After(filtered[j].Time)
	})
	if trainingRunID != "" {
		return CheckpointsResponse{Checkpoints: filtered}, nil
	}
	if limit <= 0 {
		limit = 100
	}
	if offset < 0 {
		offset = 0
	}
	total := len(filtered)
	if offset > total {
		offset = total
	}
	end := offset + limit
	if end > total {
		end = total
	}
	return CheckpointsResponse{
		Checkpoints: filtered[offset:end],
		Cursor: &Cursor{
			Offset:     offset,
			Limit:      limit,
			TotalCount: total,
		},
	}, nil
}

func (c *Coordinator) CreateModel(ctx context.Context, req CreateModelRequest) (Future, ModelInfo, error) {
	now := c.now().UTC()
	id, err := newID("model")
	if err != nil {
		return Future{}, ModelInfo{}, err
	}
	base := req.BaseModel
	if base == "" {
		base = "Qwen/Qwen3-8B"
	}
	rank := intFromMap(req.LoRAConfig, "rank")
	model := tinkerdb.Model{
		ID:          id,
		SessionID:   req.SessionID,
		BaseModel:   base,
		TokenizerID: base,
		IsLoRA:      req.LoRAConfig != nil,
		LoRARank:    rank,
		CreatedAt:   now,
	}
	if err := c.train.Create(ctx, id, tinkertrain.CreateConfig{
		BaseModel: base,
		LoRARank:  rank,
	}); err != nil {
		return Future{}, ModelInfo{}, err
	}
	if err := c.store.PutModel(ctx, model); err != nil {
		c.train.Delete(ctx, id)
		return Future{}, ModelInfo{}, err
	}
	future, err := c.CompleteFuture(ctx, map[string]any{
		"type":     "create_model",
		"model_id": id,
	}, nil)
	if err != nil {
		return Future{}, ModelInfo{}, err
	}
	return future, fromDBModel(model), nil
}

func (c *Coordinator) GetModel(ctx context.Context, id string) (ModelInfo, error) {
	model, err := c.store.GetModel(ctx, id)
	if err != nil {
		return ModelInfo{}, err
	}
	return fromDBModel(model), nil
}

func (c *Coordinator) UnloadModel(ctx context.Context, id string) (Future, error) {
	if err := c.store.DeleteModel(ctx, id); err != nil {
		return Future{}, err
	}
	c.train.Delete(ctx, id)
	return c.CompleteFuture(ctx, map[string]any{
		"type":     "unload_model",
		"model_id": id,
	}, nil)
}

func (c *Coordinator) Forward(ctx context.Context, req tinkertrain.Request) (Future, error) {
	out, err := c.train.Forward(ctx, req)
	if err != nil {
		return c.UserErrorFuture(ctx, err.Error())
	}
	return c.CompleteFuture(ctx, out, map[string]any{
		"type":     "forward",
		"model_id": req.ModelID,
		"loss_fn":  req.Input.LossFn,
	})
}

func (c *Coordinator) ForwardBackward(ctx context.Context, req tinkertrain.Request) (Future, error) {
	out, err := c.train.ForwardBackward(ctx, req)
	if err != nil {
		return c.UserErrorFuture(ctx, err.Error())
	}
	return c.CompleteFuture(ctx, out, map[string]any{
		"type":     "forward_backward",
		"model_id": req.ModelID,
		"loss_fn":  req.Input.LossFn,
	})
}

func (c *Coordinator) OptimStep(ctx context.Context, modelID string, params tinkertrain.AdamParams) (Future, error) {
	out, err := c.train.OptimStep(ctx, modelID, params)
	if err != nil {
		return c.UserErrorFuture(ctx, err.Error())
	}
	return c.CompleteFuture(ctx, out, map[string]any{
		"type":          "optim_step",
		"model_id":      modelID,
		"learning_rate": params.LearningRate,
	})
}

func (c *Coordinator) SaveWeightsForSampler(ctx context.Context, modelID, path string, samplingSessionSeqID int) (Future, error) {
	name := path
	if name == "" {
		name = fmt.Sprintf("ephemeral-%d", samplingSessionSeqID)
	}
	modelPath, err := c.train.SaveForSampler(ctx, modelID, name)
	if err != nil {
		return c.UserErrorFuture(ctx, err.Error())
	}
	result := map[string]any{
		"type": "save_weights_for_sampler",
	}
	if path == "" {
		sessionID := samplingSessionID(modelID, samplingSessionSeqID)
		if err := c.train.CreateSamplingSession(ctx, sessionID, modelPath, ""); err != nil {
			return c.UserErrorFuture(ctx, err.Error())
		}
		result["sampling_session_id"] = sessionID
	} else {
		result["path"] = modelPath
	}
	return c.CompleteFuture(ctx, result, map[string]any{
		"type":     "save_weights_for_sampler",
		"model_id": modelID,
		"path":     modelPath,
	})
}

func (c *Coordinator) SaveWeights(ctx context.Context, modelID, path string) (Future, error) {
	if path == "" {
		path = "checkpoint"
	}
	modelPath, err := c.train.SaveState(ctx, modelID, path)
	if err != nil {
		return c.UserErrorFuture(ctx, err.Error())
	}
	return c.CompleteFuture(ctx, map[string]any{
		"type": "save_weights",
		"path": modelPath,
	}, map[string]any{
		"type":     "save_weights",
		"model_id": modelID,
		"path":     modelPath,
	})
}

func (c *Coordinator) LoadWeights(ctx context.Context, modelID, path string) (Future, error) {
	if err := c.train.LoadState(ctx, modelID, path); err != nil {
		return c.UserErrorFuture(ctx, err.Error())
	}
	return c.CompleteFuture(ctx, map[string]any{
		"type": "load_weights",
		"path": path,
	}, map[string]any{
		"type":     "load_weights",
		"model_id": modelID,
		"path":     path,
	})
}

func (c *Coordinator) CreateSamplingSession(ctx context.Context, sessionID string, seq int, modelPath, baseModel string) (string, error) {
	id := samplingSessionID(sessionID, seq)
	if err := c.train.CreateSamplingSession(ctx, id, modelPath, baseModel); err != nil {
		return "", err
	}
	return id, nil
}

func (c *Coordinator) Sample(ctx context.Context, req tinkertrain.SampleRequest) (Future, error) {
	out, err := c.train.Sample(ctx, req)
	if err != nil {
		return c.UserErrorFuture(ctx, err.Error())
	}
	return c.CompleteFuture(ctx, out, map[string]any{
		"type":                "sample",
		"sampling_session_id": req.SamplingSessionID,
	})
}

func (c *Coordinator) checkpointRecords(ctx context.Context) ([]Checkpoint, error) {
	futures, err := c.store.ListFutures(ctx)
	if err != nil {
		return nil, err
	}
	seen := make(map[string]bool)
	var checkpoints []Checkpoint
	for _, future := range futures {
		path := futurePath(future.Result)
		if path == "" {
			path = futurePath(future.Metadata)
		}
		if path == "" || seen[path] || !tinkertrain.CheckpointPathExists(path) {
			continue
		}
		parsed, err := tinkertrain.ParseTinkerPath(path)
		if err != nil {
			continue
		}
		typ := "training"
		if parsed.Kind == "sampler_weights" {
			typ = "sampler"
		}
		at := future.CompletedAt
		if at.IsZero() {
			at = future.CreatedAt
		}
		checkpoints = append(checkpoints, Checkpoint{
			CheckpointID:   parsed.Kind + "/" + parsed.Name,
			CheckpointType: typ,
			Time:           at,
			TinkerPath:     path,
			Public:         false,
		})
		seen[path] = true
	}
	return checkpoints, nil
}

func (c *Coordinator) AcceptTelemetry(context.Context, json.RawMessage) error {
	return nil
}

func (c *Coordinator) CompleteFuture(ctx context.Context, result any, metadata any) (Future, error) {
	now := c.now().UTC()
	id, err := newID("fut")
	if err != nil {
		return Future{}, err
	}
	resultJSON, err := marshalRaw(result)
	if err != nil {
		return Future{}, err
	}
	metadataJSON, err := marshalRaw(metadata)
	if err != nil {
		return Future{}, err
	}
	future := tinkerdb.Future{
		ID:          id,
		State:       FutureComplete,
		Result:      resultJSON,
		Metadata:    metadataJSON,
		CreatedAt:   now,
		CompletedAt: now,
	}
	if err := c.store.PutFuture(ctx, future); err != nil {
		return Future{}, err
	}
	return fromDBFuture(future), nil
}

func (c *Coordinator) UserErrorFuture(ctx context.Context, message string) (Future, error) {
	now := c.now().UTC()
	id, err := newID("fut")
	if err != nil {
		return Future{}, err
	}
	errJSON, err := marshalRaw(map[string]any{
		"code":    "user_error",
		"message": message,
	})
	if err != nil {
		return Future{}, err
	}
	future := tinkerdb.Future{
		ID:          id,
		State:       FutureUserError,
		Error:       errJSON,
		CreatedAt:   now,
		CompletedAt: now,
	}
	if err := c.store.PutFuture(ctx, future); err != nil {
		return Future{}, err
	}
	return fromDBFuture(future), nil
}

func (c *Coordinator) RetrieveFuture(ctx context.Context, id string, allowMetadataOnly bool) (Future, error) {
	future, err := c.store.GetFuture(ctx, id)
	if err != nil {
		return Future{}, err
	}
	if allowMetadataOnly && future.State == FutureComplete && len(future.Metadata) > 0 && !future.MetadataDelivered {
		future.State = FutureCompleteMetadata
		future.MetadataDelivered = true
		if err := c.store.PutFuture(ctx, future); err != nil {
			return Future{}, err
		}
		return fromDBFuture(future), nil
	}
	if future.State == FutureCompleteMetadata {
		future.State = FutureComplete
		if err := c.store.PutFuture(ctx, future); err != nil {
			return Future{}, err
		}
	}
	out := fromDBFuture(future)
	if out.State == FutureQueued || out.State == FutureRunning {
		out.State = FutureTryAgain
	}
	return out, nil
}

func dashboardFuture(future tinkerdb.Future) DashboardFuture {
	out := DashboardFuture{
		ID:          future.ID,
		State:       future.State,
		CreatedAt:   future.CreatedAt,
		CompletedAt: future.CompletedAt,
		ResultBytes: len(future.Result),
		ErrorBytes:  len(future.Error),
	}
	var meta struct {
		Type    string `json:"type"`
		ModelID string `json:"model_id"`
	}
	_ = json.Unmarshal(future.Metadata, &meta)
	out.Operation = meta.Type
	out.ModelID = meta.ModelID

	var result struct {
		Type    string    `json:"type"`
		ModelID string    `json:"model_id"`
		Metrics MetricMap `json:"metrics"`
	}
	_ = json.Unmarshal(future.Result, &result)
	if out.Operation == "" {
		out.Operation = result.Type
	}
	if out.ModelID == "" {
		out.ModelID = result.ModelID
	}
	if len(result.Metrics) > 0 {
		out.Metrics = result.Metrics
		if out.Operation == "" {
			out.Operation = inferOperation(result.Metrics)
		}
	}

	var errPayload struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	}
	_ = json.Unmarshal(future.Error, &errPayload)
	if errPayload.Message != "" {
		out.Error = errPayload.Message
	} else if errPayload.Code != "" {
		out.Error = errPayload.Code
	}
	return out
}

func inferOperation(metrics MetricMap) string {
	if _, ok := metrics["optimizer_step:unique"]; ok {
		return "optim_step"
	}
	if _, ok := metrics["loss:mean"]; ok {
		return "forward"
	}
	return ""
}

func fromDBSession(session tinkerdb.Session) Session {
	return Session{
		ID:         session.ID,
		CreatedAt:  session.CreatedAt,
		LastSeenAt: session.LastSeenAt,
		HeartbeatN: session.HeartbeatN,
	}
}

func fromDBFuture(future tinkerdb.Future) Future {
	return Future{
		ID:       future.ID,
		State:    future.State,
		Result:   future.Result,
		Error:    future.Error,
		Metadata: future.Metadata,
	}
}

func fromDBModel(model tinkerdb.Model) ModelInfo {
	return ModelInfo{
		ID:          model.ID,
		BaseModel:   model.BaseModel,
		TokenizerID: model.TokenizerID,
		IsLoRA:      model.IsLoRA,
		LoRARank:    model.LoRARank,
	}
}

func intFromMap(m map[string]any, key string) int {
	switch v := m[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return 0
	}
}

func futurePath(raw json.RawMessage) string {
	var body struct {
		Path string `json:"path"`
	}
	_ = json.Unmarshal(raw, &body)
	return body.Path
}

func marshalRaw(v any) (json.RawMessage, error) {
	if v == nil {
		return nil, nil
	}
	if raw, ok := v.(json.RawMessage); ok {
		return raw, nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("encode future payload: %w", err)
	}
	return b, nil
}

func newID(prefix string) (string, error) {
	const n = 16
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("create id: %w", err)
	}
	return prefix + "_" + hex.EncodeToString(b), nil
}

func samplingSessionID(prefix string, seq int) string {
	if seq == 0 {
		return prefix + ":sample"
	}
	return fmt.Sprintf("%s:sample:%d", prefix, seq)
}
