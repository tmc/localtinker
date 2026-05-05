// Package tinkerhttp serves the Python SDK compatible HTTP routes.
package tinkerhttp

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerdb"
	"github.com/tmc/localtinker/internal/tinkertrain"
)

type Server struct {
	coord *tinkercoord.Coordinator
}

func New(coord *tinkercoord.Coordinator) *Server {
	return &Server{coord: coord}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/v1/client/config", s.clientConfig)
	mux.HandleFunc("GET /api/v1/client/config", s.clientConfig)
	mux.HandleFunc("GET /api/v1/healthz", s.healthz)
	mux.HandleFunc("POST /api/v1/auth/token", s.authToken)
	mux.HandleFunc("POST /api/v1/create_session", s.createSession)
	mux.HandleFunc("POST /api/v1/session_heartbeat", s.sessionHeartbeat)
	mux.HandleFunc("POST /api/v1/get_server_capabilities", s.getServerCapabilities)
	mux.HandleFunc("GET /api/v1/get_server_capabilities", s.getServerCapabilities)
	mux.HandleFunc("POST /api/v1/telemetry", s.telemetry)
	mux.HandleFunc("POST /api/v1/retrieve_future", s.retrieveFuture)
	mux.HandleFunc("POST /api/v1/cancel_future", s.cancelFuture)
	mux.HandleFunc("POST /api/v1/create_model", s.createModel)
	mux.HandleFunc("POST /api/v1/unload_model", s.unloadModel)
	mux.HandleFunc("POST /api/v1/forward", s.forward)
	mux.HandleFunc("POST /api/v1/forward_backward", s.forwardBackward)
	mux.HandleFunc("POST /api/v1/optim_step", s.optimStep)
	mux.HandleFunc("POST /api/v1/save_weights", s.saveWeights)
	mux.HandleFunc("POST /api/v1/load_weights", s.loadWeights)
	mux.HandleFunc("POST /api/v1/load_state_with_optimizer", s.loadStateWithOptimizer)
	mux.HandleFunc("POST /api/v1/save_weights_for_sampler", s.saveWeightsForSampler)
	mux.HandleFunc("POST /api/v1/create_sampling_session", s.createSamplingSession)
	mux.HandleFunc("POST /api/v1/asample", s.asample)
	mux.HandleFunc("POST /api/v1/get_info", s.getInfo)
	mux.HandleFunc("POST /api/v1/weights_info", s.weightsInfo)
	mux.HandleFunc("GET /api/v1/training_runs", s.trainingRuns)
	mux.HandleFunc("GET /api/v1/checkpoints", s.checkpoints)
	mux.HandleFunc("GET /api/v1/sessions", s.sessions)
	mux.HandleFunc("GET /api/v1/sessions/", s.sessionPath)
	mux.HandleFunc("GET /api/v1/training_runs/", s.trainingRunPath)
	mux.HandleFunc("POST /api/v1/training_runs/", s.trainingRunPath)
	mux.HandleFunc("PUT /api/v1/training_runs/", s.trainingRunPath)
	mux.HandleFunc("DELETE /api/v1/training_runs/", s.trainingRunPath)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if limitsRequestBody(r) {
			cfg := s.coord.ClientConfig(r.Context())
			r.Body = http.MaxBytesReader(w, r.Body, int64(cfg.MaxRequestBytes))
		}
		mux.ServeHTTP(w, r)
	})
}

func limitsRequestBody(r *http.Request) bool {
	if !strings.HasPrefix(r.URL.Path, "/api/v1/") {
		return false
	}
	switch r.Method {
	case http.MethodPost, http.MethodPut, http.MethodPatch:
		return true
	default:
		return false
	}
}

func (s *Server) clientConfig(w http.ResponseWriter, r *http.Request) {
	cfg := s.coord.ClientConfig(r.Context())
	writeJSON(w, http.StatusOK, ConfigResponse{
		PJWTAuthEnabled:                    false,
		CredentialDefaultSource:            "api_key",
		SampleDispatchBytesSemaphoreSize:   cfg.MaxRequestBytes,
		InflightResponseBytesSemaphoreSize: cfg.MaxRequestBytes,
		UseJWT:                             cfg.UseJWT,
		ParallelFWDBWDChunks:               cfg.ParallelFWDBWDChunks,
		MaxRequestBytes:                    cfg.MaxRequestBytes,
		Auth: map[string]any{
			"use_jwt": cfg.UseJWT,
			"mode":    "none",
		},
		Features: map[string]any{
			"parallel_fwdbwd_chunks": cfg.ParallelFWDBWDChunks,
		},
	})
}

func (s *Server) healthz(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) authToken(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"jwt": ""})
}

func (s *Server) createSession(w http.ResponseWriter, r *http.Request) {
	session, err := s.coord.CreateSession(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, CreateSessionResponse{
		Type:      "create_session",
		SessionID: session.ID,
		ID:        session.ID,
		Status:    "ok",
	})
}

func (s *Server) sessionHeartbeat(w http.ResponseWriter, r *http.Request) {
	var req HeartbeatRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	id := first(req.SessionID, req.ID)
	if id == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing session_id")
		return
	}
	session, err := s.coord.Heartbeat(r.Context(), id)
	if errors.Is(err, tinkerdb.ErrNotFound) {
		writeError(w, http.StatusNotFound, "not_found", "unknown session")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, HeartbeatResponse{
		Type:       "session_heartbeat",
		SessionID:  session.ID,
		Status:     "ok",
		HeartbeatN: session.HeartbeatN,
	})
}

func (s *Server) getServerCapabilities(w http.ResponseWriter, r *http.Request) {
	caps := s.coord.Capabilities(r.Context())
	models := make([]map[string]any, 0, len(caps.Models))
	for _, model := range caps.Models {
		models = append(models, map[string]any{
			"model_name":         model.ModelID,
			"max_context_length": model.ContextLength,
			"tokenizer_id":       model.TokenizerID,
			"supported":          model.Supported,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"supported_models": models})
}

func (s *Server) telemetry(w http.ResponseWriter, r *http.Request) {
	var raw json.RawMessage
	if err := decodeJSON(r, &raw); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	if err := s.coord.AcceptTelemetry(r.Context(), raw); err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, TelemetryResponse{Status: "accepted"})
}

func (s *Server) retrieveFuture(w http.ResponseWriter, r *http.Request) {
	var req RetrieveFutureRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	id := first(req.FutureID, req.RequestID, req.ID)
	if id == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing future_id")
		return
	}
	future, err := s.coord.RetrieveFuture(r.Context(), id, req.AllowMetadataOnly)
	if errors.Is(err, tinkerdb.ErrNotFound) {
		writeError(w, http.StatusNotFound, "not_found", "unknown future")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	s.writeFutureResult(w, future)
}

func (s *Server) cancelFuture(w http.ResponseWriter, r *http.Request) {
	var req RetrieveFutureRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	id := first(req.FutureID, req.RequestID, req.ID)
	if id == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing future_id")
		return
	}
	future, err := s.coord.CancelFuture(r.Context(), id)
	if errors.Is(err, tinkerdb.ErrNotFound) {
		writeError(w, http.StatusNotFound, "not_found", "unknown future")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	s.writeFutureResult(w, future)
}

func (s *Server) createModel(w http.ResponseWriter, r *http.Request) {
	var req tinkercoord.CreateModelRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	if req.SessionID == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing session_id")
		return
	}
	future, model, err := s.coord.CreateModel(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, FutureResponse{
		FutureID:  future.ID,
		RequestID: future.ID,
		ID:        future.ID,
		ModelID:   model.ID,
	})
}

func (s *Server) forward(w http.ResponseWriter, r *http.Request) {
	var req trainingRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	if req.ModelID == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing model_id")
		return
	}
	input := firstInput(req.ForwardInput, req.ForwardBackwardInput)
	if err := normalizeAndValidateInput(&input); err != nil {
		writeUserError(w, http.StatusBadRequest, err.Error())
		return
	}
	future, err := s.coord.Forward(r.Context(), tinkertrain.Request{
		ModelID: req.ModelID,
		Input:   input,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, FutureResponse{
		FutureID:  future.ID,
		RequestID: future.ID,
		ID:        future.ID,
		ModelID:   req.ModelID,
	})
}

func (s *Server) forwardBackward(w http.ResponseWriter, r *http.Request) {
	var req trainingRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	if req.ModelID == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing model_id")
		return
	}
	input := firstInput(req.ForwardBackwardInput, req.ForwardInput)
	if err := normalizeAndValidateInput(&input); err != nil {
		writeUserError(w, http.StatusBadRequest, err.Error())
		return
	}
	future, err := s.coord.ForwardBackward(r.Context(), tinkertrain.Request{
		ModelID: req.ModelID,
		Input:   input,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, FutureResponse{
		FutureID:  future.ID,
		RequestID: future.ID,
		ID:        future.ID,
		ModelID:   req.ModelID,
	})
}

func (s *Server) optimStep(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ModelID    string                 `json:"model_id"`
		AdamParams tinkertrain.AdamParams `json:"adam_params"`
	}
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	if req.ModelID == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing model_id")
		return
	}
	future, err := s.coord.OptimStep(r.Context(), req.ModelID, req.AdamParams)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, FutureResponse{
		FutureID:  future.ID,
		RequestID: future.ID,
		ID:        future.ID,
		ModelID:   req.ModelID,
	})
}

func (s *Server) unloadModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ModelID string `json:"model_id"`
	}
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	if req.ModelID == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing model_id")
		return
	}
	future, err := s.coord.UnloadModel(r.Context(), req.ModelID)
	if errors.Is(err, tinkerdb.ErrNotFound) {
		future, err = s.coord.UserErrorFuture(r.Context(), "unknown model")
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, FutureResponse{
		FutureID:  future.ID,
		RequestID: future.ID,
		ID:        future.ID,
		ModelID:   req.ModelID,
	})
}

func (s *Server) saveWeights(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ModelID string `json:"model_id"`
		Path    string `json:"path"`
	}
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	if req.ModelID == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing model_id")
		return
	}
	future, err := s.coord.SaveWeights(r.Context(), req.ModelID, req.Path)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, FutureResponse{
		FutureID:  future.ID,
		RequestID: future.ID,
		ID:        future.ID,
		ModelID:   req.ModelID,
	})
}

func (s *Server) loadWeights(w http.ResponseWriter, r *http.Request) {
	s.loadWeightsWithOptimizer(w, r, false)
}

func (s *Server) loadStateWithOptimizer(w http.ResponseWriter, r *http.Request) {
	s.loadWeightsWithOptimizer(w, r, true)
}

func (s *Server) loadWeightsWithOptimizer(w http.ResponseWriter, r *http.Request, optimizer bool) {
	var req struct {
		ModelID        string `json:"model_id"`
		Path           string `json:"path"`
		OptimizerState *bool  `json:"optimizer_state"`
	}
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	if req.ModelID == "" || req.Path == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing model_id or path")
		return
	}
	if req.OptimizerState != nil {
		optimizer = *req.OptimizerState
	}
	future, err := s.coord.LoadWeightsWithOptimizer(r.Context(), req.ModelID, req.Path, optimizer)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, FutureResponse{
		FutureID:  future.ID,
		RequestID: future.ID,
		ID:        future.ID,
		ModelID:   req.ModelID,
	})
}

func (s *Server) saveWeightsForSampler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ModelID              string `json:"model_id"`
		Path                 string `json:"path"`
		SamplingSessionSeqID int    `json:"sampling_session_seq_id"`
	}
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	if req.ModelID == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing model_id")
		return
	}
	future, err := s.coord.SaveWeightsForSampler(r.Context(), req.ModelID, req.Path, req.SamplingSessionSeqID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, FutureResponse{
		FutureID:  future.ID,
		RequestID: future.ID,
		ID:        future.ID,
		ModelID:   req.ModelID,
	})
}

func (s *Server) createSamplingSession(w http.ResponseWriter, r *http.Request) {
	var req struct {
		SessionID            string `json:"session_id"`
		SamplingSessionSeqID int    `json:"sampling_session_seq_id"`
		BaseModel            string `json:"base_model"`
		ModelPath            string `json:"model_path"`
	}
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	if req.SessionID == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing session_id")
		return
	}
	id, err := s.coord.CreateSamplingSession(r.Context(), req.SessionID, req.SamplingSessionSeqID, req.ModelPath, req.BaseModel)
	if err != nil {
		writeUserError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"type":                "create_sampling_session",
		"sampling_session_id": id,
	})
}

func (s *Server) asample(w http.ResponseWriter, r *http.Request) {
	var req tinkertrain.SampleRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	if err := validateSampleRequest(req); err != nil {
		writeUserError(w, http.StatusBadRequest, err.Error())
		return
	}
	future, err := s.coord.Sample(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, FutureResponse{
		FutureID:  future.ID,
		RequestID: future.ID,
		ID:        future.ID,
		ModelID:   req.ModelPath,
	})
}

func (s *Server) getInfo(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ModelID string `json:"model_id"`
	}
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	if req.ModelID == "" {
		writeError(w, http.StatusBadRequest, "bad_request", "missing model_id")
		return
	}
	model, err := s.coord.GetModel(r.Context(), req.ModelID)
	if errors.Is(err, tinkerdb.ErrNotFound) {
		writeError(w, http.StatusNotFound, "not_found", "unknown model")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"type":     "get_info",
		"model_id": model.ID,
		"model_data": map[string]any{
			"arch":         "unknown",
			"model_name":   model.BaseModel,
			"tokenizer_id": model.TokenizerID,
		},
		"is_lora":    model.IsLoRA,
		"lora_rank":  model.LoRARank,
		"model_name": model.BaseModel,
	})
}

type trainingRequest struct {
	ModelID              string                           `json:"model_id"`
	ForwardInput         tinkertrain.ForwardBackwardInput `json:"forward_input"`
	ForwardBackwardInput tinkertrain.ForwardBackwardInput `json:"forward_backward_input"`
}

func firstInput(a, b tinkertrain.ForwardBackwardInput) tinkertrain.ForwardBackwardInput {
	if len(a.Data) > 0 || a.LossFn != "" {
		return a
	}
	return b
}

func normalizeAndValidateInput(input *tinkertrain.ForwardBackwardInput) error {
	if input.LossFn != "cross_entropy" {
		return fmt.Errorf("unsupported loss function %q", input.LossFn)
	}
	if len(input.Data) == 0 {
		return errors.New("no data")
	}
	for i := range input.Data {
		datum := &input.Data[i]
		if datum.LossFnInputs == nil {
			return fmt.Errorf("datum %d missing loss_fn_inputs", i)
		}
		for name := range datum.LossFnInputs {
			switch name {
			case "target_tokens", "weights":
			default:
				return fmt.Errorf("datum %d unsupported loss_fn_inputs key %q for cross_entropy", i, name)
			}
		}
		for name, tensor := range datum.LossFnInputs {
			var err error
			tensor, err = normalizeTensorData(name, tensor)
			if err != nil {
				return fmt.Errorf("datum %d loss_fn_inputs[%q]: %w", i, name, err)
			}
			datum.LossFnInputs[name] = tensor
		}
		if err := validateCrossEntropyDatum(i, *datum); err != nil {
			return err
		}
	}
	return nil
}

func normalizeTensorData(name string, tensor tinkertrain.TensorData) (tinkertrain.TensorData, error) {
	if tensor.SparseCrowIndices != nil || tensor.SparseColIndices != nil {
		return tensor, errors.New("sparse tensors are not supported")
	}
	if tensor.Shape == nil {
		tensor.Shape = []int{len(tensor.Data)}
	}
	n := 1
	for _, dim := range tensor.Shape {
		if dim < 0 {
			return tensor, fmt.Errorf("negative shape dimension %d", dim)
		}
		n *= dim
	}
	if n != len(tensor.Data) {
		return tensor, fmt.Errorf("shape %v has %d elements but data has %d", tensor.Shape, n, len(tensor.Data))
	}
	switch name {
	case "target_tokens":
		if tensor.DType != "" && tensor.DType != "int64" {
			return tensor, fmt.Errorf("dtype %q, want int64", tensor.DType)
		}
		for i, v := range tensor.Data {
			if math.Trunc(v) != v {
				return tensor, fmt.Errorf("data[%d] = %v is not an integer", i, v)
			}
			if v < 0 || v > float64(math.MaxInt32) {
				return tensor, fmt.Errorf("data[%d] = %v is out of range", i, v)
			}
		}
		tensor.DType = "int64"
	case "weights":
		if tensor.DType != "" && tensor.DType != "float32" {
			return tensor, fmt.Errorf("dtype %q, want float32", tensor.DType)
		}
		for i, v := range tensor.Data {
			if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
				return tensor, fmt.Errorf("data[%d] = %v is not a non-negative finite number", i, v)
			}
		}
		tensor.DType = "float32"
	}
	return tensor, nil
}

func validateCrossEntropyDatum(i int, datum tinkertrain.Datum) error {
	target, ok := datum.LossFnInputs["target_tokens"]
	if !ok {
		return fmt.Errorf("datum %d missing target_tokens for cross_entropy", i)
	}
	if weight, ok := datum.LossFnInputs["weights"]; ok && len(weight.Data) != len(target.Data) {
		return fmt.Errorf("datum %d weights length %d does not match target_tokens length %d", i, len(weight.Data), len(target.Data))
	}
	if weight, ok := datum.LossFnInputs["weights"]; ok && !sameShape(weight.Shape, target.Shape) {
		return fmt.Errorf("datum %d weights shape %v does not match target_tokens shape %v", i, weight.Shape, target.Shape)
	}
	if tokens := tokenCount(datum.ModelInput); tokens != len(target.Data) {
		return fmt.Errorf("datum %d input tokens length %d does not match target_tokens length %d", i, tokens, len(target.Data))
	}
	return nil
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func tokenCount(input tinkertrain.ModelInput) int {
	var n int
	for _, chunk := range input.Chunks {
		if chunk.Type == "" || chunk.Type == "encoded_text" {
			n += len(chunk.Tokens)
		}
	}
	return n
}

func validateSampleRequest(req tinkertrain.SampleRequest) error {
	if req.SamplingSessionID == "" && req.ModelPath == "" && req.BaseModel == "" {
		return errors.New("missing sampling_session_id, model_path, or base_model")
	}
	if tokenCount(req.Prompt) == 0 {
		return errors.New("prompt is empty")
	}
	return nil
}

func (s *Server) weightsInfo(w http.ResponseWriter, r *http.Request) {
	var req struct {
		TinkerPath string `json:"tinker_path"`
	}
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	base := "Qwen/Qwen3-8B"
	rank := 8
	if req.TinkerPath != "" {
		if parsed, err := tinkertrain.ParseTinkerPath(req.TinkerPath); err == nil {
			if model, err := s.coord.GetModel(r.Context(), parsed.ModelID); err == nil {
				base = model.BaseModel
				rank = model.LoRARank
			}
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"base_model":    base,
		"is_lora":       true,
		"lora_rank":     rank,
		"train_unembed": false,
		"train_mlp":     true,
		"train_attn":    true,
	})
}

func (s *Server) trainingRuns(w http.ResponseWriter, r *http.Request) {
	resp, err := s.coord.TrainingRuns(r.Context(), intQuery(r, "limit", 20), intQuery(r, "offset", 0))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) checkpoints(w http.ResponseWriter, r *http.Request) {
	resp, err := s.coord.Checkpoints(r.Context(), "", intQuery(r, "limit", 100), intQuery(r, "offset", 0))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) sessions(w http.ResponseWriter, r *http.Request) {
	snapshot, err := s.coord.DashboardSnapshot(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	limit := intQuery(r, "limit", 20)
	offset := intQuery(r, "offset", 0)
	if limit <= 0 {
		limit = 20
	}
	if offset < 0 {
		offset = 0
	}
	total := len(snapshot.Sessions)
	if offset > total {
		offset = total
	}
	end := offset + limit
	if end > total {
		end = total
	}
	ids := make([]string, 0, end-offset)
	for _, session := range snapshot.Sessions[offset:end] {
		ids = append(ids, session.ID)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"sessions": ids,
		"cursor": tinkercoord.Cursor{
			Offset:     offset,
			Limit:      limit,
			TotalCount: total,
		},
	})
}

func (s *Server) sessionPath(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/v1/sessions/")
	if id == "" || strings.Contains(id, "/") {
		writeError(w, http.StatusNotFound, "not_found", "unknown session")
		return
	}
	snapshot, err := s.coord.DashboardSnapshot(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "system_error", err.Error())
		return
	}
	found := false
	for _, session := range snapshot.Sessions {
		if session.ID == id {
			found = true
			break
		}
	}
	if !found {
		writeError(w, http.StatusNotFound, "not_found", "unknown session")
		return
	}
	modelIDs := make([]string, 0)
	for _, model := range snapshot.Models {
		if model.SessionID == id {
			modelIDs = append(modelIDs, model.ID)
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"training_run_ids": modelIDs,
		"sampler_ids":      []string{},
	})
}

func (s *Server) trainingRunPath(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/api/v1/training_runs/")
	if handled := s.checkpointAction(w, r, path); handled {
		return
	}
	switch {
	case strings.HasSuffix(path, "/checkpoints"):
		runID := strings.TrimSuffix(path, "/checkpoints")
		resp, err := s.coord.Checkpoints(r.Context(), runID, 0, 0)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "system_error", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, resp)
	case !strings.Contains(path, "/"):
		resp, err := s.coord.TrainingRuns(r.Context(), 1000, 0)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "system_error", err.Error())
			return
		}
		for _, run := range resp.TrainingRuns {
			if run.TrainingRunID == path {
				writeJSON(w, http.StatusOK, run)
				return
			}
		}
		writeError(w, http.StatusNotFound, "not_found", "unknown training run")
	default:
		writeError(w, http.StatusNotFound, "not_found", "unsupported training run route")
	}
}

func (s *Server) checkpointAction(w http.ResponseWriter, r *http.Request, path string) bool {
	parts := strings.Split(path, "/")
	if len(parts) < 4 || parts[1] != "checkpoints" {
		return false
	}
	modelID := parts[0]
	action := ""
	checkpointParts := parts[2:]
	switch parts[len(parts)-1] {
	case "archive", "publish", "ttl":
		action = parts[len(parts)-1]
		checkpointParts = parts[2 : len(parts)-1]
	}
	if len(checkpointParts) < 2 {
		return false
	}
	tinkerPath := "tinker://" + modelID + "/" + strings.Join(checkpointParts, "/")
	if !tinkertrain.CheckpointPathExists(tinkerPath) {
		writeError(w, http.StatusNotFound, "not_found", "unknown checkpoint")
		return true
	}
	switch {
	case r.Method == http.MethodGet && action == "archive":
		expired, err := s.coord.CheckpointExpired(r.Context(), tinkerPath)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "system_error", err.Error())
			return true
		}
		if expired {
			writeError(w, http.StatusGone, "gone", "checkpoint expired")
			return true
		}
		file, err := tinkertrain.CheckpointArchive(tinkerPath)
		if err != nil {
			writeError(w, http.StatusBadRequest, "bad_request", err.Error())
			return true
		}
		expires := time.Now().UTC().Add(15 * time.Minute)
		w.Header().Set("Location", "file://"+file)
		w.Header().Set("Expires", expires.Format(http.TimeFormat))
		w.Header().Set("X-Tinker-Archive-Expires-At", expires.Format(time.RFC3339))
		w.Header().Set("X-Tinker-Archive-Owner", "local")
		w.Header().Set("X-Tinker-Archive-Visibility", "private")
		w.WriteHeader(http.StatusFound)
	case r.Method == http.MethodDelete && action == "":
		if err := tinkertrain.DeleteCheckpoint(tinkerPath); err != nil {
			writeError(w, http.StatusInternalServerError, "system_error", err.Error())
			return true
		}
		if err := s.coord.DeleteCheckpointMetadata(r.Context(), tinkerPath); err != nil {
			writeError(w, http.StatusInternalServerError, "system_error", err.Error())
			return true
		}
		writeJSON(w, http.StatusOK, map[string]any{})
	case r.Method == http.MethodPost && action == "publish":
		if err := s.coord.SetCheckpointPublic(r.Context(), tinkerPath, true); err != nil {
			writeError(w, http.StatusInternalServerError, "system_error", err.Error())
			return true
		}
		writeJSON(w, http.StatusOK, map[string]any{"public": true})
	case r.Method == http.MethodDelete && action == "publish":
		if err := s.coord.SetCheckpointPublic(r.Context(), tinkerPath, false); err != nil {
			writeError(w, http.StatusInternalServerError, "system_error", err.Error())
			return true
		}
		writeJSON(w, http.StatusOK, map[string]any{"public": false})
	case r.Method == http.MethodPut && action == "ttl":
		ttl, err := decodeTTL(r)
		if err != nil {
			writeError(w, http.StatusBadRequest, "bad_request", err.Error())
			return true
		}
		if err := s.coord.SetCheckpointTTL(r.Context(), tinkerPath, ttl); err != nil {
			writeError(w, http.StatusInternalServerError, "system_error", err.Error())
			return true
		}
		writeJSON(w, http.StatusOK, map[string]any{})
	default:
		writeError(w, http.StatusNotFound, "not_found", "unsupported checkpoint route")
	}
	return true
}

func (s *Server) writeFutureResult(w http.ResponseWriter, future tinkercoord.Future) {
	switch future.State {
	case tinkercoord.FutureTryAgain, tinkercoord.FutureQueued, tinkercoord.FutureRunning:
		writeJSON(w, http.StatusOK, RetrieveFutureResponse{
			Type:       "try_again",
			FutureID:   future.ID,
			RequestID:  future.ID,
			QueueState: "active",
		})
	case tinkercoord.FutureCompleteMetadata:
		writeJSON(w, http.StatusOK, RetrieveFutureResponse{
			Status:              "complete_metadata",
			FutureID:            future.ID,
			RequestID:           future.ID,
			Metadata:            future.Metadata,
			ResponsePayloadSize: len(future.Result),
		})
	case tinkercoord.FutureUserError:
		writeJSON(w, http.StatusOK, map[string]any{
			"error":      errorMessage(future.Error),
			"category":   "user",
			"future_id":  future.ID,
			"request_id": future.ID,
		})
	case tinkercoord.FutureSystemError, tinkercoord.FutureCanceled:
		writeJSON(w, http.StatusOK, map[string]any{
			"error":      errorMessage(future.Error),
			"category":   "server",
			"future_id":  future.ID,
			"request_id": future.ID,
		})
	default:
		if len(future.Result) == 0 {
			writeJSON(w, http.StatusOK, map[string]any{})
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(future.Result)
	}
}

func decodeJSON(r *http.Request, v any) error {
	defer r.Body.Close()
	if r.Body == http.NoBody {
		return nil
	}
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(v); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			return fmt.Errorf("request body exceeds %d bytes", maxErr.Limit)
		}
		return fmt.Errorf("decode json: %w", err)
	}
	var extra json.RawMessage
	if err := dec.Decode(&extra); err != nil {
		if errors.Is(err, io.EOF) {
			return nil
		}
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			return fmt.Errorf("request body exceeds %d bytes", maxErr.Limit)
		}
		return fmt.Errorf("decode json: %w", err)
	}
	return errors.New("decode json: multiple json values")
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, code int, errCode, message string) {
	writeJSON(w, code, ErrorResponse{
		Status:  "error",
		Code:    errCode,
		Message: message,
	})
}

func writeUserError(w http.ResponseWriter, code int, message string) {
	writeJSON(w, code, map[string]string{
		"error":    message,
		"category": "user",
	})
}

func errorMessage(raw json.RawMessage) string {
	var body struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	}
	if len(raw) == 0 || json.Unmarshal(raw, &body) != nil {
		return "request failed"
	}
	if body.Message != "" {
		return body.Message
	}
	if body.Code != "" {
		return body.Code
	}
	return "request failed"
}

func first(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func intQuery(r *http.Request, name string, def int) int {
	raw := r.URL.Query().Get(name)
	if raw == "" {
		return def
	}
	var n int
	if _, err := fmt.Sscanf(raw, "%d", &n); err != nil {
		return def
	}
	return n
}

func decodeTTL(r *http.Request) (time.Duration, error) {
	var req struct {
		TTL        *float64 `json:"ttl"`
		TTLSeconds *float64 `json:"ttl_seconds"`
		ExpiresIn  *float64 `json:"expires_in"`
		ExpiresAt  string   `json:"expires_at"`
	}
	if err := decodeJSON(r, &req); err != nil {
		return 0, err
	}
	seconds := req.TTL
	if seconds == nil {
		seconds = req.TTLSeconds
	}
	if seconds == nil {
		seconds = req.ExpiresIn
	}
	if seconds != nil {
		if *seconds < 0 {
			return 0, errors.New("ttl is negative")
		}
		return time.Duration(*seconds * float64(time.Second)), nil
	}
	if req.ExpiresAt != "" {
		at, err := time.Parse(time.RFC3339, req.ExpiresAt)
		if err != nil {
			return 0, fmt.Errorf("parse expires_at: %w", err)
		}
		ttl := time.Until(at)
		if ttl < 0 {
			return 0, nil
		}
		return ttl, nil
	}
	return 0, nil
}
