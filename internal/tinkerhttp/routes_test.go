package tinkerhttp

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerdb"
	"github.com/tmc/localtinker/internal/tinkertrain"
)

func TestHandshakeRoutes(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()

	var cfg ConfigResponse
	postJSON(t, h, "/api/v1/client/config", nil, &cfg)
	if cfg.UseJWT {
		t.Fatal("UseJWT = true, want false")
	}
	if cfg.ParallelFWDBWDChunks {
		t.Fatal("ParallelFWDBWDChunks = true, want false")
	}

	var created CreateSessionResponse
	postJSON(t, h, "/api/v1/create_session", nil, &created)
	if created.SessionID == "" {
		t.Fatal("empty session_id")
	}

	var heartbeat HeartbeatResponse
	postJSON(t, h, "/api/v1/session_heartbeat",
		HeartbeatRequest{SessionID: created.SessionID},
		&heartbeat,
	)
	if heartbeat.Status != "ok" {
		t.Fatalf("heartbeat status = %q, want ok", heartbeat.Status)
	}

	var telemetry TelemetryResponse
	postJSON(t, h, "/api/v1/telemetry", map[string]any{"events": []any{}}, &telemetry)
	if telemetry.Status != "accepted" {
		t.Fatalf("telemetry status = %q, want accepted", telemetry.Status)
	}

	var caps struct {
		SupportedModels []map[string]any `json:"supported_models"`
	}
	getJSON(t, h, "/api/v1/get_server_capabilities", &caps)
	if len(caps.SupportedModels) == 0 {
		t.Fatal("no supported models")
	}
	supported, ok := caps.SupportedModels[0]["supported"].([]any)
	if !ok || len(supported) == 0 {
		t.Fatalf("supported capabilities = %#v, want non-empty list", caps.SupportedModels[0]["supported"])
	}
}

func TestSessionRESTRoutes(t *testing.T) {
	store := tinkerdb.OpenMemory()
	c, err := tinkercoord.New(tinkercoord.Config{Store: store})
	if err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()

	var created CreateSessionResponse
	postJSON(t, h, "/api/v1/create_session", nil, &created)
	if created.SessionID == "" {
		t.Fatal("empty session_id")
	}
	if err := store.PutModel(nil, tinkerdb.Model{
		ID:          "model-a",
		SessionID:   created.SessionID,
		BaseModel:   "Qwen/Qwen3-8B",
		TokenizerID: "Qwen/Qwen3-8B",
		IsLoRA:      true,
		LoRARank:    8,
		CreatedAt:   time.Now().UTC(),
	}); err != nil {
		t.Fatal(err)
	}

	var sessions struct {
		Sessions []string           `json:"sessions"`
		Cursor   tinkercoord.Cursor `json:"cursor"`
	}
	getJSON(t, h, "/api/v1/sessions?limit=1", &sessions)
	if len(sessions.Sessions) != 1 || sessions.Sessions[0] != created.SessionID {
		t.Fatalf("sessions = %#v, want %q", sessions, created.SessionID)
	}
	if sessions.Cursor.TotalCount != 1 {
		t.Fatalf("cursor = %#v, want total_count 1", sessions.Cursor)
	}

	var session struct {
		TrainingRunIDs []string `json:"training_run_ids"`
		SamplerIDs     []string `json:"sampler_ids"`
	}
	getJSON(t, h, "/api/v1/sessions/"+created.SessionID, &session)
	if len(session.TrainingRunIDs) != 1 || session.TrainingRunIDs[0] != "model-a" {
		t.Fatalf("session = %#v, want model-a", session)
	}
	if session.SamplerIDs == nil {
		t.Fatalf("sampler_ids = nil, want empty list")
	}

	var missing ErrorResponse
	methodJSON(t, h, http.MethodGet, "/api/v1/sessions/missing", nil, http.StatusNotFound, &missing)
	if missing.Code != "not_found" {
		t.Fatalf("missing session response = %#v", missing)
	}
}

func TestSamplerRESTRoute(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()

	var sampler struct {
		SamplerID string  `json:"sampler_id"`
		BaseModel string  `json:"base_model"`
		ModelPath *string `json:"model_path"`
	}
	getJSON(t, h, "/api/v1/samplers/sess-a:sample:0", &sampler)
	if sampler.SamplerID != "sess-a:sample:0" {
		t.Fatalf("sampler_id = %q, want sess-a:sample:0", sampler.SamplerID)
	}
	if sampler.BaseModel != "Qwen/Qwen3-8B" {
		t.Fatalf("base_model = %q, want Qwen/Qwen3-8B", sampler.BaseModel)
	}
	if sampler.ModelPath != nil {
		t.Fatalf("model_path = %q, want nil", *sampler.ModelPath)
	}

	var missing ErrorResponse
	methodJSON(t, h, http.MethodGet, "/api/v1/samplers/", nil, http.StatusNotFound, &missing)
	if missing.Code != "not_found" {
		t.Fatalf("missing sampler response = %#v", missing)
	}
}

func TestRequestBodyLimit(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{
		Store:           tinkerdb.OpenMemory(),
		MaxRequestBytes: 32,
	})
	if err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()

	var cfg ConfigResponse
	postJSON(t, h, "/api/v1/client/config", nil, &cfg)
	if cfg.MaxRequestBytes != 32 {
		t.Fatalf("max_request_bytes = %d, want 32", cfg.MaxRequestBytes)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/v1/telemetry", strings.NewReader(`{"events":["012345678901234567890123456789"]}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d body = %s", rec.Code, http.StatusBadRequest, rec.Body.String())
	}

	var resp ErrorResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Code != "bad_request" || !strings.Contains(resp.Message, "request body exceeds 32 bytes") {
		t.Fatalf("response = %#v", resp)
	}
}

func TestRejectsTrailingJSON(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()

	req := httptest.NewRequest(http.MethodPost, "/api/v1/telemetry", strings.NewReader(`{"events":[]} {}`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d body = %s", rec.Code, http.StatusBadRequest, rec.Body.String())
	}

	var resp ErrorResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Code != "bad_request" || !strings.Contains(resp.Message, "multiple json values") {
		t.Fatalf("response = %#v", resp)
	}
}

func TestWeightsInfoRejectsMalformedJSON(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()

	req := httptest.NewRequest(http.MethodPost, "/api/v1/weights_info", strings.NewReader(`{"tinker_path":`))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d body = %s", rec.Code, http.StatusBadRequest, rec.Body.String())
	}

	var resp ErrorResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Code != "bad_request" {
		t.Fatalf("response = %#v", resp)
	}
}

func TestRetrieveFutureRoute(t *testing.T) {
	store := tinkerdb.OpenMemory()
	c, err := tinkercoord.New(tinkercoord.Config{Store: store})
	if err != nil {
		t.Fatal(err)
	}
	future, err := c.CompleteFuture(nil, map[string]any{"ok": true}, map[string]any{"summary": "ready"})
	if err != nil {
		t.Fatal(err)
	}

	var got RetrieveFutureResponse
	postJSON(t, New(c).Handler(), "/api/v1/retrieve_future",
		RetrieveFutureRequest{FutureID: future.ID, AllowMetadataOnly: true},
		&got,
	)
	if got.Status != tinkercoord.FutureCompleteMetadata {
		t.Fatalf("status = %q, want %q", got.Status, tinkercoord.FutureCompleteMetadata)
	}
	if got.FutureID != future.ID || got.RequestID != future.ID {
		t.Fatalf("metadata future ids = %#v, want %q", got, future.ID)
	}

	var result map[string]bool
	postJSON(t, New(c).Handler(), "/api/v1/retrieve_future",
		RetrieveFutureRequest{FutureID: future.ID},
		&result,
	)
	if !result["ok"] {
		t.Fatalf("result = %#v, want ok", result)
	}

	queued := tinkerdb.Future{ID: "fut-queued", State: tinkercoord.FutureQueued}
	if err := store.PutFuture(nil, queued); err != nil {
		t.Fatal(err)
	}
	var pending RetrieveFutureResponse
	postJSON(t, New(c).Handler(), "/api/v1/retrieve_future",
		RetrieveFutureRequest{FutureID: queued.ID, AllowMetadataOnly: true},
		&pending,
	)
	if pending.Type != "try_again" || pending.FutureID != queued.ID || pending.RequestID != queued.ID {
		t.Fatalf("pending = %#v", pending)
	}
}

func TestCancelFutureRoute(t *testing.T) {
	store := tinkerdb.OpenMemory()
	c, err := tinkercoord.New(tinkercoord.Config{Store: store})
	if err != nil {
		t.Fatal(err)
	}
	queued := tinkerdb.Future{ID: "fut-cancel", State: tinkercoord.FutureQueued}
	if err := store.PutFuture(nil, queued); err != nil {
		t.Fatal(err)
	}

	var canceled map[string]any
	postJSON(t, New(c).Handler(), "/api/v1/cancel_future",
		RetrieveFutureRequest{RequestID: queued.ID},
		&canceled,
	)
	if canceled["category"] != "server" || canceled["request_id"] != queued.ID {
		t.Fatalf("canceled = %#v", canceled)
	}
	got, err := store.GetFuture(nil, queued.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got.State != tinkercoord.FutureCanceled {
		t.Fatalf("state = %q, want %q", got.State, tinkercoord.FutureCanceled)
	}
}

func TestUnsupportedOperationReturnsFutureFailure(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}

	var future FutureResponse
	postJSON(t, New(c).Handler(), "/api/v1/save_weights",
		map[string]any{"model_id": "model-a"},
		&future,
	)
	if future.RequestID == "" || future.ModelID != "model-a" {
		t.Fatalf("future = %#v", future)
	}

	var failed map[string]string
	postJSON(t, New(c).Handler(), "/api/v1/retrieve_future",
		RetrieveFutureRequest{RequestID: future.RequestID},
		&failed,
	)
	if failed["category"] != "user" || failed["error"] == "" {
		t.Fatalf("failed = %#v", failed)
	}
	if failed["future_id"] != future.RequestID || failed["request_id"] != future.RequestID {
		t.Fatalf("failed ids = %#v, want %q", failed, future.RequestID)
	}
}

func TestForwardBackwardAndOptimStepTune(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{
		Store:        tinkerdb.OpenMemory(),
		LeaseTimeout: 2 * time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()

	var created FutureResponse
	postJSON(t, h, "/api/v1/create_model",
		map[string]any{
			"session_id":  "sess-a",
			"base_model":  "Qwen/Qwen3-8B",
			"lora_config": map[string]any{"rank": 8},
		},
		&created,
	)

	req := map[string]any{
		"model_id": created.ModelID,
		"forward_backward_input": map[string]any{
			"loss_fn": "cross_entropy",
			"data": []any{
				map[string]any{
					"model_input": map[string]any{
						"chunks": []any{
							map[string]any{
								"type":   "encoded_text",
								"tokens": []int{1, 1, 1, 1},
							},
						},
					},
					"loss_fn_inputs": map[string]any{
						"target_tokens": map[string]any{
							"data":  []int{1, 1, 1, 1},
							"dtype": "int64",
						},
					},
				},
			},
		},
	}

	before := trainingLoss(t, h, "/api/v1/forward", map[string]any{
		"model_id":      created.ModelID,
		"forward_input": req["forward_backward_input"],
	})
	for range 4 {
		var future FutureResponse
		postJSON(t, h, "/api/v1/forward_backward", req, &future)
		retrieveOK(t, h, future.RequestID)

		postJSON(t, h, "/api/v1/optim_step",
			map[string]any{
				"model_id": created.ModelID,
				"adam_params": map[string]any{
					"learning_rate": 1e-4,
				},
			},
			&future,
		)
		retrieveOK(t, h, future.RequestID)
	}
	after := trainingLoss(t, h, "/api/v1/forward", map[string]any{
		"model_id":      created.ModelID,
		"forward_input": req["forward_backward_input"],
	})
	if after >= before {
		t.Fatalf("loss did not decrease: before=%v after=%v", before, after)
	}
}

func TestTrainingInputValidationReturnsUserErrors(t *testing.T) {
	validDatum := func() map[string]any {
		return map[string]any{
			"model_input": map[string]any{
				"chunks": []any{
					map[string]any{
						"type":   "encoded_text",
						"tokens": []int{1, 1, 1, 1},
					},
				},
			},
			"loss_fn_inputs": map[string]any{
				"target_tokens": map[string]any{
					"data":  []int{1, 1, 1, 1},
					"dtype": "int64",
				},
			},
		}
	}

	tests := []struct {
		name string
		edit func(map[string]any)
		want string
	}{
		{
			name: "missing target_tokens",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"] = map[string]any{}
			},
			want: "missing target_tokens",
		},
		{
			name: "unsupported key",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["mask"] = map[string]any{
					"data":  []float64{1, 1, 1, 1},
					"dtype": "float32",
				}
			},
			want: `unsupported loss_fn_inputs key "mask"`,
		},
		{
			name: "bad weights length",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["weights"] = map[string]any{
					"data":  []float64{1, 1},
					"dtype": "float32",
				}
			},
			want: "weights length 2 does not match target_tokens length 4",
		},
		{
			name: "bad weights shape",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["target_tokens"] = map[string]any{
					"data":  []int{2, 2, 2, 2},
					"dtype": "int64",
					"shape": []int{2, 2},
				}
				d["loss_fn_inputs"].(map[string]any)["weights"] = map[string]any{
					"data":  []float64{1, 1, 1, 1},
					"dtype": "float32",
					"shape": []int{4},
				}
			},
			want: "weights shape [4] does not match target_tokens shape [2 2]",
		},
		{
			name: "broadcast-looking weights shape rejected",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["target_tokens"] = map[string]any{
					"data":  []int{2, 2, 2, 2},
					"dtype": "int64",
					"shape": []int{2, 2},
				}
				d["loss_fn_inputs"].(map[string]any)["weights"] = map[string]any{
					"data":  []float64{1, 1, 1, 1},
					"dtype": "float32",
					"shape": []int{1, 4},
				}
			},
			want: "weights shape [1 4] does not match target_tokens shape [2 2]",
		},
		{
			name: "bad shape",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["target_tokens"] = map[string]any{
					"data":  []int{2, 2, 2, 2},
					"dtype": "int64",
					"shape": []int{3},
				}
			},
			want: "shape [3] has 3 elements but data has 4",
		},
		{
			name: "negative shape",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["target_tokens"] = map[string]any{
					"data":  []int{2, 2, 2, 2},
					"dtype": "int64",
					"shape": []int{2, -2},
				}
			},
			want: "negative shape dimension -2",
		},
		{
			name: "out of range target",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["target_tokens"] = map[string]any{
					"data":  []float64{1, 1, float64(1 << 31), 1},
					"dtype": "int64",
				}
			},
			want: "is out of range",
		},
		{
			name: "fractional target",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["target_tokens"] = map[string]any{
					"data":  []float64{1, 1.5, 1, 1},
					"dtype": "int64",
				}
			},
			want: "is not an integer",
		},
		{
			name: "negative weight",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["weights"] = map[string]any{
					"data":  []float64{1, -1, 1, 1},
					"dtype": "float32",
				}
			},
			want: "is not a non-negative finite number",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			datum := validDatum()
			tt.edit(datum)
			input := firstInput(tinkertrain.ForwardBackwardInput{
				LossFn: "cross_entropy",
				Data:   []tinkertrain.Datum{decodeDatum(t, datum)},
			}, tinkertrain.ForwardBackwardInput{})
			err := normalizeAndValidateInput(&input)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestTrainingInputValidationAcceptsDenseCrossEntropyTensors(t *testing.T) {
	input := tinkertrain.ForwardBackwardInput{
		LossFn: "cross_entropy",
		Data: []tinkertrain.Datum{{
			ModelInput: tinkertrain.ModelInput{Chunks: []tinkertrain.ModelInputChunk{{
				Type:   "encoded_text",
				Tokens: []int{1, 2, 3, 4},
			}}},
			LossFnInputs: map[string]tinkertrain.TensorData{
				"target_tokens": {
					Data:  []float64{9, 8, 7, 6},
					DType: "int64",
					Shape: []int{2, 2},
				},
				"weights": {
					Data:  []float64{1, 0.5, 0, 1},
					DType: "float32",
					Shape: []int{2, 2},
				},
			},
		}},
	}

	if err := normalizeAndValidateInput(&input); err != nil {
		t.Fatal(err)
	}
}

func TestTrainingInputValidationAcceptsDenseCrossEntropy(t *testing.T) {
	input := tinkertrain.ForwardBackwardInput{
		LossFn: "cross_entropy",
		Data: []tinkertrain.Datum{
			decodeDatum(t, map[string]any{
				"model_input": map[string]any{
					"chunks": []any{
						map[string]any{
							"type":   "encoded_text",
							"tokens": []int{10, 11, 12, 13},
						},
					},
				},
				"loss_fn_inputs": map[string]any{
					"target_tokens": map[string]any{
						"data":  []int{30, 31, 32, 33},
						"dtype": "int64",
						"shape": []int{2, 2},
					},
					"weights": map[string]any{
						"data":  []float64{0, 0.25, 1, 0.5},
						"dtype": "float32",
						"shape": []int{2, 2},
					},
				},
			}),
		},
	}
	if err := normalizeAndValidateInput(&input); err != nil {
		t.Fatal(err)
	}
	target := input.Data[0].LossFnInputs["target_tokens"]
	if target.DType != "int64" || !sameShape(target.Shape, []int{2, 2}) {
		t.Fatalf("target tensor = %#v", target)
	}
	weight := input.Data[0].LossFnInputs["weights"]
	if weight.DType != "float32" || !sameShape(weight.Shape, []int{2, 2}) {
		t.Fatalf("weight tensor = %#v", weight)
	}
}

func TestTrainingInputValidationNormalizesTensorData(t *testing.T) {
	tests := []struct {
		name      string
		target    map[string]any
		weight    map[string]any
		wantShape []int
	}{
		{
			name:      "flat omitted dtype and shape",
			target:    map[string]any{"data": []int{4, 3, 2, 1}},
			weight:    map[string]any{"data": []float64{1, 0.5, 0, 1}},
			wantShape: []int{4},
		},
		{
			name: "rectangular omitted dtype",
			target: map[string]any{
				"data":  []int{4, 3, 2, 1},
				"shape": []int{2, 2},
			},
			weight: map[string]any{
				"data":  []float64{1, 0.5, 0, 1},
				"shape": []int{2, 2},
			},
			wantShape: []int{2, 2},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := tinkertrain.ForwardBackwardInput{
				LossFn: "cross_entropy",
				Data: []tinkertrain.Datum{
					decodeDatum(t, map[string]any{
						"model_input": map[string]any{
							"chunks": []any{
								map[string]any{
									"type":   "encoded_text",
									"tokens": []int{10, 11, 12, 13},
								},
							},
						},
						"loss_fn_inputs": map[string]any{
							"target_tokens": tt.target,
							"weights":       tt.weight,
						},
					}),
				},
			}
			if err := normalizeAndValidateInput(&input); err != nil {
				t.Fatal(err)
			}
			target := input.Data[0].LossFnInputs["target_tokens"]
			if target.DType != "int64" || !sameShape(target.Shape, tt.wantShape) {
				t.Fatalf("target tensor = %#v, want dtype int64 shape %v", target, tt.wantShape)
			}
			weight := input.Data[0].LossFnInputs["weights"]
			if weight.DType != "float32" || !sameShape(weight.Shape, tt.wantShape) {
				t.Fatalf("weight tensor = %#v, want dtype float32 shape %v", weight, tt.wantShape)
			}
		})
	}
}

func TestConformanceMalformedTrainingInputsReturnUserErrors(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()

	validDatum := func() map[string]any {
		return map[string]any{
			"model_input": map[string]any{
				"chunks": []any{
					map[string]any{
						"type":   "encoded_text",
						"tokens": []int{1, 1, 1, 1},
					},
				},
			},
			"loss_fn_inputs": map[string]any{
				"target_tokens": map[string]any{
					"data":  []int{1, 1, 1, 1},
					"dtype": "int64",
				},
			},
		}
	}
	request := func(datum map[string]any) map[string]any {
		return map[string]any{
			"model_id": "model-a",
			"forward_input": map[string]any{
				"loss_fn": "cross_entropy",
				"data":    []any{datum},
			},
		}
	}
	tests := []struct {
		name string
		edit func(map[string]any)
		want string
	}{
		{
			name: "missing loss inputs",
			edit: func(d map[string]any) {
				delete(d, "loss_fn_inputs")
			},
			want: "missing loss_fn_inputs",
		},
		{
			name: "unsupported loss input",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["mask"] = map[string]any{
					"data":  []float64{1, 1, 1, 1},
					"dtype": "float32",
				}
			},
			want: `unsupported loss_fn_inputs key "mask"`,
		},
		{
			name: "sparse tensor",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["target_tokens"] = map[string]any{
					"data":                []int{1, 1, 1, 1},
					"dtype":               "int64",
					"sparse_crow_indices": []int{0, 1},
				}
			},
			want: "sparse tensors are not supported",
		},
		{
			name: "bad dtype",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["target_tokens"] = map[string]any{
					"data":  []int{1, 1, 1, 1},
					"dtype": "float32",
				}
			},
			want: `dtype "float32", want int64`,
		},
		{
			name: "bad shape",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["target_tokens"] = map[string]any{
					"data":  []int{1, 1, 1, 1},
					"dtype": "int64",
					"shape": []int{2, 3},
				}
			},
			want: "shape [2 3] has 6 elements but data has 4",
		},
		{
			name: "input target mismatch",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["target_tokens"] = map[string]any{
					"data":  []int{1, 1, 1},
					"dtype": "int64",
				}
			},
			want: "input tokens length 4 does not match target_tokens length 3",
		},
		{
			name: "image chunk",
			edit: func(d map[string]any) {
				d["model_input"] = map[string]any{
					"chunks": []any{
						map[string]any{"type": "image", "tokens": []int{1, 1, 1, 1}},
					},
				}
			},
			want: `unsupported model input chunk type "image"`,
		},
		{
			name: "invalid target category",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["target_tokens"] = map[string]any{
					"data":  []float64{1, 1, float64(1 << 31), 1},
					"dtype": "int64",
				}
			},
			want: "is out of range",
		},
		{
			name: "weight broadcast category",
			edit: func(d map[string]any) {
				d["loss_fn_inputs"].(map[string]any)["target_tokens"] = map[string]any{
					"data":  []int{1, 1, 1, 1},
					"dtype": "int64",
					"shape": []int{2, 2},
				}
				d["loss_fn_inputs"].(map[string]any)["weights"] = map[string]any{
					"data":  []float64{1, 1, 1, 1},
					"dtype": "float32",
					"shape": []int{1, 4},
				}
			},
			want: "weights shape [1 4] does not match target_tokens shape [2 2]",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			datum := validDatum()
			tt.edit(datum)
			var resp map[string]string
			postJSONStatus(t, h, "/api/v1/forward", request(datum), http.StatusBadRequest, &resp)
			if resp["category"] != "user" || !strings.Contains(resp["error"], tt.want) {
				t.Fatalf("response = %#v, want user error containing %q", resp, tt.want)
			}
		})
	}
}

func TestConformanceMalformedAsyncRequestsReturnBadRequest(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()

	var errResp ErrorResponse
	postJSONStatus(t, h, "/api/v1/create_model",
		map[string]any{"base_model": "Qwen/Qwen3-8B"},
		http.StatusBadRequest,
		&errResp,
	)
	if errResp.Code != "bad_request" || errResp.Message != "missing session_id" {
		t.Fatalf("create_model response = %#v", errResp)
	}

	validInput := map[string]any{
		"loss_fn": "cross_entropy",
		"data": []any{
			map[string]any{
				"model_input": map[string]any{
					"chunks": []any{
						map[string]any{"type": "encoded_text", "tokens": []int{1}},
					},
				},
				"loss_fn_inputs": map[string]any{
					"target_tokens": map[string]any{"data": []int{1}, "dtype": "int64"},
				},
			},
		},
	}
	postJSONStatus(t, h, "/api/v1/forward",
		map[string]any{"forward_input": validInput},
		http.StatusBadRequest,
		&errResp,
	)
	if errResp.Code != "bad_request" || errResp.Message != "missing model_id" {
		t.Fatalf("forward response = %#v", errResp)
	}

	postJSONStatus(t, h, "/api/v1/optim_step",
		map[string]any{"adam_params": map[string]any{"learning_rate": 1e-4}},
		http.StatusBadRequest,
		&errResp,
	)
	if errResp.Code != "bad_request" || errResp.Message != "missing model_id" {
		t.Fatalf("optim_step response = %#v", errResp)
	}

	postJSONStatus(t, h, "/api/v1/load_state_with_optimizer",
		map[string]any{"path": "tinker://model-a/weights/ckpt"},
		http.StatusBadRequest,
		&errResp,
	)
	if errResp.Code != "bad_request" || errResp.Message != "missing model_id or path" {
		t.Fatalf("load_state_with_optimizer response = %#v", errResp)
	}

	postJSONStatus(t, h, "/api/v1/load_state_with_optimizer",
		map[string]any{"model_id": "model-a"},
		http.StatusBadRequest,
		&errResp,
	)
	if errResp.Code != "bad_request" || errResp.Message != "missing model_id or path" {
		t.Fatalf("load_state_with_optimizer missing path response = %#v", errResp)
	}

	tests := []struct {
		name string
		req  map[string]any
		want string
	}{
		{
			name: "missing model reference",
			req: map[string]any{
				"num_samples":     1,
				"prompt":          map[string]any{"chunks": []any{map[string]any{"tokens": []int{1}}}},
				"sampling_params": map[string]any{"max_tokens": 1},
			},
			want: "missing sampling_session_id, model_path, or base_model",
		},
		{
			name: "empty prompt",
			req: map[string]any{
				"sampling_session_id": "sample-a",
				"num_samples":         1,
				"prompt":              map[string]any{"chunks": []any{map[string]any{"tokens": []int{}}}},
				"sampling_params":     map[string]any{"max_tokens": 1},
			},
			want: "prompt is empty",
		},
		{
			name: "image prompt",
			req: map[string]any{
				"sampling_session_id": "sample-a",
				"num_samples":         1,
				"prompt":              map[string]any{"chunks": []any{map[string]any{"type": "image_asset_pointer"}}},
				"sampling_params":     map[string]any{"max_tokens": 1},
			},
			want: `unsupported model input chunk type "image_asset_pointer"`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var resp map[string]string
			postJSONStatus(t, h, "/api/v1/asample", tt.req, http.StatusBadRequest, &resp)
			if resp["category"] != "user" || resp["error"] != tt.want {
				t.Fatalf("response = %#v, want user error %q", resp, tt.want)
			}
		})
	}
}

func TestCreateModelGetInfoAndUnload(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()

	var future FutureResponse
	postJSON(t, h, "/api/v1/create_model",
		map[string]any{
			"session_id":  "sess-a",
			"base_model":  "Qwen/Qwen3-8B",
			"lora_config": map[string]any{"rank": 8},
		},
		&future,
	)
	if future.ModelID == "" {
		t.Fatalf("future = %#v", future)
	}

	var created map[string]string
	postJSON(t, h, "/api/v1/retrieve_future",
		RetrieveFutureRequest{RequestID: future.RequestID},
		&created,
	)
	if created["type"] != "create_model" || created["model_id"] != future.ModelID {
		t.Fatalf("created = %#v", created)
	}

	var info struct {
		ModelID   string `json:"model_id"`
		ModelData struct {
			TokenizerID string `json:"tokenizer_id"`
		} `json:"model_data"`
		LoRARank int `json:"lora_rank"`
	}
	postJSON(t, h, "/api/v1/get_info", map[string]any{"model_id": future.ModelID}, &info)
	if info.ModelData.TokenizerID != "Qwen/Qwen3-8B" || info.LoRARank != 8 {
		t.Fatalf("info = %#v", info)
	}

	var unloaded FutureResponse
	postJSON(t, h, "/api/v1/unload_model", map[string]any{"model_id": future.ModelID}, &unloaded)
	if unloaded.ModelID != future.ModelID {
		t.Fatalf("unloaded = %#v", unloaded)
	}
}

func TestCheckpointActionsTrackMetadata(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LOCALTINKER_CHECKPOINT_ROOT", root)
	if err := os.MkdirAll(filepath.Join(root, "model-a", "weights", "ckpt"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "model-a", "weights", "ckpt", "adapters.safetensors"), []byte("weights"), 0644); err != nil {
		t.Fatal(err)
	}

	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.CompleteFuture(nil, map[string]any{
		"type": "save_weights",
		"path": "tinker://model-a/weights/ckpt",
	}, map[string]any{
		"type":     "save_weights",
		"model_id": "model-a",
		"path":     "tinker://model-a/weights/ckpt",
	}); err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()
	base := "/api/v1/training_runs/model-a/checkpoints/weights/ckpt"

	methodJSON(t, h, http.MethodPost, base+"/publish", nil, http.StatusOK, &map[string]any{})
	checkpoints := checkpointList(t, h)
	if len(checkpoints) != 1 {
		t.Fatalf("checkpoints = %#v, want one", checkpoints)
	}
	if !checkpoints[0].Public {
		t.Fatalf("published checkpoint = %#v", checkpoints[0])
	}
	if checkpoints[0].SizeBytes == nil || *checkpoints[0].SizeBytes != int64(len("weights")) {
		t.Fatalf("size bytes = %#v", checkpoints[0].SizeBytes)
	}

	methodJSON(t, h, http.MethodPut, base+"/ttl", map[string]any{"ttl": 3600}, http.StatusOK, &map[string]any{})
	checkpoints = checkpointList(t, h)
	if checkpoints[0].ExpiresAt == nil {
		t.Fatalf("ttl checkpoint = %#v", checkpoints[0])
	}

	req := httptest.NewRequest(http.MethodGet, base+"/archive", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusFound {
		t.Fatalf("archive status = %d, want %d body = %s", rec.Code, http.StatusFound, rec.Body.String())
	}
	if rec.Header().Get("X-Tinker-Archive-Expires-At") == "" {
		t.Fatalf("missing archive expiration metadata")
	}
	if rec.Header().Get("X-Tinker-Archive-Owner") != "local" {
		t.Fatalf("archive owner = %q", rec.Header().Get("X-Tinker-Archive-Owner"))
	}
	if rec.Header().Get("X-Tinker-Archive-Visibility") != "private" {
		t.Fatalf("archive visibility = %q", rec.Header().Get("X-Tinker-Archive-Visibility"))
	}
	location := rec.Header().Get("Location")
	u, err := url.Parse(location)
	if err != nil {
		t.Fatalf("archive location %q: %v", location, err)
	}
	if u.Scheme != "http" || u.Host != "example.com" || u.Query().Get("download") != "1" {
		t.Fatalf("archive location = %q, want http://example.com download URL", location)
	}

	req = httptest.NewRequest(http.MethodGet, u.RequestURI(), nil)
	req.Host = u.Host
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("download status = %d, want %d body = %s", rec.Code, http.StatusOK, rec.Body.String())
	}
	if !strings.Contains(rec.Header().Get("Content-Disposition"), "attachment;") {
		t.Fatalf("content disposition = %q", rec.Header().Get("Content-Disposition"))
	}
	names := map[string]bool{}
	tr := tar.NewReader(bytes.NewReader(rec.Body.Bytes()))
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		names[header.Name] = true
	}
	if !names["adapters.safetensors"] {
		t.Fatalf("download archive missing adapters.safetensors in %v", names)
	}

	req = httptest.NewRequest(http.MethodGet, base+"/archive", nil)
	req.Header.Set("X-Forwarded-Proto", "https")
	req.Header.Set("X-Forwarded-Host", "tinker.example")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusFound {
		t.Fatalf("proxied archive status = %d, want %d body = %s", rec.Code, http.StatusFound, rec.Body.String())
	}
	if got := rec.Header().Get("Location"); !strings.HasPrefix(got, "https://tinker.example/") {
		t.Fatalf("proxied archive location = %q", got)
	}

	methodJSON(t, h, http.MethodDelete, base+"/publish", nil, http.StatusOK, &map[string]any{})
	checkpoints = checkpointList(t, h)
	if checkpoints[0].Public {
		t.Fatalf("unpublished checkpoint = %#v", checkpoints[0])
	}

	methodJSON(t, h, http.MethodDelete, base, nil, http.StatusOK, &map[string]any{})
	checkpoints = checkpointList(t, h)
	if len(checkpoints) != 0 {
		t.Fatalf("checkpoints after delete = %#v", checkpoints)
	}
}

func TestExpiredCheckpointIsHiddenAndArchiveGone(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LOCALTINKER_CHECKPOINT_ROOT", root)
	if err := os.MkdirAll(filepath.Join(root, "model-a", "weights", "ckpt"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "model-a", "weights", "ckpt", "adapters.safetensors"), []byte("weights"), 0644); err != nil {
		t.Fatal(err)
	}

	store := tinkerdb.OpenMemory()
	c, err := tinkercoord.New(tinkercoord.Config{Store: store})
	if err != nil {
		t.Fatal(err)
	}
	path := "tinker://model-a/weights/ckpt"
	if _, err := c.CompleteFuture(nil, map[string]any{
		"type": "save_weights",
		"path": path,
	}, map[string]any{
		"type":     "save_weights",
		"model_id": "model-a",
		"path":     path,
	}); err != nil {
		t.Fatal(err)
	}
	expired := time.Now().UTC().Add(-time.Minute)
	if err := store.PutCheckpoint(nil, tinkerdb.Checkpoint{
		Path:      path,
		Owner:     "local",
		ExpiresAt: &expired,
	}); err != nil {
		t.Fatal(err)
	}

	h := New(c).Handler()
	if checkpoints := checkpointList(t, h); len(checkpoints) != 0 {
		t.Fatalf("checkpoints = %#v, want none", checkpoints)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/v1/training_runs/model-a/checkpoints/weights/ckpt/archive", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusGone {
		t.Fatalf("status = %d, want %d body = %s", rec.Code, http.StatusGone, rec.Body.String())
	}
	var resp ErrorResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Code != "gone" || resp.Message != "checkpoint expired" {
		t.Fatalf("response = %#v", resp)
	}
}

func postJSON(t *testing.T, h http.Handler, path string, in any, out any) {
	t.Helper()
	postJSONStatus(t, h, path, in, http.StatusOK, out)
}

func decodeDatum(t *testing.T, in any) tinkertrain.Datum {
	t.Helper()
	data, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out tinkertrain.Datum
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func postJSONStatus(t *testing.T, h http.Handler, path string, in any, wantStatus int, out any) {
	t.Helper()
	methodJSON(t, h, http.MethodPost, path, in, wantStatus, out)
}

func methodJSON(t *testing.T, h http.Handler, method, path string, in any, wantStatus int, out any) {
	t.Helper()
	var body bytes.Buffer
	if in != nil {
		if err := json.NewEncoder(&body).Encode(in); err != nil {
			t.Fatal(err)
		}
	}
	req := httptest.NewRequest(method, path, &body)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != wantStatus {
		t.Fatalf("%s status = %d, want %d body = %s", path, rec.Code, wantStatus, rec.Body.String())
	}
	if err := json.NewDecoder(rec.Body).Decode(out); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}

func checkpointList(t *testing.T, h http.Handler) []tinkercoord.Checkpoint {
	t.Helper()
	var resp tinkercoord.CheckpointsResponse
	getJSON(t, h, "/api/v1/checkpoints", &resp)
	return resp.Checkpoints
}

func getJSON(t *testing.T, h http.Handler, path string, out any) {
	t.Helper()
	req := httptest.NewRequest("GET", path, nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("%s status = %d body = %s", path, rec.Code, rec.Body.String())
	}
	if err := json.NewDecoder(rec.Body).Decode(out); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}

func trainingLoss(t *testing.T, h http.Handler, path string, body any) float64 {
	t.Helper()
	var future FutureResponse
	postJSON(t, h, path, body, &future)
	out := retrieveOK(t, h, future.RequestID)
	metrics, _ := out["metrics"].(map[string]any)
	loss, _ := metrics["loss:mean"].(float64)
	return loss
}

func retrieveOK(t *testing.T, h http.Handler, requestID string) map[string]any {
	t.Helper()
	deadline := time.Now().Add(60 * time.Second)
	for {
		var out map[string]any
		postJSON(t, h, "/api/v1/retrieve_future",
			RetrieveFutureRequest{RequestID: requestID},
			&out,
		)
		if _, ok := out["error"]; ok {
			t.Fatalf("future failed: %#v", out)
		}
		if out["type"] != "try_again" {
			return out
		}
		if time.Now().After(deadline) {
			t.Fatalf("future %s did not complete: %#v", requestID, out)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
