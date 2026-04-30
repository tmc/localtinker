package tinkerhttp

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

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

func TestRetrieveFutureRoute(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
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

	var result map[string]bool
	postJSON(t, New(c).Handler(), "/api/v1/retrieve_future",
		RetrieveFutureRequest{FutureID: future.ID},
		&result,
	)
	if !result["ok"] {
		t.Fatalf("result = %#v, want ok", result)
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
}

func TestForwardBackwardAndOptimStepTune(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
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
	var out struct {
		Metrics map[string]float64 `json:"metrics"`
	}
	postJSON(t, h, "/api/v1/retrieve_future",
		RetrieveFutureRequest{RequestID: future.RequestID},
		&out,
	)
	return out.Metrics["loss:mean"]
}

func retrieveOK(t *testing.T, h http.Handler, requestID string) {
	t.Helper()
	var out map[string]any
	postJSON(t, h, "/api/v1/retrieve_future",
		RetrieveFutureRequest{RequestID: requestID},
		&out,
	)
	if _, ok := out["error"]; ok {
		t.Fatalf("future failed: %#v", out)
	}
}
