package tinkerhttp

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/klauspost/compress/zstd"
	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerdb"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkertrain"
	"google.golang.org/protobuf/proto"
)

func float32Bytes(vals ...float32) []byte {
	b := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

func int64Bytes(vals ...int64) []byte {
	b := make([]byte, 8*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint64(b[i*8:], uint64(v))
	}
	return b
}

func int32Bytes(vals ...int32) []byte {
	b := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], uint32(v))
	}
	return b
}

// TestDecodeForwardBackwardProtoParity proves a proto ForwardBackwardRequest
// decodes into the same ForwardBackwardInput the JSON path builds, so the proto
// branch runs the existing training path unchanged.
func TestDecodeForwardBackwardProtoParity(t *testing.T) {
	msg := &tinkerv1.ForwardBackwardRequest{
		ModelId: "model-a",
		SeqId:   1,
		LossFn:  "cross_entropy",
		Data: []*tinkerv1.Datum{{
			ModelInput: []*tinkerv1.Chunk{{
				Chunk: &tinkerv1.Chunk_EncodedText{
					EncodedText: &tinkerv1.EncodedTextChunk{Tokens: int32Bytes(1, 2, 3, 4)},
				},
			}},
			LossFnInputs: map[string]*tinkerv1.Tensor{
				"target_tokens": {
					Encoding: &tinkerv1.Tensor_Dense{Dense: int64Bytes(9, 8, 7, 6)},
					Dtype:    tinkerv1.DType_DTYPE_INT64,
					Shape:    []int64{2, 2},
				},
				"weights": {
					Encoding: &tinkerv1.Tensor_Dense{Dense: float32Bytes(1, 0.5, 0, 1)},
					Dtype:    tinkerv1.DType_DTYPE_FLOAT32,
					Shape:    []int64{2, 2},
				},
			},
		}},
	}

	modelID, got, forwardOnly, err := decodeForwardBackwardProto(msg)
	if err != nil {
		t.Fatal(err)
	}
	if modelID != "model-a" {
		t.Fatalf("model_id = %q, want model-a", modelID)
	}
	if forwardOnly {
		t.Fatal("forward_only = true, want false")
	}

	want := tinkertrain.ForwardBackwardInput{
		LossFn: "cross_entropy",
		Data: []tinkertrain.Datum{{
			ModelInput: tinkertrain.ModelInput{Chunks: []tinkertrain.ModelInputChunk{{
				Type:   "encoded_text",
				Tokens: []int{1, 2, 3, 4},
			}}},
			LossFnInputs: map[string]tinkertrain.TensorData{
				"target_tokens": {Data: []float64{9, 8, 7, 6}, DType: "int64", Shape: []int{2, 2}},
				"weights":       {Data: []float64{1, 0.5, 0, 1}, DType: "float32", Shape: []int{2, 2}},
			},
		}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("decoded input mismatch:\n got = %#v\nwant = %#v", got, want)
	}

	// The decoded input passes the same validation as the JSON path.
	if err := normalizeAndValidateInput(&got); err != nil {
		t.Fatalf("decoded input failed validation: %v", err)
	}
}

// TestDecodeProtoBFloat16Widening locks the bfloat16 -> float32 widening: a
// bfloat16 value is the upper 16 bits of its float32 bit pattern.
func TestDecodeProtoBFloat16Widening(t *testing.T) {
	// 1.5f is 0x3FC00000; its bfloat16 form is the upper half 0x3FC0.
	bf16 := []byte{0xC0, 0x3F} // little-endian uint16 0x3FC0
	got, err := bytesToFloat64s(bf16, tinkerv1.DType_DTYPE_BFLOAT16)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0] != 1.5 {
		t.Fatalf("bfloat16 widen = %v, want [1.5]", got)
	}
}

// TestDecodeProtoSparseCsr locks the CSR decode: values become float64, crow and
// col indices stay as int slices.
func TestDecodeProtoSparseCsr(t *testing.T) {
	tensor := &tinkerv1.Tensor{
		Encoding: &tinkerv1.Tensor_SparseCsr{SparseCsr: &tinkerv1.SparseCsr{
			Values:      float32Bytes(2.5, -1.0),
			CrowIndices: int64Bytes(0, 1, 2),
			ColIndices:  int64Bytes(0, 1),
		}},
		Dtype: tinkerv1.DType_DTYPE_FLOAT32,
		Shape: []int64{2, 2},
	}
	got, err := protoTensor(tensor)
	if err != nil {
		t.Fatal(err)
	}
	want := tinkertrain.TensorData{
		Data:              []float64{2.5, -1.0},
		DType:             "float32",
		Shape:             []int{2, 2},
		SparseCrowIndices: []int{0, 1, 2},
		SparseColIndices:  []int{0, 1},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("csr decode mismatch:\n got = %#v\nwant = %#v", got, want)
	}
}

// TestForwardBackwardProtoRoute submits a protobuf forward_backward request over
// HTTP, plain and zstd-compressed, and checks the server accepts both and
// creates a future.
//
// The route enqueues a future against any non-empty model_id without loading the
// model, so the test uses a synthetic model_id and needs no MLX weights — the
// model-gated training behavior is covered by the JSON forward_backward test.
func TestForwardBackwardProtoRoute(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()

	const modelID = "model-a"

	msg := &tinkerv1.ForwardBackwardRequest{
		ModelId: modelID,
		SeqId:   1,
		LossFn:  "cross_entropy",
		Data: []*tinkerv1.Datum{{
			ModelInput: []*tinkerv1.Chunk{{
				Chunk: &tinkerv1.Chunk_EncodedText{
					EncodedText: &tinkerv1.EncodedTextChunk{Tokens: int32Bytes(1, 2, 3, 4)},
				},
			}},
			LossFnInputs: map[string]*tinkerv1.Tensor{
				"target_tokens": {
					Encoding: &tinkerv1.Tensor_Dense{Dense: int64Bytes(9, 8, 7, 6)},
					Dtype:    tinkerv1.DType_DTYPE_INT64,
					Shape:    []int64{2, 2},
				},
				"weights": {
					Encoding: &tinkerv1.Tensor_Dense{Dense: float32Bytes(1, 0.5, 0, 1)},
					Dtype:    tinkerv1.DType_DTYPE_FLOAT32,
					Shape:    []int64{2, 2},
				},
			},
		}},
	}
	body, err := proto.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	// Plain proto body.
	var future FutureResponse
	postProto(t, h, body, "", &future)
	if future.RequestID == "" {
		t.Fatalf("proto forward_backward returned no future: %#v", future)
	}

	// zstd-compressed proto body.
	var compressed bytes.Buffer
	enc, err := zstd.NewWriter(&compressed)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := enc.Write(body); err != nil {
		t.Fatal(err)
	}
	if err := enc.Close(); err != nil {
		t.Fatal(err)
	}
	var future2 FutureResponse
	postProto(t, h, compressed.Bytes(), "zstd", &future2)
	if future2.RequestID == "" {
		t.Fatalf("zstd proto forward_backward returned no future: %#v", future2)
	}
}

// TestForwardBackwardProtoZstdBomb proves the decompressed-size cap rejects a
// small zstd body that inflates past the request limit, instead of buffering it
// all into memory. MaxBytesReader bounds only the compressed bytes, so the cap
// in readMaybeCompressed is what protects this path.
func TestForwardBackwardProtoZstdBomb(t *testing.T) {
	c, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	h := New(c).Handler()

	limit := int(c.ClientConfig(context.Background()).MaxRequestBytes)
	if limit <= 0 {
		t.Fatalf("MaxRequestBytes = %d, want > 0", limit)
	}

	// A highly compressible payload larger than the limit: a few bytes of zstd
	// that inflate well past MaxRequestBytes.
	var compressed bytes.Buffer
	enc, err := zstd.NewWriter(&compressed)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := enc.Write(bytes.Repeat([]byte{0}, limit+1024)); err != nil {
		t.Fatal(err)
	}
	if err := enc.Close(); err != nil {
		t.Fatal(err)
	}
	if compressed.Len() >= limit {
		t.Fatalf("compressed payload %d not below limit %d; test would not exercise the cap", compressed.Len(), limit)
	}

	rec := postProtoStatus(t, h, compressed.Bytes(), "zstd")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("zstd bomb status = %d, want 400; body = %s", rec.Code, rec.Body.String())
	}
}

func postProto(t *testing.T, h http.Handler, body []byte, encoding string, out any) {
	t.Helper()
	rec := postProtoStatus(t, h, body, encoding)
	if rec.Code != http.StatusOK {
		t.Fatalf("proto forward_backward status = %d body = %s", rec.Code, rec.Body.String())
	}
	if err := json.NewDecoder(rec.Body).Decode(out); err != nil {
		t.Fatalf("decode proto forward_backward response: %v", err)
	}
}

func postProtoStatus(t *testing.T, h http.Handler, body []byte, encoding string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/forward_backward", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/x-protobuf")
	if encoding != "" {
		req.Header.Set("Content-Encoding", encoding)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}
