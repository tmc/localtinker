package tinkerhttp

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"mime"
	"net/http"
	"strings"

	"github.com/klauspost/compress/zstd"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkertrain"
)

// isProtoContent reports whether the request body is a protobuf
// ForwardBackwardRequest (Content-Type: application/x-protobuf).
func isProtoContent(r *http.Request) bool {
	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil {
		return false
	}
	return mediaType == "application/x-protobuf"
}

// readMaybeCompressed reads the request body, decompressing it first when the
// request sets Content-Encoding: zstd (mirroring the upstream ASGI middleware).
func readMaybeCompressed(r *http.Request) ([]byte, error) {
	if strings.EqualFold(strings.TrimSpace(r.Header.Get("Content-Encoding")), "zstd") {
		dec, err := zstd.NewReader(r.Body)
		if err != nil {
			return nil, fmt.Errorf("init zstd reader: %w", err)
		}
		defer dec.Close()
		return io.ReadAll(dec)
	}
	return io.ReadAll(r.Body)
}

// decodeForwardBackwardProto converts a public-wire ForwardBackwardRequest into
// the same ForwardBackwardInput the JSON path produces, so the proto request
// runs through the existing training path unchanged. It returns the model ID,
// the input, and whether the request is forward-only.
func decodeForwardBackwardProto(req *tinkerv1.ForwardBackwardRequest) (modelID string, input tinkertrain.ForwardBackwardInput, forwardOnly bool, err error) {
	input.LossFn = req.GetLossFn()
	if cfg := req.GetLossFnConfig(); len(cfg) > 0 {
		input.LossFnConfig = make(map[string]float64, len(cfg))
		for k, v := range cfg {
			input.LossFnConfig[k] = v
		}
	}
	for i, datum := range req.GetData() {
		converted, derr := protoDatum(datum)
		if derr != nil {
			return "", tinkertrain.ForwardBackwardInput{}, false, fmt.Errorf("datum %d: %w", i, derr)
		}
		input.Data = append(input.Data, converted)
	}
	return req.GetModelId(), input, req.GetForwardOnly(), nil
}

func protoDatum(datum *tinkerv1.Datum) (tinkertrain.Datum, error) {
	var out tinkertrain.Datum
	for i, chunk := range datum.GetModelInput() {
		converted, err := protoChunk(chunk)
		if err != nil {
			return tinkertrain.Datum{}, fmt.Errorf("model_input chunk %d: %w", i, err)
		}
		out.ModelInput.Chunks = append(out.ModelInput.Chunks, converted)
	}
	if inputs := datum.GetLossFnInputs(); len(inputs) > 0 {
		out.LossFnInputs = make(map[string]tinkertrain.TensorData, len(inputs))
		for name, tensor := range inputs {
			converted, err := protoTensor(tensor)
			if err != nil {
				return tinkertrain.Datum{}, fmt.Errorf("loss_fn_inputs[%q]: %w", name, err)
			}
			out.LossFnInputs[name] = converted
		}
	}
	return out, nil
}

func protoChunk(chunk *tinkerv1.Chunk) (tinkertrain.ModelInputChunk, error) {
	switch c := chunk.GetChunk().(type) {
	case *tinkerv1.Chunk_EncodedText:
		tokens, err := bytesToInt32s(c.EncodedText.GetTokens())
		if err != nil {
			return tinkertrain.ModelInputChunk{}, fmt.Errorf("encoded_text tokens: %w", err)
		}
		ints := make([]int, len(tokens))
		for i, t := range tokens {
			ints[i] = int(t)
		}
		return tinkertrain.ModelInputChunk{Type: "encoded_text", Tokens: ints}, nil
	case *tinkerv1.Chunk_Image:
		img := c.Image
		mc := tinkertrain.ModelInputChunk{
			Type:   "image",
			Format: img.GetFormat(),
			Data:   img.GetData(),
		}
		if img.ExpectedTokens != nil {
			n := int(img.GetExpectedTokens())
			mc.ExpectedTokens = &n
		}
		return mc, nil
	default:
		return tinkertrain.ModelInputChunk{}, fmt.Errorf("empty or unknown chunk")
	}
}

// protoTensor converts a public-wire Tensor into a TensorData matching the JSON
// path: values become float64, sparse CSR indices stay as int slices, and
// integer/bfloat16 dtypes collapse the same way the JSON wire does.
func protoTensor(tensor *tinkerv1.Tensor) (tinkertrain.TensorData, error) {
	out := tinkertrain.TensorData{DType: protoDTypeToString(tensor.GetDtype())}
	for _, dim := range tensor.GetShape() {
		out.Shape = append(out.Shape, int(dim))
	}
	switch enc := tensor.GetEncoding().(type) {
	case *tinkerv1.Tensor_Dense:
		values, err := bytesToFloat64s(enc.Dense, tensor.GetDtype())
		if err != nil {
			return tinkertrain.TensorData{}, err
		}
		out.Data = values
	case *tinkerv1.Tensor_SparseCsr:
		csr := enc.SparseCsr
		values, err := bytesToFloat64s(csr.GetValues(), tensor.GetDtype())
		if err != nil {
			return tinkertrain.TensorData{}, fmt.Errorf("sparse values: %w", err)
		}
		crow, err := bytesToInts(csr.GetCrowIndices())
		if err != nil {
			return tinkertrain.TensorData{}, fmt.Errorf("sparse crow_indices: %w", err)
		}
		col, err := bytesToInts(csr.GetColIndices())
		if err != nil {
			return tinkertrain.TensorData{}, fmt.Errorf("sparse col_indices: %w", err)
		}
		out.Data = values
		out.SparseCrowIndices = crow
		out.SparseColIndices = col
	default:
		return tinkertrain.TensorData{}, fmt.Errorf("tensor has no encoding")
	}
	return out, nil
}

// protoDTypeToString collapses proto dtypes the same way the JSON wire does:
// integer widths to int64 and bfloat16 to float32.
func protoDTypeToString(dt tinkerv1.DType) string {
	switch dt {
	case tinkerv1.DType_DTYPE_INT64, tinkerv1.DType_DTYPE_INT32:
		return "int64"
	default:
		return "float32"
	}
}

// bytesToFloat64s reads a dense byte buffer in the given dtype into float64s,
// matching the JSON path where all tensor values are float64.
func bytesToFloat64s(raw []byte, dt tinkerv1.DType) ([]float64, error) {
	switch dt {
	case tinkerv1.DType_DTYPE_FLOAT32:
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("float32 buffer length %d not a multiple of 4", len(raw))
		}
		out := make([]float64, len(raw)/4)
		for i := range out {
			out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
		}
		return out, nil
	case tinkerv1.DType_DTYPE_INT64:
		if len(raw)%8 != 0 {
			return nil, fmt.Errorf("int64 buffer length %d not a multiple of 8", len(raw))
		}
		out := make([]float64, len(raw)/8)
		for i := range out {
			out[i] = float64(int64(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out, nil
	case tinkerv1.DType_DTYPE_INT32:
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("int32 buffer length %d not a multiple of 4", len(raw))
		}
		out := make([]float64, len(raw)/4)
		for i := range out {
			out[i] = float64(int32(binary.LittleEndian.Uint32(raw[i*4:])))
		}
		return out, nil
	case tinkerv1.DType_DTYPE_BFLOAT16:
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("bfloat16 buffer length %d not a multiple of 2", len(raw))
		}
		out := make([]float64, len(raw)/2)
		for i := range out {
			// bfloat16 is the upper 16 bits of a float32.
			bits := uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16
			out[i] = float64(math.Float32frombits(bits))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported tensor dtype %v", dt)
	}
}

// bytesToInt32s reads a little-endian int32 buffer (encoded text tokens).
func bytesToInt32s(raw []byte) ([]int32, error) {
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("int32 buffer length %d not a multiple of 4", len(raw))
	}
	out := make([]int32, len(raw)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, nil
}

// bytesToInts reads a little-endian int64 buffer (CSR crow/col indices) into
// int slices.
func bytesToInts(raw []byte) ([]int, error) {
	if len(raw)%8 != 0 {
		return nil, fmt.Errorf("int64 buffer length %d not a multiple of 8", len(raw))
	}
	out := make([]int, len(raw)/8)
	for i := range out {
		out[i] = int(int64(binary.LittleEndian.Uint64(raw[i*8:])))
	}
	return out, nil
}
