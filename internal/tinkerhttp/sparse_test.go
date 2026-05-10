package tinkerhttp

import (
	"strings"
	"testing"

	"github.com/tmc/localtinker/internal/tinkertrain"
)

// TestRehydrateCSRMatchesDense pins the CSR-to-dense rehydration contract for
// target_tokens and weights. The sparse encoding mirrors the SDK's
// TensorData.from_torch_sparse output: data carries nnz values,
// sparse_crow_indices is the row pointer of length nrows+1, and
// sparse_col_indices is the column index of length nnz.
func TestRehydrateCSRMatchesDense(t *testing.T) {
	tests := []struct {
		name      string
		tensor    tinkertrain.TensorData
		wantData  []float64
		wantShape []int
	}{
		{
			// 2x3 sparse:
			//   row 0: col 1 = 5
			//   row 1: col 0 = 7, col 2 = 9
			// dense: [0 5 0 7 0 9]
			name: "weights 2x3 csr",
			tensor: tinkertrain.TensorData{
				Data:              []float64{5, 7, 9},
				DType:             "float32",
				Shape:             []int{2, 3},
				SparseCrowIndices: []int{0, 1, 3},
				SparseColIndices:  []int{1, 0, 2},
			},
			wantData:  []float64{0, 5, 0, 7, 0, 9},
			wantShape: []int{2, 3},
		},
		{
			// All-zero rows are valid (crow has equal consecutive entries).
			name: "weights 3x2 with empty middle row",
			tensor: tinkertrain.TensorData{
				Data:              []float64{1, 2},
				DType:             "float32",
				Shape:             []int{3, 2},
				SparseCrowIndices: []int{0, 1, 1, 2},
				SparseColIndices:  []int{0, 1},
			},
			wantData:  []float64{1, 0, 0, 0, 0, 2},
			wantShape: []int{3, 2},
		},
		{
			// Sparse target_tokens: integer dtype, fill-with-zero is the
			// padding token id by convention. Caller must align weights so
			// that padded positions carry zero weight.
			name: "target_tokens 1x4 csr",
			tensor: tinkertrain.TensorData{
				Data:              []float64{42, 17},
				DType:             "int64",
				Shape:             []int{1, 4},
				SparseCrowIndices: []int{0, 2},
				SparseColIndices:  []int{1, 3},
			},
			wantData:  []float64{0, 42, 0, 17},
			wantShape: []int{1, 4},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tinkertrain.RehydrateCSR(tt.tensor)
			if err != nil {
				t.Fatalf("RehydrateCSR error = %v", err)
			}
			if got.SparseCrowIndices != nil || got.SparseColIndices != nil {
				t.Errorf("sparse fields not cleared after rehydration: %+v", got)
			}
			if got.DType != tt.tensor.DType {
				t.Errorf("dtype = %q, want %q", got.DType, tt.tensor.DType)
			}
			if len(got.Shape) != len(tt.wantShape) {
				t.Fatalf("shape len = %d, want %d", len(got.Shape), len(tt.wantShape))
			}
			for i, d := range tt.wantShape {
				if got.Shape[i] != d {
					t.Fatalf("shape = %v, want %v", got.Shape, tt.wantShape)
				}
			}
			if len(got.Data) != len(tt.wantData) {
				t.Fatalf("data len = %d, want %d", len(got.Data), len(tt.wantData))
			}
			for i, v := range tt.wantData {
				if got.Data[i] != v {
					t.Fatalf("data[%d] = %v, want %v (full=%v)", i, got.Data[i], v, got.Data)
				}
			}
		})
	}
}

// TestRehydrateCSRRejectsMalformed pins the malformed-CSR error contract.
// Each case is a distinct way the SDK could send broken sparse metadata; all
// of them must produce an explicit user-error message.
func TestRehydrateCSRRejectsMalformed(t *testing.T) {
	tests := []struct {
		name    string
		tensor  tinkertrain.TensorData
		wantErr string
	}{
		{
			name: "missing col indices",
			tensor: tinkertrain.TensorData{
				Data: []float64{1}, DType: "float32", Shape: []int{1, 1},
				SparseCrowIndices: []int{0, 1},
			},
			wantErr: "requires both sparse_crow_indices and sparse_col_indices",
		},
		{
			name: "missing crow indices",
			tensor: tinkertrain.TensorData{
				Data: []float64{1}, DType: "float32", Shape: []int{1, 1},
				SparseColIndices: []int{0},
			},
			wantErr: "requires both sparse_crow_indices and sparse_col_indices",
		},
		{
			name: "1-D shape rejected",
			tensor: tinkertrain.TensorData{
				Data: []float64{1}, DType: "float32", Shape: []int{1},
				SparseCrowIndices: []int{0, 1}, SparseColIndices: []int{0},
			},
			wantErr: "requires 2-D shape",
		},
		{
			name: "3-D shape rejected",
			tensor: tinkertrain.TensorData{
				Data: []float64{1}, DType: "float32", Shape: []int{1, 1, 1},
				SparseCrowIndices: []int{0, 1}, SparseColIndices: []int{0},
			},
			wantErr: "requires 2-D shape",
		},
		{
			name: "crow length mismatch",
			tensor: tinkertrain.TensorData{
				Data: []float64{1}, DType: "float32", Shape: []int{2, 2},
				SparseCrowIndices: []int{0, 1}, SparseColIndices: []int{0},
			},
			wantErr: "sparse_crow_indices length 2, want 3",
		},
		{
			name: "crow does not start at zero",
			tensor: tinkertrain.TensorData{
				Data: []float64{1}, DType: "float32", Shape: []int{1, 2},
				SparseCrowIndices: []int{1, 2}, SparseColIndices: []int{0},
			},
			wantErr: "sparse_crow_indices[0] = 1, want 0",
		},
		{
			name: "col indices length mismatch",
			tensor: tinkertrain.TensorData{
				Data: []float64{1, 2}, DType: "float32", Shape: []int{1, 2},
				SparseCrowIndices: []int{0, 2}, SparseColIndices: []int{0},
			},
			wantErr: "sparse_col_indices length 1, want 2",
		},
		{
			name: "data length mismatch",
			tensor: tinkertrain.TensorData{
				Data: []float64{1}, DType: "float32", Shape: []int{1, 2},
				SparseCrowIndices: []int{0, 2}, SparseColIndices: []int{0, 1},
			},
			wantErr: "sparse data length 1, want 2",
		},
		{
			// nrows=3, crow=[0,2,1,2]: middle element decreases. nnz=crow[3]=2,
			// matching data and col lengths so the monotonic check is reached.
			name: "crow not monotonically non-decreasing",
			tensor: tinkertrain.TensorData{
				Data: []float64{1, 2}, DType: "float32", Shape: []int{3, 2},
				SparseCrowIndices: []int{0, 2, 1, 2}, SparseColIndices: []int{0, 1},
			},
			wantErr: "not monotonically non-decreasing",
		},
		{
			name: "col index out of range",
			tensor: tinkertrain.TensorData{
				Data: []float64{1}, DType: "float32", Shape: []int{1, 2},
				SparseCrowIndices: []int{0, 1}, SparseColIndices: []int{5},
			},
			wantErr: "out of range",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tinkertrain.RehydrateCSR(tt.tensor)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

// TestNormalizeTensorDataRehydratesNamedSparse verifies that
// normalizeTensorData rehydrates sparse target_tokens and sparse weights to
// the same dense values that an equivalent dense TensorData would carry, and
// that other tensor names (which are not actually reachable through
// cross_entropy today, but form a defense-in-depth boundary) keep the
// sparse-rejection contract.
func TestNormalizeTensorDataRehydratesNamedSparse(t *testing.T) {
	t.Run("target_tokens csr matches dense", func(t *testing.T) {
		sparse := tinkertrain.TensorData{
			Data:              []float64{42, 17},
			DType:             "int64",
			Shape:             []int{1, 4},
			SparseCrowIndices: []int{0, 2},
			SparseColIndices:  []int{1, 3},
		}
		got, err := normalizeTensorData("target_tokens", sparse)
		if err != nil {
			t.Fatalf("normalize sparse error = %v", err)
		}
		want := []float64{0, 42, 0, 17}
		if len(got.Data) != len(want) {
			t.Fatalf("data = %v, want %v", got.Data, want)
		}
		for i, v := range want {
			if got.Data[i] != v {
				t.Fatalf("data[%d] = %v, want %v", i, got.Data[i], v)
			}
		}
		if got.SparseCrowIndices != nil || got.SparseColIndices != nil {
			t.Errorf("sparse fields not cleared: %+v", got)
		}
	})

	t.Run("weights csr matches dense", func(t *testing.T) {
		sparse := tinkertrain.TensorData{
			Data:              []float64{0.5, 0.25, 1.0},
			DType:             "float32",
			Shape:             []int{2, 3},
			SparseCrowIndices: []int{0, 1, 3},
			SparseColIndices:  []int{2, 0, 1},
		}
		got, err := normalizeTensorData("weights", sparse)
		if err != nil {
			t.Fatalf("normalize sparse error = %v", err)
		}
		want := []float64{0, 0, 0.5, 0.25, 1.0, 0}
		for i, v := range want {
			if got.Data[i] != v {
				t.Fatalf("data[%d] = %v, want %v (full=%v)", i, got.Data[i], v, got.Data)
			}
		}
	})

	t.Run("sparse weights end to end matches dense", func(t *testing.T) {
		// 1x4 sparse weights at columns 0 and 2: values 0.5 and 0.25.
		// Dense equivalent: [0.5, 0, 0.25, 0].
		sparseInput := tinkertrain.ForwardBackwardInput{
			LossFn: "cross_entropy",
			Data: []tinkertrain.Datum{{
				ModelInput: tinkertrain.ModelInput{Chunks: []tinkertrain.ModelInputChunk{{
					Type:   "encoded_text",
					Tokens: []int{1, 2, 3, 4},
				}}},
				LossFnInputs: map[string]tinkertrain.TensorData{
					"target_tokens": {Data: []float64{1, 2, 3, 4}, DType: "int64", Shape: []int{1, 4}},
					"weights": {
						Data:              []float64{0.5, 0.25},
						DType:             "float32",
						Shape:             []int{1, 4},
						SparseCrowIndices: []int{0, 2},
						SparseColIndices:  []int{0, 2},
					},
				},
			}},
		}
		denseInput := tinkertrain.ForwardBackwardInput{
			LossFn: "cross_entropy",
			Data: []tinkertrain.Datum{{
				ModelInput: tinkertrain.ModelInput{Chunks: []tinkertrain.ModelInputChunk{{
					Type:   "encoded_text",
					Tokens: []int{1, 2, 3, 4},
				}}},
				LossFnInputs: map[string]tinkertrain.TensorData{
					"target_tokens": {Data: []float64{1, 2, 3, 4}, DType: "int64", Shape: []int{1, 4}},
					"weights":       {Data: []float64{0.5, 0, 0.25, 0}, DType: "float32", Shape: []int{1, 4}},
				},
			}},
		}
		if err := normalizeAndValidateInput(&sparseInput); err != nil {
			t.Fatalf("normalize sparse error = %v", err)
		}
		if err := normalizeAndValidateInput(&denseInput); err != nil {
			t.Fatalf("normalize dense error = %v", err)
		}
		sw := sparseInput.Data[0].LossFnInputs["weights"]
		dw := denseInput.Data[0].LossFnInputs["weights"]
		if sw.SparseCrowIndices != nil || sw.SparseColIndices != nil {
			t.Fatalf("sparse fields not cleared after normalize: %+v", sw)
		}
		if len(sw.Data) != len(dw.Data) {
			t.Fatalf("sparse data len %d != dense data len %d", len(sw.Data), len(dw.Data))
		}
		for i := range sw.Data {
			if sw.Data[i] != dw.Data[i] {
				t.Fatalf("data[%d]: sparse=%v dense=%v", i, sw.Data[i], dw.Data[i])
			}
		}
	})

	t.Run("unknown name keeps sparse rejection", func(t *testing.T) {
		sparse := tinkertrain.TensorData{
			Data:              []float64{1},
			DType:             "float32",
			Shape:             []int{1, 1},
			SparseCrowIndices: []int{0, 1},
			SparseColIndices:  []int{0},
		}
		_, err := normalizeTensorData("advantages", sparse)
		if err == nil {
			t.Fatal("normalizeTensorData(advantages, sparse) = nil, want error")
		}
		if !strings.Contains(err.Error(), `sparse tensors are not supported for "advantages"`) {
			t.Fatalf("error = %v, want sparse-rejection for non-CE tensor", err)
		}
	})
}
