package tinkertrain

import (
	"reflect"
	"strings"
	"testing"
)

// TestDatumTargetsRehydratesSparse pins that direct Go API callers can pass
// 2-D CSR sparse target_tokens and get the same dense int slice as the
// equivalent dense TensorData.
func TestDatumTargetsRehydratesSparse(t *testing.T) {
	sparse := Datum{LossFnInputs: map[string]TensorData{
		"target_tokens": {
			Data:              []float64{7, 5},
			DType:             "int64",
			Shape:             []int{1, 4},
			SparseCrowIndices: []int{0, 2},
			SparseColIndices:  []int{0, 2},
		},
	}}
	dense := Datum{LossFnInputs: map[string]TensorData{
		"target_tokens": {
			Data:  []float64{7, 0, 5, 0},
			DType: "int64",
			Shape: []int{1, 4},
		},
	}}
	gotSparse, err := sparse.targets()
	if err != nil {
		t.Fatalf("sparse targets() error = %v", err)
	}
	gotDense, err := dense.targets()
	if err != nil {
		t.Fatalf("dense targets() error = %v", err)
	}
	if !reflect.DeepEqual(gotSparse, gotDense) {
		t.Fatalf("sparse targets = %v, dense targets = %v", gotSparse, gotDense)
	}
	want := []int{7, 0, 5, 0}
	if !reflect.DeepEqual(gotSparse, want) {
		t.Fatalf("targets = %v, want %v", gotSparse, want)
	}
}

// TestDatumWeightsRehydratesSparse pins that direct Go API callers can pass
// 2-D CSR sparse weights and get the same dense float slice as the equivalent
// dense TensorData.
func TestDatumWeightsRehydratesSparse(t *testing.T) {
	sparse := Datum{LossFnInputs: map[string]TensorData{
		"weights": {
			Data:              []float64{0.5, 0.25},
			DType:             "float32",
			Shape:             []int{1, 4},
			SparseCrowIndices: []int{0, 2},
			SparseColIndices:  []int{0, 2},
		},
	}}
	dense := Datum{LossFnInputs: map[string]TensorData{
		"weights": {
			Data:  []float64{0.5, 0, 0.25, 0},
			DType: "float32",
			Shape: []int{1, 4},
		},
	}}
	gotSparse, err := sparse.weights(4)
	if err != nil {
		t.Fatalf("sparse weights() error = %v", err)
	}
	gotDense, err := dense.weights(4)
	if err != nil {
		t.Fatalf("dense weights() error = %v", err)
	}
	if !reflect.DeepEqual(gotSparse, gotDense) {
		t.Fatalf("sparse weights = %v, dense weights = %v", gotSparse, gotDense)
	}
	want := []float64{0.5, 0, 0.25, 0}
	if !reflect.DeepEqual(gotSparse, want) {
		t.Fatalf("weights = %v, want %v", gotSparse, want)
	}
}

// TestRehydrateCSRMatchesDense pins the package-level RehydrateCSR contract
// for a small set of representative shapes including an empty middle row.
func TestRehydrateCSRMatchesDense(t *testing.T) {
	cases := []struct {
		name   string
		tensor TensorData
		want   []float64
		shape  []int
	}{
		{
			name: "2x3 csr",
			tensor: TensorData{
				Data:              []float64{1, 2, 3},
				DType:             "float32",
				Shape:             []int{2, 3},
				SparseCrowIndices: []int{0, 2, 3},
				SparseColIndices:  []int{0, 2, 1},
			},
			want:  []float64{1, 0, 2, 0, 3, 0},
			shape: []int{2, 3},
		},
		{
			name: "3x2 with empty middle row",
			tensor: TensorData{
				Data:              []float64{0.5, 0.25},
				DType:             "float32",
				Shape:             []int{3, 2},
				SparseCrowIndices: []int{0, 1, 1, 2},
				SparseColIndices:  []int{0, 1},
			},
			want:  []float64{0.5, 0, 0, 0, 0, 0.25},
			shape: []int{3, 2},
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RehydrateCSR(tt.tensor)
			if err != nil {
				t.Fatalf("RehydrateCSR error = %v", err)
			}
			if !reflect.DeepEqual(got.Data, tt.want) {
				t.Fatalf("Data = %v, want %v", got.Data, tt.want)
			}
			if !reflect.DeepEqual(got.Shape, tt.shape) {
				t.Fatalf("Shape = %v, want %v", got.Shape, tt.shape)
			}
			if got.SparseCrowIndices != nil || got.SparseColIndices != nil {
				t.Fatalf("sparse fields not cleared: %+v", got)
			}
			if got.DType != tt.tensor.DType {
				t.Fatalf("DType = %q, want %q", got.DType, tt.tensor.DType)
			}
		})
	}
}

// TestRehydrateCSRRejectsMalformed pins the malformed-CSR error contract for
// the package-level helper.
func TestRehydrateCSRRejectsMalformed(t *testing.T) {
	cases := []struct {
		name    string
		tensor  TensorData
		wantErr string
	}{
		{
			name: "missing col indices",
			tensor: TensorData{
				Data:              []float64{1},
				Shape:             []int{1, 2},
				SparseCrowIndices: []int{0, 1},
			},
			wantErr: "sparse tensor requires both sparse_crow_indices and sparse_col_indices",
		},
		{
			name: "1-D shape",
			tensor: TensorData{
				Data:              []float64{1},
				Shape:             []int{4},
				SparseCrowIndices: []int{0, 1},
				SparseColIndices:  []int{0},
			},
			wantErr: "sparse tensor requires 2-D shape",
		},
		{
			name: "crow not starting at zero",
			tensor: TensorData{
				Data:              []float64{1},
				Shape:             []int{1, 2},
				SparseCrowIndices: []int{1, 2},
				SparseColIndices:  []int{0},
			},
			wantErr: "sparse_crow_indices[0] = 1, want 0",
		},
		{
			name: "col index out of range",
			tensor: TensorData{
				Data:              []float64{1},
				Shape:             []int{1, 2},
				SparseCrowIndices: []int{0, 1},
				SparseColIndices:  []int{5},
			},
			wantErr: "out of range",
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			_, err := RehydrateCSR(tt.tensor)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}
