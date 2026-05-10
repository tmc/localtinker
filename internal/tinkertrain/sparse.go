package tinkertrain

import (
	"errors"
	"fmt"
)

// RehydrateCSR converts a 2-D CSR-encoded TensorData into the equivalent dense
// TensorData with zero fill at unset positions. The Python Tinker SDK encodes
// 2-D weights and (rarely) target_tokens as CSR via TensorData.from_torch_sparse:
// data carries nnz values, sparse_crow_indices is the row pointer of length
// nrows+1, sparse_col_indices is the column index of length nnz, and shape is
// required and 2-D. The returned TensorData has the same dtype and a dense
// flattened payload of length nrows*ncols, with the sparse fields cleared so
// downstream code sees a dense tensor.
//
// Higher-rank or otherwise non-2-D sparse tensors are rejected; MLX-native
// sparse ops are not supported.
func RehydrateCSR(tensor TensorData) (TensorData, error) {
	if tensor.SparseCrowIndices == nil || tensor.SparseColIndices == nil {
		return tensor, errors.New("sparse tensor requires both sparse_crow_indices and sparse_col_indices")
	}
	if len(tensor.Shape) != 2 {
		return tensor, fmt.Errorf("sparse tensor requires 2-D shape, got %v", tensor.Shape)
	}
	nrows, ncols := tensor.Shape[0], tensor.Shape[1]
	if nrows < 0 || ncols < 0 {
		return tensor, fmt.Errorf("sparse tensor shape has negative dimension: %v", tensor.Shape)
	}
	crow := tensor.SparseCrowIndices
	col := tensor.SparseColIndices
	if len(crow) != nrows+1 {
		return tensor, fmt.Errorf("sparse_crow_indices length %d, want %d (nrows+1)", len(crow), nrows+1)
	}
	if crow[0] != 0 {
		return tensor, fmt.Errorf("sparse_crow_indices[0] = %d, want 0", crow[0])
	}
	nnz := crow[nrows]
	if nnz != len(col) {
		return tensor, fmt.Errorf("sparse_col_indices length %d, want %d (nnz from crow)", len(col), nnz)
	}
	if nnz != len(tensor.Data) {
		return tensor, fmt.Errorf("sparse data length %d, want %d (nnz from crow)", len(tensor.Data), nnz)
	}
	for i := 1; i < len(crow); i++ {
		if crow[i] < crow[i-1] {
			return tensor, fmt.Errorf("sparse_crow_indices not monotonically non-decreasing at %d", i)
		}
	}
	dense := make([]float64, nrows*ncols)
	for r := 0; r < nrows; r++ {
		start, end := crow[r], crow[r+1]
		for k := start; k < end; k++ {
			c := col[k]
			if c < 0 || c >= ncols {
				return tensor, fmt.Errorf("sparse_col_indices[%d] = %d out of range [0, %d)", k, c, ncols)
			}
			dense[r*ncols+c] = tensor.Data[k]
		}
	}
	return TensorData{
		Data:  dense,
		DType: tensor.DType,
		Shape: []int{nrows, ncols},
	}, nil
}
