package tinkertrain

import (
	"math"
	"testing"

	"github.com/tmc/mlx-go/mlx"
)

func TestDenseCrossEntropyBatchAcceptsArbitraryTargetsAndWeights(t *testing.T) {
	input := ForwardBackwardInput{
		LossFn: "cross_entropy",
		Data: []Datum{
			{
				ModelInput: ModelInput{Chunks: []ModelInputChunk{{
					Type:   "encoded_text",
					Tokens: []int{10, 11, 12, 13},
				}}},
				LossFnInputs: map[string]TensorData{
					"target_tokens": {
						Data:  []float64{30, 31, 32, 33},
						DType: "int64",
						Shape: []int{2, 2},
					},
					"weights": {
						Data:  []float64{0, 0.25, 1, 0.5},
						DType: "float32",
						Shape: []int{2, 2},
					},
				},
			},
			{
				ModelInput: ModelInput{Chunks: []ModelInputChunk{{
					Type:   "encoded_text",
					Tokens: []int{20, 21},
				}}},
				LossFnInputs: map[string]TensorData{
					"target_tokens": {
						Data:  []float64{40, 41},
						DType: "int64",
					},
				},
			},
		},
	}

	batch, err := newDenseBatch(input)
	if err != nil {
		t.Fatal(err)
	}
	if batch.seqLen != 4 {
		t.Fatalf("seqLen = %d, want 4", batch.seqLen)
	}
	if batch.weightSum != 3.75 {
		t.Fatalf("weightSum = %v, want 3.75", batch.weightSum)
	}
	if got, want := batch.rows[0].outputShape, []int{2, 2}; !sameInts(got, want) {
		t.Fatalf("row 0 shape = %v, want %v", got, want)
	}
	if got, want := batch.rows[1].outputShape, []int{2}; !sameInts(got, want) {
		t.Fatalf("row 1 shape = %v, want %v", got, want)
	}
	if got, want := batch.rows[0].targets, []int32{30, 31, 32, 33}; !sameInt32s(got, want) {
		t.Fatalf("row 0 targets = %v, want %v", got, want)
	}
}

func TestDenseCrossEntropyBatchRejectsBadTargetsAndWeights(t *testing.T) {
	base := func() ForwardBackwardInput {
		return ForwardBackwardInput{
			LossFn: "cross_entropy",
			Data: []Datum{{
				ModelInput: ModelInput{Chunks: []ModelInputChunk{{Tokens: []int{1, 2}}}},
				LossFnInputs: map[string]TensorData{
					"target_tokens": {Data: []float64{3, 4}, DType: "int64"},
				},
			}},
		}
	}
	tests := []struct {
		name string
		edit func(*ForwardBackwardInput)
	}{
		{
			name: "fractional target",
			edit: func(in *ForwardBackwardInput) {
				in.Data[0].LossFnInputs["target_tokens"] = TensorData{Data: []float64{3.5, 4}, DType: "int64"}
			},
		},
		{
			name: "negative weight",
			edit: func(in *ForwardBackwardInput) {
				in.Data[0].LossFnInputs["weights"] = TensorData{Data: []float64{1, -1}, DType: "float32"}
			},
		},
		{
			name: "zero total weight",
			edit: func(in *ForwardBackwardInput) {
				in.Data[0].LossFnInputs["weights"] = TensorData{Data: []float64{0, 0}, DType: "float32"}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := base()
			tt.edit(&in)
			if _, err := newDenseBatch(in); err == nil {
				t.Fatal("newDenseBatch succeeded, want error")
			}
		})
	}
}

func TestDenseCrossEntropyReturnsWeightedLossAndLogprobs(t *testing.T) {
	logits, err := mlx.FromSlice([]float32{
		0, 1, 2,
		2, 0, -2,
	}, []int{1, 2, 3}, mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer logits.Free()
	targets, err := mlx.FromSlice([]int32{2, 0}, []int{1, 2}, mlx.Int32)
	if err != nil {
		t.Fatal(err)
	}
	defer targets.Free()
	weights, err := mlx.FromSlice([]float32{1, 0.5}, []int{1, 2}, mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Free()

	loss, logprobs, err := denseCrossEntropy(logits, targets, weights)
	if err != nil {
		t.Fatal(err)
	}
	defer loss.Free()
	defer logprobs.Free()
	if err := mlx.Eval(loss, logprobs); err != nil {
		t.Fatal(err)
	}
	gotLoss, err := mlx.ItemAs[float32](loss)
	if err != nil {
		t.Fatal(err)
	}
	gotLogprobs, err := mlx.ToSlice[float32](logprobs)
	if err != nil {
		t.Fatal(err)
	}
	wantLogprobs := []float64{
		2 - logsumexp(0, 1, 2),
		2 - logsumexp(2, 0, -2),
	}
	if len(gotLogprobs) != len(wantLogprobs) {
		t.Fatalf("logprobs = %v, want %v", gotLogprobs, wantLogprobs)
	}
	for i := range gotLogprobs {
		if !near(float64(gotLogprobs[i]), wantLogprobs[i]) {
			t.Fatalf("logprobs[%d] = %v, want %v", i, gotLogprobs[i], wantLogprobs[i])
		}
	}
	wantLoss := (-wantLogprobs[0] - 0.5*wantLogprobs[1]) / 1.5
	if !near(float64(gotLoss), wantLoss) {
		t.Fatalf("loss = %v, want %v", gotLoss, wantLoss)
	}
}

func sameInts(a, b []int) bool {
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

func sameInt32s(a, b []int32) bool {
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

func logsumexp(xs ...float64) float64 {
	max := xs[0]
	for _, x := range xs[1:] {
		if x > max {
			max = x
		}
	}
	var sum float64
	for _, x := range xs {
		sum += math.Exp(x - max)
	}
	return max + math.Log(sum)
}

func near(a, b float64) bool {
	return math.Abs(a-b) < 1e-5
}
