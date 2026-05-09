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
		name    string
		edit    func(*ForwardBackwardInput)
		wantErr bool
	}{
		{
			name: "fractional target",
			edit: func(in *ForwardBackwardInput) {
				in.Data[0].LossFnInputs["target_tokens"] = TensorData{Data: []float64{3.5, 4}, DType: "int64"}
			},
			wantErr: true,
		},
		{
			name: "out of range target",
			edit: func(in *ForwardBackwardInput) {
				in.Data[0].LossFnInputs["target_tokens"] = TensorData{Data: []float64{float64(math.MaxInt32) + 1, 4}, DType: "int64"}
			},
			wantErr: true,
		},
		{
			name: "negative weight",
			edit: func(in *ForwardBackwardInput) {
				in.Data[0].LossFnInputs["weights"] = TensorData{Data: []float64{1, -1}, DType: "float32"}
			},
			wantErr: true,
		},
		{
			name: "non finite weight",
			edit: func(in *ForwardBackwardInput) {
				in.Data[0].LossFnInputs["weights"] = TensorData{Data: []float64{1, math.Inf(1)}, DType: "float32"}
			},
			wantErr: true,
		},
		{
			name: "zero total weight",
			edit: func(in *ForwardBackwardInput) {
				in.Data[0].LossFnInputs["weights"] = TensorData{Data: []float64{0, 0}, DType: "float32"}
			},
		},
		{
			// Well-formed image chunk: parse layer accepts it, but the MLX
			// executor refuses multimodal at the boundary.
			name: "well formed image chunk refused at executor",
			edit: func(in *ForwardBackwardInput) {
				two := 2
				in.Data[0].ModelInput.Chunks = []ModelInputChunk{{
					Type:           "image",
					Format:         "png",
					Data:           validPNG,
					ExpectedTokens: &two,
				}}
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := base()
			tt.edit(&in)
			_, err := newDenseBatch(in)
			if tt.wantErr && err == nil {
				t.Fatal("newDenseBatch succeeded, want error")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("newDenseBatch error = %v, want nil", err)
			}
		})
	}
}

func TestDenseCrossEntropyBatchShapeTable(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
	}{
		{name: "flat", shape: []int{4}},
		{name: "rectangular", shape: []int{2, 2}},
		{name: "row", shape: []int{1, 4}},
		{name: "column", shape: []int{4, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := ForwardBackwardInput{
				LossFn: "cross_entropy",
				Data: []Datum{{
					ModelInput: ModelInput{Chunks: []ModelInputChunk{{
						Type:   "encoded_text",
						Tokens: []int{10, 11, 12, 13},
					}}},
					LossFnInputs: map[string]TensorData{
						"target_tokens": {Data: []float64{4, 3, 2, 1}, DType: "int64", Shape: tt.shape},
						"weights":       {Data: []float64{1, 0, 0.5, 2}, DType: "float32", Shape: tt.shape},
					},
				}},
			}
			batch, err := newDenseBatch(input)
			if err != nil {
				t.Fatal(err)
			}
			if got := batch.rows[0].outputShape; !sameInts(got, tt.shape) {
				t.Fatalf("output shape = %v, want %v", got, tt.shape)
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

func TestDenseCrossEntropyFractionalWeights(t *testing.T) {
	// Interleaved non-prefix fractional weights: zeros and fractions appear in
	// the middle of the sequence, not as a leading mask. Locks the contract
	// that training accepts arbitrary dense weights, not only zero-prefix
	// followed by ones.
	t.Run("batch accepts interleaved fractional weights", func(t *testing.T) {
		input := ForwardBackwardInput{
			LossFn: "cross_entropy",
			Data: []Datum{{
				ModelInput: ModelInput{Chunks: []ModelInputChunk{{
					Type:   "encoded_text",
					Tokens: []int{10, 11, 12, 13, 14, 15},
				}}},
				LossFnInputs: map[string]TensorData{
					"target_tokens": {Data: []float64{20, 21, 22, 23, 24, 25}, DType: "int64"},
					"weights":       {Data: []float64{1, 0, 0.3, 0, 0.7, 1}, DType: "float32"},
				},
			}},
		}
		batch, err := newDenseBatch(input)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := batch.weightSum, 3.0; !near(got, want) {
			t.Fatalf("weightSum = %v, want %v", got, want)
		}
		if got, want := batch.rows[0].weights, []float32{1, 0, 0.3, 0, 0.7, 1}; !sameFloat32s(got, want) {
			t.Fatalf("row weights = %v, want %v", got, want)
		}
	})

	t.Run("loss is weighted mean over arbitrary fractional weights", func(t *testing.T) {
		// 1 batch, 4 sequence positions, 3-token vocab.
		logits, err := mlx.FromSlice([]float32{
			0, 1, 2,
			2, 0, -2,
			-1, 0, 1,
			0, 2, 1,
		}, []int{1, 4, 3}, mlx.Float32)
		if err != nil {
			t.Fatal(err)
		}
		defer logits.Free()
		targets, err := mlx.FromSlice([]int32{2, 0, 1, 1}, []int{1, 4}, mlx.Int32)
		if err != nil {
			t.Fatal(err)
		}
		defer targets.Free()
		// Non-prefix pattern: 0.25, 1, 0, 0.75 — zero in the middle, fractions
		// at non-prefix positions.
		weightVals := []float32{0.25, 1, 0, 0.75}
		weights, err := mlx.FromSlice(weightVals, []int{1, 4}, mlx.Float32)
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

		lp := []float64{
			2 - logsumexp(0, 1, 2),
			2 - logsumexp(2, 0, -2),
			0 - logsumexp(-1, 0, 1),
			2 - logsumexp(0, 2, 1),
		}
		var num, denom float64
		for i, w := range weightVals {
			num += float64(w) * (-lp[i])
			denom += float64(w)
		}
		want := num / denom
		if !near(float64(gotLoss), want) {
			t.Fatalf("loss = %v, want %v", gotLoss, want)
		}
	})

	t.Run("rejects sparse weights", func(t *testing.T) {
		input := ForwardBackwardInput{
			LossFn: "cross_entropy",
			Data: []Datum{{
				ModelInput: ModelInput{Chunks: []ModelInputChunk{{Tokens: []int{1, 2}}}},
				LossFnInputs: map[string]TensorData{
					"target_tokens": {Data: []float64{3, 4}, DType: "int64"},
					"weights": {
						Data:              []float64{0.5, 0.5},
						DType:             "float32",
						SparseCrowIndices: []int{0, 2},
					},
				},
			}},
		}
		if _, err := newDenseBatch(input); err == nil {
			t.Fatal("newDenseBatch succeeded with sparse weights, want error")
		}
	})
}

func TestDenseCrossEntropyShapeErrors(t *testing.T) {
	logits, err := mlx.FromSlice([]float32{
		0, 1, 2,
		2, 0, -2,
		1, 0, 1,
		-1, 2, 0,
		0, 0, 1,
		1, 1, 0,
	}, []int{2, 3, 3}, mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer logits.Free()
	targets, err := mlx.FromSlice([]int32{0, 1, 2, 0, 1, 2}, []int{2, 3}, mlx.Int32)
	if err != nil {
		t.Fatal(err)
	}
	defer targets.Free()
	weights, err := mlx.FromSlice([]float32{1, 1, 1, 1, 1, 1}, []int{2, 3}, mlx.Float32)
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
	if got, want := logprobs.Shape(), []int{2, 3}; !sameInts(got, want) {
		t.Fatalf("logprobs shape = %v, want %v", got, want)
	}

	badTargets, err := mlx.FromSlice([]int32{0, 1}, []int{1, 2}, mlx.Int32)
	if err != nil {
		t.Fatal(err)
	}
	defer badTargets.Free()
	if _, _, err := denseCrossEntropy(logits, badTargets, weights); err == nil {
		t.Fatal("denseCrossEntropy with bad targets succeeded, want error")
	}

	badWeights, err := mlx.FromSlice([]float32{1, 1}, []int{1, 2}, mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer badWeights.Free()
	if _, _, err := denseCrossEntropy(logits, targets, badWeights); err == nil {
		t.Fatal("denseCrossEntropy with bad weights succeeded, want error")
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

func sameFloat32s(a, b []float32) bool {
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
