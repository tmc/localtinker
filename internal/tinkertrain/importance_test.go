package tinkertrain

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/mlx-go/mlx"
)

func TestDenseImportanceSamplingBatchInputs(t *testing.T) {
	input := ForwardBackwardInput{
		LossFn: "importance_sampling",
		Data: []Datum{{
			ModelInput: ModelInput{Chunks: []ModelInputChunk{{
				Type:   "encoded_text",
				Tokens: []int{10, 11, 12, 13},
			}}},
			LossFnInputs: map[string]TensorData{
				"target_tokens": {
					Data:              []float64{2, 1},
					DType:             "int64",
					Shape:             []int{1, 4},
					SparseCrowIndices: []int{0, 2},
					SparseColIndices:  []int{0, 2},
				},
				"weights": {
					Data:              []float64{0.5, 1.5},
					DType:             "float32",
					Shape:             []int{1, 4},
					SparseCrowIndices: []int{0, 2},
					SparseColIndices:  []int{0, 2},
				},
				"logprobs":   {Data: []float64{-0.5, -0.25, -0.125, -1}, DType: "float32", Shape: []int{1, 4}},
				"advantages": {Data: []float64{1, -2, 0.5, -1.5}, DType: "float32", Shape: []int{1, 4}},
			},
		}},
	}

	batch, err := newDenseBatch(input)
	if err != nil {
		t.Fatal(err)
	}
	if batch.lossFn != "importance_sampling" {
		t.Fatalf("lossFn = %q, want importance_sampling", batch.lossFn)
	}
	if got, want := batch.weightSum, 2.0; !near(got, want) {
		t.Fatalf("weightSum = %v, want %v", got, want)
	}
	row := batch.rows[0]
	if got, want := row.outputShape, []int{1, 4}; !sameInts(got, want) {
		t.Fatalf("output shape = %v, want %v", got, want)
	}
	if got, want := row.targets, []int32{2, 0, 1, 0}; !sameInt32s(got, want) {
		t.Fatalf("targets = %v, want %v", got, want)
	}
	if got, want := row.weights, []float32{0.5, 0, 1.5, 0}; !sameFloat32s(got, want) {
		t.Fatalf("weights = %v, want %v", got, want)
	}
	if got, want := row.logprobs, []float32{-0.5, -0.25, -0.125, -1}; !sameFloat32s(got, want) {
		t.Fatalf("logprobs = %v, want %v", got, want)
	}
	if got, want := row.advantages, []float32{1, -2, 0.5, -1.5}; !sameFloat32s(got, want) {
		t.Fatalf("advantages = %v, want %v", got, want)
	}
}

func TestDenseImportanceSamplingReturnsWeightedLossAndLogprobs(t *testing.T) {
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
	weightVals := []float32{0.25, 1, 0, 0.75}
	weights, err := mlx.FromSlice(weightVals, []int{1, 4}, mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Free()
	oldVals := []float32{-0.75, -0.25, -0.5, -1.25}
	oldLogprobs, err := mlx.FromSlice(oldVals, []int{1, 4}, mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer oldLogprobs.Free()
	advantageVals := []float32{1.5, -0.5, 2, 0.25}
	advantages, err := mlx.FromSlice(advantageVals, []int{1, 4}, mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer advantages.Free()

	loss, logprobs, err := denseImportanceSampling(logits, targets, weights, oldLogprobs, advantages)
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
	if got, want := logprobs.Shape(), []int{1, 4}; !sameInts(got, want) {
		t.Fatalf("logprobs shape = %v, want %v", got, want)
	}

	wantLogprobs := []float64{
		2 - logsumexp(0, 1, 2),
		2 - logsumexp(2, 0, -2),
		0 - logsumexp(-1, 0, 1),
		2 - logsumexp(0, 2, 1),
	}
	var num float64
	for i, lp := range wantLogprobs {
		if !near(float64(gotLogprobs[i]), lp) {
			t.Fatalf("logprobs[%d] = %v, want %v", i, gotLogprobs[i], lp)
		}
		w := float64(weightVals[i])
		num += w * math.Exp(lp-float64(oldVals[i])) * float64(advantageVals[i])
	}
	wantLoss := -num
	if !near(float64(gotLoss), wantLoss) {
		t.Fatalf("loss = %v, want %v", gotLoss, wantLoss)
	}
}

func TestDensePolicyLossesReturnWeightedSumAndLogprobs(t *testing.T) {
	logitVals := []float32{
		0, 1, 2,
		2, 0, -2,
		-1, 0, 1,
		0, 2, 1,
	}
	targetVals := []int32{2, 0, 1, 1}
	weightVals := []float32{0.25, 1, 0, 0.75}
	oldVals := []float32{-0.75, -0.25, -0.5, -1.25}
	advantageVals := []float32{1.5, -0.5, 2, 0.25}
	wantLogprobs := []float64{
		2 - logsumexp(0, 1, 2),
		2 - logsumexp(2, 0, -2),
		0 - logsumexp(-1, 0, 1),
		2 - logsumexp(0, 2, 1),
	}
	tests := []struct {
		lossFn string
		config policyLossConfig
		want   func(lp, old, adv, w float64) float64
	}{
		{
			lossFn: "importance_sampling",
			want: func(lp, old, adv, w float64) float64 {
				return -w * math.Exp(lp-old) * adv
			},
		},
		{
			lossFn: "ppo",
			config: policyLossConfig{clipLow: 0.8, clipHigh: 1.2},
			want: func(lp, old, adv, w float64) float64 {
				ratio := math.Exp(lp - old)
				clipped := math.Min(math.Max(ratio, 0.8), 1.2)
				return -w * math.Min(ratio*adv, clipped*adv)
			},
		},
		{
			lossFn: "cispo",
			config: policyLossConfig{clipLow: 0.8, clipHigh: 1.2},
			want: func(lp, old, adv, w float64) float64 {
				ratio := math.Exp(lp - old)
				clipped := math.Min(math.Max(ratio, 0.8), 1.2)
				return -w * clipped * lp * adv
			},
		},
		{
			lossFn: "dro",
			config: policyLossConfig{beta: 0.05},
			want: func(lp, old, adv, w float64) float64 {
				delta := lp - old
				return -w * (lp*adv - 0.5*0.05*delta*delta)
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.lossFn, func(t *testing.T) {
			logits, targets, weights, oldLogprobs, advantages := policyLossArrays(t, logitVals, targetVals, weightVals, oldVals, advantageVals)
			defer logits.Free()
			defer targets.Free()
			defer weights.Free()
			defer oldLogprobs.Free()
			defer advantages.Free()

			loss, logprobs, err := densePolicyLoss(tt.lossFn, logits, targets, weights, oldLogprobs, advantages, tt.config)
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
			var wantLoss float64
			for i, lp := range wantLogprobs {
				if !near(float64(gotLogprobs[i]), lp) {
					t.Fatalf("logprobs[%d] = %v, want %v", i, gotLogprobs[i], lp)
				}
				wantLoss += tt.want(lp, float64(oldVals[i]), float64(advantageVals[i]), float64(weightVals[i]))
			}
			if !near(float64(gotLoss), wantLoss) {
				t.Fatalf("loss = %v, want %v", gotLoss, wantLoss)
			}
		})
	}
}

func policyLossArrays(t *testing.T, logits []float32, targetVals []int32, weights, oldVals, advantageVals []float32) (logitArr, targetArr, weightArr, oldArr, advantageArr *mlx.Array) {
	t.Helper()
	logitArr, err := mlx.FromSlice(logits, []int{1, len(targetVals), 3}, mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	targetArr, err = mlx.FromSlice(targetVals, []int{1, len(targetVals)}, mlx.Int32)
	if err != nil {
		logitArr.Free()
		t.Fatal(err)
	}
	weightArr, err = mlx.FromSlice(weights, []int{1, len(targetVals)}, mlx.Float32)
	if err != nil {
		logitArr.Free()
		targetArr.Free()
		t.Fatal(err)
	}
	oldArr, err = mlx.FromSlice(oldVals, []int{1, len(targetVals)}, mlx.Float32)
	if err != nil {
		logitArr.Free()
		targetArr.Free()
		weightArr.Free()
		t.Fatal(err)
	}
	advantageArr, err = mlx.FromSlice(advantageVals, []int{1, len(targetVals)}, mlx.Float32)
	if err != nil {
		logitArr.Free()
		targetArr.Free()
		weightArr.Free()
		oldArr.Free()
		t.Fatal(err)
	}
	return logitArr, targetArr, weightArr, oldArr, advantageArr
}

func TestDenseImportanceSamplingRejectsBadInputs(t *testing.T) {
	base := func() ForwardBackwardInput {
		return ForwardBackwardInput{
			LossFn: "importance_sampling",
			Data: []Datum{{
				ModelInput: ModelInput{Chunks: []ModelInputChunk{{Tokens: []int{1, 2, 3, 4}}}},
				LossFnInputs: map[string]TensorData{
					"target_tokens": {Data: []float64{1, 1, 1, 1}, DType: "int64", Shape: []int{2, 2}},
					"weights":       {Data: []float64{1, 1, 1, 1}, DType: "float32", Shape: []int{2, 2}},
					"logprobs":      {Data: []float64{-1, -1, -1, -1}, DType: "float32", Shape: []int{2, 2}},
					"advantages":    {Data: []float64{1, 1, 1, 1}, DType: "float32", Shape: []int{2, 2}},
				},
			}},
		}
	}
	tests := []struct {
		name string
		edit func(*ForwardBackwardInput)
		want string
	}{
		{
			name: "missing logprobs",
			edit: func(in *ForwardBackwardInput) {
				delete(in.Data[0].LossFnInputs, "logprobs")
			},
			want: "missing logprobs",
		},
		{
			name: "bad advantages shape",
			edit: func(in *ForwardBackwardInput) {
				in.Data[0].LossFnInputs["advantages"] = TensorData{Data: []float64{1, 1, 1, 1}, DType: "float32", Shape: []int{4}}
			},
			want: "advantages shape [4] does not match target_tokens shape [2 2]",
		},
		{
			name: "sparse logprobs",
			edit: func(in *ForwardBackwardInput) {
				in.Data[0].LossFnInputs["logprobs"] = TensorData{
					Data:              []float64{-1},
					DType:             "float32",
					Shape:             []int{1, 4},
					SparseCrowIndices: []int{0, 1},
					SparseColIndices:  []int{0},
				}
			},
			want: "sparse tensors are not supported for logprobs",
		},
		{
			name: "non finite advantage",
			edit: func(in *ForwardBackwardInput) {
				in.Data[0].LossFnInputs["advantages"] = TensorData{Data: []float64{1, math.Inf(1), 1, 1}, DType: "float32", Shape: []int{2, 2}}
			},
			want: "advantages[1] = +Inf is not finite",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := base()
			tt.edit(&input)
			_, err := newDenseBatch(input)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want containing %q", err, tt.want)
			}
		})
	}
}
