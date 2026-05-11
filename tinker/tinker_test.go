package tinker

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
)

type registry map[string]ModelSpec

func (r registry) Resolve(_ context.Context, name string) (ModelSpec, error) {
	spec, ok := r[name]
	if !ok {
		return ModelSpec{}, ErrNotFound
	}
	return spec, nil
}

func (r registry) List(context.Context) ([]ModelInfo, error) {
	var out []ModelInfo
	for _, spec := range r {
		out = append(out, spec.info())
	}
	return out, nil
}

func TestFromTokensCopies(t *testing.T) {
	tokens := []int{1, 2, 3}
	in := FromTokens(tokens)
	tokens[0] = 99
	if got := in.Tokens()[0]; got != 1 {
		t.Fatalf("Tokens()[0] = %d, want 1", got)
	}
	flat := in.Tokens()
	flat[0] = 88
	if got := in.Tokens()[0]; got != 1 {
		t.Fatalf("Tokens()[0] after mutating flat copy = %d, want 1", got)
	}
}

func TestForwardValidatesBeforeUnsupported(t *testing.T) {
	tr := &Trainer{}
	_, err := tr.Forward(context.Background(), nil, CrossEntropy{})
	if err == nil || errors.Is(err, ErrUnsupported) {
		t.Fatalf("Forward(nil batch) error = %v, want validation error", err)
	}

	batch := []Datum{{
		Input: FromTokens([]int{1}),
		LossInput: LossInput{
			TargetTokens: []int{2},
		},
	}}
	_, err = tr.Forward(context.Background(), batch, CrossEntropy{})
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("Forward valid batch error = %v, want ErrUnsupported", err)
	}
}

func TestCrossEntropyAcceptsTensorData(t *testing.T) {
	batch := []Datum{{
		Input: FromTokens([]int{10, 11, 12, 13}),
		LossInput: LossInput{
			TargetTokensTensor: TensorData{
				Data:  []float64{42, 7, 42, 9},
				DType: "int64",
				Shape: []int{2, 2},
			},
			WeightsTensor: TensorData{
				Data:  []float64{1, 0.5, 0, 2},
				DType: "float32",
				Shape: []int{2, 2},
			},
		},
	}}
	if err := validateBatch(batch, CrossEntropy{}); err != nil {
		t.Fatalf("validateBatch() = %v, want nil", err)
	}
}

func TestCrossEntropyTensorDataValidation(t *testing.T) {
	valid := func() LossInput {
		return LossInput{
			TargetTokensTensor: TensorData{
				Data:  []float64{1, 2, 3, 4},
				DType: "int64",
				Shape: []int{2, 2},
			},
		}
	}
	tests := []struct {
		name string
		edit func(*LossInput)
		want string
	}{
		{
			name: "infer flat shape",
			edit: func(in *LossInput) {
				in.TargetTokensTensor.Shape = nil
			},
		},
		{
			name: "omitted dtypes",
			edit: func(in *LossInput) {
				in.TargetTokensTensor.DType = ""
				in.WeightsTensor = TensorData{
					Data:  []float64{1, 0.5, 0, 1},
					Shape: []int{2, 2},
				}
			},
		},
		{
			name: "bad target shape",
			edit: func(in *LossInput) {
				in.TargetTokensTensor.Shape = []int{3}
			},
			want: "target tokens shape does not match data",
		},
		{
			name: "bad target dtype",
			edit: func(in *LossInput) {
				in.TargetTokensTensor.DType = "float32"
			},
			want: `target tokens dtype "float32", want int64`,
		},
		{
			name: "fractional target",
			edit: func(in *LossInput) {
				in.TargetTokensTensor.Data[1] = 2.5
			},
			want: "target tokens tensor contains invalid token",
		},
		{
			name: "infinite target",
			edit: func(in *LossInput) {
				in.TargetTokensTensor.Data[1] = math.Inf(1)
			},
			want: "target tokens tensor contains invalid token",
		},
		{
			name: "out of range target",
			edit: func(in *LossInput) {
				in.TargetTokensTensor.Data[1] = float64(math.MaxInt32) + 1
			},
			want: "target tokens tensor contains invalid token",
		},
		{
			name: "sparse target",
			edit: func(in *LossInput) {
				in.TargetTokensTensor.SparseCrowIndices = []int{0}
			},
			want: "target tokens sparse tensors are not supported",
		},
		{
			name: "bad weight length",
			edit: func(in *LossInput) {
				in.WeightsTensor = TensorData{
					Data:  []float64{1, 1},
					DType: "float32",
				}
			},
			want: "weights length does not match target tokens",
		},
		{
			name: "bad weight shape",
			edit: func(in *LossInput) {
				in.WeightsTensor = TensorData{
					Data:  []float64{1, 1, 1, 1},
					DType: "float32",
					Shape: []int{4},
				}
			},
			want: "weights shape does not match target tokens",
		},
		{
			name: "bad weight dtype",
			edit: func(in *LossInput) {
				in.WeightsTensor = TensorData{
					Data:  []float64{1, 1, 1, 1},
					DType: "float64",
					Shape: []int{2, 2},
				}
			},
			want: `weights dtype "float64", want float32`,
		},
		{
			name: "negative weight tensor",
			edit: func(in *LossInput) {
				in.WeightsTensor = TensorData{
					Data:  []float64{1, -1, 1, 1},
					DType: "float32",
					Shape: []int{2, 2},
				}
			},
			want: "weights tensor contains invalid weight",
		},
		{
			name: "non finite weight tensor",
			edit: func(in *LossInput) {
				in.WeightsTensor = TensorData{
					Data:  []float64{1, math.Inf(1), 1, 1},
					DType: "float32",
					Shape: []int{2, 2},
				}
			},
			want: "weights tensor contains invalid weight",
		},
		{
			name: "negative legacy weight",
			edit: func(in *LossInput) {
				in.Weights = []float32{1, 1, -1, 1}
			},
			want: "weights contain invalid weight",
		},
		{
			name: "legacy and tensor target",
			edit: func(in *LossInput) {
				in.TargetTokens = []int{1, 2, 3, 4}
			},
			want: "target tokens and target tokens tensor are both set",
		},
		{
			name: "input target length mismatch",
			edit: func(in *LossInput) {
				in.TargetTokensTensor.Data = []float64{1, 2, 3}
				in.TargetTokensTensor.Shape = []int{3}
			},
			want: "input tokens length does not match target tokens",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := valid()
			tt.edit(&in)
			batch := []Datum{{
				Input:     FromTokens([]int{10, 11, 12, 13}),
				LossInput: in,
			}}
			err := validateBatch(batch, CrossEntropy{})
			if tt.want == "" {
				if err != nil {
					t.Fatalf("validateBatch() = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("validateBatch() = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestCrossEntropyTensorShapeTable(t *testing.T) {
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
			batch := []Datum{{
				Input: FromTokens([]int{10, 11, 12, 13}),
				LossInput: LossInput{
					TargetTokensTensor: TensorData{
						Data:  []float64{1, 2, 3, 4},
						DType: "int64",
						Shape: tt.shape,
					},
					WeightsTensor: TensorData{
						Data:  []float64{1, 0, 0.5, 2},
						DType: "float32",
						Shape: tt.shape,
					},
				},
			}}
			if err := validateBatch(batch, CrossEntropy{}); err != nil {
				t.Fatalf("validateBatch() = %v, want nil", err)
			}
		})
	}
}

func TestCreateLoRAAndClose(t *testing.T) {
	ctx := context.Background()
	client, err := New(Config{
		RootDir: t.TempDir(),
		Models: registry{"m": {
			Name:       "m",
			Path:       "/tmp/m",
			MaxContext: 16,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	tr, err := client.CreateLoRA(ctx, CreateLoRARequest{BaseModel: "m", Rank: 4})
	if err != nil {
		t.Fatal(err)
	}
	info, err := tr.Info(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !info.IsLoRA || info.LoRARank != 4 || info.Model.Name != "m" {
		t.Fatalf("Info() = %+v, want lora rank 4 model m", info)
	}
	if err := tr.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := tr.Info(ctx); !errors.Is(err, ErrClosed) {
		t.Fatalf("Info after Close error = %v, want ErrClosed", err)
	}
}

func TestCapabilitiesReportExecutableLosses(t *testing.T) {
	ctx := context.Background()
	client, err := New(Config{
		RootDir: t.TempDir(),
		Models: registry{"m": {
			Name:       "m",
			Path:       "/tmp/m",
			MaxContext: 16,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	caps, err := client.Capabilities(ctx)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"cross_entropy", "importance_sampling", "ppo", "cispo", "dro"}
	if len(caps.Losses) != len(want) {
		t.Fatalf("Losses = %v, want %v", caps.Losses, want)
	}
	for i := range want {
		if caps.Losses[i] != want[i] {
			t.Fatalf("Losses = %v, want %v", caps.Losses, want)
		}
	}
}
