package tinker

import (
	"context"
	"errors"
	"fmt"
	"math"
)

var (
	// ErrUnsupported reports functionality that is part of the API but not
	// implemented by this experimental package yet.
	ErrUnsupported = errors.New("unsupported")

	// ErrNotFound reports a missing model, trainer, sampler, or checkpoint.
	ErrNotFound = errors.New("not found")

	// ErrClosed reports use after Close.
	ErrClosed = errors.New("closed")
)

// Capabilities describes the local Tinker API surface available to a client.
type Capabilities struct {
	Training bool
	Sampling bool
	// Losses lists hosted-compatible built-in losses advertised by the local
	// client. Experimental local-only losses may still validate and execute.
	Losses []string
}

// ModelRegistry resolves Tinker model names to local model assets.
type ModelRegistry interface {
	Resolve(ctx context.Context, name string) (ModelSpec, error)
	List(ctx context.Context) ([]ModelInfo, error)
}

// ModelSpec describes a locally available model.
type ModelSpec struct {
	Name       string
	Path       string
	Tokenizer  string
	MaxContext int
}

// ModelInfo describes a model without exposing its local asset path.
type ModelInfo struct {
	Name       string
	Tokenizer  string
	MaxContext int
}

func (s ModelSpec) info() ModelInfo {
	return ModelInfo{
		Name:       s.Name,
		Tokenizer:  s.Tokenizer,
		MaxContext: s.MaxContext,
	}
}

func validateModelSpec(s ModelSpec) error {
	if s.Name == "" {
		return errors.New("model name is empty")
	}
	if s.Path == "" {
		return errors.New("model path is empty")
	}
	if s.MaxContext < 0 {
		return errors.New("model max context is negative")
	}
	return nil
}

// ModelInput is a tokenized model input.
type ModelInput struct {
	Chunks []Chunk
}

// Chunk is a contiguous token chunk.
type Chunk struct {
	Tokens []int
}

// FromTokens returns a ModelInput containing a copy of tokens.
func FromTokens(tokens []int) ModelInput {
	out := append([]int(nil), tokens...)
	return ModelInput{Chunks: []Chunk{{Tokens: out}}}
}

// Tokens returns a flattened copy of all input tokens.
func (in ModelInput) Tokens() []int {
	var n int
	for _, ch := range in.Chunks {
		n += len(ch.Tokens)
	}
	out := make([]int, 0, n)
	for _, ch := range in.Chunks {
		out = append(out, ch.Tokens...)
	}
	return out
}

// Len returns the flattened token count.
func (in ModelInput) Len() int {
	var n int
	for _, ch := range in.Chunks {
		n += len(ch.Tokens)
	}
	return n
}

// Datum is one training example.
type Datum struct {
	Input     ModelInput
	LossInput LossInput
}

// LossInput holds typed loss data for one datum.
type LossInput struct {
	TargetTokens       []int
	TargetTokensTensor TensorData
	Weights            []float32
	WeightsTensor      TensorData
	Logprobs           []float32
	Advantages         []float32
	Mask               []bool
}

// TensorData is a dense tensor value passed to SDK-shaped loss inputs and
// outputs.
//
// Data is row-major. When Shape is nil, validation treats the tensor as a
// one-dimensional tensor with len(Data) elements. Sparse tensors are not
// supported by the built-in losses.
type TensorData struct {
	Data              []float64
	DType             string
	Shape             []int
	SparseCrowIndices []int
	SparseColIndices  []int
}

// Loss is a built-in training loss.
type Loss interface {
	loss()
}

// CrossEntropy is token cross entropy.
type CrossEntropy struct{}

// ImportanceSampling is the Tinker importance-sampling loss.
type ImportanceSampling struct{}

// PPO is the clipped PPO loss.
type PPO struct {
	ClipLow  float32
	ClipHigh float32
}

// CISPO is the clipped CISPO loss.
type CISPO struct {
	ClipLow  float32
	ClipHigh float32
}

// DRO is the Tinker DRO loss.
type DRO struct {
	Beta float32
}

func (CrossEntropy) loss()       {}
func (ImportanceSampling) loss() {}
func (PPO) loss()                {}
func (CISPO) loss()              {}
func (DRO) loss()                {}

func validateBatch(batch []Datum, loss Loss) error {
	if len(batch) == 0 {
		return errors.New("batch is empty")
	}
	if loss == nil {
		return errors.New("loss is nil")
	}
	for i, d := range batch {
		if d.Input.Len() == 0 {
			return fmt.Errorf("datum %d input is empty", i)
		}
		n, err := validateLossInput(d.LossInput, loss)
		if err != nil {
			return fmt.Errorf("datum %d: %w", i, err)
		}
		if d.Input.Len() != n {
			return fmt.Errorf("datum %d input tokens length does not match target tokens", i)
		}
	}
	return nil
}

func validateLossInput(in LossInput, loss Loss) (int, error) {
	n, targetShape, err := validateTargetTokens(in)
	if err != nil {
		return 0, err
	}
	if err := validateWeights(in, n, targetShape); err != nil {
		return 0, err
	}
	switch loss.(type) {
	case CrossEntropy:
		return n, nil
	case ImportanceSampling, PPO, CISPO, DRO:
		if len(in.Logprobs) != n {
			return 0, errors.New("logprobs length does not match target tokens")
		}
		if len(in.Advantages) != n {
			return 0, errors.New("advantages length does not match target tokens")
		}
		if len(in.Mask) != 0 && len(in.Mask) != n {
			return 0, errors.New("mask length does not match target tokens")
		}
		return n, nil
	default:
		return 0, fmt.Errorf("%w: loss %T", ErrUnsupported, loss)
	}
}

func validateTargetTokens(in LossInput) (int, []int, error) {
	hasTokens := len(in.TargetTokens) != 0
	hasTensor := tensorDataSet(in.TargetTokensTensor)
	if hasTokens && hasTensor {
		return 0, nil, errors.New("target tokens and target tokens tensor are both set")
	}
	if hasTensor {
		n, shape, err := validateTensorData("target tokens", in.TargetTokensTensor, "int64")
		if err != nil {
			return 0, nil, err
		}
		if n == 0 {
			return 0, nil, errors.New("target tokens are empty")
		}
		for _, v := range in.TargetTokensTensor.Data {
			if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > float64(math.MaxInt32) || math.Trunc(v) != v {
				return 0, nil, errors.New("target tokens tensor contains invalid token")
			}
		}
		return n, shape, nil
	}
	n := len(in.TargetTokens)
	if n == 0 {
		return 0, nil, errors.New("target tokens are empty")
	}
	for _, v := range in.TargetTokens {
		if v < 0 {
			return 0, nil, errors.New("target tokens contain invalid token")
		}
	}
	return n, []int{n}, nil
}

func validateWeights(in LossInput, n int, targetShape []int) error {
	hasWeights := len(in.Weights) != 0
	hasTensor := tensorDataSet(in.WeightsTensor)
	if hasWeights && hasTensor {
		return errors.New("weights and weights tensor are both set")
	}
	if hasTensor {
		wn, weightShape, err := validateTensorData("weights", in.WeightsTensor, "float32")
		if err != nil {
			return err
		}
		for _, v := range in.WeightsTensor.Data {
			if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
				return errors.New("weights tensor contains invalid weight")
			}
		}
		if wn != n {
			return errors.New("weights length does not match target tokens")
		}
		if !sameShape(weightShape, targetShape) {
			return errors.New("weights shape does not match target tokens")
		}
		return nil
	}
	if len(in.Weights) != 0 && len(in.Weights) != n {
		return errors.New("weights length does not match target tokens")
	}
	for _, v := range in.Weights {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v < 0 {
			return errors.New("weights contain invalid weight")
		}
	}
	return nil
}

func validateTensorData(name string, tensor TensorData, dtype string) (int, []int, error) {
	if tensor.SparseCrowIndices != nil || tensor.SparseColIndices != nil {
		return 0, nil, fmt.Errorf("%s sparse tensors are not supported", name)
	}
	if tensor.DType != "" && tensor.DType != dtype {
		return 0, nil, fmt.Errorf("%s dtype %q, want %s", name, tensor.DType, dtype)
	}
	shape := tensor.Shape
	if shape == nil {
		shape = []int{len(tensor.Data)}
	}
	shape = append([]int(nil), shape...)
	n := 1
	for _, dim := range shape {
		if dim < 0 {
			return 0, nil, fmt.Errorf("%s shape has negative dimension", name)
		}
		n *= dim
	}
	if n != len(tensor.Data) {
		return 0, nil, fmt.Errorf("%s shape does not match data", name)
	}
	return n, shape, nil
}

func tensorDataSet(t TensorData) bool {
	return t.Data != nil || t.DType != "" || t.Shape != nil ||
		t.SparseCrowIndices != nil || t.SparseColIndices != nil
}

func sameShape(a, b []int) bool {
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

// AdamW configures one AdamW optimizer step.
type AdamW struct {
	LearningRate float32
	Beta1        float32
	Beta2        float32
	Eps          float32
	WeightDecay  float32
	GradClipNorm float32
}

// DefaultAdamW returns Tinker-compatible AdamW defaults.
func DefaultAdamW() AdamW {
	return AdamW{
		LearningRate: 1e-4,
		Beta1:        0.9,
		Beta2:        0.95,
		Eps:          1e-12,
	}
}

func validateAdamW(opt AdamW) error {
	if opt.LearningRate <= 0 {
		return errors.New("learning rate must be positive")
	}
	if opt.Beta1 < 0 || opt.Beta1 >= 1 {
		return errors.New("beta1 must be in [0, 1)")
	}
	if opt.Beta2 < 0 || opt.Beta2 >= 1 {
		return errors.New("beta2 must be in [0, 1)")
	}
	if opt.Eps <= 0 {
		return errors.New("eps must be positive")
	}
	if opt.WeightDecay < 0 {
		return errors.New("weight decay is negative")
	}
	if opt.GradClipNorm < 0 {
		return errors.New("grad clip norm is negative")
	}
	return nil
}

// CustomTensor is an opaque tensor handle for custom loss hooks.
//
// It is intentionally not backed by a public MLX type while this package has
// no execution backend. Values passed to custom hooks, once implemented, will
// be owned by the package and valid only for the duration of the call.
type CustomTensor struct{}

// CustomLossFunc computes a loss and gradient over model logprobs.
type CustomLossFunc func(context.Context, CustomLossInput) (CustomLossOutput, error)

// CustomLossInput is passed to a custom loss function.
type CustomLossInput struct {
	Logprobs *CustomTensor
	Batch    []Datum
}

// CustomLossOutput is returned by a custom loss function.
type CustomLossOutput struct {
	Loss         *CustomTensor
	GradLogprobs *CustomTensor
	Metrics      map[string]float64
}
