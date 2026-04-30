package tinker

import (
	"context"
	"errors"
	"fmt"
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
	Losses   []string
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
	TargetTokens []int
	Weights      []float32
	Logprobs     []float32
	Advantages   []float32
	Mask         []bool
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

func (CrossEntropy) loss()       {}
func (ImportanceSampling) loss() {}
func (PPO) loss()                {}
func (CISPO) loss()              {}

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
		if err := validateLossInput(d.LossInput, loss); err != nil {
			return fmt.Errorf("datum %d: %w", i, err)
		}
	}
	return nil
}

func validateLossInput(in LossInput, loss Loss) error {
	n := len(in.TargetTokens)
	if n == 0 {
		return errors.New("target tokens are empty")
	}
	if len(in.Weights) != 0 && len(in.Weights) != n {
		return errors.New("weights length does not match target tokens")
	}
	switch loss.(type) {
	case CrossEntropy:
		return nil
	case ImportanceSampling, PPO, CISPO:
		if len(in.Logprobs) != n {
			return errors.New("logprobs length does not match target tokens")
		}
		if len(in.Advantages) != n {
			return errors.New("advantages length does not match target tokens")
		}
		if len(in.Mask) != 0 && len(in.Mask) != n {
			return errors.New("mask length does not match target tokens")
		}
		return nil
	default:
		return fmt.Errorf("%w: loss %T", ErrUnsupported, loss)
	}
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
