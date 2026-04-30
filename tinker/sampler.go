package tinker

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// Sampler is a local Tinker sampling handle.
type Sampler struct {
	mu     sync.Mutex
	model  ModelInfo
	path   string
	closed bool
}

// SamplingRequest configures a sampler.
type SamplingRequest struct {
	BaseModel string
	Path      string
}

// SampleParams controls generation.
type SampleParams struct {
	MaxTokens          int
	Temperature        float32
	TopP               float32
	TopK               int
	TopKPromptLogprobs int
	Stop               [][]int
}

// SampleResult is returned by Sample.
type SampleResult struct {
	Sequences          []Sequence
	PromptLogprobs     []float32
	TopKPromptLogprobs [][]TokenLogprob
}

// Sequence is one generated token sequence.
type Sequence struct {
	Tokens     []int
	Logprobs   []float32
	StopReason StopReason
}

// StopReason explains why generation stopped.
type StopReason string

const (
	StopLength StopReason = "length"
	StopStop   StopReason = "stop"
)

// TokenLogprob is one token alternative and its log probability.
type TokenLogprob struct {
	Token   int
	Logprob float32
}

func (s *Sampler) checkOpen() error {
	if s == nil {
		return ErrClosed
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	return nil
}

// Sample validates sampling inputs and returns ErrUnsupported until MLX
// execution is implemented.
func (s *Sampler) Sample(ctx context.Context, prompt ModelInput, params SampleParams) (SampleResult, error) {
	if err := ctx.Err(); err != nil {
		return SampleResult{}, err
	}
	if err := s.checkOpen(); err != nil {
		return SampleResult{}, err
	}
	if err := validatePrompt(prompt); err != nil {
		return SampleResult{}, err
	}
	if err := validateSampleParams(params); err != nil {
		return SampleResult{}, err
	}
	return SampleResult{}, fmt.Errorf("%w: sample", ErrUnsupported)
}

// Logprobs validates prompt input and returns ErrUnsupported until MLX
// execution is implemented.
func (s *Sampler) Logprobs(ctx context.Context, prompt ModelInput) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := s.checkOpen(); err != nil {
		return nil, err
	}
	if err := validatePrompt(prompt); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("%w: logprobs", ErrUnsupported)
}

// Close releases sampler resources. It is idempotent.
func (s *Sampler) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return nil
}

func validatePrompt(prompt ModelInput) error {
	if prompt.Len() == 0 {
		return errors.New("prompt is empty")
	}
	return nil
}

func validateSampleParams(params SampleParams) error {
	if params.MaxTokens < 0 {
		return errors.New("max tokens is negative")
	}
	if params.Temperature < 0 {
		return errors.New("temperature is negative")
	}
	if params.TopP < 0 || params.TopP > 1 {
		return errors.New("top p must be in [0, 1]")
	}
	if params.TopK < 0 {
		return errors.New("top k is negative")
	}
	if params.TopKPromptLogprobs < 0 {
		return errors.New("top k prompt logprobs is negative")
	}
	for i, stop := range params.Stop {
		if len(stop) == 0 {
			return fmt.Errorf("stop sequence %d is empty", i)
		}
	}
	return nil
}
