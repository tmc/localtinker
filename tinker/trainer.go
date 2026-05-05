package tinker

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// Trainer is a local Tinker training handle.
type Trainer struct {
	mu     sync.Mutex
	model  ModelInfo
	info   TrainingInfo
	closed bool
}

// CreateLoRARequest configures LoRA training.
type CreateLoRARequest struct {
	BaseModel    string
	Rank         int
	Seed         int64
	TrainMLP     bool
	TrainAttn    bool
	TrainUnembed bool
	Metadata     map[string]string
}

// TrainingInfo describes a trainer.
type TrainingInfo struct {
	Model        ModelInfo
	ID           string
	IsLoRA       bool
	LoRARank     int
	TrainMLP     bool
	TrainAttn    bool
	TrainUnembed bool
}

// ForwardResult is returned by forward passes.
type ForwardResult struct {
	Loss             float32
	Logprobs         [][]float32
	LossFnOutputType string
	LossFnOutputs    []map[string]TensorData
	Metrics          map[string]float64
}

// OptimResult is returned by optimizer steps.
type OptimResult struct {
	Metrics map[string]float64
}

func (t *Trainer) checkOpen() error {
	if t == nil {
		return ErrClosed
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return ErrClosed
	}
	return nil
}

// Info reports trainer metadata.
func (t *Trainer) Info(ctx context.Context) (TrainingInfo, error) {
	if err := ctx.Err(); err != nil {
		return TrainingInfo{}, err
	}
	if err := t.checkOpen(); err != nil {
		return TrainingInfo{}, err
	}
	return t.info, nil
}

// Forward validates a batch and returns ErrUnsupported until MLX execution is
// implemented.
func (t *Trainer) Forward(ctx context.Context, batch []Datum, loss Loss) (ForwardResult, error) {
	if err := ctx.Err(); err != nil {
		return ForwardResult{}, err
	}
	if err := t.checkOpen(); err != nil {
		return ForwardResult{}, err
	}
	if err := validateBatch(batch, loss); err != nil {
		return ForwardResult{}, err
	}
	return ForwardResult{}, fmt.Errorf("%w: forward", ErrUnsupported)
}

// ForwardBackward validates a batch and returns ErrUnsupported until MLX
// execution is implemented.
func (t *Trainer) ForwardBackward(ctx context.Context, batch []Datum, loss Loss) (ForwardResult, error) {
	if err := ctx.Err(); err != nil {
		return ForwardResult{}, err
	}
	if err := t.checkOpen(); err != nil {
		return ForwardResult{}, err
	}
	if err := validateBatch(batch, loss); err != nil {
		return ForwardResult{}, err
	}
	return ForwardResult{}, fmt.Errorf("%w: forward backward", ErrUnsupported)
}

// ForwardBackwardCustom returns ErrUnsupported until custom MLX loss execution
// is implemented.
func (t *Trainer) ForwardBackwardCustom(ctx context.Context, batch []Datum, fn CustomLossFunc) (ForwardResult, error) {
	if err := ctx.Err(); err != nil {
		return ForwardResult{}, err
	}
	if err := t.checkOpen(); err != nil {
		return ForwardResult{}, err
	}
	if len(batch) == 0 {
		return ForwardResult{}, errors.New("batch is empty")
	}
	for i, d := range batch {
		if d.Input.Len() == 0 {
			return ForwardResult{}, fmt.Errorf("datum %d input is empty", i)
		}
	}
	if fn == nil {
		return ForwardResult{}, errors.New("custom loss func is nil")
	}
	return ForwardResult{}, fmt.Errorf("%w: forward backward custom", ErrUnsupported)
}

// OptimStep validates optimizer parameters and returns ErrUnsupported until
// MLX execution is implemented.
func (t *Trainer) OptimStep(ctx context.Context, opt AdamW) (OptimResult, error) {
	if err := ctx.Err(); err != nil {
		return OptimResult{}, err
	}
	if err := t.checkOpen(); err != nil {
		return OptimResult{}, err
	}
	if err := validateAdamW(opt); err != nil {
		return OptimResult{}, err
	}
	return OptimResult{}, fmt.Errorf("%w: optim step", ErrUnsupported)
}

// Save saves a training checkpoint.
func (t *Trainer) Save(ctx context.Context, name string, opts ...SaveOption) (Checkpoint, error) {
	return t.save(ctx, name, CheckpointTrain, opts...)
}

// SaveForSampler saves a sampler checkpoint.
func (t *Trainer) SaveForSampler(ctx context.Context, name string, opts ...SaveOption) (Checkpoint, error) {
	return t.save(ctx, name, CheckpointSampler, opts...)
}

func (t *Trainer) save(ctx context.Context, name string, kind CheckpointKind, opts ...SaveOption) (Checkpoint, error) {
	if err := ctx.Err(); err != nil {
		return Checkpoint{}, err
	}
	if err := t.checkOpen(); err != nil {
		return Checkpoint{}, err
	}
	if name == "" {
		return Checkpoint{}, errors.New("name is empty")
	}
	var so saveOptions
	for _, opt := range opts {
		if opt != nil {
			opt(&so)
		}
	}
	if so.ttl < 0 {
		return Checkpoint{}, errors.New("ttl is negative")
	}
	return Checkpoint{}, fmt.Errorf("%w: save %s checkpoint", ErrUnsupported, kind)
}

// Load loads a training checkpoint.
func (t *Trainer) Load(ctx context.Context, path string, opts ...LoadOption) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := t.checkOpen(); err != nil {
		return err
	}
	if path == "" {
		return errors.New("path is empty")
	}
	var lo loadOptions
	for _, opt := range opts {
		if opt != nil {
			opt(&lo)
		}
	}
	return fmt.Errorf("%w: load checkpoint", ErrUnsupported)
}

// Sampler creates a sampler over the trainer's current weights.
func (t *Trainer) Sampler(ctx context.Context) (*Sampler, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := t.checkOpen(); err != nil {
		return nil, err
	}
	return &Sampler{model: t.model}, nil
}

// Close releases trainer resources. It is idempotent.
func (t *Trainer) Close() error {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.closed = true
	return nil
}
