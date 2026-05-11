package tinker

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
)

// Client owns model registry access, local run state, and checkpoint
// resolution.
//
// The zero value is not usable; use [New].
type Client struct {
	mu     sync.Mutex
	root   string
	models ModelRegistry
	closed bool
}

// Config configures a Client.
type Config struct {
	RootDir string
	Models  ModelRegistry
}

// New constructs a Client.
func New(cfg Config) (*Client, error) {
	if cfg.RootDir == "" {
		return nil, errors.New("root dir is empty")
	}
	if cfg.Models == nil {
		return nil, errors.New("model registry is nil")
	}
	if err := os.MkdirAll(cfg.RootDir, 0o755); err != nil {
		return nil, fmt.Errorf("create root dir: %w", err)
	}
	return &Client{root: cfg.RootDir, models: cfg.Models}, nil
}

// Close releases client resources. It is idempotent.
func (c *Client) Close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	return nil
}

func (c *Client) checkOpen() error {
	if c == nil {
		return ErrClosed
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return ErrClosed
	}
	if c.models == nil {
		return errors.New("client is not initialized")
	}
	return nil
}

// Capabilities reports the currently exposed API surface.
func (c *Client) Capabilities(ctx context.Context) (Capabilities, error) {
	if err := ctx.Err(); err != nil {
		return Capabilities{}, err
	}
	if err := c.checkOpen(); err != nil {
		return Capabilities{}, err
	}
	return Capabilities{
		Training: true,
		Sampling: true,
		Losses:   []string{"cross_entropy", "importance_sampling"},
	}, nil
}

// CreateLoRA starts a LoRA trainer.
func (c *Client) CreateLoRA(ctx context.Context, req CreateLoRARequest) (*Trainer, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := c.checkOpen(); err != nil {
		return nil, err
	}
	if req.BaseModel == "" {
		return nil, errors.New("base model is empty")
	}
	if req.Rank <= 0 {
		return nil, errors.New("rank must be positive")
	}
	spec, err := c.models.Resolve(ctx, req.BaseModel)
	if err != nil {
		return nil, fmt.Errorf("resolve model: %w", err)
	}
	if err := validateModelSpec(spec); err != nil {
		return nil, fmt.Errorf("resolve model: %w", err)
	}
	return &Trainer{
		model: spec.info(),
		info: TrainingInfo{
			Model:        spec.info(),
			ID:           req.BaseModel,
			IsLoRA:       true,
			LoRARank:     req.Rank,
			TrainMLP:     req.TrainMLP,
			TrainAttn:    req.TrainAttn,
			TrainUnembed: req.TrainUnembed,
		},
	}, nil
}

// OpenTrainer opens a saved trainer checkpoint.
func (c *Client) OpenTrainer(ctx context.Context, path string) (*Trainer, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := c.checkOpen(); err != nil {
		return nil, err
	}
	if path == "" {
		return nil, errors.New("path is empty")
	}
	return nil, fmt.Errorf("%w: open trainer", ErrUnsupported)
}

// NewSampler opens a sampler from a base model and optional checkpoint path.
func (c *Client) NewSampler(ctx context.Context, req SamplingRequest) (*Sampler, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := c.checkOpen(); err != nil {
		return nil, err
	}
	if req.BaseModel == "" {
		return nil, errors.New("base model is empty")
	}
	spec, err := c.models.Resolve(ctx, req.BaseModel)
	if err != nil {
		return nil, fmt.Errorf("resolve model: %w", err)
	}
	if err := validateModelSpec(spec); err != nil {
		return nil, fmt.Errorf("resolve model: %w", err)
	}
	return &Sampler{model: spec.info(), path: req.Path}, nil
}

// Checkpoints lists local checkpoints.
func (c *Client) Checkpoints(ctx context.Context) ([]Checkpoint, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := c.checkOpen(); err != nil {
		return nil, err
	}
	return nil, nil
}

// CheckpointInfo describes a local checkpoint.
func (c *Client) CheckpointInfo(ctx context.Context, path string) (CheckpointInfo, error) {
	if err := ctx.Err(); err != nil {
		return CheckpointInfo{}, err
	}
	if err := c.checkOpen(); err != nil {
		return CheckpointInfo{}, err
	}
	if path == "" {
		return CheckpointInfo{}, errors.New("path is empty")
	}
	return CheckpointInfo{}, fmt.Errorf("%w: checkpoint %q", ErrNotFound, path)
}
