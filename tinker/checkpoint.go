package tinker

import "time"

// Checkpoint is a local portable checkpoint name.
type Checkpoint struct {
	Path string
	Kind CheckpointKind
}

// CheckpointKind identifies checkpoint contents.
type CheckpointKind string

const (
	CheckpointTrain   CheckpointKind = "train"
	CheckpointSampler CheckpointKind = "sampler"
)

// CheckpointInfo describes a checkpoint.
type CheckpointInfo struct {
	Path         string
	Kind         CheckpointKind
	BaseModel    string
	IsLoRA       bool
	LoRARank     int
	TrainMLP     bool
	TrainAttn    bool
	TrainUnembed bool
}

// SaveOption configures checkpoint saving.
type SaveOption func(*saveOptions)

type saveOptions struct {
	ttl time.Duration
}

// WithTTL records an advisory checkpoint lifetime.
func WithTTL(ttl time.Duration) SaveOption {
	return func(o *saveOptions) {
		o.ttl = ttl
	}
}

// LoadOption configures checkpoint loading.
type LoadOption func(*loadOptions)

type loadOptions struct {
	withOptimizer bool
}

// WithOptimizer loads optimizer state from a training checkpoint.
func WithOptimizer() LoadOption {
	return func(o *loadOptions) {
		o.withOptimizer = true
	}
}
