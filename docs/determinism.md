# Determinism

## Inputs that control determinism

Symbols below live in `internal/tinkertrain` (`train.go`, `mlx.go`,
`sample_test.go`); they are named rather than cited by line so the references
survive refactoring.

- Sampler seed. `SamplingParams.Seed`; the RNG key is constructed and split per
  token in `sampleToken`. A zero seed leaves the key nil and the underlying
  sampler is unseeded.
- Sampler temperature, top-p, top-k. `SamplingParams` fields, with defaults of
  temperature 1 and top-p 1 applied in `sampleToken`. Temperature 0 with a fixed
  seed is the deterministic configuration used in
  `sample_test.go`.
- Base model and MLX resolution. `CreateConfig.BaseModel`; the default
  `Qwen/Qwen3-8B` maps to `mlx-community/Qwen3-8B-4bit`, overridable with
  `LOCALTINKER_QWEN3_8B_MLX_BASE`.
- Tokenizer. Loaded from the resolved model bundle path. Stop strings are
  tokenized through it, so tokenizer version affects which sequences trigger
  `stop`.
- LoRA adapter config. Rank from `CreateConfig.LoRARank` (default 8);
  alpha = rank*4, dropout 0, key patterns from
  `lmtrain.DefaultAdapterKeyPatterns()`.
- Optimizer. AdamW. The caller supplies `AdamParams` (learning rate default
  1e-4, weight decay, grad clip norm); unset learning rate, betas, and eps are
  filled by `AdamParams.withDefaults`.
- Data order. No shuffling. Batches are consumed in caller order: each
  `forwardBackward` stores the pending batch and the next `optimStep` consumes
  exactly that batch. Within a batch, rows keep input order (`newDenseBatch`).
- Loss. Local execution accepts `cross_entropy`, `importance_sampling`, `ppo`,
  `cispo`, and `dro` (`newDenseBatch`). `denseCrossEntropy` uses weighted-mean
  reduction for non-negative weights, and the unnormalized sum
  `sum(-logprobs*weights)` when any weight is negative — the signed-weight case
  the SDK uses to backpropagate a `forward_backward_custom` loss. Policy losses
  use `logprobs`, `advantages`, weights, and loss-specific config; PPO/CISPO clip
  thresholds and DRO beta are parsed from `LossFnConfig`.
- MLX backend. Selected at link time via `MLX_LIB_PATH`. Execution flags
  (`FastSDPA`, `FastRoPE`) are set when the model is created.
- Checkpoint root. `LOCALTINKER_CHECKPOINT_ROOT` controls where adapter weights
  and optimizer state are written and loaded.

## Not guaranteed deterministic across

- MLX library versions and Metal driver versions.
- Hosted Tinker vs localtinker. Hosted numerics differ; see the hosted
  comparison notes in `docs/internal/conformance.md` and `docs/internal/roadmap.md`.
- Hardware: Metal GPU vs CPU, and across Apple Silicon generations.
- Concurrent runs sharing a model: sampling and training take
  `mlxModel.mu`, but interleaved `forward_backward` / `optim_step` from
  different callers can change which batch the next step consumes.

## Reproducing a run locally

    export MLX_LIB_PATH=/path/to/mlx-go/mlxc/lib
    export GOWORK=off
    export LOCALTINKER_CHECKPOINT_ROOT=$(mktemp -d)
    # then, in SamplingParams:
    #   Seed != 0, Temperature = 0, fixed TopK / TopP
    # and in CreateConfig: pin BaseModel and LoRARank
    go test ./...

Pin `AdamParams` (learning rate, weight decay, grad clip norm) explicitly
in the caller. Submit `forward_backward` and `optim_step` from a single
goroutine to keep batch-to-step pairing deterministic.
