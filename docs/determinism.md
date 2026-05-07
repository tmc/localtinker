# Determinism

## Inputs that control determinism

- Sampler seed. `SamplingParams.Seed` at `internal/tinkertrain/train.go:97`;
  RNG key constructed and split per token at
  `internal/tinkertrain/mlx.go:343` and `internal/tinkertrain/mlx.go:350`. A
  zero seed leaves `key` nil and the underlying sampler is unseeded.
- Sampler temperature, top-p, top-k. `SamplingParams` at
  `internal/tinkertrain/train.go:95`; defaults applied at
  `internal/tinkertrain/mlx.go:421` (temperature=1) and
  `internal/tinkertrain/mlx.go:426` (top-p=1). Temperature 0 with a fixed
  seed is the deterministic configuration used in
  `internal/tinkertrain/sample_test.go:129`.
- Base model and MLX resolution. `CreateConfig.BaseModel` at
  `internal/tinkertrain/train.go:21`; default `Qwen/Qwen3-8B` mapped to
  `mlx-community/Qwen3-8B-4bit` at `internal/tinkertrain/mlx.go:71` and
  `internal/tinkertrain/mlx.go:127`.
- Tokenizer. Loaded from the model bundle path at
  `internal/tinkertrain/mlx.go:117`. Stop strings are tokenized through it
  at `internal/tinkertrain/mlx.go:1112`, so tokenizer version affects which
  sequences trigger `stop`.
- LoRA adapter config. Rank from `CreateConfig.LoRARank`
  (`internal/tinkertrain/train.go:22`), default 8 at
  `internal/tinkertrain/mlx.go:88`; alpha = rank*4, dropout 0, key patterns
  from `lmtrain.DefaultAdapterKeyPatterns()` at
  `internal/tinkertrain/mlx.go:92`.
- Optimizer. AdamW, fixed at `internal/tinkertrain/mlx.go:759`. Caller
  supplies `AdamParams` (`internal/tinkertrain/train.go:70`): learning rate
  (default 1e-4 at `internal/tinkertrain/mlx.go:174`), weight decay, grad
  clip norm. Betas and eps are not plumbed; defaults come from
  `training.DefaultTrainParameters()` at `internal/tinkertrain/mlx.go:758`.
- Data order. No shuffling. Batches are consumed in caller order: each
  `forward_backward` stores the batch at `internal/tinkertrain/mlx.go:152`
  and the next `optim_step` consumes exactly that pending batch at
  `internal/tinkertrain/mlx.go:168`. Within a batch, rows keep input order
  (`internal/tinkertrain/mlx.go:586`).
- Loss. Only `cross_entropy` is accepted
  (`internal/tinkertrain/mlx.go:579`); weighted mean reduction at
  `internal/tinkertrain/mlx.go:816`.
- MLX backend. Selected at link time via `MLX_LIB_PATH`. Execution flags
  (`FastSDPA`, `FastRoPE`) set at `internal/tinkertrain/mlx.go:81`.
- Checkpoint root. `LOCALTINKER_CHECKPOINT_ROOT`
  (`internal/tinkertrain/mlx.go:1054`) controls where adapter weights and
  optimizer state are written and loaded.

## Not guaranteed deterministic across

- MLX library versions and Metal driver versions.
- Hosted Tinker vs localtinker. Hosted numerics differ
  (`docs/internal/roadmap.md:197`).
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
