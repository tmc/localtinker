# localtinker

`localtinker` runs a local Tinker-compatible coordinator backed by MLX. It
serves the Python Tinker SDK HTTP API on your machine so SDK jobs can train,
save checkpoints, and sample without using the hosted service.

It includes:

- `cmd/localtinker`: coordinator, Python SDK HTTP API, Connect RPC API, and dashboard.
- `cmd/localtinker-node`: node and artifact cache tools.
- `cmd/localtinker-tray`: macOS menu bar monitor.
- `tinker`: experimental Go API types.

## Requirements

- Go 1.26 or newer.
- Local MLX support through `github.com/tmc/mlx-go`.
- A Python environment with the Thinking Machines Tinker SDK installed.
- The base model must be available to the local MLX/Hugging Face cache. The
  default SDK example uses `Qwen/Qwen3-8B`, mapped locally to
  `mlx-community/Qwen3-8B-4bit`.

## First 10 minutes

Build the local runner:

```sh
go build -o /tmp/localtinker ./cmd/localtinker
```

Start it on localhost:

```sh
/tmp/localtinker serve \
  -addr 127.0.0.1:8080 \
  -home /tmp/localtinker-home
```

In another shell, point the Tinker SDK at the local runner:

```sh
export TINKER_SDK_DIR=$HOME/go/src/github.com/thinking-machines-lab/tinker
export PYTHONPATH=$TINKER_SDK_DIR/src
export TINKER_BASE_URL=http://127.0.0.1:8080
export TINKER_API_KEY=tml-local-test
```

If your SDK virtualenv is elsewhere, set `LOCALTINKER_SDK_PYTHON`:

```sh
export LOCALTINKER_SDK_PYTHON=/path/to/python
```

Run the SDK smoke suite:

```sh
go test ./cmd/localtinker -run TestPythonSDKScript -count=1
```

Run a minimal SDK training job:

```sh
${LOCALTINKER_SDK_PYTHON:-$TINKER_SDK_DIR/.venv/bin/python} \
  ./cmd/localtinker/examples/tinker_job.py --preset short
```

Open the dashboard:

```text
http://127.0.0.1:8080/
```

Use `/runs` for run summaries, `/checkpoints` for local checkpoint paths,
`/nodes` for node health, and `/artifacts` for artifact state.

Beta limits: local checkpoint archive URLs are local HTTP download URLs, hosted
authorization behavior is not reproduced, hosted numeric parity is not
guaranteed, and model execution depends on local MLX libraries plus a cached
mapped base model.

## Start localtinker

Run the coordinator from this repository:

```sh
go run ./cmd/localtinker serve \
  -addr 127.0.0.1:8080 \
  -home .localtinker
```

The server writes coordinator state below `-home`; checkpoints are written
below the system temp directory unless `LOCALTINKER_CHECKPOINT_ROOT` is set.

Open the dashboard at:

```text
http://127.0.0.1:8080/
```

For a multi-process local demo, start a node with an artifact peer address:

```sh
go run ./cmd/localtinker-node run \
  -coordinator http://127.0.0.1:8080 \
  -peer-addr 127.0.0.1:8091 \
  -root .localtinker-node
```

On macOS, start the menu bar monitor in another shell:

```sh
go run ./cmd/localtinker-tray -coordinator http://127.0.0.1:8080
```

The dashboard pages at `/runs`, `/checkpoints`, `/nodes`, and `/artifacts`
show the same live coordinator snapshot with sections ordered for operators.

## Point the Tinker SDK at localtinker

In another shell, point SDK jobs at the local endpoint:

```sh
export TINKER_BASE_URL=http://127.0.0.1:8080
```

Set `TINKER_API_KEY` too if your SDK environment requires one; localtinker does
not validate the value.

## Run the included SDK job

With the coordinator still running:

```sh
python ./cmd/localtinker/examples/tinker_job.py --preset short
```

The script prints JSON events for session setup, futures, loss values,
optimizer steps, and a final summary. The loss should decrease over the short
run. The dashboard shows the model, futures, metrics, and recent run activity.

## Use your own SDK code

Most SDK code only needs `TINKER_BASE_URL` set before creating the client:

```python
import tinker
from tinker import ServiceClient

client = ServiceClient()
training = client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
    rank=8,
)
info = training.get_info()
print(info.model_id)
client.holder.close()
```

Supported local routes include session creation, heartbeat, REST session
listing, model creation, futures, future cancellation, `forward`,
`forward_backward`, `optim_step`, save/load weights,
`load_state_with_optimizer`, sampler sessions, sampling, training run listing,
checkpoint listing, archive URLs, publish/unpublish, TTL, and delete. Sampling
returns generated-token logprobs, prompt logprobs, and top-k prompt logprobs.
Unsupported hosted features return local user errors instead of silently
falling back to the hosted service.

See `docs/` for the local SDK reference and `docs/internal/conformance.md` for
the current SDK coverage and hosted comparison checklist.

## Development

```sh
go test ./...
```

For a clean-checkout release gate, run without any local workspace overrides
and with an isolated build cache:

```sh
GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) GOWORK=off go test ./...
```

Release builds are ordinary Go builds:

```sh
GOWORK=off go build ./cmd/localtinker
GOWORK=off go build ./cmd/localtinker-node
GOOS=darwin GOWORK=off go build ./cmd/localtinker-tray
```

The MLX runtime must be installed or `MLX_LIB_PATH` must point at a directory
containing `libmlxc.dylib`. See [docs/mlx-setup.md](docs/mlx-setup.md) for the
verified path on the development machine, the metallib caveat, and the exact
clean-checkout test command. SDK smoke tests also need a Python environment
with the Tinker SDK installed, and model runs need the mapped base model in the
local Hugging Face or MLX cache.
