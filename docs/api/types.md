# Types

localtinker accepts the upstream SDK request and response types for the
supported local routes.

## ModelInput

`ModelInput` carries token IDs for training and sampling. Sampling stop
sequences may be token sequences or strings. String stops are tokenized by the
local model tokenizer.

## Datum

`Datum` carries model input and loss function inputs. Dense cross entropy data
is supported. Multimodal chunks and sparse TensorData are rejected locally.

## SamplingParams

Supported fields include:

- `max_tokens`;
- `seed`;
- `stop`;
- `temperature`;
- `top_k`;
- `top_p`;
- `prompt_logprobs`.

## Checkpoint

Checkpoint records include ID, type, time, local `tinker://` path, size, public
state, and optional expiration time.

## Optimizer State

When saved with optimizer state, local checkpoints can restore Adam state
through `load_state_with_optimizer` or
`ServiceClient.create_training_client_from_state_with_optimizer`.

## Capabilities

The live server capability response is the source of truth for model IDs,
context lengths, max request size, and supported local features.
