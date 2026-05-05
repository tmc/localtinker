# ServiceClient

`ServiceClient` is the normal upstream Tinker SDK entry point. With
`TINKER_BASE_URL` set to a localtinker server, SDK calls stay on the local
coordinator.

```python
from tinker import ServiceClient

client = ServiceClient()
caps = client.get_server_capabilities()
print(caps.supported_models)
client.holder.close()
```

## Server Capabilities

`get_server_capabilities()` reports the local model map, request limits, and
supported local features. localtinker currently advertises LoRA training,
sampling, prompt logprobs, top-k prompt logprobs, and string stop sequences.

## Training Clients

Use `create_lora_training_client` for a new local LoRA run.

```python
training = client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
    rank=8,
)
```

Use `create_training_client_from_state` or
`create_training_client_from_state_with_optimizer` to resume from a local
`tinker://` checkpoint path. The optimizer variant restores saved Adam state.

## Sampling Clients

Use `create_sampling_client` to sample from a base model, or use
`TrainingClient.save_weights_and_get_sampling_client` after training.

```python
sampling = client.create_sampling_client(base_model="Qwen/Qwen3-8B")
```

## REST Clients

`create_rest_client` returns an SDK REST client for listing sessions, runs,
checkpoints, archive URLs, and future results.

## Local Requirements

The server process must have local MLX support and the mapped base model in the
local cache. localtinker does not validate `TINKER_API_KEY`; it only needs the
variable when the SDK expects one.
