# localtinker Documentation

This directory documents the localtinker SDK-compatible server.

localtinker serves the Thinking Machines Tinker Python SDK HTTP API on a local
MLX-backed coordinator. It is useful for local development, SDK smoke tests, and
offline experiments that should not call the hosted service.

## Start Here

1. Start the coordinator.

```sh
go run ./cmd/localtinker serve -addr 127.0.0.1:8080 -home .localtinker
```

2. Point the Python SDK at the local endpoint.

```sh
export TINKER_BASE_URL=http://127.0.0.1:8080
export TINKER_API_KEY=tml-local-test
```

3. Use the normal upstream SDK clients.

```python
from tinker import ServiceClient

client = ServiceClient()
training = client.create_lora_training_client(base_model="Qwen/Qwen3-8B")
print(training.get_info().model_id)
client.holder.close()
```

## API Reference

- [ServiceClient](api/serviceclient.md): entry point, capabilities, training,
  sampling, and REST client creation.
- [TrainingClient](api/trainingclient.md): forward, backward, optimizer, save,
  load, and custom loss flows.
- [SamplingClient](api/samplingclient.md): sampling, prompt logprobs, top-k
  prompt logprobs, and stop sequences.
- [RestClient](api/restclient.md): runs, checkpoints, archive URLs, publish,
  TTL, delete, and future lookup.
- [APIFuture](api/apifuture.md): local future states and result lookup.
- [Types](api/types.md): request, response, checkpoint, sampling, and tensor
  data notes.
- [Exceptions](api/exceptions.md): local user and system error behavior.

## Dashboard

The coordinator also serves an embedded dashboard and docs pages:

- `/`: live coordinator, node, queue, future, run, and artifact status.
- `/docs`: local SDK compatibility overview.
- `/quickstart`: commands for a short local SDK run.
- `/api`: browser API reference summary.

## Compatibility Notes

localtinker intentionally differs from the hosted service in a few places:

- checkpoint archive URLs are local HTTP download URLs;
- hosted authorization and signed URL behavior are not reproduced;
- numeric results depend on the local MLX libraries and cached model;
- multimodal chunks are parsed and validated, then refused at MLX execution
  because the local runtime has no vision backend;
- CSR sparse `target_tokens` and `weights` are rehydrated locally; unsupported
  sparse tensor names are rejected before MLX execution;
- hosted fleet scheduling is replaced by the local coordinator and optional
  local nodes.

See [internal conformance](internal/conformance.md) for the current SDK coverage
and hosted comparison evidence.
