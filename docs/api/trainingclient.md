# TrainingClient

`TrainingClient` represents one local model training run.

You normally create it with `ServiceClient.create_lora_training_client`.

```python
training = client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
    rank=8,
)
```

## Forward

`forward(data, "cross_entropy")` computes local loss without applying
gradients.

localtinker supports dense cross entropy inputs through the upstream SDK data
types. Unsupported tensor forms, including sparse TensorData, return local user
errors.

## Forward And Backward

`forward_backward(data, "cross_entropy")` computes loss and gradients for a
subsequent optimizer step.

```python
future = training.forward_backward(data, "cross_entropy")
result = future.result()
print(result.loss)
```

## Custom Loss

`forward_backward_custom` supports custom loss functions over logprobs, matching
the upstream SDK call shape. The local server computes logprobs through the MLX
adapter and returns metrics through the same future path.

## Optimizer Step

`optim_step(types.AdamParams(...))` applies a local Adam update and records
optimizer state for later save/load.

```python
optim = training.optim_step(types.AdamParams(learning_rate=1e-4))
optim.result()
```

## Save And Load

`save_weights`, `save_weights_and_get_sampling_client`, `load_state`, and
`load_state_with_optimizer` operate on local `tinker://` checkpoint paths.

Checkpoint archive download URLs are served by the local HTTP server. Hosted
signed URL authorization is not reproduced.

## Metrics

The dashboard and future records include local metrics such as loss, token
count, gradient norm, max update, optimizer step, and backend markers when
available.
