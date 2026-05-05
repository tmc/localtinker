# SamplingClient

`SamplingClient` samples from either a base model or a saved local training
checkpoint.

```python
sampling = client.create_sampling_client(base_model="Qwen/Qwen3-8B")
```

## Sample

`sample(prompt, num_samples, sampling_params, ...)` returns generated tokens,
generated-token logprobs, and a stop reason.

```python
params = tinker.SamplingParams(
    max_tokens=32,
    temperature=0.7,
    stop=["\n\n"],
)
future = sampling.sample(
    prompt=prompt,
    num_samples=1,
    sampling_params=params,
)
sample = future.result().samples[0]
```

Supported sampling parameters include:

- `max_tokens`;
- `temperature`;
- `top_k`;
- `top_p`;
- `seed`;
- token stop sequences;
- string stop sequences.

## Prompt Logprobs

Set `include_prompt_logprobs=True` to return prompt-token logprobs.

Set `topk_prompt_logprobs` to a positive value to return top-k prompt logprobs
per prompt position.

```python
future = sampling.sample(
    prompt=prompt,
    num_samples=1,
    sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0),
    include_prompt_logprobs=True,
    topk_prompt_logprobs=2,
)
```

## Compute Logprobs

`compute_logprobs(prompt)` is served through the same local sampler session and
returns per-token prompt logprobs.

## Local Notes

Sampling runs through the local MLX adapter, tokenizer, and model cache. Numeric
values can differ from hosted Tinker even when the API shape matches.
