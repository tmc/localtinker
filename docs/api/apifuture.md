# APIFuture

SDK methods that schedule coordinator work return an `APIFuture`.

The future can be awaited through the upstream SDK, or retrieved later through
the REST route by future ID.

## States

localtinker records futures in these local states:

- `queued`;
- `running`;
- `complete`;
- `user_error`;
- `system_error`;
- `canceled`.

## Results

Completed futures carry operation-specific result JSON. Training futures include
loss and metrics when available. Sampling futures include samples, generated
tokens, logprobs, and stop reasons.

## Cancellation

Future cancellation is local to the coordinator. Hosted fleet cancellation
behavior is not reproduced exactly; check the future state and error body for
the final local outcome.
