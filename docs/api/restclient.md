# RestClient

`RestClient` exposes the SDK's REST-style inspection and checkpoint operations.

Create it with `ServiceClient.create_rest_client()`.

```python
rest = client.create_rest_client()
print(rest.list_training_runs())
```

## Runs And Sessions

localtinker supports listing SDK sessions and training runs recorded by the
local coordinator. Training runs can also be looked up by their local
`tinker://` paths.

## Checkpoints

Checkpoint listing, publish, unpublish, TTL update, delete, and archive URL
operations are backed by the local checkpoint store.

Archive URLs are local HTTP download URLs. They are not hosted signed URLs and
do not reproduce hosted authorization behavior.

## Futures

The REST surface can retrieve futures and their final local results. Canceled
and failed futures preserve the same local state visible in the dashboard.

## Error Behavior

Unsupported hosted-only behavior returns a local user error. System failures are
reported as local system errors with the coordinator's message.
