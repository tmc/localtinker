# localtinker

`localtinker` runs a local Tinker-compatible coordinator backed by MLX.

It includes:

- `cmd/localtinker`: coordinator, Python SDK HTTP API, Connect RPC API, and dashboard.
- `cmd/localtinker-node`: node and artifact cache tools.
- `cmd/localtinker-tray`: macOS menu bar monitor.
- `tinker`: experimental Go API types.

Run:

```sh
go test ./...
go run ./cmd/localtinker serve
```
