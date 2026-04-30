# localtinker-tray

`localtinker-tray` is a macOS menu bar monitor for an localtinker
coordinator. It uses `github.com/tmc/apple` for the AppKit status item and the
existing Connect admin API for node and artifact status.

Run it from this module:

```sh
go run . -coordinator http://127.0.0.1:8080
```

Useful flags:

```sh
-coordinator string  coordinator base URL
-interval duration   poll interval
-node-id string      node id or name to highlight
```

The status item title is intentionally compact:

- `T!`: coordinator query failed
- `T?`: selected node is not present
- `T0`: coordinator has no registered nodes
- `Tn`: active leases when any are running, otherwise node count
