# MLX library setup

localtinker links against MLX through `github.com/tmc/mlx-go`, which loads
`libmlxc.dylib` at runtime. Builds succeed without MLX present, but tests and
any model run fail at first call into MLX with a dynamic-loader error.

## MLX_LIB_PATH

Set `MLX_LIB_PATH` to a directory containing `libmlxc.dylib`. The loader uses
this directly; no symlink or `DYLD_LIBRARY_PATH` is needed.

On this development machine, the verified value is:

```
MLX_LIB_PATH=/Users/tmc/ml-explore/mlx-go/mlxc/lib
```

That directory must contain a built `libmlxc.dylib`. If it is missing, build it
in the `mlx-go` checkout (see `mlx-go/mlxc/README.md`).

## Metallib caveat

The `mlx-go-libs` darwin-arm64 distribution under
`/Volumes/tmc/go/src/github.com/tmc/mlx-go-libs/dist/darwin-arm64` ships
`mlx.metallib.gz`. `libmlxc.dylib` loads from there, but the current tests
look up an uncompressed `mlx.metallib` and fail when only the gzipped form is
present. See `docs/internal/conformance.md` for the recorded behavior.

Use the `mlx-go` source tree's library directory, not the `mlx-go-libs` dist,
until the metallib lookup accepts the gzipped form.

## Clean-checkout test command

Run the full suite with an isolated build cache, no workspace overrides, and
the MLX library directory pinned:

```sh
MLX_LIB_PATH=/path/to/mlx/lib \
  GOCACHE=$(mktemp -d /tmp/localtinker-gocache.XXXXXX) \
  GOWORK=off \
  go test ./...
```

This is the gate used in `docs/internal/conformance.md`.
