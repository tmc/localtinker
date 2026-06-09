# Docs drift audit (2026-06-08, @ bbaf898)

A drift audit of the `docs/` tree after the ecosystem-parity arc landed
(`c5cba5d..bbaf898`). Every claim below was triaged against code ground truth.

## Method

Every claim was triaged directly against the repository — each cited symbol,
flag, file, and behavior was confirmed in code before the docs were edited,
rather than trusted from memory or an external index.

## Findings

| Doc | Verdict | Action |
| --- | --- | --- |
| `internal/conformance.md` | Current (updated @ bbaf898) | none |
| `mlx-setup.md`, `api/*.md` | Current | none |
| `internal/roadmap.md` | Stale (May 11; predates the arc) | add ecosystem-parity coverage |
| `README.md` | Missing the headline capability | surface cookbook-recipe parity |
| `determinism.md` | Drifted line citations + one stale claim | symbol refs; fix CE claim |

### roadmap.md

Predates the ecosystem-parity arc entirely. Omits: running unmodified
`tinker-cookbook` recipes, the `forward_backward_custom` custom-loss
signed-weights path, the optimizer-resume `load_weights` response-type fix, the
cookbook recipe-test pattern (`cmd/localtinker/cookbook_script_test.go`), and the
`-max-operations` coordinator flag (which exists at `cmd/localtinker/main.go`
but is undocumented).

### README.md

No mention that unmodified cookbook recipes run against localtinker — the
strongest recent capability. The "Compatibility Notes" list is accurate but
incomplete.

### determinism.md

The exact `file.go:NNN` citations drifted as `internal/tinkertrain/mlx.go` grew
(~988 to ~1592 lines). Demonstrated wrong: cross-entropy reduction cited at
`mlx.go:988` (actually the `Divide(total, weightTotal)` in `denseCrossEntropy`),
`LOCALTINKER_CHECKPOINT_ROOT` cited at `mlx.go:1054` (actually ~`:1375`), and the
`roadmap.md:197` cross-ref for "hosted numerics differ" now points at unrelated
fixture text. The cross-entropy claim is also incomplete post-fix: it reads
"weighted mean reduction" but `denseCrossEntropy` now returns the unnormalized
sum when weights are signed (the `forward_backward_custom` contract). Resolution:
cite by symbol name rather than line number (matches conformance.md), and correct
the reduction claim.

## Verified NOT drift

`conformance.md` Ecosystem Parity section (2026-06-08) is accurate against code:
`denseCrossEntropy(..., signed bool)`, `denseBatch.weightsSigned`, the four
`testdata/recipe_*.txt` files, and the `-max-operations` flag all exist as
described.
