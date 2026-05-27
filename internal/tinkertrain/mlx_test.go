package tinkertrain

import "testing"

func TestResolveMLXBaseOverride(t *testing.T) {
	t.Setenv("LOCALTINKER_QWEN3_8B_MLX_BASE", "mlx-community/Qwen3-0.6B-bf16")
	if got := resolveMLXBase("Qwen/Qwen3-8B"); got != "mlx-community/Qwen3-0.6B-bf16" {
		t.Fatalf("resolveMLXBase override = %q, want mlx-community/Qwen3-0.6B-bf16", got)
	}
	if got := resolveMLXBase("other/model"); got != "other/model" {
		t.Fatalf("resolveMLXBase other = %q, want other/model", got)
	}
}
