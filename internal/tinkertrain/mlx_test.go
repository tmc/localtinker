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

func TestAdamParamsWithDefaults(t *testing.T) {
	// Empty params take the SDK AdamParams defaults.
	got := AdamParams{}.withDefaults()
	want := AdamParams{LearningRate: 1e-4, Beta1: 0.9, Beta2: 0.95, Eps: 1e-12}
	if got != want {
		t.Fatalf("withDefaults empty = %#v, want %#v", got, want)
	}

	// Explicit values are preserved; weight_decay/grad_clip_norm stay at 0.
	in := AdamParams{LearningRate: 5e-5, Beta1: 0.8, Beta2: 0.99, Eps: 1e-8, WeightDecay: 0.01, GradClipNorm: 1.0}
	if got := in.withDefaults(); got != in {
		t.Fatalf("withDefaults explicit = %#v, want %#v", got, in)
	}

	// Partial: only the unset fields are filled.
	got = AdamParams{Beta2: 0.9}.withDefaults()
	want = AdamParams{LearningRate: 1e-4, Beta1: 0.9, Beta2: 0.9, Eps: 1e-12}
	if got != want {
		t.Fatalf("withDefaults partial = %#v, want %#v", got, want)
	}
}
