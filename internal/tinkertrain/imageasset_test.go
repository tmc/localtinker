package tinkertrain

import (
	"errors"
	"strings"
	"testing"
)

// TestDefaultImageAssetResolverRefuses pins that a fresh Manager refuses
// pointer resolution with the typed boundary error rather than a string
// match. This is the surface callers should use to detect "no store
// configured" instead of comparing error.Error().
func TestDefaultImageAssetResolverRefuses(t *testing.T) {
	four := 4
	c := ModelInputChunk{
		Type: "image_asset_pointer", Format: "png", Location: "tinker://x",
		ExpectedTokens: &four,
	}
	_, err := ResolveImageAssetPointer(DefaultImageAssetResolver{}, c)
	if err == nil {
		t.Fatal("ResolveImageAssetPointer with default resolver succeeded, want refusal")
	}
	if !errors.Is(err, ErrImageAssetStoreNotConfigured) {
		t.Fatalf("error = %v, want errors.Is(_, ErrImageAssetStoreNotConfigured)", err)
	}
}

// TestMapImageAssetResolverResolvesAndRevalidates pins the happy path
// and the magic-byte revalidation contract: a resolver may store any
// bytes, but ResolveImageAssetPointer rejects bytes that fail the
// shared header check.
func TestMapImageAssetResolverResolvesAndRevalidates(t *testing.T) {
	four := 4
	r := NewMapImageAssetResolver()
	r.Set("tinker://good-png", "png", validPNG)
	r.Set("tinker://good-jpeg", "jpeg", validJPEG)
	r.Set("tinker://header-mismatch", "png", validJPEG)
	r.Set("tinker://garbage", "png", []byte("not-an-image"))

	tests := []struct {
		name       string
		location   string
		wantFormat string
		wantErr    string
	}{
		{name: "png resolves", location: "tinker://good-png", wantFormat: "png"},
		{name: "jpeg resolves", location: "tinker://good-jpeg", wantFormat: "jpeg"},
		{
			name:     "header mismatch revalidated",
			location: "tinker://header-mismatch",
			wantErr:  "data does not start with PNG magic bytes",
		},
		{
			name:     "garbage revalidated",
			location: "tinker://garbage",
			wantErr:  "data does not start with PNG magic bytes",
		},
		{
			name:     "missing key surfaces typed error",
			location: "tinker://nope",
			wantErr:  "image asset not found",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := ModelInputChunk{
				Type: "image_asset_pointer", Format: "png", Location: tt.location,
				ExpectedTokens: &four,
			}
			got, err := ResolveImageAssetPointer(r, c)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("ResolveImageAssetPointer error = %v, want nil", err)
				}
				if got.Type != "image" {
					t.Fatalf("resolved type = %q, want image", got.Type)
				}
				if got.Format != tt.wantFormat {
					t.Fatalf("resolved format = %q, want %q", got.Format, tt.wantFormat)
				}
				if got.ExpectedTokens == nil || *got.ExpectedTokens != four {
					t.Fatalf("resolved expected_tokens = %v, want %d", got.ExpectedTokens, four)
				}
				if err := ValidateImageChunk(got); err != nil {
					t.Fatalf("revalidation of resolved chunk failed: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

// TestResolveImageAssetPointerRejectsWrongChunkType pins that the
// resolver entry point only consumes image_asset_pointer; passing an
// inline image is a programmer error and is reported as one.
func TestResolveImageAssetPointerRejectsWrongChunkType(t *testing.T) {
	four := 4
	c := ModelInputChunk{Type: "image", Format: "png", Data: validPNG, ExpectedTokens: &four}
	_, err := ResolveImageAssetPointer(NewMapImageAssetResolver(), c)
	if err == nil || !strings.Contains(err.Error(), `chunk type "image"`) {
		t.Fatalf("error = %v, want chunk type complaint", err)
	}
}

// TestManagerImageAssetResolverDefaultsAndOverrides pins that a fresh
// Manager exposes the refusing default and that SetImageAssetResolver
// installs a replacement (and nil restores the default).
func TestManagerImageAssetResolverDefaultsAndOverrides(t *testing.T) {
	m := &Manager{}
	if _, ok := m.ImageAssetResolver().(DefaultImageAssetResolver); !ok {
		t.Fatalf("fresh Manager resolver = %T, want DefaultImageAssetResolver", m.ImageAssetResolver())
	}
	r := NewMapImageAssetResolver()
	m.SetImageAssetResolver(r)
	if got := m.ImageAssetResolver(); got != r {
		t.Fatalf("after SetImageAssetResolver, resolver = %v, want installed instance", got)
	}
	m.SetImageAssetResolver(nil)
	if _, ok := m.ImageAssetResolver().(DefaultImageAssetResolver); !ok {
		t.Fatalf("after SetImageAssetResolver(nil), resolver = %T, want DefaultImageAssetResolver",
			m.ImageAssetResolver())
	}
}

// TestExecutorRefusesImageAssetPointerEvenWithResolver pins that a
// resolver does not weaken the executor refusal: even when bytes can
// be resolved, MLX execution still refuses image_asset_pointer chunks
// because no vision backend exists. This guards against accidentally
// turning the resolver into an execution path.
func TestExecutorRefusesImageAssetPointerEvenWithResolver(t *testing.T) {
	four := 4
	r := NewMapImageAssetResolver()
	r.Set("tinker://good", "png", validPNG)
	input := ForwardBackwardInput{
		LossFn: "cross_entropy",
		Data: []Datum{{
			ModelInput: ModelInput{Chunks: []ModelInputChunk{
				{Type: "encoded_text", Tokens: []int{1, 2}},
				{Type: "image_asset_pointer", Format: "png", Location: "tinker://good", ExpectedTokens: &four},
			}},
			LossFnInputs: map[string]TensorData{
				"target_tokens": {Data: []float64{1, 1, 1, 1, 1, 1}, DType: "int64"},
			},
		}},
	}
	_, err := newDenseBatch(input)
	if err == nil {
		t.Fatal("newDenseBatch succeeded, want executor refusal")
	}
	if !strings.Contains(err.Error(), "image_asset_pointer chunks require a local image asset store") {
		t.Fatalf("error = %v, want executor refusal substring", err)
	}
}
