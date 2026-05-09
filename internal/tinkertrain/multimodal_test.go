package tinkertrain

import (
	"strings"
	"testing"
)

// TestMultimodalChunkParsesAndCounts pins the parse-layer contract for
// well-formed image and image_asset_pointer chunks: tokens() accepts them
// and contributes expected_tokens placeholder slots so a multimodal request
// has a coherent shape on the SDK side.
func TestMultimodalChunkParsesAndCounts(t *testing.T) {
	four := 4
	two := 2
	tests := []struct {
		name  string
		input ModelInput
		want  int
	}{
		{
			name: "image only",
			input: ModelInput{Chunks: []ModelInputChunk{{
				Type: "image", Format: "png", Data: []byte("img"), ExpectedTokens: &four,
			}}},
			want: 4,
		},
		{
			name: "image_asset_pointer only",
			input: ModelInput{Chunks: []ModelInputChunk{{
				Type: "image_asset_pointer", Format: "jpeg", Location: "tinker://a",
				ExpectedTokens: &two,
			}}},
			want: 2,
		},
		{
			name: "text plus image",
			input: ModelInput{Chunks: []ModelInputChunk{
				{Type: "encoded_text", Tokens: []int{10, 11, 12}},
				{Type: "image", Format: "png", Data: []byte("img"), ExpectedTokens: &two},
				{Type: "encoded_text", Tokens: []int{13}},
			}},
			want: 6,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.input.tokens()
			if err != nil {
				t.Fatalf("tokens() error = %v", err)
			}
			if len(got) != tt.want {
				t.Fatalf("tokens() len = %d, want %d", len(got), tt.want)
			}
			if !tt.input.hasMultimodalChunks() {
				t.Fatal("hasMultimodalChunks = false, want true")
			}
		})
	}
}

// TestMultimodalExecutionRejected pins the executor-boundary contract:
// even when chunks parse cleanly, newDenseBatch refuses multimodal input
// before MLX is touched, with a typed user error that names the boundary.
func TestMultimodalExecutionRejected(t *testing.T) {
	four := 4
	input := ForwardBackwardInput{
		LossFn: "cross_entropy",
		Data: []Datum{{
			ModelInput: ModelInput{Chunks: []ModelInputChunk{
				{Type: "encoded_text", Tokens: []int{10, 11}},
				{Type: "image", Format: "png", Data: []byte("img"), ExpectedTokens: &four},
			}},
			LossFnInputs: map[string]TensorData{
				"target_tokens": {Data: []float64{1, 1, 1, 1, 1, 1}, DType: "int64"},
			},
		}},
	}
	_, err := newDenseBatch(input)
	if err == nil {
		t.Fatal("newDenseBatch succeeded with image chunk, want executor refusal")
	}
	if !strings.Contains(err.Error(), "multimodal chunks not supported by local MLX backend") {
		t.Fatalf("error = %v, want substring 'multimodal chunks not supported by local MLX backend'", err)
	}
}
