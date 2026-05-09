package tinkertrain

import (
	"strings"
	"testing"
)

// validPNG is the PNG magic prefix; enough to satisfy header validation
// without carrying a full image payload.
var validPNG = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00}

// validJPEG is the JPEG SOI marker plus a byte; enough to satisfy header
// validation without carrying a full image payload.
var validJPEG = []byte{0xFF, 0xD8, 0xFF, 0x00}

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
				Type: "image", Format: "png", Data: validPNG, ExpectedTokens: &four,
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
				{Type: "image", Format: "png", Data: validPNG, ExpectedTokens: &two},
				{Type: "encoded_text", Tokens: []int{13}},
			}},
			want: 6,
		},
		{
			name: "image jpeg",
			input: ModelInput{Chunks: []ModelInputChunk{{
				Type: "image", Format: "jpeg", Data: validJPEG, ExpectedTokens: &two,
			}}},
			want: 2,
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

// TestMultimodalExecutionRejected pins the executor-boundary contract.
// The two refusal messages are distinct: image chunks lack a vision
// backend; image_asset_pointer chunks lack a local image asset store.
func TestMultimodalExecutionRejected(t *testing.T) {
	four := 4
	tests := []struct {
		name     string
		chunks   []ModelInputChunk
		seq      int
		wantPart string
	}{
		{
			name: "image refused as no vision backend",
			chunks: []ModelInputChunk{
				{Type: "encoded_text", Tokens: []int{10, 11}},
				{Type: "image", Format: "png", Data: validPNG, ExpectedTokens: &four},
			},
			seq:      6,
			wantPart: "image chunks require a vision backend, which the local MLX runtime does not provide",
		},
		{
			name: "image_asset_pointer refused as no asset store",
			chunks: []ModelInputChunk{
				{Type: "encoded_text", Tokens: []int{10, 11}},
				{Type: "image_asset_pointer", Format: "jpeg", Location: "tinker://x", ExpectedTokens: &four},
			},
			seq:      6,
			wantPart: "image_asset_pointer chunks require a local image asset store, which is not configured",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			targets := make([]float64, tt.seq)
			for i := range targets {
				targets[i] = 1
			}
			input := ForwardBackwardInput{
				LossFn: "cross_entropy",
				Data: []Datum{{
					ModelInput: ModelInput{Chunks: tt.chunks},
					LossFnInputs: map[string]TensorData{
						"target_tokens": {Data: targets, DType: "int64"},
					},
				}},
			}
			_, err := newDenseBatch(input)
			if err == nil {
				t.Fatalf("newDenseBatch succeeded, want refusal containing %q", tt.wantPart)
			}
			if !strings.Contains(err.Error(), tt.wantPart) {
				t.Fatalf("error = %v, want substring %q", err, tt.wantPart)
			}
		})
	}
}

// TestImageChunkHeaderValidation pins that image bytes must begin with
// the magic prefix matching the declared format. Arbitrary nonempty
// bytes are no longer enough.
func TestImageChunkHeaderValidation(t *testing.T) {
	four := 4
	tests := []struct {
		name    string
		chunk   ModelInputChunk
		wantErr string
	}{
		{
			name: "png matches",
			chunk: ModelInputChunk{
				Type: "image", Format: "png", Data: validPNG, ExpectedTokens: &four,
			},
		},
		{
			name: "jpeg matches",
			chunk: ModelInputChunk{
				Type: "image", Format: "jpeg", Data: validJPEG, ExpectedTokens: &four,
			},
		},
		{
			name: "png header on jpeg-declared chunk",
			chunk: ModelInputChunk{
				Type: "image", Format: "jpeg", Data: validPNG, ExpectedTokens: &four,
			},
			wantErr: "data does not start with JPEG magic bytes",
		},
		{
			name: "jpeg header on png-declared chunk",
			chunk: ModelInputChunk{
				Type: "image", Format: "png", Data: validJPEG, ExpectedTokens: &four,
			},
			wantErr: "data does not start with PNG magic bytes",
		},
		{
			name: "garbage rejected for png",
			chunk: ModelInputChunk{
				Type: "image", Format: "png", Data: []byte("not-an-image"), ExpectedTokens: &four,
			},
			wantErr: "data does not start with PNG magic bytes",
		},
		{
			name: "garbage rejected for jpeg",
			chunk: ModelInputChunk{
				Type: "image", Format: "jpeg", Data: []byte{0x00, 0x00, 0x00}, ExpectedTokens: &four,
			},
			wantErr: "data does not start with JPEG magic bytes",
		},
		{
			name: "data shorter than magic",
			chunk: ModelInputChunk{
				Type: "image", Format: "png", Data: []byte{0x89, 0x50}, ExpectedTokens: &four,
			},
			wantErr: "data does not start with PNG magic bytes",
		},
		{
			name: "image_asset_pointer skips header check",
			chunk: ModelInputChunk{
				Type: "image_asset_pointer", Format: "png", Location: "tinker://x",
				ExpectedTokens: &four,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateImageChunk(tt.chunk)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("ValidateImageChunk error = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ValidateImageChunk error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}
