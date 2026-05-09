package tinkertrain

import (
	"strings"
	"testing"
)

func TestDatumTargetsRejects(t *testing.T) {
	tests := []struct {
		name    string
		datum   Datum
		wantErr string
	}{
		{
			name: "sparse target via crow indices",
			datum: Datum{LossFnInputs: map[string]TensorData{
				"target_tokens": {Data: []float64{1, 2}, SparseCrowIndices: []int{0, 1}},
			}},
			wantErr: "sparse target_tokens are not supported",
		},
		{
			name: "sparse target via col indices",
			datum: Datum{LossFnInputs: map[string]TensorData{
				"target_tokens": {Data: []float64{1, 2}, SparseColIndices: []int{0, 1}},
			}},
			wantErr: "sparse target_tokens are not supported",
		},
		{
			name: "dense target succeeds",
			datum: Datum{LossFnInputs: map[string]TensorData{
				"target_tokens": {Data: []float64{1, 2, 3}},
			}},
			wantErr: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.datum.targets()
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("targets() error = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("targets() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestDatumWeightsRejects(t *testing.T) {
	tests := []struct {
		name    string
		datum   Datum
		n       int
		wantErr string
	}{
		{
			name: "sparse weights via crow indices",
			datum: Datum{LossFnInputs: map[string]TensorData{
				"weights": {Data: []float64{1, 1}, SparseCrowIndices: []int{0, 1}},
			}},
			n:       2,
			wantErr: "sparse weights are not supported",
		},
		{
			name: "sparse weights via col indices",
			datum: Datum{LossFnInputs: map[string]TensorData{
				"weights": {Data: []float64{1, 1}, SparseColIndices: []int{0, 1}},
			}},
			n:       2,
			wantErr: "sparse weights are not supported",
		},
		{
			name: "dense weights succeed",
			datum: Datum{LossFnInputs: map[string]TensorData{
				"weights": {Data: []float64{1, 1, 1}},
			}},
			n:       3,
			wantErr: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.datum.weights(tt.n)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("weights() error = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("weights() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestModelInputTokensRejects(t *testing.T) {
	four := 4
	zero := 0
	tests := []struct {
		name    string
		input   ModelInput
		wantErr string
	}{
		{
			name: "image chunk missing format",
			input: ModelInput{Chunks: []ModelInputChunk{{
				Type: "image", Data: []byte("not-empty"), ExpectedTokens: &four,
			}}},
			wantErr: `image chunk: format "", want png or jpeg`,
		},
		{
			name: "image chunk missing expected_tokens",
			input: ModelInput{Chunks: []ModelInputChunk{{
				Type: "image", Format: "png", Data: []byte("x"),
			}}},
			wantErr: "image chunk: expected_tokens is required",
		},
		{
			name: "image chunk non-positive expected_tokens",
			input: ModelInput{Chunks: []ModelInputChunk{{
				Type: "image", Format: "png", Data: []byte("x"), ExpectedTokens: &zero,
			}}},
			wantErr: "expected_tokens = 0, want positive",
		},
		{
			name: "image chunk missing data",
			input: ModelInput{Chunks: []ModelInputChunk{{
				Type: "image", Format: "png", ExpectedTokens: &four,
			}}},
			wantErr: "image chunk: data is required",
		},
		{
			name: "image_asset_pointer missing location",
			input: ModelInput{Chunks: []ModelInputChunk{{
				Type: "image_asset_pointer", Format: "jpeg", ExpectedTokens: &four,
			}}},
			wantErr: "image_asset_pointer chunk: location is required",
		},
		{
			name:    "encoded_text chunk succeeds",
			input:   ModelInput{Chunks: []ModelInputChunk{{Type: "encoded_text", Tokens: []int{1, 2, 3}}}},
			wantErr: "",
		},
		{
			name:    "unknown chunk type",
			input:   ModelInput{Chunks: []ModelInputChunk{{Type: "audio", Tokens: []int{1, 2}}}},
			wantErr: `unknown model input chunk type "audio"`,
		},
		{
			name:    "typo'd chunk type rejected",
			input:   ModelInput{Chunks: []ModelInputChunk{{Type: "encoded_txt", Tokens: []int{1, 2}}}},
			wantErr: `unknown model input chunk type "encoded_txt"`,
		},
		{
			name: "well-formed image chunk parses",
			input: ModelInput{Chunks: []ModelInputChunk{{
				Type: "image", Format: "png", Data: []byte("fake"), ExpectedTokens: &four,
			}}},
			wantErr: "",
		},
		{
			name: "well-formed image_asset_pointer parses",
			input: ModelInput{Chunks: []ModelInputChunk{{
				Type: "image_asset_pointer", Format: "jpeg", Location: "tinker://asset/x",
				ExpectedTokens: &four,
			}}},
			wantErr: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.input.tokens()
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("tokens() error = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("tokens() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}
