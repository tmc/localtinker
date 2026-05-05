package tinkertrain

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/mlx-go-lm/mlxlm"
)

func TestStopTokenSequencesScalarInteger(t *testing.T) {
	tests := []struct {
		name string
		in   any
		want [][]int
	}{
		{name: "int", in: 42, want: [][]int{{42}}},
		{name: "json number", in: json.Number("42"), want: [][]int{{42}}},
		{name: "float64 integer", in: float64(42), want: [][]int{{42}}},
		{name: "float64 fraction", in: 42.5, want: nil},
		{name: "negative", in: -1, want: nil},
		{name: "sequence", in: []int{1, 2}, want: [][]int{{1, 2}}},
		{name: "sequences", in: [][]int{{1}, {2, 3}}, want: [][]int{{1}, {2, 3}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := stopTokenSequences(tt.in, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("stopTokenSequences(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestStopTokenSequencesString(t *testing.T) {
	tok := testTokenizer{
		"stop":  {11, 12},
		"again": {13},
	}
	tests := []struct {
		name string
		in   any
		want [][]int
	}{
		{name: "string", in: "stop", want: [][]int{{11, 12}}},
		{name: "strings", in: []string{"stop", "again"}, want: [][]int{{11, 12}, {13}}},
		{name: "json strings", in: []any{"stop", "again"}, want: [][]int{{11, 12}, {13}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := stopTokenSequences(tt.in, tok)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("stopTokenSequences(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestStopTokenSequencesStringRequiresTokenizer(t *testing.T) {
	_, err := stopTokenSequences("stop", nil)
	if err == nil {
		t.Fatal("stopTokenSequences string error = nil, want tokenizer error")
	}
}

func TestSampleDeterministicSmallCachedModel(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model smoke in short mode")
	}
	base := os.Getenv("LOCALTINKER_SMALL_MODEL")
	if base == "" {
		base = "mlx-community/Qwen3-0.6B-4bit"
	}
	if !cachedHuggingFaceModel(base) {
		t.Skipf("%s is not cached", base)
	}
	ctx := context.Background()
	m := NewManager()
	if err := m.Create(ctx, base, CreateConfig{BaseModel: base, LoRARank: 1}); err != nil {
		t.Fatal(err)
	}
	req := SampleRequest{
		BaseModel:  base,
		NumSamples: 1,
		Prompt:     ModelInput{Chunks: []ModelInputChunk{{Tokens: []int{1, 1, 1}}}},
		SamplingParams: SamplingParams{
			MaxTokens:      1,
			Seed:           7,
			Temperature:    float64Ptr(0),
			PromptLogprobs: true,
		},
	}
	first, err := m.Sample(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	second, err := m.Sample(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first.Sequences[0].Tokens, second.Sequences[0].Tokens) {
		t.Fatalf("tokens differ: %v vs %v", first.Sequences[0].Tokens, second.Sequences[0].Tokens)
	}
	if len(first.Sequences[0].Logprobs) != len(first.Sequences[0].Tokens) {
		t.Fatalf("logprobs = %v tokens = %v", first.Sequences[0].Logprobs, first.Sequences[0].Tokens)
	}
	if len(first.PromptLogprobs) != 3 {
		t.Fatalf("prompt logprobs = %v, want 3 values", first.PromptLogprobs)
	}
}

func cachedHuggingFaceModel(model string) bool {
	home, err := os.UserHomeDir()
	if err != nil {
		return false
	}
	name := "models--" + strings.NewReplacer("/", "--").Replace(model)
	info, err := os.Stat(filepath.Join(home, ".cache", "huggingface", "hub", name))
	return err == nil && info.IsDir()
}

func float64Ptr(v float64) *float64 {
	return &v
}

type testTokenizer map[string][]int32

func (t testTokenizer) Encode(text string) ([]int32, error) {
	tokens, ok := t[text]
	if !ok {
		return nil, fmt.Errorf("unknown text %q", text)
	}
	return append([]int32(nil), tokens...), nil
}

func (testTokenizer) Decode([]int32) (string, error) { return "", nil }
func (testTokenizer) DecodeWithOptions([]int32, bool) (string, error) {
	return "", nil
}
func (testTokenizer) VocabSize() int                                           { return 0 }
func (testTokenizer) EOSToken() int32                                          { return -1 }
func (testTokenizer) EOSTokenIDs() []int32                                     { return nil }
func (testTokenizer) BOSToken() int32                                          { return -1 }
func (testTokenizer) ImageTokenID() (int32, bool)                              { return 0, false }
func (testTokenizer) VisionStartTokenID() (int32, bool)                        { return 0, false }
func (testTokenizer) VisionEndTokenID() (int32, bool)                          { return 0, false }
func (testTokenizer) ApplyChatTemplate([]mlxlm.Message, bool) ([]int32, error) { return nil, nil }
func (testTokenizer) SetChatTemplate(mlxlm.ChatTemplateApplier)                {}
