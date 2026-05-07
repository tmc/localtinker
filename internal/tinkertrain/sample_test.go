package tinkertrain

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
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

func TestSampleOutputPromptLogprobsMarshalNull(t *testing.T) {
	value := 1.25
	out := SampleOutput{
		Type:               "sample",
		Sequences:          []SampledSequence{{StopReason: "length", Tokens: []int{1}, Logprobs: []float64{-0.5}}},
		PromptLogprobs:     []*float64{nil, &value},
		TopKPromptLogprobs: []any{nil, [][]any{{7, -0.25}, {3, -1.5}}},
	}
	data, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	const want = `"prompt_logprobs":[null,1.25]`
	if !strings.Contains(string(data), want) {
		t.Fatalf("json = %s, want %s", data, want)
	}
	const topK = `"topk_prompt_logprobs":[null,[[7,-0.25],[3,-1.5]]]`
	if !strings.Contains(string(data), topK) {
		t.Fatalf("json = %s, want %s", data, topK)
	}
}

func TestTopKLogprobs(t *testing.T) {
	got := topKLogprobs([]float64{-2, -0.5, -0.5, -4}, 3)
	want := [][]any{{1, -0.5}, {2, -0.5}, {0, -2.0}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("topKLogprobs = %#v, want %#v", got, want)
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
	if first.PromptLogprobs[0] != nil {
		t.Fatalf("prompt logprobs[0] = %v, want nil", *first.PromptLogprobs[0])
	}
	for i, logprob := range first.PromptLogprobs[1:] {
		if logprob == nil {
			t.Fatalf("prompt logprobs[%d] = nil, want value", i+1)
		}
	}
}

// TestSampleDeterministicRepeats exercises seeded-sample repeatability across
// a few configurations. Tokens must match exactly across runs with identical
// inputs; logprobs are compared within an epsilon to absorb any
// non-determinism in MLX reductions across builds.
func TestSampleDeterministicRepeats(t *testing.T) {
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
	prompt := ModelInput{Chunks: []ModelInputChunk{{Tokens: []int{1, 1, 1}}}}
	mkReq := func(maxTokens int, temp float64, seed int) SampleRequest {
		t := temp
		return SampleRequest{
			BaseModel:  base,
			NumSamples: 1,
			Prompt:     prompt,
			SamplingParams: SamplingParams{
				MaxTokens:   maxTokens,
				Seed:        seed,
				Temperature: &t,
			},
		}
	}
	const epsilon = 1e-4

	cases := []struct {
		name string
		req  SampleRequest
	}{
		{name: "greedy temp0 seed7", req: mkReq(3, 0, 7)},
		{name: "stochastic temp0.7 seed7", req: mkReq(3, 0.7, 7)},
		{name: "stochastic temp0.7 seed11", req: mkReq(3, 0.7, 11)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a, err := m.Sample(ctx, tc.req)
			if err != nil {
				t.Fatal(err)
			}
			b, err := m.Sample(ctx, tc.req)
			if err != nil {
				t.Fatal(err)
			}
			ta, tb := a.Sequences[0].Tokens, b.Sequences[0].Tokens
			if !reflect.DeepEqual(ta, tb) {
				t.Fatalf("tokens differ across repeats: %v vs %v", ta, tb)
			}
			la, lb := a.Sequences[0].Logprobs, b.Sequences[0].Logprobs
			if len(la) != len(lb) {
				t.Fatalf("logprob lengths differ: %d vs %d", len(la), len(lb))
			}
			for i := range la {
				if math.Abs(la[i]-lb[i]) > epsilon {
					t.Fatalf("logprob[%d] differ beyond epsilon: %v vs %v", i, la[i], lb[i])
				}
			}
		})
	}
}

// TestSampleDeterministicPrefix verifies that with the same seed and params,
// a shorter generation is a prefix of a longer one. This catches sampler
// state regressions where MaxTokens incorrectly affects the random stream.
func TestSampleDeterministicPrefix(t *testing.T) {
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
	temp := 0.7
	mkReq := func(maxTokens int) SampleRequest {
		return SampleRequest{
			BaseModel:  base,
			NumSamples: 1,
			Prompt:     ModelInput{Chunks: []ModelInputChunk{{Tokens: []int{1, 1, 1}}}},
			SamplingParams: SamplingParams{
				MaxTokens:   maxTokens,
				Seed:        7,
				Temperature: &temp,
			},
		}
	}
	short, err := m.Sample(ctx, mkReq(2))
	if err != nil {
		t.Fatal(err)
	}
	long, err := m.Sample(ctx, mkReq(4))
	if err != nil {
		t.Fatal(err)
	}
	st, lt := short.Sequences[0].Tokens, long.Sequences[0].Tokens
	if short.Sequences[0].StopReason != "length" {
		t.Skipf("short run stopped early (%s); prefix property not applicable", short.Sequences[0].StopReason)
	}
	if len(st) > len(lt) {
		t.Fatalf("short tokens longer than long: %v vs %v", st, lt)
	}
	for i := range st {
		if st[i] != lt[i] {
			t.Fatalf("short is not a prefix of long: short=%v long=%v", st, lt)
		}
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
