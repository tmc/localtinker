package tinker

import (
	"context"
	"errors"
	"testing"
)

type registry map[string]ModelSpec

func (r registry) Resolve(_ context.Context, name string) (ModelSpec, error) {
	spec, ok := r[name]
	if !ok {
		return ModelSpec{}, ErrNotFound
	}
	return spec, nil
}

func (r registry) List(context.Context) ([]ModelInfo, error) {
	var out []ModelInfo
	for _, spec := range r {
		out = append(out, spec.info())
	}
	return out, nil
}

func TestFromTokensCopies(t *testing.T) {
	tokens := []int{1, 2, 3}
	in := FromTokens(tokens)
	tokens[0] = 99
	if got := in.Tokens()[0]; got != 1 {
		t.Fatalf("Tokens()[0] = %d, want 1", got)
	}
	flat := in.Tokens()
	flat[0] = 88
	if got := in.Tokens()[0]; got != 1 {
		t.Fatalf("Tokens()[0] after mutating flat copy = %d, want 1", got)
	}
}

func TestForwardValidatesBeforeUnsupported(t *testing.T) {
	tr := &Trainer{}
	_, err := tr.Forward(context.Background(), nil, CrossEntropy{})
	if err == nil || errors.Is(err, ErrUnsupported) {
		t.Fatalf("Forward(nil batch) error = %v, want validation error", err)
	}

	batch := []Datum{{
		Input: FromTokens([]int{1}),
		LossInput: LossInput{
			TargetTokens: []int{2},
		},
	}}
	_, err = tr.Forward(context.Background(), batch, CrossEntropy{})
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("Forward valid batch error = %v, want ErrUnsupported", err)
	}
}

func TestCreateLoRAAndClose(t *testing.T) {
	ctx := context.Background()
	client, err := New(Config{
		RootDir: t.TempDir(),
		Models: registry{"m": {
			Name:       "m",
			Path:       "/tmp/m",
			MaxContext: 16,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	tr, err := client.CreateLoRA(ctx, CreateLoRARequest{BaseModel: "m", Rank: 4})
	if err != nil {
		t.Fatal(err)
	}
	info, err := tr.Info(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !info.IsLoRA || info.LoRARank != 4 || info.Model.Name != "m" {
		t.Fatalf("Info() = %+v, want lora rank 4 model m", info)
	}
	if err := tr.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := tr.Info(ctx); !errors.Is(err, ErrClosed) {
		t.Fatalf("Info after Close error = %v, want ErrClosed", err)
	}
}
