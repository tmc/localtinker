package tinker_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/localtinker/tinker"
)

type exampleRegistry struct{}

func (exampleRegistry) Resolve(context.Context, string) (tinker.ModelSpec, error) {
	return tinker.ModelSpec{
		Name:       "local",
		Path:       "/models/local",
		Tokenizer:  "tokenizer.json",
		MaxContext: 4096,
	}, nil
}

func (exampleRegistry) List(context.Context) ([]tinker.ModelInfo, error) {
	return []tinker.ModelInfo{{Name: "local", Tokenizer: "tokenizer.json", MaxContext: 4096}}, nil
}

func Example() {
	ctx := context.Background()
	root := filepath.Join(os.TempDir(), "tinker-example")
	defer os.RemoveAll(root)

	client, err := tinker.New(tinker.Config{
		RootDir: root,
		Models:  exampleRegistry{},
	})
	if err != nil {
		panic(err)
	}
	defer client.Close()

	trainer, err := client.CreateLoRA(ctx, tinker.CreateLoRARequest{
		BaseModel: "local",
		Rank:      8,
		TrainMLP:  true,
		TrainAttn: true,
	})
	if err != nil {
		panic(err)
	}
	defer trainer.Close()

	batch := []tinker.Datum{{
		Input: tinker.FromTokens([]int{1, 2, 3}),
		LossInput: tinker.LossInput{
			TargetTokens: []int{2, 3, 4},
		},
	}}
	_, err = trainer.Forward(ctx, batch, tinker.CrossEntropy{})
	fmt.Println(errors.Is(err, tinker.ErrUnsupported))
	// Output: true
}
