package tinkernode

import (
	"context"
	"fmt"
)

func ExampleMemoryResultCache() {
	ctx := context.Background()
	cache := NewMemoryResultCache()
	_ = cache.Store(ctx, OperationResult{
		OperationID: "op-1",
		Kind:        OperationOptimStep,
		State:       ResultSucceeded,
	})
	result, ok, _ := cache.Get(ctx, "op-1")
	fmt.Println(ok, result.Kind, result.State)
	// Output: true optim_step succeeded
}
