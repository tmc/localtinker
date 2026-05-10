package tinkerhttp

import (
	"net/http"
	"testing"

	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerdb"
	"github.com/tmc/localtinker/internal/tinkertrain"
)

// TestImageAssetResolverReachableThroughHTTPConfig pins the programmatic
// wiring path: a caller builds a tinkertrain.Manager, installs a
// resolver, and hands it to tinkercoord.New via Config.Train. After
// tinkercoord owns the Manager the same resolver is still reachable via
// Manager.ImageAssetResolver, and the HTTP server constructed on top of
// the Coordinator routes against that wired Manager.
//
// The HTTP request layer never invokes the resolver itself — multimodal
// execution is refused at the MLX boundary before resolution would run
// — so the proof is two-part: (1) the Manager flows through Config.Train
// and exposes the installed resolver out-of-band; (2) a well-formed
// image_asset_pointer prompt clears the parse layer through the wired
// Server (it fails later for a non-multimodal reason, not for the chunk).
func TestImageAssetResolverReachableThroughHTTPConfig(t *testing.T) {
	resolver := tinkertrain.NewMapImageAssetResolver()
	mgr := tinkertrain.NewManager()
	mgr.SetImageAssetResolver(resolver)

	coord, err := tinkercoord.New(tinkercoord.Config{
		Store: tinkerdb.OpenMemory(),
		Train: mgr,
	})
	if err != nil {
		t.Fatal(err)
	}
	h := New(coord).Handler()

	if got := mgr.ImageAssetResolver(); got != resolver {
		t.Fatalf("after tinkercoord.New consumed the Manager, ImageAssetResolver = %v, want installed instance", got)
	}

	four := 4
	chunk := map[string]any{
		"type":            "image_asset_pointer",
		"format":          "png",
		"location":        "tinker://asset/x",
		"expected_tokens": four,
	}
	req := map[string]any{
		"sampling_session_id": "sample-missing",
		"num_samples":         1,
		"prompt":              map[string]any{"chunks": []any{chunk}},
		"sampling_params":     map[string]any{"max_tokens": 1},
	}
	var future FutureResponse
	postJSONStatus(t, h, "/api/v1/asample", req, http.StatusOK, &future)
	if future.RequestID == "" {
		t.Fatalf("/asample returned no request_id: %#v", future)
	}
}

// TestImageAssetResolverDefaultRefusalThroughHTTPConfig pins that a
// Coordinator built without an explicit resolver still surfaces the
// typed default-refusal sentinel through Manager.ImageAssetResolver.
// This is the negative side of the wiring contract: HTTP runtime users
// who never call SetImageAssetResolver get the refusing default rather
// than an unconfigured nil.
func TestImageAssetResolverDefaultRefusalThroughHTTPConfig(t *testing.T) {
	mgr := tinkertrain.NewManager()
	coord, err := tinkercoord.New(tinkercoord.Config{
		Store: tinkerdb.OpenMemory(),
		Train: mgr,
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = New(coord).Handler()

	if _, ok := mgr.ImageAssetResolver().(tinkertrain.DefaultImageAssetResolver); !ok {
		t.Fatalf("default Manager resolver = %T, want DefaultImageAssetResolver", mgr.ImageAssetResolver())
	}
}
