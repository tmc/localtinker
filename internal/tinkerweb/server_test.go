package tinkerweb

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"connectrpc.com/connect"

	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerdb"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkerrpc"
)

func TestDashboardRoutes(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := tinkerrpc.New(coord)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rpc.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
		Name:   "mac-studio",
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := rpc.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-a",
		Load: &tinkerv1.NodeLoad{
			ActiveLeases:         1,
			QueuedOperations:     2,
			MemoryAvailableBytes: 64 << 30,
			TemperatureCelsius:   48.5,
		},
	})); err != nil {
		t.Fatal(err)
	}

	h := New(coord, rpc).Handler()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET / status = %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "preact") {
		t.Fatalf("index does not reference preact:\n%s", rec.Body.String())
	}

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/api/web/dashboard", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET dashboard status = %d body = %s", rec.Code, rec.Body.String())
	}
	var got Dashboard
	if err := json.NewDecoder(rec.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if len(got.Mesh.Nodes) != 1 || got.Mesh.Nodes[0].Name != "mac-studio" {
		t.Fatalf("nodes = %#v", got.Mesh.Nodes)
	}
}
