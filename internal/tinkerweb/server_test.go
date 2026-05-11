package tinkerweb

import (
	"bytes"
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
	for _, path := range []string{"/docs", "/quickstart", "/api", "/runs", "/checkpoints", "/nodes", "/artifacts"} {
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("GET", path, nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s status = %d", path, rec.Code)
		}
		if !strings.Contains(rec.Body.String(), "preact") {
			t.Fatalf("%s does not serve dashboard index:\n%s", path, rec.Body.String())
		}
	}

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/app.js", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /app.js status = %d", rec.Code)
	}
	for _, want := range []string{"Queue", "Recent Failures", "ArtifactsTable", "ServiceClient", "SDK API"} {
		if !strings.Contains(rec.Body.String(), want) {
			t.Fatalf("app.js missing %q", want)
		}
	}

	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/api/web/dashboard", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET dashboard status = %d body = %s", rec.Code, rec.Body.String())
	}
	raw := rec.Body.Bytes()
	var got Dashboard
	if err := json.NewDecoder(bytes.NewReader(raw)).Decode(&got); err != nil {
		t.Fatal(err)
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(raw, &envelope); err != nil {
		t.Fatal(err)
	}
	var mesh map[string]json.RawMessage
	if err := json.Unmarshal(envelope["mesh"], &mesh); err != nil {
		t.Fatal(err)
	}
	if _, ok := mesh["nodes"]; ok {
		t.Fatalf("mesh contains duplicate nodes: %s", mesh["nodes"])
	}
	if len(got.Coordinator.Nodes) != 2 {
		t.Fatalf("coordinator nodes = %#v, want local and node-a", got.Coordinator.Nodes)
	}
	var node tinkercoord.DashboardNode
	for _, gotNode := range got.Coordinator.Nodes {
		if gotNode.ID == "node-a" {
			node = gotNode
			break
		}
	}
	if node.Name != "mac-studio" || node.Running != 1 {
		t.Fatalf("coordinator node = %#v", node)
	}
}
