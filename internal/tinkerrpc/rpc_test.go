package tinkerrpc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"connectrpc.com/connect"

	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerdb"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1/tinkerv1connect"
)

func newTestRPC(t *testing.T) *Server {
	t.Helper()
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}
	return rpc
}

func operationSnapshot(t *testing.T, snap Snapshot, id string) OperationSnapshot {
	t.Helper()
	for _, op := range snap.Operations {
		if op.OperationID == id {
			return op
		}
	}
	t.Fatalf("operation %s not found in snapshot %+v", id, snap.Operations)
	return OperationSnapshot{}
}

func TestRegisterNodeAndListNodes(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	adminClient := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)

	reg, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
		Name:   "Node A",
		Labels: map[string]string{"rack": "desk"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if got := reg.Msg.GetAssignedNodeId(); got != "node-a" {
		t.Fatalf("assigned node id = %q, want node-a", got)
	}
	if got := reg.Msg.GetConfig()["rpc_max_bytes"]; got != "134217728" {
		t.Fatalf("rpc_max_bytes = %q", got)
	}

	if _, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-a",
		Load:   &tinkerv1.NodeLoad{ActiveLeases: 1},
	})); err != nil {
		t.Fatal(err)
	}

	nodes, err := adminClient.ListNodes(context.Background(), connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes.Msg.GetNodes()) != 1 {
		t.Fatalf("nodes = %d, want 1", len(nodes.Msg.GetNodes()))
	}
	node := nodes.Msg.GetNodes()[0]
	if node.GetNodeId() != "node-a" || node.GetName() != "Node A" || node.GetLoad().GetActiveLeases() != 1 {
		t.Fatalf("node summary = %+v", node)
	}
	snap, err := coord.DashboardSnapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var stored tinkercoord.DashboardNode
	for _, node := range snap.Nodes {
		if node.ID == "node-a" {
			stored = node
			break
		}
	}
	if stored.ID == "" {
		t.Fatalf("dashboard nodes = %#v, want node-a", snap.Nodes)
	}
	if stored.Name != "Node A" || stored.Labels["rack"] != "desk" || stored.Running != 1 {
		t.Fatalf("stored dashboard node = %#v", stored)
	}
}

func TestDrainNodeRequestsDrainOnHeartbeat(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	adminClient := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)

	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
		Name:   "Node A",
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-a",
		Load:   &tinkerv1.NodeLoad{ActiveLeases: 1},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := adminClient.DrainNode(context.Background(), connect.NewRequest(&tinkerv1.DrainNodeRequest{
		NodeId: "node-a",
		Reason: "test",
	})); err != nil {
		t.Fatal(err)
	}

	nodes, err := adminClient.ListNodes(context.Background(), connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if got := nodes.Msg.GetNodes()[0].GetState(); got != "draining" {
		t.Fatalf("state = %q, want draining", got)
	}

	hb, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-a",
		Load:   &tinkerv1.NodeLoad{ActiveLeases: 0},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if !hb.Msg.GetDrainRequested() {
		t.Fatal("drain_requested = false, want true")
	}

	nodes, err = adminClient.ListNodes(context.Background(), connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if got := nodes.Msg.GetNodes()[0].GetState(); got != "drained" {
		t.Fatalf("state = %q, want drained", got)
	}
}

func TestWatchStreamsDrainCommand(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	adminClient := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)

	reg, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
		Name:   "Node A",
	}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-a",
		Load:   &tinkerv1.NodeLoad{ActiveLeases: 1},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := adminClient.DrainNode(context.Background(), connect.NewRequest(&tinkerv1.DrainNodeRequest{
		NodeId: "node-a",
		Reason: "test drain",
	})); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	stream, err := coordClient.Watch(ctx, connect.NewRequest(&tinkerv1.WatchRequest{
		NodeId:        "node-a",
		CoordinatorId: reg.Msg.GetCoordinatorId(),
	}))
	if err != nil {
		t.Fatal(err)
	}
	if !stream.Receive() {
		t.Fatalf("receive drain command: %v", stream.Err())
	}
	cmd := stream.Msg()
	if cmd.GetKind() != "drain" || cmd.GetDrain() == nil {
		t.Fatalf("command = %+v, want drain", cmd)
	}
	if cmd.GetCommandId() == "" || cmd.GetSeqId() == 0 || cmd.GetDeadlineUnixNano() == 0 {
		t.Fatalf("command metadata = %+v", cmd)
	}
	if got := cmd.GetDrain().GetReason(); got != "test drain" {
		t.Fatalf("drain reason = %q, want test drain", got)
	}
	if !cmd.GetDrain().GetCheckpoint() {
		t.Fatal("checkpoint = false, want true")
	}
}

func TestReportLifecycleUpdatesNodeState(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	adminClient := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)

	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
		Name:   "Node A",
	})); err != nil {
		t.Fatal(err)
	}
	stream := coordClient.Report(context.Background())
	if err := stream.Send(&tinkerv1.NodeEvent{
		NodeId: "node-a",
		Payload: &tinkerv1.NodeEvent_Lifecycle{
			Lifecycle: &tinkerv1.LifecycleEvent{
				Event:    "draining",
				Metadata: map[string]string{"reason": "test"},
			},
		},
	}); err != nil {
		t.Fatal(err)
	}
	if err := stream.Send(&tinkerv1.NodeEvent{
		NodeId: "node-a",
		Payload: &tinkerv1.NodeEvent_Telemetry{
			Telemetry: &tinkerv1.NodeTelemetry{
				Labels: map[string]string{"role": "worker"},
			},
		},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.CloseAndReceive(); err != nil {
		t.Fatal(err)
	}

	nodes, err := adminClient.ListNodes(context.Background(), connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	node := nodes.Msg.GetNodes()[0]
	if node.GetState() != "draining" || node.GetLabels()["reason"] != "test" || node.GetLabels()["role"] != "worker" {
		t.Fatalf("node = %+v", node)
	}
}

func TestReportCommandAckUpdatesNodeLabels(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	adminClient := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)

	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
		Name:   "Node A",
	})); err != nil {
		t.Fatal(err)
	}
	stream := coordClient.Report(context.Background())
	if err := stream.Send(&tinkerv1.NodeEvent{
		NodeId:    "node-a",
		CommandId: "cmd-1",
		Kind:      "drain",
		Payload:   &tinkerv1.NodeEvent_Ack{Ack: &tinkerv1.CommandAck{}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.CloseAndReceive(); err != nil {
		t.Fatal(err)
	}

	nodes, err := adminClient.ListNodes(context.Background(), connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	labels := nodes.Msg.GetNodes()[0].GetLabels()
	if labels["last_command_ack_id"] != "cmd-1" || labels["last_command_ack_kind"] != "drain" {
		t.Fatalf("labels = %+v", labels)
	}
}

func TestReportOperationEventsUpdateLoad(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	adminClient := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)

	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-a",
		Load:   &tinkerv1.NodeLoad{QueuedOperations: 1},
	})); err != nil {
		t.Fatal(err)
	}
	stream := coordClient.Report(context.Background())
	for _, event := range []*tinkerv1.NodeEvent{
		{
			NodeId:  "node-a",
			Payload: &tinkerv1.NodeEvent_Started{Started: &tinkerv1.OperationStarted{}},
		},
		{
			NodeId:  "node-a",
			Payload: &tinkerv1.NodeEvent_Completed{Completed: &tinkerv1.OperationCompleted{}},
		},
		{
			NodeId:  "node-a",
			Payload: &tinkerv1.NodeEvent_Failed{Failed: &tinkerv1.OperationFailed{}},
		},
	} {
		if err := stream.Send(event); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := stream.CloseAndReceive(); err != nil {
		t.Fatal(err)
	}

	nodes, err := adminClient.ListNodes(context.Background(), connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	load := nodes.Msg.GetNodes()[0].GetLoad()
	if load.GetActiveLeases() != 0 || load.GetQueuedOperations() != 0 {
		t.Fatalf("load = %+v", load)
	}
}

func TestWatchAssignsRunOperationsAcrossHealthyNodes(t *testing.T) {
	rpc := newTestRPC(t)
	rpc.nodes["node-a"] = &nodeState{
		state:     nodeHealthy,
		load:      &tinkerv1.NodeLoad{MemoryAvailableBytes: 16},
		artifacts: make(map[string]*tinkerv1.ArtifactInventory),
	}
	rpc.nodes["node-b"] = &nodeState{
		state:     nodeHealthy,
		load:      &tinkerv1.NodeLoad{MemoryAvailableBytes: 16},
		artifacts: make(map[string]*tinkerv1.ArtifactInventory),
	}
	first, err := rpc.EnqueueOperation("forward", []byte(`{"n":1}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := rpc.EnqueueOperation("forward", []byte(`{"n":2}`), nil)
	if err != nil {
		t.Fatal(err)
	}

	cmdA, err := rpc.watchCommand("node-a")
	if err != nil {
		t.Fatal(err)
	}
	if cmdA.GetRun() == nil || cmdA.GetOperationId() != first {
		t.Fatalf("node-a command = %+v, want first run operation", cmdA)
	}
	cmdA2, err := rpc.watchCommand("node-a")
	if err != nil {
		t.Fatal(err)
	}
	if cmdA2 != nil {
		t.Fatalf("second node-a command = %+v, want nil while node-b is less loaded", cmdA2)
	}
	cmdB, err := rpc.watchCommand("node-b")
	if err != nil {
		t.Fatal(err)
	}
	if cmdB.GetRun() == nil || cmdB.GetOperationId() != second {
		t.Fatalf("node-b command = %+v, want second run operation", cmdB)
	}

	snap := rpc.Snapshot()
	if len(snap.Operations) != 2 {
		t.Fatalf("operations = %d, want 2", len(snap.Operations))
	}
	firstSnap := operationSnapshot(t, snap, first)
	if firstSnap.State != operationLeased || firstSnap.NodeID != "node-a" {
		t.Fatalf("first snapshot = %+v", firstSnap)
	}
	secondSnap := operationSnapshot(t, snap, second)
	if secondSnap.State != operationLeased || secondSnap.NodeID != "node-b" {
		t.Fatalf("second snapshot = %+v", secondSnap)
	}
}

func TestWatchSkipsDrainingNodeForRunOperation(t *testing.T) {
	rpc := newTestRPC(t)
	rpc.nodes["node-a"] = &nodeState{
		state:       nodeDraining,
		drainReason: "test",
		load:        &tinkerv1.NodeLoad{},
		artifacts:   make(map[string]*tinkerv1.ArtifactInventory),
	}
	rpc.nodes["node-b"] = &nodeState{
		state:     nodeHealthy,
		load:      &tinkerv1.NodeLoad{ActiveLeases: 3},
		artifacts: make(map[string]*tinkerv1.ArtifactInventory),
	}
	opID, err := rpc.EnqueueOperation("forward", nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	drain, err := rpc.watchCommand("node-a")
	if err != nil {
		t.Fatal(err)
	}
	if drain.GetRun() != nil || drain.GetDrain() == nil {
		t.Fatalf("node-a command = %+v, want drain only", drain)
	}
	run, err := rpc.watchCommand("node-b")
	if err != nil {
		t.Fatal(err)
	}
	if run.GetRun() == nil || run.GetOperationId() != opID {
		t.Fatalf("node-b command = %+v, want run operation %s", run, opID)
	}
}

func TestWatchRequeuesExpiredOperationLease(t *testing.T) {
	rpc := newTestRPC(t)
	now := time.Date(2026, 5, 5, 10, 0, 0, 0, time.UTC)
	rpc.now = func() time.Time { return now }
	rpc.leaseTimeout = time.Second
	rpc.nodes["node-a"] = &nodeState{
		state:     nodeHealthy,
		load:      &tinkerv1.NodeLoad{},
		artifacts: make(map[string]*tinkerv1.ArtifactInventory),
	}
	rpc.nodes["node-b"] = &nodeState{
		state:     nodeHealthy,
		load:      &tinkerv1.NodeLoad{},
		artifacts: make(map[string]*tinkerv1.ArtifactInventory),
	}
	opID, err := rpc.EnqueueOperation("forward", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	first, err := rpc.watchCommand("node-a")
	if err != nil {
		t.Fatal(err)
	}
	if first.GetOperationId() != opID {
		t.Fatalf("first operation = %q, want %q", first.GetOperationId(), opID)
	}

	rpc.nodes["node-a"].load.ActiveLeases = 10
	now = now.Add(2 * time.Second)
	second, err := rpc.watchCommand("node-b")
	if err != nil {
		t.Fatal(err)
	}
	if second.GetOperationId() != opID || second.GetLeaseId() == first.GetLeaseId() {
		t.Fatalf("reassigned command = %+v, first lease %q", second, first.GetLeaseId())
	}
	snap := rpc.Snapshot()
	got := operationSnapshot(t, snap, opID)
	if got.Attempts != 2 || got.NodeID != "node-b" {
		t.Fatalf("operation snapshot = %+v, want second attempt on node-b", got)
	}
}

func TestCancelOperationRevokesLeaseOnWatch(t *testing.T) {
	rpc := newTestRPC(t)
	rpc.nodes["node-a"] = &nodeState{
		state:     nodeHealthy,
		load:      &tinkerv1.NodeLoad{},
		artifacts: make(map[string]*tinkerv1.ArtifactInventory),
	}
	opID, err := rpc.EnqueueOperation("forward", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	run, err := rpc.watchCommand("node-a")
	if err != nil {
		t.Fatal(err)
	}
	if run.GetRun() == nil || run.GetOperationId() != opID {
		t.Fatalf("run command = %+v, want operation %s", run, opID)
	}
	if !rpc.CancelOperation(opID, "test cancel") {
		t.Fatal("CancelOperation returned false")
	}
	revoke, err := rpc.watchCommand("node-a")
	if err != nil {
		t.Fatal(err)
	}
	if revoke.GetRevoke() == nil {
		t.Fatalf("revoke command = %+v, want revoke", revoke)
	}
	if revoke.GetOperationId() != opID || revoke.GetLeaseId() != run.GetLeaseId() {
		t.Fatalf("revoke command = {OperationID:%q LeaseID:%q}, want {%q %q}", revoke.GetOperationId(), revoke.GetLeaseId(), opID, run.GetLeaseId())
	}
	if got := revoke.GetRevoke().GetReason(); got != "test cancel" {
		t.Fatalf("revoke reason = %q, want test cancel", got)
	}
	again, err := rpc.watchCommand("node-a")
	if err != nil {
		t.Fatal(err)
	}
	if again != nil {
		t.Fatalf("second command = %+v, want nil", again)
	}
}

func TestCanceledOperationTerminalReportQueuesAck(t *testing.T) {
	rpc := newTestRPC(t)
	rpc.nodes["node-a"] = &nodeState{
		state:     nodeHealthy,
		load:      &tinkerv1.NodeLoad{},
		artifacts: make(map[string]*tinkerv1.ArtifactInventory),
	}
	opID, err := rpc.EnqueueOperation("forward", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cmd, err := rpc.watchCommand("node-a")
	if err != nil {
		t.Fatal(err)
	}
	if !rpc.CancelOperation(opID, "test cancel") {
		t.Fatal("CancelOperation returned false")
	}
	if _, err := rpc.watchCommand("node-a"); err != nil {
		t.Fatal(err)
	}
	if err := rpc.applyNodeEvent(&tinkerv1.NodeEvent{
		NodeId:      "node-a",
		CommandId:   cmd.GetCommandId(),
		LeaseId:     cmd.GetLeaseId(),
		OperationId: opID,
		Kind:        cmd.GetKind(),
		Payload:     &tinkerv1.NodeEvent_Completed{Completed: &tinkerv1.OperationCompleted{}},
	}); err != nil {
		t.Fatal(err)
	}
	snap := rpc.Snapshot()
	got := operationSnapshot(t, snap, opID)
	if got.State != operationFailed || got.LastErrorCode != "canceled" || !got.AckPending {
		t.Fatalf("operation snapshot = %+v, want canceled failure with ack pending", got)
	}
	ack, err := rpc.watchCommand("node-a")
	if err != nil {
		t.Fatal(err)
	}
	if ack.GetAckOperation() == nil || len(ack.GetAckOperation().GetOperationIds()) != 1 || ack.GetAckOperation().GetOperationIds()[0] != opID {
		t.Fatalf("ack command = %+v, want operation %s", ack, opID)
	}
}

func TestReportTerminalOperationQueuesAck(t *testing.T) {
	rpc := newTestRPC(t)
	rpc.nodes["node-a"] = &nodeState{
		state:     nodeHealthy,
		load:      &tinkerv1.NodeLoad{},
		artifacts: make(map[string]*tinkerv1.ArtifactInventory),
	}
	opID, err := rpc.EnqueueOperation("forward", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cmd, err := rpc.watchCommand("node-a")
	if err != nil {
		t.Fatal(err)
	}
	if err := rpc.applyNodeEvent(&tinkerv1.NodeEvent{
		NodeId:      "node-a",
		CommandId:   cmd.GetCommandId(),
		LeaseId:     cmd.GetLeaseId(),
		OperationId: opID,
		Kind:        cmd.GetKind(),
		Payload:     &tinkerv1.NodeEvent_Started{Started: &tinkerv1.OperationStarted{}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := rpc.applyNodeEvent(&tinkerv1.NodeEvent{
		NodeId:      "node-a",
		CommandId:   cmd.GetCommandId(),
		LeaseId:     cmd.GetLeaseId(),
		OperationId: opID,
		Kind:        cmd.GetKind(),
		Payload:     &tinkerv1.NodeEvent_Completed{Completed: &tinkerv1.OperationCompleted{}},
	}); err != nil {
		t.Fatal(err)
	}
	snap := rpc.Snapshot()
	got := operationSnapshot(t, snap, opID)
	if got.State != operationComplete || !got.AckPending {
		t.Fatalf("operation snapshot = %+v, want complete with ack pending", got)
	}
	ack, err := rpc.watchCommand("node-a")
	if err != nil {
		t.Fatal(err)
	}
	if ack.GetAckOperation() == nil || len(ack.GetAckOperation().GetOperationIds()) != 1 || ack.GetAckOperation().GetOperationIds()[0] != opID {
		t.Fatalf("ack command = %+v, want operation %s", ack, opID)
	}
	snap = rpc.Snapshot()
	got = operationSnapshot(t, snap, opID)
	if got.AckPending {
		t.Fatalf("operation snapshot = %+v, want ack cleared", got)
	}
}

func TestHeartbeatReturnsPrewarmRoots(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	tracker := tinkerv1connect.NewArtifactTrackerClient(server.Client(), server.URL)

	if _, err := tracker.PublishManifest(context.Background(), connect.NewRequest(&tinkerv1.PublishManifestRequest{
		Manifest: &tinkerv1.Manifest{RootHash: "root-a", Kind: "model", Storage: "tinker"},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.PublishManifest(context.Background(), connect.NewRequest(&tinkerv1.PublishManifestRequest{
		Manifest: &tinkerv1.Manifest{RootHash: "root-b", Kind: "model", Storage: "tinker"},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
	})); err != nil {
		t.Fatal(err)
	}

	hb, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-a",
		Artifacts: []*tinkerv1.ArtifactInventory{{
			RootHash: "root-b",
			State:    "complete",
		}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	got := hb.Msg.GetPrewarmRoots()
	if len(got) != 1 || got[0] != "root-a" {
		t.Fatalf("prewarm roots = %v, want [root-a]", got)
	}
}

func TestHeartbeatBalancesPrewarmRootsAcrossNodes(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	tracker := tinkerv1connect.NewArtifactTrackerClient(server.Client(), server.URL)

	for _, root := range []string{"root-a", "root-b"} {
		if _, err := tracker.PublishManifest(context.Background(), connect.NewRequest(&tinkerv1.PublishManifestRequest{
			Manifest: &tinkerv1.Manifest{RootHash: root, Kind: "model", Storage: "tinker"},
		})); err != nil {
			t.Fatal(err)
		}
	}
	for _, nodeID := range []string{"node-a", "node-b"} {
		if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
			NodeId: nodeID,
		})); err != nil {
			t.Fatal(err)
		}
	}

	hbA, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-a",
		Load:   &tinkerv1.NodeLoad{MemoryAvailableBytes: 8},
	}))
	if err != nil {
		t.Fatal(err)
	}
	hbB, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-b",
		Load:   &tinkerv1.NodeLoad{MemoryAvailableBytes: 8},
	}))
	if err != nil {
		t.Fatal(err)
	}
	gotA := hbA.Msg.GetPrewarmRoots()
	gotB := hbB.Msg.GetPrewarmRoots()
	if len(gotA) != 1 || len(gotB) != 1 || gotA[0] == gotB[0] {
		t.Fatalf("prewarm roots: node-a=%v node-b=%v, want split roots", gotA, gotB)
	}
}

func TestHeartbeatAssignsPrewarmToLessLoadedNode(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	tracker := tinkerv1connect.NewArtifactTrackerClient(server.Client(), server.URL)

	if _, err := tracker.PublishManifest(context.Background(), connect.NewRequest(&tinkerv1.PublishManifestRequest{
		Manifest: &tinkerv1.Manifest{RootHash: "root-a", Kind: "model", Storage: "tinker"},
	})); err != nil {
		t.Fatal(err)
	}
	for _, nodeID := range []string{"node-a", "node-b"} {
		if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
			NodeId: nodeID,
		})); err != nil {
			t.Fatal(err)
		}
	}

	hbA, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-a",
		Load:   &tinkerv1.NodeLoad{ActiveLeases: 2},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if got := hbA.Msg.GetPrewarmRoots(); len(got) != 0 {
		t.Fatalf("node-a prewarm roots = %v, want none", got)
	}
	hbB, err := coordClient.Heartbeat(context.Background(), connect.NewRequest(&tinkerv1.HeartbeatRequest{
		NodeId: "node-b",
		Load:   &tinkerv1.NodeLoad{},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if got := hbB.Msg.GetPrewarmRoots(); len(got) != 1 || got[0] != "root-a" {
		t.Fatalf("node-b prewarm roots = %v, want [root-a]", got)
	}
}

func TestAdminRunRoutes(t *testing.T) {
	ctx := context.Background()
	store := tinkerdb.OpenMemory()
	coord, err := tinkercoord.New(tinkercoord.Config{Store: store})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.PutModel(ctx, tinkerdb.Model{
		ID:          "model-a",
		SessionID:   "sess-a",
		BaseModel:   "Qwen/Qwen3-8B",
		TokenizerID: "Qwen/Qwen3-8B",
		IsLoRA:      true,
		LoRARank:    8,
		CreatedAt:   time.Unix(1, 0).UTC(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := coord.CompleteFuture(ctx, map[string]any{"ok": true}, map[string]any{
		"type":     "forward",
		"model_id": "model-a",
	}); err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	adminClient := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)
	runs, err := adminClient.ListRuns(ctx, connect.NewRequest(&tinkerv1.ListRunsRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if len(runs.Msg.GetRuns()) != 1 {
		t.Fatalf("runs = %d, want 1", len(runs.Msg.GetRuns()))
	}
	run := runs.Msg.GetRuns()[0]
	if run.GetRunId() != "model-a" || run.GetAlgorithm() != "lora" || run.GetState() != "ready" {
		t.Fatalf("run = %+v", run)
	}

	inspect, err := adminClient.InspectRun(ctx, connect.NewRequest(&tinkerv1.InspectRunRequest{RunId: "model-a"}))
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Run tinkercoord.TrainingRun `json:"run"`
	}
	if err := json.Unmarshal(inspect.Msg.GetRunJson(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Run.TrainingRunID != "model-a" {
		t.Fatalf("inspected run = %#v", body.Run)
	}
	if len(inspect.Msg.GetEventsJson()) != 1 {
		t.Fatalf("events = %d, want 1", len(inspect.Msg.GetEventsJson()))
	}
}

func TestArtifactTrackerManifestInventoryAndPeers(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	tracker := tinkerv1connect.NewArtifactTrackerClient(server.Client(), server.URL)
	admin := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)

	manifest := &tinkerv1.Manifest{
		Kind:      "training_checkpoint",
		Storage:   "tinker",
		Name:      "ckpt",
		RootHash:  "abc123",
		ChunkSize: 4,
		Files: []*tinkerv1.ManifestFile{{
			Path:   "weights.bin",
			Size:   4,
			Sha256: "file",
			Chunks: []*tinkerv1.ChunkRef{{Index: 0, Size: 4, Sha256: "chunk"}},
		}},
	}
	pub, err := tracker.PublishManifest(context.Background(), connect.NewRequest(&tinkerv1.PublishManifestRequest{
		Manifest: manifest,
		Alias:    "latest",
	}))
	if err != nil {
		t.Fatal(err)
	}
	if pub.Msg.GetRootHash() != "abc123" {
		t.Fatalf("root hash = %q", pub.Msg.GetRootHash())
	}
	got, err := tracker.GetManifest(context.Background(), connect.NewRequest(&tinkerv1.GetManifestRequest{RootHashOrAlias: "latest"}))
	if err != nil {
		t.Fatal(err)
	}
	if got.Msg.GetManifest().GetRootHash() != "abc123" {
		t.Fatalf("manifest root = %q", got.Msg.GetManifest().GetRootHash())
	}

	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
		Labels: map[string]string{
			"artifact_peer_url": "http://127.0.0.1:9000",
			"rack":              "desk",
		},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.ReportInventory(context.Background(), connect.NewRequest(&tinkerv1.ReportInventoryRequest{
		NodeId: "node-a",
		Artifacts: []*tinkerv1.ArtifactInventory{{
			RootHash:     "abc123",
			State:        "complete",
			BytesPresent: 4,
		}},
	})); err != nil {
		t.Fatal(err)
	}
	peers, err := tracker.ListPeers(context.Background(), connect.NewRequest(&tinkerv1.ListPeersRequest{
		RootHash:        "abc123",
		PreferredLabels: []string{"rack=desk"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if len(peers.Msg.GetPeers()) != 1 {
		t.Fatalf("peers = %d, want 1", len(peers.Msg.GetPeers()))
	}
	if peers.Msg.GetPeers()[0].GetAddress() != "http://127.0.0.1:9000" {
		t.Fatalf("peer address = %q", peers.Msg.GetPeers()[0].GetAddress())
	}

	artifacts, err := admin.ListArtifacts(context.Background(), connect.NewRequest(&tinkerv1.ListArtifactsRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if len(artifacts.Msg.GetArtifacts()) != 1 || artifacts.Msg.GetArtifacts()[0].GetAlias() != "latest" {
		t.Fatalf("artifacts = %+v", artifacts.Msg.GetArtifacts())
	}
}

func TestArtifactTrackerDeleteAlias(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	tracker := tinkerv1connect.NewArtifactTrackerClient(server.Client(), server.URL)
	manifest := &tinkerv1.Manifest{
		Kind:      "training_checkpoint",
		Storage:   "tinker",
		Name:      "ckpt",
		RootHash:  "abc123",
		ChunkSize: 4,
		Files: []*tinkerv1.ManifestFile{{
			Path:   "weights.bin",
			Size:   4,
			Sha256: "file",
			Chunks: []*tinkerv1.ChunkRef{{Index: 0, Size: 4, Sha256: "chunk"}},
		}},
	}
	if _, err := tracker.PublishManifest(context.Background(), connect.NewRequest(&tinkerv1.PublishManifestRequest{
		Manifest: manifest,
		Alias:    "latest",
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.GetManifest(context.Background(), connect.NewRequest(&tinkerv1.GetManifestRequest{
		RootHashOrAlias: "latest",
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.DeleteAlias(context.Background(), connect.NewRequest(&tinkerv1.DeleteAliasRequest{
		Alias: " latest ",
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.GetManifest(context.Background(), connect.NewRequest(&tinkerv1.GetManifestRequest{
		RootHashOrAlias: "latest",
	})); err == nil {
		t.Fatal("get deleted alias succeeded")
	}
	if _, err := tracker.GetManifest(context.Background(), connect.NewRequest(&tinkerv1.GetManifestRequest{
		RootHashOrAlias: "abc123",
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.DeleteAlias(context.Background(), connect.NewRequest(&tinkerv1.DeleteAliasRequest{})); err == nil {
		t.Fatal("delete empty alias succeeded")
	}
}

func TestApplyRetentionUpdatesNodeInventory(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	tracker := tinkerv1connect.NewArtifactTrackerClient(server.Client(), server.URL)

	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
		Labels: map[string]string{
			"artifact_peer_url": "http://127.0.0.1:9000",
		},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.ReportInventory(context.Background(), connect.NewRequest(&tinkerv1.ReportInventoryRequest{
		NodeId: "node-a",
		Artifacts: []*tinkerv1.ArtifactInventory{
			{RootHash: "root-a", State: "complete", BytesPresent: 4},
			{RootHash: "root-b", State: "complete", BytesPresent: 8},
		},
	})); err != nil {
		t.Fatal(err)
	}

	retention, err := tracker.ApplyRetention(context.Background(), connect.NewRequest(&tinkerv1.ApplyRetentionRequest{
		NodeId:              "node-a",
		ProtectedRootHashes: []string{"root-b"},
		TargetFreeBytes:     4,
	}))
	if err != nil {
		t.Fatal(err)
	}
	if got := retention.Msg.GetDeletedRootHashes(); len(got) != 1 || got[0] != "root-a" {
		t.Fatalf("deleted roots = %v, want [root-a]", got)
	}
	if got := retention.Msg.GetBytesDeleted(); got != 4 {
		t.Fatalf("bytes deleted = %d, want 4", got)
	}

	peers, err := tracker.ListPeers(context.Background(), connect.NewRequest(&tinkerv1.ListPeersRequest{RootHash: "root-a"}))
	if err != nil {
		t.Fatal(err)
	}
	if len(peers.Msg.GetPeers()) != 0 {
		t.Fatalf("root-a peers = %d, want 0", len(peers.Msg.GetPeers()))
	}
	peers, err = tracker.ListPeers(context.Background(), connect.NewRequest(&tinkerv1.ListPeersRequest{RootHash: "root-b"}))
	if err != nil {
		t.Fatal(err)
	}
	if len(peers.Msg.GetPeers()) != 1 {
		t.Fatalf("root-b peers = %d, want 1", len(peers.Msg.GetPeers()))
	}
}

func TestReportTransferUpdatesNodeInventory(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	coordClient := tinkerv1connect.NewTinkerCoordinatorClient(server.Client(), server.URL)
	tracker := tinkerv1connect.NewArtifactTrackerClient(server.Client(), server.URL)
	adminClient := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)

	if _, err := coordClient.RegisterNode(context.Background(), connect.NewRequest(&tinkerv1.RegisterNodeRequest{
		NodeId: "node-a",
		Labels: map[string]string{
			"artifact_peer_url": "http://127.0.0.1:9000",
		},
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := tracker.ReportTransfer(context.Background(), connect.NewRequest(&tinkerv1.ReportTransferRequest{
		NodeId:     "node-a",
		RootHash:   "root-a",
		PeerNodeId: "peer-a",
		State:      "complete",
		Bytes:      12,
	})); err != nil {
		t.Fatal(err)
	}

	peers, err := tracker.ListPeers(context.Background(), connect.NewRequest(&tinkerv1.ListPeersRequest{RootHash: "root-a"}))
	if err != nil {
		t.Fatal(err)
	}
	if len(peers.Msg.GetPeers()) != 1 {
		t.Fatalf("peers = %d, want 1", len(peers.Msg.GetPeers()))
	}
	nodes, err := adminClient.ListNodes(context.Background(), connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	labels := nodes.Msg.GetNodes()[0].GetLabels()
	if labels["last_transfer_state"] != "complete" || labels["last_transfer_peer_node_id"] != "peer-a" || labels["last_transfer_bytes"] != "12" {
		t.Fatalf("labels = %#v", labels)
	}
}

func TestReportTransferRecordsFailure(t *testing.T) {
	coord, err := tinkercoord.New(tinkercoord.Config{Store: tinkerdb.OpenMemory()})
	if err != nil {
		t.Fatal(err)
	}
	rpc, err := New(coord)
	if err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	rpc.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	tracker := tinkerv1connect.NewArtifactTrackerClient(server.Client(), server.URL)
	adminClient := tinkerv1connect.NewTinkerAdminClient(server.Client(), server.URL)
	if _, err := tracker.ReportTransfer(context.Background(), connect.NewRequest(&tinkerv1.ReportTransferRequest{
		NodeId:     "node-a",
		RootHash:   "root-a",
		PeerNodeId: "peer-a",
		State:      "failed",
		Error: &tinkerv1.ErrorInfo{
			Code:    "transfer_failed",
			Message: "peer fetch failed",
		},
	})); err != nil {
		t.Fatal(err)
	}

	nodes, err := adminClient.ListNodes(context.Background(), connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	labels := nodes.Msg.GetNodes()[0].GetLabels()
	if labels["last_transfer_state"] != "failed" || labels["last_transfer_error_code"] != "transfer_failed" || labels["last_transfer_error_message"] != "peer fetch failed" {
		t.Fatalf("labels = %#v", labels)
	}
	peers, err := tracker.ListPeers(context.Background(), connect.NewRequest(&tinkerv1.ListPeersRequest{RootHash: "root-a"}))
	if err != nil {
		t.Fatal(err)
	}
	if len(peers.Msg.GetPeers()) != 0 {
		t.Fatalf("peers = %d, want 0", len(peers.Msg.GetPeers()))
	}
}
