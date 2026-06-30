package tinkermesh_test

import (
	"testing"

	"github.com/tmc/localtinker/internal/tinkerid"
	"github.com/tmc/localtinker/internal/tinkermesh"
)

func key(t *testing.T) tinkerid.Key {
	t.Helper()
	k, err := tinkerid.LoadOrCreate(t.TempDir())
	if err != nil {
		t.Fatalf("key: %v", err)
	}
	return k
}

func TestHeartbeatSignVerify(t *testing.T) {
	k := key(t)
	hb := tinkermesh.SignHeartbeat(tinkermesh.Heartbeat{
		UnixNano: 123,
		Load:     tinkermesh.Load{ActiveLeases: 1, QueuedOperations: 2, MemoryAvailableBytes: 99, TemperatureCelsius: 42.5},
	}, k)

	if hb.NodeID != k.ID() {
		t.Fatal("sign did not set NodeID from the key")
	}
	if err := tinkermesh.VerifyHeartbeat(hb); err != nil {
		t.Fatalf("verify: %v", err)
	}
	tampered := hb
	tampered.Load.ActiveLeases = 999
	if err := tinkermesh.VerifyHeartbeat(tampered); err == nil {
		t.Fatal("verify accepted a tampered load")
	}
	forged := hb
	forged.NodeID = key(t).ID()
	if err := tinkermesh.VerifyHeartbeat(forged); err == nil {
		t.Fatal("verify accepted a forged signer")
	}
}

func TestCommandSignVerify(t *testing.T) {
	coord := key(t)
	cmd := tinkermesh.SignCommand(tinkermesh.Command{
		To:        key(t).ID(),
		CommandID: "cmd1",
		Kind:      "run",
		Payload:   []byte(`{"op":"forward"}`),
	}, coord)

	if err := tinkermesh.VerifyCommand(cmd, coord.ID()); err != nil {
		t.Fatalf("verify: %v", err)
	}
	// A node verifies a command against the coordinator id it trusts; a
	// different signer must fail.
	if err := tinkermesh.VerifyCommand(cmd, key(t).ID()); err == nil {
		t.Fatal("verify accepted a command from the wrong coordinator")
	}
	tampered := cmd
	tampered.Payload = []byte(`{"op":"evil"}`)
	if err := tinkermesh.VerifyCommand(tampered, coord.ID()); err == nil {
		t.Fatal("verify accepted a tampered payload")
	}
}

func TestEventSignVerify(t *testing.T) {
	k := key(t)
	ev := tinkermesh.SignEvent(tinkermesh.NodeEvent{
		CommandID: "cmd1",
		Kind:      "completed",
		UnixNano:  456,
		Payload:   []byte(`{"metrics":{"loss":0.5}}`),
	}, k)

	if ev.From != k.ID() {
		t.Fatal("sign did not set From from the key")
	}
	if err := tinkermesh.VerifyEvent(ev); err != nil {
		t.Fatalf("verify: %v", err)
	}
	tampered := ev
	tampered.Kind = "failed"
	if err := tinkermesh.VerifyEvent(tampered); err == nil {
		t.Fatal("verify accepted a tampered kind")
	}
}
