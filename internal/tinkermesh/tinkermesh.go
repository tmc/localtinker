// Package tinkermesh is the transport seam between a localtinker coordinator and
// its nodes. The coordinator and training engine never learn what carries their
// control messages: an HTTP/Connect-RPC backend and an iroh gossip backend both
// satisfy [Transport], and the coordinator does not know which is in use.
//
// The seam lifts the four node↔coordinator control operations — heartbeats,
// commands, and events — into transport-neutral types. The existing
// Connect-RPC server is the default backend (see [github.com/tmc/localtinker/internal/tinkermeshloop]
// for an in-process backend used in tests); the iroh backend in
// [github.com/tmc/localtinker/internal/tinkermeshiroh] carries the same messages
// over signed gossip, behind an opt-in flag.
//
// Messages are signed by the publishing node's key so a relayed heartbeat or a
// command can be trusted without trusting the relay. This mirrors the mesh
// principle that liveness and assignments are self-attested, verified before a
// scheduling decision acts on them.
package tinkermesh

import (
	"context"

	"github.com/tmc/localtinker/internal/tinkerid"
)

// Load is a node's self-reported capacity and health, the basis for the
// coordinator's scheduling and freshness decisions. It mirrors the existing
// NodeLoad wire fields so the HTTP backend maps onto it without loss.
type Load struct {
	ActiveLeases         int     `json:"active_leases"`
	QueuedOperations     int     `json:"queued_operations"`
	MemoryAvailableBytes uint64  `json:"memory_available_bytes"`
	TemperatureCelsius   float64 `json:"temperature_celsius,omitempty"`
}

// Heartbeat is a node's signed liveness report. NodeID and Signature let the
// coordinator trust the reported Load even when the heartbeat is relayed past
// its origin neighbor on the mesh. UnixNano is the node's send time, used for
// staleness.
type Heartbeat struct {
	NodeID    tinkerid.NodeID `json:"node_id"`
	UnixNano  int64           `json:"unix_nano"`
	Load      Load            `json:"load"`
	Signature []byte          `json:"signature,omitempty"`
}

// Command is a coordinator-issued work assignment or control directive for one
// node. Kind names the directive (run, drain, revoke); Payload carries the
// directive-specific JSON. CommandID and LeaseID tie it to a Future lease.
type Command struct {
	To          tinkerid.NodeID `json:"to"`
	CommandID   string          `json:"command_id"`
	LeaseID     string          `json:"lease_id,omitempty"`
	OperationID string          `json:"operation_id,omitempty"`
	Kind        string          `json:"kind"`
	Payload     []byte          `json:"payload,omitempty"`
	Signature   []byte          `json:"signature,omitempty"`
}

// NodeEvent is a node's report of an operation result or telemetry. It is the
// node→coordinator direction: started, completed, failed, telemetry. Kind names
// the event; Payload carries the event-specific JSON.
type NodeEvent struct {
	From        tinkerid.NodeID `json:"from"`
	CommandID   string          `json:"command_id,omitempty"`
	LeaseID     string          `json:"lease_id,omitempty"`
	OperationID string          `json:"operation_id,omitempty"`
	Kind        string          `json:"kind"`
	UnixNano    int64           `json:"unix_nano"`
	Payload     []byte          `json:"payload,omitempty"`
	Signature   []byte          `json:"signature,omitempty"`
}

// Transport carries coordinator↔node control traffic. The HTTP/Connect-RPC
// server and an iroh gossip backend both implement it; the coordinator does not
// know which is in use. A backend signs outbound messages with the publishing
// node's key and verifies inbound ones, dropping any that fail verification, so
// the channels yield only authenticated messages.
//
// The receive channels are closed when the transport is closed. Publish methods
// return an error if the transport is closed or the message cannot be sent.
type Transport interface {
	// PublishHeartbeat sends this node's signed heartbeat to the coordinator.
	PublishHeartbeat(ctx context.Context, hb Heartbeat) error
	// Heartbeats receives heartbeats (coordinator side).
	Heartbeats() <-chan Heartbeat

	// PublishCommand sends a signed command to a node (coordinator side).
	PublishCommand(ctx context.Context, cmd Command) error
	// Commands receives commands addressed to this node (node side).
	Commands() <-chan Command

	// PublishEvent sends a node event to the coordinator (node side).
	PublishEvent(ctx context.Context, ev NodeEvent) error
	// Events receives node events (coordinator side).
	Events() <-chan NodeEvent

	// Close shuts the transport down and closes the receive channels.
	Close() error
}
