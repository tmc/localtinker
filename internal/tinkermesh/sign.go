package tinkermesh

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/tmc/localtinker/internal/tinkerid"
)

// Domain separators keep each message type's signature distinct, so a signed
// heartbeat can never be replayed as a command or event.
const (
	heartbeatDomain = "localtinker/mesh-heartbeat/v1\x00"
	commandDomain   = "localtinker/mesh-command/v1\x00"
	eventDomain     = "localtinker/mesh-event/v1\x00"
)

// ErrUnsigned reports a message that is missing or carries an invalid signature.
var ErrUnsigned = errors.New("tinkermesh: message not validly signed")

func putU64(buf *bytes.Buffer, v uint64) {
	var u [8]byte
	binary.BigEndian.PutUint64(u[:], v)
	buf.Write(u[:])
}

func putField(buf *bytes.Buffer, b []byte) {
	putU64(buf, uint64(len(b)))
	buf.Write(b)
}

func (hb Heartbeat) payload() []byte {
	var buf bytes.Buffer
	buf.WriteString(heartbeatDomain)
	buf.Write(hb.NodeID[:])
	putU64(&buf, uint64(hb.UnixNano))
	putU64(&buf, uint64(hb.Load.ActiveLeases))
	putU64(&buf, uint64(hb.Load.QueuedOperations))
	putU64(&buf, hb.Load.MemoryAvailableBytes)
	putU64(&buf, uint64(int64(hb.Load.TemperatureCelsius*1000)))
	return buf.Bytes()
}

// SignHeartbeat signs hb with the node key and returns it; NodeID is set from
// the key so the signer and the claimed id always agree.
func SignHeartbeat(hb Heartbeat, k tinkerid.Key) Heartbeat {
	hb.NodeID = k.ID()
	hb.Signature = k.Sign(hb.payload())
	return hb
}

// VerifyHeartbeat reports whether hb is signed by hb.NodeID.
func VerifyHeartbeat(hb Heartbeat) error {
	if !tinkerid.VerifyNodeID(hb.NodeID, hb.payload(), hb.Signature) {
		return fmt.Errorf("%w: heartbeat from %s", ErrUnsigned, hb.NodeID)
	}
	return nil
}

func (c Command) payload() []byte {
	var buf bytes.Buffer
	buf.WriteString(commandDomain)
	buf.Write(c.To[:])
	putField(&buf, []byte(c.CommandID))
	putField(&buf, []byte(c.LeaseID))
	putField(&buf, []byte(c.OperationID))
	putField(&buf, []byte(c.Kind))
	putField(&buf, c.Payload)
	return buf.Bytes()
}

// SignCommand signs cmd with the coordinator's node key.
func SignCommand(cmd Command, k tinkerid.Key) Command {
	cmd.Signature = k.Sign(cmd.payload())
	return cmd
}

// VerifyCommand reports whether cmd is signed by coordID, the coordinator key
// the node trusts. A command is the one message signed by the coordinator, not
// the sending node, since the coordinator issues assignments.
func VerifyCommand(cmd Command, coordID tinkerid.NodeID) error {
	if !tinkerid.VerifyNodeID(coordID, cmd.payload(), cmd.Signature) {
		return fmt.Errorf("%w: command %s", ErrUnsigned, cmd.CommandID)
	}
	return nil
}

func (e NodeEvent) payload() []byte {
	var buf bytes.Buffer
	buf.WriteString(eventDomain)
	buf.Write(e.From[:])
	putField(&buf, []byte(e.CommandID))
	putField(&buf, []byte(e.LeaseID))
	putField(&buf, []byte(e.OperationID))
	putField(&buf, []byte(e.Kind))
	putU64(&buf, uint64(e.UnixNano))
	putField(&buf, e.Payload)
	return buf.Bytes()
}

// SignEvent signs ev with the node key; From is set from the key.
func SignEvent(ev NodeEvent, k tinkerid.Key) NodeEvent {
	ev.From = k.ID()
	ev.Signature = k.Sign(ev.payload())
	return ev
}

// VerifyEvent reports whether ev is signed by ev.From.
func VerifyEvent(ev NodeEvent) error {
	if !tinkerid.VerifyNodeID(ev.From, ev.payload(), ev.Signature) {
		return fmt.Errorf("%w: event from %s", ErrUnsigned, ev.From)
	}
	return nil
}
