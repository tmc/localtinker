// Package tinkermeshiroh carries localtinker control traffic over iroh gossip,
// implementing [tinkermesh.Transport] with the same four operations the
// Connect-RPC backend serves. It is the opt-in mesh transport: nodes discover
// each other and exchange signed heartbeats, commands, and events over gossip,
// with no central coordinator socket.
//
// Each control message travels in two signature layers: the message itself is
// signed by its origin node (the tinkermesh sign helpers), and the gossip
// envelope is signed again by the publishing key (irohmesh). The backend
// verifies both and the message-level signature before delivering, so a relayed
// message is trusted only if it is authentically signed end to end.
//
// Topics follow the routine/critical split: heartbeats and events ride a
// routine topic, commands ride a critical topic, matching the mesh convention
// that liveness chatter and work assignments are separated.
package tinkermeshiroh

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/tmc/go-iroh/gossip"
	"github.com/tmc/localtinker/internal/tinkerid"
	"github.com/tmc/localtinker/internal/tinkermesh"
	irohmesh "github.com/tmc/mlx-go-iroh"
)

// ALPN is the application protocol the mesh endpoint serves alongside gossip.
const ALPN = "localtinker/mesh/v1"

// Config configures an iroh mesh backend.
type Config struct {
	// Key is this node's identity; it binds the endpoint and signs messages.
	Key tinkerid.Key
	// CoordinatorID is the coordinator's node id, used by nodes to verify the
	// commands they receive. A coordinator leaves it as its own id.
	CoordinatorID tinkerid.NodeID
	// RoutineTopic carries heartbeats and events; CriticalTopic carries
	// commands. Derive both per run so separate runs do not cross-talk.
	RoutineTopic  gossip.TopicID
	CriticalTopic gossip.TopicID
	// Bootstrap seeds the gossip swarm; empty starts a new swarm.
	Bootstrap []string
	// BindAddr is the endpoint bind address; empty uses the go-iroh default.
	BindAddr string
	// Buffer sizes the receive channels.
	Buffer int
}

// Transport is the iroh-backed [tinkermesh.Transport]. Construct with [Dial].
type Transport struct {
	cfg      Config
	ep       *irohmesh.Endpoint
	routine  *gossip.Topic
	critical *gossip.Topic

	heartbeats chan tinkermesh.Heartbeat
	commands   chan tinkermesh.Command
	events     chan tinkermesh.NodeEvent

	cancel context.CancelFunc
	wg     sync.WaitGroup
	clos   sync.Once
}

// Dial binds an iroh endpoint with the node's identity, joins the routine and
// critical gossip topics, and starts the receive loops. The returned Transport
// is ready to publish and its channels begin delivering verified messages.
func Dial(ctx context.Context, cfg Config) (*Transport, error) {
	buf := cfg.Buffer
	if buf <= 0 {
		buf = 64
	}
	boot, err := irohmesh.ParseBootstraps(cfg.Bootstrap)
	if err != nil {
		return nil, fmt.Errorf("mesh dial: bootstrap: %w", err)
	}
	ep, err := irohmesh.Bind(ctx, irohmesh.Config{
		BindAddr: cfg.BindAddr,
		Identity: cfg.Key.Mesh().Ed25519(),
	})
	if err != nil {
		return nil, fmt.Errorf("mesh dial: bind: %w", err)
	}
	// Serve gossip discovery on the routine topic so peers find each other.
	disc := irohmesh.NewGossipDiscovery(ep, cfg.RoutineTopic, boot)
	if err := ep.Serve(ALPN, func(context.Context, *irohmesh.Conn) error { return nil }, disc); err != nil {
		ep.Close()
		return nil, fmt.Errorf("mesh dial: serve: %w", err)
	}

	routine, err := ep.Subscribe(ctx, cfg.RoutineTopic, boot)
	if err != nil {
		ep.Close()
		return nil, fmt.Errorf("mesh dial: subscribe routine: %w", err)
	}
	critical, err := ep.Subscribe(ctx, cfg.CriticalTopic, boot)
	if err != nil {
		routine.Close()
		ep.Close()
		return nil, fmt.Errorf("mesh dial: subscribe critical: %w", err)
	}

	runCtx, cancel := context.WithCancel(context.Background())
	t := &Transport{
		cfg:        cfg,
		ep:         ep,
		routine:    routine,
		critical:   critical,
		heartbeats: make(chan tinkermesh.Heartbeat, buf),
		commands:   make(chan tinkermesh.Command, buf),
		events:     make(chan tinkermesh.NodeEvent, buf),
		cancel:     cancel,
	}
	t.wg.Add(2)
	go t.consumeRoutine(runCtx)
	go t.consumeCritical(runCtx)
	return t, nil
}

// PublishHeartbeat signs and broadcasts hb on the routine topic.
func (t *Transport) PublishHeartbeat(ctx context.Context, hb tinkermesh.Heartbeat) error {
	hb = tinkermesh.SignHeartbeat(hb, t.cfg.Key)
	return t.publish(ctx, t.routine, hb)
}

// PublishCommand signs and broadcasts cmd on the critical topic.
func (t *Transport) PublishCommand(ctx context.Context, cmd tinkermesh.Command) error {
	cmd = tinkermesh.SignCommand(cmd, t.cfg.Key)
	return t.publish(ctx, t.critical, cmd)
}

// PublishEvent signs and broadcasts ev on the routine topic.
func (t *Transport) PublishEvent(ctx context.Context, ev tinkermesh.NodeEvent) error {
	ev = tinkermesh.SignEvent(ev, t.cfg.Key)
	return t.publish(ctx, t.routine, ev)
}

func (t *Transport) publish(ctx context.Context, topic *gossip.Topic, msg any) error {
	payload, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("mesh publish: marshal: %w", err)
	}
	if err := irohmesh.PublishSigned(ctx, topic, t.cfg.Key.Mesh(), payload); err != nil {
		return fmt.Errorf("mesh publish: %w", err)
	}
	return nil
}

// SeedAddr returns this transport's dialable bootstrap string in the
// endpointID@ip:port form another node passes in [Config.Bootstrap] to join its
// swarm without external discovery — the seed-list and test case.
func (t *Transport) SeedAddr() string {
	addr := t.ep.DialableAddr()
	ips := addr.IPAddrs()
	if len(ips) == 0 {
		return t.ep.ID().String()
	}
	return fmt.Sprintf("%s@ip:%s", t.ep.ID(), ips[0])
}

// Heartbeats returns the channel of verified heartbeats.
func (t *Transport) Heartbeats() <-chan tinkermesh.Heartbeat { return t.heartbeats }

// Commands returns the channel of verified commands addressed to this node.
func (t *Transport) Commands() <-chan tinkermesh.Command { return t.commands }

// Events returns the channel of verified node events.
func (t *Transport) Events() <-chan tinkermesh.NodeEvent { return t.events }

func (t *Transport) consumeRoutine(ctx context.Context) {
	defer t.wg.Done()
	for env, err := range irohmesh.VerifiedEnvelopes(t.routine) {
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			continue
		}
		t.dispatchRoutine(ctx, env.Payload)
	}
}

func (t *Transport) consumeCritical(ctx context.Context) {
	defer t.wg.Done()
	for env, err := range irohmesh.VerifiedEnvelopes(t.critical) {
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			continue
		}
		t.dispatchCommand(ctx, env.Payload)
	}
}

// dispatchRoutine decodes a routine-topic payload as either a heartbeat or an
// event, verifies its message-level signature, and delivers it. A payload that
// is neither, or fails verification, is dropped.
func (t *Transport) dispatchRoutine(ctx context.Context, payload []byte) {
	var hb tinkermesh.Heartbeat
	if json.Unmarshal(payload, &hb) == nil && !hb.NodeID.IsZero() && len(hb.Signature) > 0 {
		if tinkermesh.VerifyHeartbeat(hb) == nil {
			send(ctx, t.heartbeats, hb)
			return
		}
	}
	var ev tinkermesh.NodeEvent
	if json.Unmarshal(payload, &ev) == nil && !ev.From.IsZero() && ev.Kind != "" {
		if tinkermesh.VerifyEvent(ev) == nil {
			send(ctx, t.events, ev)
		}
	}
}

func (t *Transport) dispatchCommand(ctx context.Context, payload []byte) {
	var cmd tinkermesh.Command
	if json.Unmarshal(payload, &cmd) != nil || cmd.Kind == "" {
		return
	}
	if tinkermesh.VerifyCommand(cmd, t.cfg.CoordinatorID) != nil {
		return
	}
	send(ctx, t.commands, cmd)
}

// send delivers v on ch, dropping it if ctx is cancelled (transport closing).
func send[T any](ctx context.Context, ch chan T, v T) {
	select {
	case ch <- v:
	case <-ctx.Done():
	}
}

// Close stops the receive loops, closes the topics and endpoint, and closes the
// receive channels. It is idempotent.
func (t *Transport) Close() error {
	var err error
	t.clos.Do(func() {
		t.cancel()
		if t.routine != nil {
			t.routine.Close()
		}
		if t.critical != nil {
			t.critical.Close()
		}
		if t.ep != nil {
			err = t.ep.Close()
		}
		t.wg.Wait()
		close(t.heartbeats)
		close(t.commands)
		close(t.events)
	})
	return err
}
