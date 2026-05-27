package tinkermgccl

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/tmc/localtinker/internal/tinkercoord"
	tinkerv1 "github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"google.golang.org/protobuf/encoding/protojson"
)

const defaultLeaseTTL = 30 * time.Second

type Epoch string
type NodeID string
type DeviceKind string
type DType string
type Collective string
type ReduceOp string
type Backend string
type BackendState string
type IslandID string
type IslandKind string
type LinkKind string

const (
	DeviceCPU   DeviceKind = "cpu"
	DeviceMetal DeviceKind = "metal"
	DeviceCUDA  DeviceKind = "cuda"
	DeviceROCm  DeviceKind = "rocm"
	DeviceApple DeviceKind = "apple"

	Bool      DType = "bool"
	Int8      DType = "int8"
	Int16     DType = "int16"
	Int32     DType = "int32"
	Int64     DType = "int64"
	Uint8     DType = "uint8"
	Uint16    DType = "uint16"
	Uint32    DType = "uint32"
	Uint64    DType = "uint64"
	Float32   DType = "float32"
	Float64   DType = "float64"
	Complex64 DType = "complex64"

	CollectiveBarrier       Collective = "barrier"
	CollectiveAllReduce     Collective = "all_reduce"
	CollectiveReduceScatter Collective = "reduce_scatter"
	CollectiveAllGather     Collective = "all_gather"
	CollectiveAllToAll      Collective = "all_to_all"
	CollectiveBroadcast     Collective = "broadcast"
	CollectiveSendRecv      Collective = "send_recv"

	Sum  ReduceOp = "sum"
	Prod ReduceOp = "prod"
	Min  ReduceOp = "min"
	Max  ReduceOp = "max"

	BackendInproc Backend      = "inproc"
	BackendReady  BackendState = "ready"

	IslandAppleLocal IslandKind = "apple_local"
	LinkLocalPeer    LinkKind   = "local_peer"
)

type PlanSnapshot struct {
	Epoch   Epoch
	Now     time.Time
	Nodes   []Node
	Islands []Island
	Links   []Link
}

func (s PlanSnapshot) Validate() error {
	seen := make(map[NodeID]bool)
	for _, node := range s.Nodes {
		if node.ID == "" {
			return fmt.Errorf("empty node id")
		}
		if seen[node.ID] {
			return fmt.Errorf("duplicate node %s", node.ID)
		}
		seen[node.ID] = true
		if len(node.Devices) == 0 {
			return fmt.Errorf("node %s has no devices", node.ID)
		}
		if len(node.Backends) == 0 {
			return fmt.Errorf("node %s has no backends", node.ID)
		}
	}
	for _, island := range s.Islands {
		if island.ID == "" {
			return fmt.Errorf("empty island id")
		}
		for _, id := range island.Nodes {
			if !seen[id] {
				return fmt.Errorf("island %s references unknown node %s", island.ID, id)
			}
		}
	}
	for _, link := range s.Links {
		if !seen[link.From] || !seen[link.To] {
			return fmt.Errorf("link references unknown node")
		}
	}
	return nil
}

type Node struct {
	ID          NodeID
	PublicKey   PublicKey
	Machine     string
	Rack        string
	Region      string
	TrustDomain string
	Devices     []Device
	Backends    []BackendAdvert
	Models      []ModelResidency
	Load        Load
	Lease       Lease
	Labels      map[string]string
}

type PublicKey struct {
	Type  string
	Value string
}

type Device struct {
	Kind            DeviceKind
	Ordinal         int
	MemoryBytes     uint64
	FreeMemoryBytes uint64
}

type BackendAdvert struct {
	Backend      Backend
	Capabilities Capabilities
	Readiness    BackendReadiness
}

type Capabilities struct {
	Backend     Backend
	DeviceKinds []DeviceKind
	DTypes      []DType
	Collectives []Collective
	ReduceOps   []ReduceOp
	MinRanks    int
	MaxRanks    int
	SupportsP2P bool
	Notes       []string
}

type BackendReadiness struct {
	State      BackendState
	CheckedAt  time.Time
	QueueDepth int
	Inflight   int
}

type ModelResidency struct {
	Name      string
	Warm      bool
	Precision string
}

type Load struct {
	ActiveBatches int
	QueuedBatches int
}

type Lease struct {
	Epoch     Epoch
	NotBefore time.Time
	Expires   time.Time
	Signature Signature
}

type Signature struct {
	KeyID   string
	Payload []byte
}

type Island struct {
	ID           IslandID
	Kind         IslandKind
	Nodes        []NodeID
	Backend      Backend
	Capabilities Capabilities
	TrustDomain  string
	Labels       map[string]string
}

type Link struct {
	From       NodeID
	To         NodeID
	Kind       LinkKind
	Bandwidth  float64
	MeasuredAt time.Time
	Labels     map[string]string
}

// Options configures Snapshot.
type Options struct {
	Epoch    Epoch
	Now      time.Time
	LeaseTTL time.Duration
}

// Snapshot converts the coordinator's current node view into an offline
// planning snapshot.
func Snapshot(ctx context.Context, coord *tinkercoord.Coordinator, opts Options) (PlanSnapshot, error) {
	if coord == nil {
		return PlanSnapshot{}, fmt.Errorf("nil coordinator")
	}
	view, err := coord.DashboardSnapshot(ctx)
	if err != nil {
		return PlanSnapshot{}, err
	}
	return FromDashboard(view.Nodes, opts)
}

// FromDashboard converts dashboard node records into a validated planning
// snapshot.
func FromDashboard(nodes []tinkercoord.DashboardNode, opts Options) (PlanSnapshot, error) {
	now := opts.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	ttl := opts.LeaseTTL
	if ttl <= 0 {
		ttl = defaultLeaseTTL
	}
	epoch := opts.Epoch
	if epoch == "" {
		epoch = Epoch("localtinker-" + now.Format("20060102T150405Z"))
	}

	nodes = append([]tinkercoord.DashboardNode(nil), nodes...)
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].ID < nodes[j].ID
	})
	snap := PlanSnapshot{Epoch: epoch, Now: now}
	for _, node := range nodes {
		if node.ID == "" || node.State != "healthy" {
			continue
		}
		n, err := convertNode(node, epoch, now, ttl)
		if err != nil {
			return PlanSnapshot{}, err
		}
		snap.Nodes = append(snap.Nodes, n)
	}
	if len(snap.Nodes) != 0 {
		snap.Islands = []Island{localIsland(snap.Nodes)}
		snap.Links = localLinks(snap.Nodes, now)
	}
	if err := snap.Validate(); err != nil {
		return PlanSnapshot{}, err
	}
	return snap, nil
}

func convertNode(node tinkercoord.DashboardNode, epoch Epoch, now time.Time, ttl time.Duration) (Node, error) {
	caps, err := nodeCapabilities(node)
	if err != nil {
		return Node{}, err
	}
	load, err := nodeLoad(node)
	if err != nil {
		return Node{}, err
	}
	labels := cloneLabels(node.Labels)
	machine := labels["machine"]
	if machine == "" {
		machine = node.ID
	}
	out := Node{
		ID:          NodeID(node.ID),
		PublicKey:   PublicKey{Type: "localtinker", Value: node.ID},
		Machine:     machine,
		Rack:        labels["rack"],
		Region:      labels["region"],
		TrustDomain: labels["trust_domain"],
		Devices:     devices(caps),
		Backends:    []BackendAdvert{backendAdvert(caps, load, now)},
		Models:      models(caps),
		Load: Load{
			ActiveBatches: int(load.GetActiveLeases()),
			QueuedBatches: int(load.GetQueuedOperations()),
		},
		Lease: Lease{
			Epoch:     epoch,
			NotBefore: now.Add(-time.Nanosecond),
			Expires:   now.Add(ttl),
			Signature: Signature{
				KeyID:   "localtinker:" + node.ID,
				Payload: []byte("localtinker-shape-only"),
			},
		},
		Labels: labels,
	}
	if out.TrustDomain == "" {
		out.TrustDomain = "local"
	}
	return out, nil
}

func nodeCapabilities(node tinkercoord.DashboardNode) (*tinkerv1.NodeCapabilities, error) {
	if len(node.Capabilities) == 0 {
		return &tinkerv1.NodeCapabilities{MaxConcurrency: int32(node.MaxConcurrency)}, nil
	}
	var caps tinkerv1.NodeCapabilities
	if err := protojson.Unmarshal(node.Capabilities, &caps); err != nil {
		return nil, fmt.Errorf("decode node %s capabilities: %w", node.ID, err)
	}
	if caps.GetMaxConcurrency() == 0 {
		caps.MaxConcurrency = int32(node.MaxConcurrency)
	}
	return &caps, nil
}

func nodeLoad(node tinkercoord.DashboardNode) (*tinkerv1.NodeLoad, error) {
	if len(node.Load) == 0 {
		return &tinkerv1.NodeLoad{ActiveLeases: int32(node.Running)}, nil
	}
	var load tinkerv1.NodeLoad
	if err := protojson.Unmarshal(node.Load, &load); err != nil {
		return nil, fmt.Errorf("decode node %s load: %w", node.ID, err)
	}
	if load.GetActiveLeases() == 0 {
		load.ActiveLeases = int32(node.Running)
	}
	return &load, nil
}

func backendAdvert(caps *tinkerv1.NodeCapabilities, load *tinkerv1.NodeLoad, now time.Time) BackendAdvert {
	maxRanks := int(caps.GetMaxConcurrency())
	if maxRanks <= 0 {
		maxRanks = 1
	}
	return BackendAdvert{
		Backend: BackendInproc,
		Capabilities: Capabilities{
			Backend:     BackendInproc,
			DeviceKinds: deviceKinds(caps),
			DTypes: []DType{
				Bool, Int8, Int16, Int32, Int64,
				Uint8, Uint16, Uint32, Uint64,
				Float32, Float64, Complex64,
			},
			Collectives: []Collective{
				CollectiveBarrier, CollectiveAllReduce,
				CollectiveReduceScatter, CollectiveAllGather,
				CollectiveAllToAll, CollectiveBroadcast,
				CollectiveSendRecv,
			},
			ReduceOps:   []ReduceOp{Sum, Prod, Min, Max},
			MinRanks:    1,
			MaxRanks:    maxRanks,
			SupportsP2P: true,
			Notes:       []string{"localtinker offline planning adapter"},
		},
		Readiness: BackendReadiness{
			State:      BackendReady,
			CheckedAt:  now,
			QueueDepth: int(load.GetQueuedOperations()),
			Inflight:   int(load.GetActiveLeases()),
		},
	}
}

func devices(caps *tinkerv1.NodeCapabilities) []Device {
	var out []Device
	for i, backend := range caps.GetBackends() {
		kind := deviceKind(backend.GetName())
		if kind == "" {
			continue
		}
		out = append(out, Device{
			Kind:            kind,
			Ordinal:         i,
			MemoryBytes:     caps.GetMemory().GetTotalBytes(),
			FreeMemoryBytes: caps.GetMemory().GetAvailableBytes(),
		})
	}
	if len(out) == 0 {
		out = append(out, Device{Kind: DeviceCPU})
	}
	return out
}

func deviceKinds(caps *tinkerv1.NodeCapabilities) []DeviceKind {
	seen := make(map[DeviceKind]bool)
	var out []DeviceKind
	for _, dev := range devices(caps) {
		if !seen[dev.Kind] {
			seen[dev.Kind] = true
			out = append(out, dev.Kind)
		}
	}
	return out
}

func deviceKind(name string) DeviceKind {
	switch strings.ToLower(name) {
	case "metal":
		return DeviceMetal
	case "cpu":
		return DeviceCPU
	case "cuda", "nccl":
		return DeviceCUDA
	case "rocm", "rccl":
		return DeviceROCm
	case "apple", "jaccl":
		return DeviceApple
	default:
		return ""
	}
}

func models(caps *tinkerv1.NodeCapabilities) []ModelResidency {
	out := make([]ModelResidency, 0, len(caps.GetModels()))
	for _, model := range caps.GetModels() {
		out = append(out, ModelResidency{
			Name:      model.GetName(),
			Warm:      model.GetRootHash() != "",
			Precision: model.GetDtype(),
		})
	}
	return out
}

func localIsland(nodes []Node) Island {
	ids := make([]NodeID, 0, len(nodes))
	caps := nodes[0].Backends[0].Capabilities
	caps.MaxRanks = len(nodes)
	for _, node := range nodes {
		ids = append(ids, node.ID)
	}
	return Island{
		ID:           IslandID("localtinker-local"),
		Kind:         IslandAppleLocal,
		Nodes:        ids,
		Backend:      BackendInproc,
		Capabilities: caps,
		TrustDomain:  "local",
		Labels:       map[string]string{"source": "localtinker"},
	}
}

func localLinks(nodes []Node, now time.Time) []Link {
	var links []Link
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			links = append(links, Link{
				From:       nodes[i].ID,
				To:         nodes[j].ID,
				Kind:       LinkLocalPeer,
				Bandwidth:  1,
				MeasuredAt: now,
				Labels:     map[string]string{"source": "localtinker"},
			})
		}
	}
	return links
}

func cloneLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	out := make(map[string]string, len(labels))
	for k, v := range labels {
		out[k] = v
	}
	return out
}
