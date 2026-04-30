package main

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"connectrpc.com/connect"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1"
	"github.com/tmc/localtinker/internal/tinkerproto/tinkerv1/tinkerv1connect"
)

type nodeInfo struct {
	ID           string
	Name         string
	State        string
	ActiveLeases int32
	QueuedOps    int32
	MemoryBytes  uint64
	Temperature  float64
	Labels       map[string]string
}

type artifactInfo struct {
	Alias    string
	Kind     string
	Storage  string
	RootHash string
}

type snapshot struct {
	Coordinator string
	CheckedAt   time.Time
	Nodes       []nodeInfo
	Artifacts   []artifactInfo
	Err         error
}

func fetchSnapshot(ctx context.Context, client *http.Client, coordinator string) snapshot {
	s := snapshot{
		Coordinator: coordinator,
		CheckedAt:   time.Now(),
	}
	admin := tinkerv1connect.NewTinkerAdminClient(client, coordinator)

	nodes, err := admin.ListNodes(ctx, connect.NewRequest(&tinkerv1.ListNodesRequest{}))
	if err != nil {
		s.Err = fmt.Errorf("list nodes: %w", err)
		return s
	}
	for _, n := range nodes.Msg.GetNodes() {
		load := n.GetLoad()
		s.Nodes = append(s.Nodes, nodeInfo{
			ID:           n.GetNodeId(),
			Name:         n.GetName(),
			State:        n.GetState(),
			ActiveLeases: load.GetActiveLeases(),
			QueuedOps:    load.GetQueuedOperations(),
			MemoryBytes:  load.GetMemoryAvailableBytes(),
			Temperature:  load.GetTemperatureCelsius(),
			Labels:       cloneMap(n.GetLabels()),
		})
	}
	sort.Slice(s.Nodes, func(i, j int) bool {
		return s.Nodes[i].ID < s.Nodes[j].ID
	})

	artifacts, err := admin.ListArtifacts(ctx, connect.NewRequest(&tinkerv1.ListArtifactsRequest{}))
	if err != nil {
		s.Err = fmt.Errorf("list artifacts: %w", err)
		return s
	}
	for _, a := range artifacts.Msg.GetArtifacts() {
		s.Artifacts = append(s.Artifacts, artifactInfo{
			Alias:    a.GetAlias(),
			Kind:     a.GetKind(),
			Storage:  a.GetStorage(),
			RootHash: a.GetRootHash(),
		})
	}
	sort.Slice(s.Artifacts, func(i, j int) bool {
		if s.Artifacts[i].Alias != s.Artifacts[j].Alias {
			return s.Artifacts[i].Alias < s.Artifacts[j].Alias
		}
		return s.Artifacts[i].RootHash < s.Artifacts[j].RootHash
	})
	return s
}

func cloneMap(m map[string]string) map[string]string {
	if len(m) == 0 {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func (s snapshot) title(nodeID string) string {
	if s.Err != nil {
		return "T!"
	}
	if nodeID != "" {
		n, ok := s.node(nodeID)
		if !ok {
			return "T?"
		}
		if n.ActiveLeases > 0 {
			return fmt.Sprintf("T%d", n.ActiveLeases)
		}
		return "T"
	}
	active := int32(0)
	for _, n := range s.Nodes {
		active += n.ActiveLeases
	}
	if active > 0 {
		return fmt.Sprintf("T%d", active)
	}
	if len(s.Nodes) == 0 {
		return "T0"
	}
	return fmt.Sprintf("T%d", len(s.Nodes))
}

func (s snapshot) tooltip(nodeID string) string {
	lines := []string{"localtinker", s.Coordinator}
	if s.Err != nil {
		lines = append(lines, "error: "+s.Err.Error())
		return strings.Join(lines, "\n")
	}
	if nodeID != "" {
		if n, ok := s.node(nodeID); ok {
			lines = append(lines, nodeLine(n))
		} else {
			lines = append(lines, "node not found: "+nodeID)
		}
	} else {
		lines = append(lines, fmt.Sprintf("%d nodes, %d artifacts", len(s.Nodes), len(s.Artifacts)))
	}
	if !s.CheckedAt.IsZero() {
		lines = append(lines, "checked "+s.CheckedAt.Format(time.Kitchen))
	}
	return strings.Join(lines, "\n")
}

func (s snapshot) node(id string) (nodeInfo, bool) {
	for _, n := range s.Nodes {
		if n.ID == id || n.Name == id {
			return n, true
		}
	}
	return nodeInfo{}, false
}

func (s snapshot) menuLines(nodeID string) []string {
	lines := []string{
		"Coordinator: " + s.Coordinator,
	}
	if !s.CheckedAt.IsZero() {
		lines = append(lines, "Last check: "+s.CheckedAt.Format(time.RFC3339))
	}
	if s.Err != nil {
		return append(lines, "Error: "+s.Err.Error())
	}
	if nodeID != "" {
		if n, ok := s.node(nodeID); ok {
			lines = append(lines, "Node: "+nodeLine(n))
		} else {
			lines = append(lines, "Node not found: "+nodeID)
		}
	} else {
		lines = append(lines, fmt.Sprintf("Nodes: %d", len(s.Nodes)))
		for _, n := range s.Nodes {
			lines = append(lines, "  "+nodeLine(n))
		}
	}
	lines = append(lines, fmt.Sprintf("Artifacts: %d", len(s.Artifacts)))
	for i, a := range s.Artifacts {
		if i == 5 {
			lines = append(lines, fmt.Sprintf("  ... %d more", len(s.Artifacts)-i))
			break
		}
		lines = append(lines, "  "+artifactLine(a))
	}
	return lines
}

func nodeLine(n nodeInfo) string {
	name := n.Name
	if name == "" {
		name = n.ID
	}
	state := n.State
	if state == "" {
		state = "unknown"
	}
	mem := formatBytes(n.MemoryBytes)
	return fmt.Sprintf("%s %s leases=%d queue=%d mem=%s temp=%.1fC", name, state, n.ActiveLeases, n.QueuedOps, mem, n.Temperature)
}

func artifactLine(a artifactInfo) string {
	name := a.Alias
	if name == "" {
		name = shortHash(a.RootHash)
	}
	kind := a.Kind
	if kind == "" {
		kind = "artifact"
	}
	storage := a.Storage
	if storage == "" {
		storage = "unknown"
	}
	return fmt.Sprintf("%s %s %s", name, kind, storage)
}

func shortHash(s string) string {
	if len(s) <= 12 {
		return s
	}
	return s[:12]
}

func formatBytes(n uint64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%dB", n)
	}
	div, exp := uint64(unit), 0
	for n/div >= unit && exp < 5 {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f%cB", float64(n)/float64(div), "KMGTPE"[exp])
}
