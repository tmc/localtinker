package main

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestSnapshotTitle(t *testing.T) {
	tests := []struct {
		name   string
		s      snapshot
		nodeID string
		want   string
	}{
		{name: "error", s: snapshot{Err: errors.New("down")}, want: "T!"},
		{name: "empty", s: snapshot{}, want: "T0"},
		{name: "nodes", s: snapshot{Nodes: []nodeInfo{{ID: "a"}, {ID: "b"}}}, want: "T2"},
		{name: "active leases", s: snapshot{Nodes: []nodeInfo{{ID: "a", ActiveLeases: 2}, {ID: "b", ActiveLeases: 3}}}, want: "T5"},
		{name: "queued operations", s: snapshot{Nodes: []nodeInfo{{ID: "a", QueuedOps: 2}, {ID: "b", QueuedOps: 3}}}, want: "Tq5"},
		{name: "named node", nodeID: "a", s: snapshot{Nodes: []nodeInfo{{ID: "a", ActiveLeases: 4}}}, want: "T4"},
		{name: "named node queued", nodeID: "a", s: snapshot{Nodes: []nodeInfo{{ID: "a", QueuedOps: 4}}}, want: "Tq4"},
		{name: "missing node", nodeID: "a", s: snapshot{Nodes: []nodeInfo{{ID: "b"}}}, want: "T?"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.s.title(tt.nodeID); got != tt.want {
				t.Fatalf("title() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestSnapshotMenuLines(t *testing.T) {
	s := snapshot{
		Coordinator: "http://127.0.0.1:8080",
		CheckedAt:   time.Date(2026, 4, 29, 12, 0, 0, 0, time.UTC),
		Nodes: []nodeInfo{{
			ID:           "node-a",
			Name:         "mac-studio",
			State:        "ready",
			ActiveLeases: 1,
			QueuedOps:    2,
			MemoryBytes:  64 << 30,
			Temperature:  48.5,
		}},
		Artifacts: []artifactInfo{{Alias: "qwen", Kind: "model", Storage: "hf-cache"}},
	}
	got := strings.Join(s.menuLines(""), "\n")
	for _, want := range []string{
		"Coordinator: http://127.0.0.1:8080",
		"Load: 1 active leases, 2 queued operations",
		"mac-studio ready leases=1 queue=2 mem=64.0GB temp=48.5C",
		"Artifacts: 1",
		"qwen model hf-cache",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("menuLines missing %q in:\n%s", want, got)
		}
	}
}

func TestDashboardURL(t *testing.T) {
	tests := []struct {
		name        string
		coordinator string
		path        string
		want        string
	}{
		{name: "root", coordinator: "http://127.0.0.1:8080", path: "/", want: "http://127.0.0.1:8080/"},
		{name: "trim slash", coordinator: "http://127.0.0.1:8080/", path: "/nodes", want: "http://127.0.0.1:8080/nodes"},
		{name: "add slash", coordinator: "http://127.0.0.1:8080", path: "artifacts", want: "http://127.0.0.1:8080/artifacts"},
		{name: "default", coordinator: "http://127.0.0.1:8080/", want: "http://127.0.0.1:8080/"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := dashboardURL(tt.coordinator, tt.path); got != tt.want {
				t.Fatalf("dashboardURL(%q, %q) = %q, want %q", tt.coordinator, tt.path, got, tt.want)
			}
		})
	}
}
