package tinkernodecap

import (
	"os"
	"runtime"
)

// ThermalLabel is a coarse node health bucket for scheduling.
type ThermalLabel string

const (
	ThermalHealthy   ThermalLabel = "healthy"
	ThermalWarm      ThermalLabel = "warm"
	ThermalThrottled ThermalLabel = "throttled"
	ThermalUnknown   ThermalLabel = "unknown"
)

// Backend describes one compute backend visible to a node.
type Backend struct {
	Name       string
	Device     string
	UnifiedMem bool
}

// MemoryInfo describes host memory available for scheduling.
type MemoryInfo struct {
	TotalBytes     uint64
	AvailableBytes uint64
}

// DiskInfo describes the node root filesystem.
type DiskInfo struct {
	RootBytes      uint64
	AvailableBytes uint64
}

// Model describes one locally available model.
type Model struct {
	Name       string
	Path       string
	Tokenizer  string
	MaxContext int
	DType      string
	Quant      string
	CanTrain   bool
	CanSample  bool
}

// Features describes node behavior supported by the local implementation.
type Features struct {
	LoRA               bool
	OptimizerState     bool
	Sampling           bool
	TopKPromptLogprobs bool
	CustomLossArrays   bool
}

// Capabilities is the stable registration payload advertised by a node.
type Capabilities struct {
	Backends       []Backend
	Memory         MemoryInfo
	Disk           DiskInfo
	Models         []Model
	MaxConcurrency int
	Features       Features
	Labels         map[string]string
	Thermal        ThermalLabel
}

// ProbeOptions configures a best-effort capability probe.
type ProbeOptions struct {
	Root           string
	Models         []Model
	MaxConcurrency int
	Features       Features
	Labels         map[string]string
}

// Probe returns best-effort node capabilities using only local stdlib probes.
func Probe(opts ProbeOptions) (Capabilities, error) {
	root := opts.Root
	if root == "" {
		root = "."
	}
	disk, err := probeDisk(root)
	if err != nil {
		return Capabilities{}, err
	}
	c := Capabilities{
		Backends:       defaultBackends(),
		Disk:           disk,
		Models:         append([]Model(nil), opts.Models...),
		MaxConcurrency: opts.MaxConcurrency,
		Features:       opts.Features,
		Labels:         copyLabels(opts.Labels),
		Thermal:        ThermalUnknown,
	}
	if c.MaxConcurrency == 0 {
		c.MaxConcurrency = 1
	}
	mem, ok := probeMemory()
	if ok {
		c.Memory = mem
	}
	if c.Labels == nil {
		c.Labels = make(map[string]string)
	}
	c.Labels["thermal"] = string(c.Thermal)
	return c, nil
}

func defaultBackends() []Backend {
	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm64" || runtime.GOARCH == "amd64") {
		return []Backend{{Name: "metal", Device: "default", UnifiedMem: true}, {Name: "cpu", Device: runtime.GOARCH}}
	}
	return []Backend{{Name: "cpu", Device: runtime.GOARCH}}
}

func copyLabels(labels map[string]string) map[string]string {
	if labels == nil {
		return nil
	}
	out := make(map[string]string, len(labels))
	for k, v := range labels {
		out[k] = v
	}
	return out
}

func rootStatPath(root string) string {
	if root == "" {
		return "."
	}
	if _, err := os.Stat(root); err == nil {
		return root
	}
	return "."
}
