package tinkernodecap

import "testing"

func TestProbe(t *testing.T) {
	c, err := Probe(ProbeOptions{
		Root:           t.TempDir(),
		MaxConcurrency: 2,
		Labels:         map[string]string{"pool": "test"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Backends) == 0 {
		t.Fatal("no backends")
	}
	if c.MaxConcurrency != 2 {
		t.Fatalf("MaxConcurrency = %d, want 2", c.MaxConcurrency)
	}
	if c.Thermal != ThermalUnknown {
		t.Fatalf("Thermal = %q, want unknown", c.Thermal)
	}
	if c.Labels["thermal"] != string(ThermalUnknown) || c.Labels["pool"] != "test" {
		t.Fatalf("Labels = %#v", c.Labels)
	}
}
