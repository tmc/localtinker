package tinkerid_test

import (
	"path/filepath"
	"testing"

	"github.com/tmc/localtinker/internal/tinkerid"
)

func TestLoadOrCreateStableIdentity(t *testing.T) {
	home := t.TempDir()

	first, err := tinkerid.LoadOrCreate(home)
	if err != nil {
		t.Fatalf("first load: %v", err)
	}
	if first.ID().IsZero() {
		t.Fatal("new identity has zero id")
	}

	second, err := tinkerid.LoadOrCreate(home)
	if err != nil {
		t.Fatalf("second load: %v", err)
	}
	if first.ID() != second.ID() {
		t.Fatalf("identity not stable across reload: %s != %s", first.ID(), second.ID())
	}

	if _, err := filepath.Abs(filepath.Join(home, tinkerid.KeyFile)); err != nil {
		t.Fatalf("key path: %v", err)
	}
}

func TestSignVerify(t *testing.T) {
	k, err := tinkerid.LoadOrCreate(t.TempDir())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	other, err := tinkerid.LoadOrCreate(t.TempDir())
	if err != nil {
		t.Fatalf("load other: %v", err)
	}

	msg := []byte("ledger receipt payload")
	sig := k.Sign(msg)

	tests := []struct {
		name string
		id   tinkerid.NodeID
		msg  []byte
		sig  []byte
		want bool
	}{
		{"valid", k.ID(), msg, sig, true},
		{"tampered message", k.ID(), []byte("tampered"), sig, false},
		{"wrong signer", other.ID(), msg, sig, false},
		{"empty signature", k.ID(), msg, nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tinkerid.VerifyNodeID(tt.id, tt.msg, tt.sig); got != tt.want {
				t.Fatalf("VerifyNodeID = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIDMatchesPublicKey(t *testing.T) {
	k, err := tinkerid.LoadOrCreate(t.TempDir())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	id := k.ID()
	pub := k.Public()
	if string(id.Bytes()) != string(pub) {
		t.Fatal("id bytes do not equal public key")
	}
	if id.String() == "" {
		t.Fatal("id string is empty")
	}
}
