package tinkerledger_test

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"testing"

	"github.com/tmc/localtinker/internal/tinkerdb"
	"github.com/tmc/localtinker/internal/tinkerid"
	"github.com/tmc/localtinker/internal/tinkerledger"
)

func TestLedgerOpenAccruePersistReload(t *testing.T) {
	ctx := context.Background()
	store := tinkerdb.OpenMemory()
	_, coord, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("gen key: %v", err)
	}
	policy := tinkerledger.Policy{Kind: tinkerledger.PointsPerWorkUnit, Rate: 2}

	led, err := tinkerledger.Open(ctx, store, coord, policy)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if !led.Enabled() {
		t.Fatal("enabled policy yielded a disabled ledger")
	}

	a, err := tinkerid.LoadOrCreate(t.TempDir())
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	if _, err := led.Accrue(ctx, a.ID(), "f1", "m", "forward_backward", 100, [32]byte{}, tinkerledger.VerdictPassed, 1); err != nil {
		t.Fatalf("accrue: %v", err)
	}
	if got := led.Balance(a.ID()); got != 200 { // 100 work units * rate 2
		t.Fatalf("balance = %d, want 200", got)
	}

	// Reopen against the same store: the persisted entry reloads and verifies.
	reopened, err := tinkerledger.Open(ctx, store, coord, policy)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if got := reopened.Balance(a.ID()); got != 200 {
		t.Fatalf("reloaded balance = %d, want 200", got)
	}
	if reopened.Root() != led.Root() {
		t.Fatal("reloaded root differs")
	}
	if reopened.Len() != 1 {
		t.Fatalf("reloaded len = %d, want 1", reopened.Len())
	}
}

func TestDisabledLedgerIsNilNoOp(t *testing.T) {
	ctx := context.Background()
	store := tinkerdb.OpenMemory()
	_, coord, _ := ed25519.GenerateKey(rand.Reader)

	led, err := tinkerledger.Open(ctx, store, coord, tinkerledger.Policy{}) // None
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if led != nil {
		t.Fatal("disabled policy did not yield a nil ledger")
	}

	// Every method must be a safe no-op on the nil ledger.
	a, _ := tinkerid.LoadOrCreate(t.TempDir())
	if led.Enabled() {
		t.Fatal("nil ledger reports enabled")
	}
	if _, err := led.Accrue(ctx, a.ID(), "f1", "m", "forward_backward", 100, [32]byte{}, tinkerledger.VerdictPassed, 1); err != nil {
		t.Fatalf("nil accrue errored: %v", err)
	}
	if led.Balance(a.ID()) != 0 || led.Len() != 0 || led.Root() != ([32]byte{}) {
		t.Fatal("nil ledger is not an empty no-op")
	}
	if len(led.Balances()) != 0 {
		t.Fatal("nil ledger balances not empty")
	}

	// Nothing was persisted.
	raws, err := store.ListLedgerEntries(ctx)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(raws) != 0 {
		t.Fatalf("disabled ledger persisted %d entries", len(raws))
	}
}
