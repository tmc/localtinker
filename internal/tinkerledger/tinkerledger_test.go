package tinkerledger_test

import (
	"crypto/ed25519"
	"crypto/rand"
	"testing"

	"github.com/tmc/localtinker/internal/tinkerid"
	"github.com/tmc/localtinker/internal/tinkerledger"
)

func mustKey(t *testing.T) ed25519.PrivateKey {
	t.Helper()
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("gen key: %v", err)
	}
	return priv
}

func node(t *testing.T) tinkerid.NodeID {
	t.Helper()
	k, err := tinkerid.LoadOrCreate(t.TempDir())
	if err != nil {
		t.Fatalf("node id: %v", err)
	}
	return k.ID()
}

func TestSignVerifyRoundTrip(t *testing.T) {
	coord := mustKey(t)
	e := tinkerledger.Entry{Seq: 1, NodeID: node(t), FutureID: "f1", WorkUnits: 100, Credits: 100, Verdict: tinkerledger.VerdictPassed, AtUnixNano: 1}
	signed, err := tinkerledger.Sign(e, coord)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	if err := tinkerledger.Verify(signed, coord.Public().(ed25519.PublicKey)); err != nil {
		t.Fatalf("verify: %v", err)
	}

	// Tamper with a credited amount: signature must no longer verify.
	tampered := signed
	tampered.Credits = 1_000_000
	if err := tinkerledger.Verify(tampered, coord.Public().(ed25519.PublicKey)); err == nil {
		t.Fatal("verify accepted a tampered entry")
	}

	// A different coordinator key must not verify.
	other := mustKey(t)
	if err := tinkerledger.Verify(signed, other.Public().(ed25519.PublicKey)); err == nil {
		t.Fatal("verify accepted a wrong-key signature")
	}
}

func TestAccrueAssignsSeqAndBalance(t *testing.T) {
	coord := mustKey(t)
	log, err := tinkerledger.NewLog(coord, nil)
	if err != nil {
		t.Fatalf("new log: %v", err)
	}
	a, b := node(t), node(t)

	e1, err := log.Accrue(a, "f1", "m", "forward_backward", 100, 100, [32]byte{}, tinkerledger.VerdictPassed, 1)
	if err != nil {
		t.Fatalf("accrue 1: %v", err)
	}
	e2, err := log.Accrue(a, "f2", "m", "forward_backward", 50, 50, [32]byte{}, tinkerledger.VerdictPassed, 2)
	if err != nil {
		t.Fatalf("accrue 2: %v", err)
	}
	if e1.Seq != 1 || e2.Seq != 2 {
		t.Fatalf("seqs = %d,%d want 1,2", e1.Seq, e2.Seq)
	}
	if got := log.Balance(a); got != 150 {
		t.Fatalf("balance(a) = %d, want 150", got)
	}
	if got := log.Balance(b); got != 0 {
		t.Fatalf("balance(b) = %d, want 0", got)
	}
}

func TestRejectedVerdictCreditsZero(t *testing.T) {
	coord := mustKey(t)
	log, _ := tinkerledger.NewLog(coord, nil)
	a := node(t)
	e, err := log.Accrue(a, "f1", "m", "forward_backward", 100, 100, [32]byte{}, tinkerledger.VerdictRejected, 1)
	if err != nil {
		t.Fatalf("accrue: %v", err)
	}
	if e.Credits != 0 {
		t.Fatalf("rejected credits = %d, want 0", e.Credits)
	}
	if got := log.Balance(a); got != 0 {
		t.Fatalf("balance after rejection = %d, want 0", got)
	}
	if log.Len() != 1 {
		t.Fatalf("rejection not recorded: len = %d", log.Len())
	}
}

func TestRootIsDeterministicAndTamperEvident(t *testing.T) {
	coord := mustKey(t)
	a := node(t)

	build := func() *tinkerledger.Log {
		l, _ := tinkerledger.NewLog(coord, nil)
		for i := 0; i < 5; i++ {
			if _, err := l.Accrue(a, "f", "m", "forward_backward", 10, 10, [32]byte{}, tinkerledger.VerdictPassed, int64(i+1)); err != nil {
				t.Fatalf("accrue: %v", err)
			}
		}
		return l
	}

	r1 := build().Root()
	r2 := build().Root()
	if r1 != r2 {
		t.Fatal("root not deterministic across identical ledgers")
	}

	// An empty ledger roots to zero; a non-empty one does not.
	empty, _ := tinkerledger.NewLog(coord, nil)
	if empty.Root() != ([32]byte{}) {
		t.Fatal("empty ledger root is not zero")
	}
	if r1 == ([32]byte{}) {
		t.Fatal("non-empty ledger root is zero")
	}

	// A ledger with one extra entry has a different root.
	more := build()
	more.Accrue(a, "f6", "m", "forward_backward", 10, 10, [32]byte{}, tinkerledger.VerdictPassed, 99)
	if more.Root() == r1 {
		t.Fatal("appending an entry did not change the root")
	}
}

func TestNewLogReloadsBalancesAndRejectsBadSig(t *testing.T) {
	coord := mustKey(t)
	a := node(t)
	src, _ := tinkerledger.NewLog(coord, nil)
	var entries []tinkerledger.Entry
	for i := 0; i < 3; i++ {
		e, _ := src.Accrue(a, "f", "m", "optim_step", 1, 5, [32]byte{}, tinkerledger.VerdictPassed, int64(i+1))
		entries = append(entries, e)
	}

	// Reload from the persisted entries: balances recompute, signatures verify.
	reloaded, err := tinkerledger.NewLog(coord, entries)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if got := reloaded.Balance(a); got != 15 {
		t.Fatalf("reloaded balance = %d, want 15", got)
	}
	if reloaded.Root() != src.Root() {
		t.Fatal("reloaded root differs from source")
	}

	// A tampered persisted entry must fail the reload's signature check.
	bad := append([]tinkerledger.Entry(nil), entries...)
	bad[1].Credits = 999
	if _, err := tinkerledger.NewLog(coord, bad); err == nil {
		t.Fatal("reload accepted a tampered entry")
	}

	// An out-of-order entry must fail.
	gap := []tinkerledger.Entry{entries[0], entries[2]}
	if _, err := tinkerledger.NewLog(coord, gap); err == nil {
		t.Fatal("reload accepted a seq gap")
	}
}

func TestPolicy(t *testing.T) {
	tests := []struct {
		name      string
		policy    tinkerledger.Policy
		enabled   bool
		workUnits int64
		credits   int64
	}{
		{"zero value is off", tinkerledger.Policy{}, false, 100, 0},
		{"none is off", tinkerledger.Policy{Kind: tinkerledger.None}, false, 100, 0},
		{"points per work unit", tinkerledger.Policy{Kind: tinkerledger.PointsPerWorkUnit, Rate: 2}, true, 100, 200},
		{"escrow accrues", tinkerledger.Policy{Kind: tinkerledger.Escrow, Rate: 1}, true, 50, 50},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.policy.Enabled(); got != tt.enabled {
				t.Fatalf("Enabled() = %v, want %v", got, tt.enabled)
			}
			if got := tt.policy.CreditsFor(tt.workUnits); got != tt.credits {
				t.Fatalf("CreditsFor(%d) = %d, want %d", tt.workUnits, got, tt.credits)
			}
		})
	}
}

func TestWorkUnitClass(t *testing.T) {
	tests := []struct {
		op        string
		class     tinkerledger.Class
		expensive bool
	}{
		{"forward_backward", tinkerledger.ClassTokens, true},
		{"forward", tinkerledger.ClassTokens, true},
		{"compute_logprobs", tinkerledger.ClassTokens, true},
		{"optim_step", tinkerledger.ClassSteps, false},
		{"save_state", tinkerledger.ClassBytes, false},
		{"sample", tinkerledger.ClassSamples, false},
		{"unknown_op", tinkerledger.ClassNone, false},
	}
	for _, tt := range tests {
		t.Run(tt.op, func(t *testing.T) {
			if got := tinkerledger.ClassFor(tt.op); got != tt.class {
				t.Fatalf("ClassFor(%q) = %q, want %q", tt.op, got, tt.class)
			}
			if got := tinkerledger.Expensive(tt.op); got != tt.expensive {
				t.Fatalf("Expensive(%q) = %v, want %v", tt.op, got, tt.expensive)
			}
		})
	}
}
