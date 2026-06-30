package tinkerledger

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/json"
	"fmt"

	"github.com/tmc/localtinker/internal/tinkerid"
)

// EntryStore is the persistence the ledger needs: an append-only log of opaque
// entry bytes. tinkerdb.JSONStore satisfies it. The ledger owns the entry
// schema; the store only persists ordered bytes.
type EntryStore interface {
	AppendLedgerEntry(context.Context, json.RawMessage) error
	ListLedgerEntries(context.Context) ([]json.RawMessage, error)
}

// Ledger is a store-backed [Log]: it loads persisted entries on open and writes
// each accrued entry through to the store. A nil Ledger is a disabled ledger —
// every method is a no-op that credits nothing — so a coordinator with rewards
// off holds a nil *Ledger and never branches on policy at the call site.
type Ledger struct {
	log    *Log
	store  EntryStore
	policy Policy
}

// Open loads the ledger from store and returns a store-backed Ledger signed by
// coordKey under policy. When policy is disabled it returns a nil *Ledger (a
// no-op ledger), so the caller can always hold a *Ledger and call Accrue
// unconditionally.
func Open(ctx context.Context, store EntryStore, coordKey ed25519.PrivateKey, policy Policy) (*Ledger, error) {
	if !policy.Enabled() {
		return nil, nil
	}
	raws, err := store.ListLedgerEntries(ctx)
	if err != nil {
		return nil, fmt.Errorf("open ledger: list entries: %w", err)
	}
	entries := make([]Entry, 0, len(raws))
	for i, raw := range raws {
		var e Entry
		if err := json.Unmarshal(raw, &e); err != nil {
			return nil, fmt.Errorf("open ledger: decode entry %d: %w", i, err)
		}
		entries = append(entries, e)
	}
	log, err := NewLog(coordKey, entries)
	if err != nil {
		return nil, fmt.Errorf("open ledger: %w", err)
	}
	return &Ledger{log: log, store: store, policy: policy}, nil
}

// Enabled reports whether the ledger accrues credit. A nil ledger is disabled.
func (l *Ledger) Enabled() bool { return l != nil }

// Policy returns the ledger's reward policy. A nil ledger reports the zero
// (disabled) policy.
func (l *Ledger) Policy() Policy {
	if l == nil {
		return Policy{}
	}
	return l.policy
}

// Accrue credits nodeID for a verified Future and persists the entry. workUnits
// is the operation's measured work; the policy converts it to credits. proofRoot
// commits to the audit receipts behind the verdict. A nil ledger is a no-op
// returning the zero entry. A rejected verdict records the rejection with zero
// credit.
func (l *Ledger) Accrue(ctx context.Context, nodeID tinkerid.NodeID, futureID, modelID, operation string, workUnits int64, proofRoot [sha256.Size]byte, verdict Verdict, atUnixNano int64) (Entry, error) {
	if l == nil {
		return Entry{}, nil
	}
	credits := l.policy.CreditsFor(workUnits)
	entry, err := l.log.Accrue(nodeID, futureID, modelID, operation, workUnits, credits, proofRoot, verdict, atUnixNano)
	if err != nil {
		return Entry{}, err
	}
	raw, err := json.Marshal(entry)
	if err != nil {
		return Entry{}, fmt.Errorf("accrue: marshal entry: %w", err)
	}
	if err := l.store.AppendLedgerEntry(ctx, raw); err != nil {
		return Entry{}, fmt.Errorf("accrue: persist entry: %w", err)
	}
	return entry, nil
}

// Balance returns id's accrued credits. A nil ledger reports zero.
func (l *Ledger) Balance(id tinkerid.NodeID) int64 {
	if l == nil {
		return 0
	}
	return l.log.Balance(id)
}

// Balances returns every node's credit total. A nil ledger reports an empty map.
func (l *Ledger) Balances() map[tinkerid.NodeID]int64 {
	if l == nil {
		return map[tinkerid.NodeID]int64{}
	}
	return l.log.Balances()
}

// Root returns the ledger's merkle root, the commitment external auditors check.
// A nil ledger reports the zero root.
func (l *Ledger) Root() [sha256.Size]byte {
	if l == nil {
		return [sha256.Size]byte{}
	}
	return l.log.Root()
}

// Len returns the number of ledger entries. A nil ledger reports zero.
func (l *Ledger) Len() int {
	if l == nil {
		return 0
	}
	return l.log.Len()
}
