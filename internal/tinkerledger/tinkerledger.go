// Package tinkerledger is a signed, append-only, merkle-rooted ledger of
// verified training work. Each [Entry] credits one node for one completed,
// verified Future; the whole accrual history is independently auditable —
// anyone can recompute [Log.Root] and check every entry's signature.
//
// The ledger is the accounting surface for localtinker's reward track. It is
// not a token or a chain: credit is a signed local counter the coordinator
// attests, accruing only on verified work. Rewards default off (Policy kind
// [None]), so a single-user localtinker writes no ledger entries and pays no
// overhead.
package tinkerledger

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/localtinker/internal/tinkerid"
)

// entryDomain separates ledger-entry signatures from any other signature the
// coordinator key produces.
const entryDomain = "localtinker/ledger-entry/v1\x00"

// Verdict records how a claim of work was adjudicated before it was credited.
type Verdict string

const (
	// VerdictPassed means an auditor quorum re-scored the work and it matched
	// within tolerance.
	VerdictPassed Verdict = "passed"
	// VerdictSpotChecked means the work was credited on trust (cheap op, or a
	// run with no auditor quorum available), subject to sampled spot checks.
	VerdictSpotChecked Verdict = "spot_checked"
	// VerdictRejected means an audit failed; the claim earns no credit.
	VerdictRejected Verdict = "rejected"
)

// Entry is a signed, verified unit of credited work, appended in Seq order.
type Entry struct {
	Seq        uint64            `json:"seq"`
	NodeID     tinkerid.NodeID   `json:"node_id"`
	FutureID   string            `json:"future_id"`
	ModelID    string            `json:"model_id,omitempty"`
	Operation  string            `json:"operation,omitempty"`
	WorkUnits  int64             `json:"work_units"`
	Credits    int64             `json:"credits"`
	ProofRoot  [sha256.Size]byte `json:"proof_root"`
	Verdict    Verdict           `json:"verdict"`
	AtUnixNano int64             `json:"at_unix_nano"`
	Signature  []byte            `json:"signature,omitempty"`
}

// payload is the canonical, signed-over byte form of e, excluding the
// signature. It is deterministic and domain-separated so a signature is stable
// and cannot be replayed as a different record.
func (e Entry) payload() []byte {
	var buf bytes.Buffer
	buf.WriteString(entryDomain)
	var u [8]byte
	binary.BigEndian.PutUint64(u[:], e.Seq)
	buf.Write(u[:])
	buf.Write(e.NodeID[:])
	writeField(&buf, []byte(e.FutureID))
	writeField(&buf, []byte(e.ModelID))
	writeField(&buf, []byte(e.Operation))
	binary.BigEndian.PutUint64(u[:], uint64(e.WorkUnits))
	buf.Write(u[:])
	binary.BigEndian.PutUint64(u[:], uint64(e.Credits))
	buf.Write(u[:])
	buf.Write(e.ProofRoot[:])
	writeField(&buf, []byte(e.Verdict))
	binary.BigEndian.PutUint64(u[:], uint64(e.AtUnixNano))
	buf.Write(u[:])
	return buf.Bytes()
}

// writeField writes a length-prefixed field so variable-length strings cannot
// be confused across the canonical encoding.
func writeField(buf *bytes.Buffer, b []byte) {
	var u [8]byte
	binary.BigEndian.PutUint64(u[:], uint64(len(b)))
	buf.Write(u[:])
	buf.Write(b)
}

// Sign returns e signed by the coordinator key. The coordinator is the ledger's
// single trusted attestor in v1.
func Sign(e Entry, coordKey ed25519.PrivateKey) (Entry, error) {
	if len(coordKey) != ed25519.PrivateKeySize {
		return Entry{}, fmt.Errorf("sign ledger entry: bad coordinator key length %d", len(coordKey))
	}
	e.Signature = ed25519.Sign(coordKey, e.payload())
	return e, nil
}

// Verify reports whether e carries a valid signature by coordPub.
func Verify(e Entry, coordPub ed25519.PublicKey) error {
	if len(coordPub) != ed25519.PublicKeySize {
		return fmt.Errorf("verify ledger entry: bad coordinator key length %d", len(coordPub))
	}
	if len(e.Signature) != ed25519.SignatureSize {
		return fmt.Errorf("verify ledger entry %d: signature length %d", e.Seq, len(e.Signature))
	}
	if !ed25519.Verify(coordPub, e.payload(), e.Signature) {
		return fmt.Errorf("verify ledger entry %d: invalid signature", e.Seq)
	}
	return nil
}

// leafHash is the merkle leaf hash of e: the SHA-256 of its signed payload.
func leafHash(e Entry) [sha256.Size]byte {
	return sha256.Sum256(e.payload())
}

// Log is an in-memory append-only ledger. It assigns Seq numbers, signs
// entries, and computes a merkle root over the accrual history. Persistence is
// the caller's concern: load entries with [NewLog] and persist each appended
// entry. The zero value is not usable; construct with [NewLog].
type Log struct {
	mu       sync.Mutex
	coordKey ed25519.PrivateKey
	coordPub ed25519.PublicKey
	entries  []Entry
	balances map[tinkerid.NodeID]int64
}

// NewLog returns a Log over existing entries, signed by coordKey. The entries
// are taken as already-verified (loaded from store); NewLog recomputes balances
// from them. Entries must be in Seq order starting at 1.
func NewLog(coordKey ed25519.PrivateKey, existing []Entry) (*Log, error) {
	if len(coordKey) != ed25519.PrivateKeySize {
		return nil, fmt.Errorf("new ledger: bad coordinator key length %d", len(coordKey))
	}
	l := &Log{
		coordKey: coordKey,
		coordPub: coordKey.Public().(ed25519.PublicKey),
		entries:  make([]Entry, 0, len(existing)),
		balances: make(map[tinkerid.NodeID]int64),
	}
	for _, e := range existing {
		if err := l.appendLocked(e, true); err != nil {
			return nil, err
		}
	}
	return l, nil
}

// Accrue signs and appends an entry crediting nodeID for a verified Future,
// assigning the next Seq and timestamp. It returns the appended, signed entry so
// the caller can persist it. A rejected verdict credits zero but is still
// recorded, so the audit trail keeps the rejection.
func (l *Log) Accrue(nodeID tinkerid.NodeID, futureID, modelID, operation string, workUnits, credits int64, proofRoot [sha256.Size]byte, verdict Verdict, atUnixNano int64) (Entry, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if verdict == VerdictRejected {
		credits = 0
	}
	e := Entry{
		Seq:        uint64(len(l.entries)) + 1,
		NodeID:     nodeID,
		FutureID:   futureID,
		ModelID:    modelID,
		Operation:  operation,
		WorkUnits:  workUnits,
		Credits:    credits,
		ProofRoot:  proofRoot,
		Verdict:    verdict,
		AtUnixNano: atUnixNano,
	}
	signed, err := Sign(e, l.coordKey)
	if err != nil {
		return Entry{}, err
	}
	if err := l.appendLocked(signed, false); err != nil {
		return Entry{}, err
	}
	return signed, nil
}

// appendLocked validates and records e. When verifySig is true it checks the
// signature (the loaded-from-store path); Accrue passes false since it just
// signed. The caller holds l.mu.
func (l *Log) appendLocked(e Entry, verifySig bool) error {
	if want := uint64(len(l.entries)) + 1; e.Seq != want {
		return fmt.Errorf("ledger append: entry seq %d, want %d", e.Seq, want)
	}
	if verifySig {
		if err := Verify(e, l.coordPub); err != nil {
			return err
		}
	}
	l.entries = append(l.entries, e)
	l.balances[e.NodeID] += e.Credits
	return nil
}

// Balance returns the total credits accrued to id.
func (l *Log) Balance(id tinkerid.NodeID) int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.balances[id]
}

// Balances returns a snapshot of every node's credit total.
func (l *Log) Balances() map[tinkerid.NodeID]int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make(map[tinkerid.NodeID]int64, len(l.balances))
	for id, c := range l.balances {
		out[id] = c
	}
	return out
}

// Len returns the number of entries in the ledger.
func (l *Log) Len() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.entries)
}

// Root returns the merkle root over all entries — a single hash that commits to
// the entire accrual history. An external auditor recomputes it from the
// entries and checks it against a published root. The empty ledger roots to the
// zero hash.
func (l *Log) Root() [sha256.Size]byte {
	l.mu.Lock()
	defer l.mu.Unlock()
	return merkleRoot(l.entries)
}

// merkleRoot computes a binary merkle root over the entries' leaf hashes. An odd
// node at a level is promoted unchanged (duplicate-free), a common, simple
// convention; the root commits to leaf order, which Seq fixes.
func merkleRoot(entries []Entry) [sha256.Size]byte {
	if len(entries) == 0 {
		return [sha256.Size]byte{}
	}
	level := make([][sha256.Size]byte, len(entries))
	for i, e := range entries {
		level[i] = leafHash(e)
	}
	for len(level) > 1 {
		next := make([][sha256.Size]byte, 0, (len(level)+1)/2)
		for i := 0; i < len(level); i += 2 {
			if i+1 == len(level) {
				next = append(next, level[i])
				continue
			}
			h := sha256.New()
			h.Write(level[i][:])
			h.Write(level[i+1][:])
			var sum [sha256.Size]byte
			copy(sum[:], h.Sum(nil))
			next = append(next, sum)
		}
		level = next
	}
	return level[0]
}

// ErrEmpty reports an operation that requires at least one entry.
var ErrEmpty = errors.New("tinkerledger: empty")
