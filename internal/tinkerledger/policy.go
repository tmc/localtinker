package tinkerledger

// Kind selects how work accrues credit. It mirrors the reward vocabulary the
// ecosystem shares so localtinker and its mesh siblings account work the same
// way.
type Kind string

const (
	// None disables the ledger entirely: no entries, no auditors, zero
	// overhead. This is the default — a single-user localtinker is unchanged.
	None Kind = "none"
	// PointsPerWorkUnit accrues Rate credits per verified work unit. The
	// Tier-1 compute-market default: a signed local ledger the coordinator
	// attests, no token and no chain.
	PointsPerWorkUnit Kind = "points_per_work_unit"
	// Escrow hooks future external settlement. Left as a seam in v1; the
	// ledger records credit the same way, settlement is out of scope.
	Escrow Kind = "escrow"
)

// Quorum sizes the auditor set for verify-gated accrual: an entry is credited
// only when At least M of N independently selected auditors pass it.
type Quorum struct {
	M int `json:"m"`
	N int `json:"n"`
}

// Policy configures reward accrual. The zero value is [None] — off — so a
// coordinator that never sets a policy writes no ledger entries.
type Policy struct {
	Kind      Kind    `json:"kind"`
	Rate      int64   `json:"rate"`       // credits per work unit
	SpotCheck float64 `json:"spot_check"` // fraction of cheap ops audited, [0,1]
	Quorum    Quorum  `json:"quorum"`     // auditor m-of-n for expensive ops
	Slash     int64   `json:"slash"`      // stake slashed on a rejected claim
}

// Enabled reports whether the policy accrues any credit.
func (p Policy) Enabled() bool { return p.Kind != None && p.Kind != "" }

// CreditsFor returns the credits a work unit count earns under the policy. A
// disabled policy earns nothing.
func (p Policy) CreditsFor(workUnits int64) int64 {
	if !p.Enabled() {
		return 0
	}
	return workUnits * p.Rate
}
