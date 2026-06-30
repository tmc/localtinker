package tinkermeshiroh

import (
	"crypto/sha256"

	"github.com/tmc/go-iroh/gossip"
	"github.com/tmc/localtinker/internal/tinkermesh"
)

// Ensure the iroh backend satisfies the transport seam.
var _ tinkermesh.Transport = (*Transport)(nil)

// Topic domain separators bind a derived topic to its purpose and version, so a
// routine topic and a critical topic for the same run never collide and a topic
// from a different protocol version never matches.
const (
	routineDomain  = "localtinker/mesh-routine/v1\x00"
	criticalDomain = "localtinker/mesh-critical/v1\x00"
)

// DeriveTopics derives the routine and critical gossip topics for a run from its
// id, so separate runs use separate topics without coordination. The derivation
// is deterministic: every node computing it from the same run id joins the same
// topics.
func DeriveTopics(runID string) (routine, critical gossip.TopicID) {
	routine = topicID(routineDomain, runID)
	critical = topicID(criticalDomain, runID)
	return routine, critical
}

func topicID(domain, runID string) gossip.TopicID {
	h := sha256.New()
	h.Write([]byte(domain))
	h.Write([]byte(runID))
	var id gossip.TopicID
	copy(id[:], h.Sum(nil))
	return id
}
