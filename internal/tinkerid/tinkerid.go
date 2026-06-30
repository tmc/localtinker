// Package tinkerid is a localtinker node's stable ed25519 identity. The public
// key is the node's id on the mesh and the signer of reward-ledger receipts, so
// one key serves both the iroh endpoint and the ledger.
//
// It wraps [irohmesh.NodeKey] so the iroh dependency stays behind a
// localtinker-flavored seam: the rest of localtinker depends on tinkerid, not on
// go-iroh or mlx-go-iroh directly. Identity is opt-in — a single-machine
// localtinker never materializes a key unless mesh or rewards is enabled.
package tinkerid

import (
	"crypto/ed25519"
	"encoding/hex"
	"fmt"
	"path/filepath"

	irohmesh "github.com/tmc/mlx-go-iroh"
)

// KeyFile is the node-key filename under the localtinker home directory.
const KeyFile = "node.key"

// NodeID is a node's ed25519 public key, the stable identity on the mesh. It
// mirrors the fixed-size key id used across the ecosystem so a NodeID compares
// and maps cleanly.
type NodeID [ed25519.PublicKeySize]byte

// String returns the NodeID as lowercase hex.
func (id NodeID) String() string { return hex.EncodeToString(id[:]) }

// Bytes returns a copy of the id's raw bytes.
func (id NodeID) Bytes() []byte { return append([]byte(nil), id[:]...) }

// IsZero reports whether the NodeID is the zero value (no identity).
func (id NodeID) IsZero() bool { return id == NodeID{} }

// Key is a localtinker node's long-lived identity. The zero value is unusable;
// construct with [LoadOrCreate].
type Key struct {
	inner irohmesh.NodeKey
}

// LoadOrCreate loads the node key under home, creating and persisting a fresh
// one if none exists. The key lives at home/[KeyFile]. It is the lazy-identity
// path: a node calls it the first time mesh or rewards is enabled.
func LoadOrCreate(home string) (Key, error) {
	k, err := irohmesh.LoadOrCreate(filepath.Join(home, KeyFile))
	if err != nil {
		return Key{}, fmt.Errorf("load node identity: %w", err)
	}
	return Key{inner: k}, nil
}

// FromEd25519 wraps an existing ed25519 private key as a node identity, for a
// caller that already manages a key.
func FromEd25519(priv ed25519.PrivateKey) (Key, error) {
	k, err := irohmesh.NodeKeyFromEd25519(priv)
	if err != nil {
		return Key{}, fmt.Errorf("node identity from key: %w", err)
	}
	return Key{inner: k}, nil
}

// ID returns the node's id, equal to its ed25519 public key.
func (k Key) ID() NodeID {
	var id NodeID
	copy(id[:], k.inner.Public())
	return id
}

// Public returns the node's ed25519 public key.
func (k Key) Public() ed25519.PublicKey { return k.inner.Public() }

// Sign returns the node's signature of msg.
func (k Key) Sign(msg []byte) []byte { return k.inner.Sign(msg) }

// Mesh returns the underlying mesh node key, for the transport layer that binds
// an iroh endpoint with this identity.
func (k Key) Mesh() irohmesh.NodeKey { return k.inner }

// VerifyNodeID reports whether sig is a valid signature of msg under id.
func VerifyNodeID(id NodeID, msg, sig []byte) bool {
	return irohmesh.Verify(id[:], msg, sig)
}
