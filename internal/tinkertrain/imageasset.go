package tinkertrain

import (
	"errors"
	"fmt"
	"sync"
)

// ErrImageAssetStoreNotConfigured is returned by [DefaultImageAssetResolver]
// and by [ResolveImageAssetPointer] when no image asset store has been
// installed via [Manager.SetImageAssetResolver]. Callers can distinguish
// the missing-store boundary from a transient lookup miss with [errors.Is].
var ErrImageAssetStoreNotConfigured = errors.New("image asset store not configured")

// ErrImageAssetNotFound is returned by an [ImageAssetResolver] implementation
// when the named location is unknown to the store.
var ErrImageAssetNotFound = errors.New("image asset not found")

// ImageAssetResolver resolves an image_asset_pointer chunk's location to
// the raw image bytes plus the on-disk format. The returned format must
// be either "png" or "jpeg" so that the bytes can be revalidated through
// [ValidateImageChunk] against the magic-byte contract.
type ImageAssetResolver interface {
	ResolveImageAsset(location string) (data []byte, format string, err error)
}

// DefaultImageAssetResolver always refuses with [ErrImageAssetStoreNotConfigured].
// It is the resolver installed on a fresh [Manager] so that local
// execution refuses image_asset_pointer chunks with a typed boundary
// error rather than a stringly-typed message.
type DefaultImageAssetResolver struct{}

// ResolveImageAsset implements [ImageAssetResolver].
func (DefaultImageAssetResolver) ResolveImageAsset(string) ([]byte, string, error) {
	return nil, "", ErrImageAssetStoreNotConfigured
}

// MapImageAssetResolver is an in-memory [ImageAssetResolver] keyed on
// location. It exists so tests and small embeddings can exercise the
// resolver path without standing up a real store.
type MapImageAssetResolver struct {
	mu      sync.RWMutex
	entries map[string]mapEntry
}

type mapEntry struct {
	data   []byte
	format string
}

// NewMapImageAssetResolver returns an empty [MapImageAssetResolver].
func NewMapImageAssetResolver() *MapImageAssetResolver {
	return &MapImageAssetResolver{entries: make(map[string]mapEntry)}
}

// Set records data at location with the declared format ("png" or "jpeg").
// Set does not validate magic bytes; that happens at resolve time through
// [ValidateImageChunk] so a single source of truth governs both inline
// image chunks and resolved image_asset_pointer chunks.
func (r *MapImageAssetResolver) Set(location, format string, data []byte) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.entries == nil {
		r.entries = make(map[string]mapEntry)
	}
	r.entries[location] = mapEntry{data: data, format: format}
}

// ResolveImageAsset implements [ImageAssetResolver].
func (r *MapImageAssetResolver) ResolveImageAsset(location string) ([]byte, string, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.entries[location]
	if !ok {
		return nil, "", fmt.Errorf("%w: %q", ErrImageAssetNotFound, location)
	}
	return e.data, e.format, nil
}

// SetImageAssetResolver installs r as the resolver for image_asset_pointer
// chunks. Passing nil restores [DefaultImageAssetResolver]. The resolver
// only governs pointer resolution; the MLX executor still refuses any
// multimodal chunk because no vision backend is wired up.
func (m *Manager) SetImageAssetResolver(r ImageAssetResolver) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if r == nil {
		m.imageAssets = DefaultImageAssetResolver{}
		return
	}
	m.imageAssets = r
}

// ImageAssetResolver returns the installed resolver, falling back to
// [DefaultImageAssetResolver] if none has been set. Callers wiring a
// [Manager] into the HTTP runtime via tinkercoord.Config.Train can
// retrieve the live resolver here to drive [ResolveImageAssetPointer]
// out-of-band, since the HTTP handlers refuse multimodal execution
// before the resolver would ever run.
func (m *Manager) ImageAssetResolver() ImageAssetResolver {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.imageAssets == nil {
		return DefaultImageAssetResolver{}
	}
	return m.imageAssets
}

// ResolveImageAssetPointer resolves c through r, then revalidates the
// resolved bytes through [ValidateImageChunk] against c.ExpectedTokens.
// c must be an image_asset_pointer chunk that has already passed
// [ValidateImageChunk]; the returned chunk has Type "image" with the
// resolved Format and Data so callers can route it through the same
// inline-image code path.
func ResolveImageAssetPointer(r ImageAssetResolver, c ModelInputChunk) (ModelInputChunk, error) {
	if c.Type != "image_asset_pointer" {
		return ModelInputChunk{}, fmt.Errorf("ResolveImageAssetPointer: chunk type %q, want image_asset_pointer", c.Type)
	}
	if r == nil {
		r = DefaultImageAssetResolver{}
	}
	data, format, err := r.ResolveImageAsset(c.Location)
	if err != nil {
		return ModelInputChunk{}, fmt.Errorf("resolve %q: %w", c.Location, err)
	}
	resolved := ModelInputChunk{
		Type:           "image",
		Format:         format,
		Data:           data,
		ExpectedTokens: c.ExpectedTokens,
	}
	if err := ValidateImageChunk(resolved); err != nil {
		return ModelInputChunk{}, fmt.Errorf("resolved asset %q: %w", c.Location, err)
	}
	return resolved, nil
}
