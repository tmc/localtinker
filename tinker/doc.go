// Package tinker provides an experimental local API for Tinker-shaped
// training and sampling workflows.
//
// The package is token based. Callers provide tokenized inputs and choose a
// typed loss value such as [CrossEntropy]. Training and sampling execution is
// not implemented yet; methods validate inputs and return [ErrUnsupported]
// where MLX model execution is required.
package tinker
