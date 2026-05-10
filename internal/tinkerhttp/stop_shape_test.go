package tinkerhttp

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestValidateStopShapeAccepts pins the SDK-supported stop shapes that
// validateSampleRequest must allow through to the sampler.
func TestValidateStopShapeAccepts(t *testing.T) {
	cases := []struct {
		name string
		v    any
	}{
		{name: "nil", v: nil},
		{name: "string", v: "STOP"},
		{name: "[]string", v: []string{"a", "b"}},
		{name: "[]any of strings", v: []any{"a", "b"}},
		{name: "scalar int", v: 42},
		{name: "scalar int64", v: int64(42)},
		{name: "scalar float64 integer", v: float64(42)},
		{name: "scalar json.Number", v: json.Number("42")},
		{name: "[]int", v: []int{1, 2, 3}},
		{name: "[][]int", v: [][]int{{1}, {2, 3}}},
		{name: "[]any of ints", v: []any{1, 2, 3}},
		{name: "[]any of float64 ints", v: []any{float64(1), float64(2)}},
		{name: "[]any of []any int sequences", v: []any{[]any{1, 2}, []any{3}}},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if err := validateStopShape(tt.v); err != nil {
				t.Fatalf("validateStopShape(%v) = %v, want nil", tt.v, err)
			}
		})
	}
}

// TestValidateStopShapeRejects pins the explicit user-error contract for stop
// shapes outside the SDK contract.
func TestValidateStopShapeRejects(t *testing.T) {
	cases := []struct {
		name    string
		v       any
		wantErr string
	}{
		{name: "object", v: map[string]any{"text": "stop"}, wantErr: "object stop value is not supported"},
		{name: "fractional float scalar", v: 42.5, wantErr: "not an integer"},
		{name: "negative scalar int", v: -1, wantErr: "negative"},
		{name: "negative scalar float", v: -1.0, wantErr: "negative"},
		{name: "fractional json.Number", v: json.Number("1.5"), wantErr: "not an integer"},
		{name: "empty string in []string", v: []string{""}, wantErr: "empty string"},
		{name: "negative in []int", v: []int{1, -2}, wantErr: "negative"},
		{name: "negative in [][]int", v: [][]int{{1}, {-2}}, wantErr: "negative"},
		{name: "mixed string and int in []any", v: []any{"a", 1}, wantErr: "mixes string"},
		{name: "mixed int and string in []any", v: []any{1, "a"}, wantErr: "unsupported scalar stop type string"},
		{name: "fractional float in []any", v: []any{1.5}, wantErr: "not an integer"},
		{name: "negative in []any", v: []any{-1, 2}, wantErr: "negative"},
		{name: "[]any with object", v: []any{map[string]any{"x": 1}}, wantErr: "unsupported"},
		{name: "doubly nested []any", v: []any{[]any{[]any{1}}}, wantErr: "unsupported"},
		{name: "function", v: func() {}, wantErr: "unsupported stop value"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			err := validateStopShape(tt.v)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("validateStopShape(%v) error = %v, want substring %q", tt.v, err, tt.wantErr)
			}
		})
	}
}
