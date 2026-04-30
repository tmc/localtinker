package tinkertrain

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestStopTokenSequencesScalarInteger(t *testing.T) {
	tests := []struct {
		name string
		in   any
		want [][]int
	}{
		{name: "int", in: 42, want: [][]int{{42}}},
		{name: "json number", in: json.Number("42"), want: [][]int{{42}}},
		{name: "float64 integer", in: float64(42), want: [][]int{{42}}},
		{name: "float64 fraction", in: 42.5, want: nil},
		{name: "negative", in: -1, want: nil},
		{name: "sequence", in: []int{1, 2}, want: [][]int{{1, 2}}},
		{name: "sequences", in: [][]int{{1}, {2, 3}}, want: [][]int{{1}, {2, 3}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := stopTokenSequences(tt.in); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("stopTokenSequences(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}
