package main

import (
	"errors"
	"flag"
	"testing"
)

func TestRunHelp(t *testing.T) {
	for _, args := range [][]string{
		{"help"},
		{"--help"},
		{"serve", "--help"},
	} {
		if err := run(args); err != nil {
			t.Fatalf("run(%q) = %v, want nil", args, err)
		}
	}
}

func TestRunUsageErrors(t *testing.T) {
	if err := run(nil); !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("run(nil) = %v, want ErrHelp", err)
	}
	if err := run([]string{"bogus"}); err == nil {
		t.Fatal(`run(["bogus"]) = nil, want error`)
	}
}
