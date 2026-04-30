package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerdb"
	"github.com/tmc/localtinker/internal/tinkerhttp"
	"github.com/tmc/localtinker/internal/tinkerrpc"
	"github.com/tmc/localtinker/internal/tinkerweb"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return usage()
	}
	switch args[0] {
	case "serve":
		return serve(args[1:])
	case "-h", "--help", "help":
		_ = usage()
		return nil
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func serve(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	addr := fs.String("addr", "127.0.0.1:8080", "HTTP listen address")
	home := fs.String("home", defaultHome(), "state directory")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}

	store, err := tinkerdb.OpenJSON(filepath.Join(*home, "tinker.json"))
	if err != nil {
		return err
	}
	defer store.Close()

	coord, err := tinkercoord.New(tinkercoord.Config{Store: store})
	if err != nil {
		return err
	}
	mux := http.NewServeMux()
	rpc, err := tinkerrpc.New(coord)
	if err != nil {
		return err
	}
	rpc.Register(mux)
	mux.Handle("/api/v1/", tinkerhttp.New(coord).Handler())
	mux.Handle("/", tinkerweb.New(coord, rpc).Handler())

	server := &http.Server{
		Addr:              *addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	errc := make(chan error, 1)
	go func() {
		log.Printf("localtinker serving on http://%s", *addr)
		errc <- server.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return server.Shutdown(shutdownCtx)
	case err := <-errc:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	}
}

func usage() error {
	fmt.Fprintf(os.Stderr, "usage: localtinker serve [-addr host:port] [-home dir]\n")
	return flag.ErrHelp
}

func defaultHome() string {
	if home := os.Getenv("LOCALTINKER_HOME"); home != "" {
		return home
	}
	if config, err := os.UserConfigDir(); err == nil {
		return filepath.Join(config, "localtinker")
	}
	return ".localtinker"
}
