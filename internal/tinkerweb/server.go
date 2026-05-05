// Package tinkerweb serves the coordinator dashboard.
package tinkerweb

import (
	"embed"
	"encoding/json"
	"io/fs"
	"net/http"
	"strings"
	"time"

	"github.com/tmc/localtinker/internal/tinkercoord"
	"github.com/tmc/localtinker/internal/tinkerrpc"
)

//go:embed static/*
var assets embed.FS

type Server struct {
	coord *tinkercoord.Coordinator
	rpc   *tinkerrpc.Server
}

type Dashboard struct {
	GeneratedAt time.Time                     `json:"generated_at"`
	Coordinator tinkercoord.DashboardSnapshot `json:"coordinator"`
	Mesh        tinkerrpc.Snapshot            `json:"mesh"`
}

func New(coord *tinkercoord.Coordinator, rpc *tinkerrpc.Server) *Server {
	return &Server{coord: coord, rpc: rpc}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/web/dashboard", s.dashboard)
	static, err := fs.Sub(assets, "static")
	if err != nil {
		panic(err)
	}
	files := http.FileServer(http.FS(static))
	mux.Handle("/", dashboardPages(files))
	return mux
}

func dashboardPages(files http.Handler) http.Handler {
	pages := map[string]bool{
		"/":            true,
		"/docs":        true,
		"/quickstart":  true,
		"/api":         true,
		"/runs":        true,
		"/checkpoints": true,
		"/nodes":       true,
		"/artifacts":   true,
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if pages[r.URL.Path] {
			r2 := r.Clone(r.Context())
			r2.URL.Path = "/"
			files.ServeHTTP(w, r2)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/") {
			http.NotFound(w, r)
			return
		}
		files.ServeHTTP(w, r)
	})
}

func (s *Server) dashboard(w http.ResponseWriter, r *http.Request) {
	if s.coord == nil || s.rpc == nil {
		writeError(w, http.StatusServiceUnavailable, "dashboard is not configured")
		return
	}
	coord, err := s.coord.DashboardSnapshot(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, Dashboard{
		GeneratedAt: time.Now().UTC(),
		Coordinator: coord,
		Mesh:        s.rpc.Snapshot(),
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}
