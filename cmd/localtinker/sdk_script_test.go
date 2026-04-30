package main

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"rsc.io/script"
	"rsc.io/script/scripttest"
)

const defaultSDKPython = "/Users/tmc/.local/homebrew/bin/python3"

func TestPythonSDKScript(t *testing.T) {
	sdk := os.Getenv("TINKER_SDK_DIR")
	if sdk == "" {
		sdk = filepath.Join(os.Getenv("HOME"), "go/src/github.com/thinking-machines-lab/tinker")
	}
	if _, err := os.Stat(filepath.Join(sdk, "src/tinker/__init__.py")); err != nil {
		t.Skipf("local Tinker SDK checkout not available: %v", err)
	}

	python := findPythonSDK(t, sdk)

	bin := filepath.Join(t.TempDir(), "localtinker")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}

	addr := freeAddr(t)
	serverLog, stop := startServer(t, bin, addr)
	t.Cleanup(stop)
	waitHealthy(t, "http://"+addr+"/api/v1/healthz", serverLog)

	cmds := scripttest.DefaultCmds()
	delete(cmds, "exec")
	cmds["python"] = script.Program(python, nil, 0)

	engine := &script.Engine{
		Cmds:  cmds,
		Conds: scripttest.DefaultConds(),
	}
	env := []string{
		"HOME=" + t.TempDir(),
		"PATH=" + os.Getenv("PATH"),
		"PYTHONPATH=" + filepath.Join(sdk, "src"),
		"TINKER_API_KEY=tml-local-test",
		"TINKER_BASE_URL=http://" + addr,
	}

	scripttest.Test(t, context.Background(), engine, env, "testdata/sdk_*.txt")
}

func checkPythonSDK(python, sdk string) error {
	cmd := exec.Command(python, "-c", "import tinker, pydantic, httpx; from tinker import ServiceClient")
	cmd.Env = append(os.Environ(), "PYTHONPATH="+filepath.Join(sdk, "src"))
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%w: %s", err, bytes.TrimSpace(out))
	}
	return nil
}

func findPythonSDK(t *testing.T, sdk string) string {
	t.Helper()

	if python := os.Getenv("LOCALTINKER_SDK_PYTHON"); python != "" {
		if err := checkPythonSDK(python, sdk); err != nil {
			t.Skipf("python SDK dependencies not available with %s: %v", python, err)
		}
		return python
	}

	var lastErr error
	for _, python := range []string{
		filepath.Join(sdk, ".venv/bin/python"),
		defaultSDKPython,
	} {
		if _, err := os.Stat(python); err != nil {
			lastErr = err
			continue
		}
		if err := checkPythonSDK(python, sdk); err != nil {
			lastErr = err
			continue
		}
		return python
	}
	t.Skipf("python SDK dependencies not available; set LOCALTINKER_SDK_PYTHON to a Tinker SDK environment: %v", lastErr)
	return ""
}

func startServer(t *testing.T, bin, addr string) (*bytes.Buffer, func()) {
	t.Helper()

	ctx, cancel := context.WithCancel(context.Background())
	home := filepath.Join(t.TempDir(), "home")
	cmd := exec.CommandContext(ctx, bin, "serve", "-addr", addr, "-home", home)
	var log bytes.Buffer
	cmd.Stdout = &log
	cmd.Stderr = &log
	if err := cmd.Start(); err != nil {
		cancel()
		t.Fatalf("start server: %v", err)
	}

	return &log, func() {
		cancel()
		if err := cmd.Wait(); err != nil && ctx.Err() == nil {
			t.Logf("server exited: %v\n%s", err, log.String())
		}
	}
}

func waitHealthy(t *testing.T, url string, serverLog *bytes.Buffer) {
	t.Helper()

	client := &http.Client{Timeout: time.Second}
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("server did not become healthy\n%s", serverLog.String())
}

func freeAddr(t *testing.T) string {
	t.Helper()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	return ln.Addr().String()
}
