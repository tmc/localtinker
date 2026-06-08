package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"rsc.io/script"
	"rsc.io/script/scripttest"
)

// cookbookPython is the persistent venv the orchestrator provisions with
// tinker_cookbook (chat_sl + math_rl), chz, datasets, blobfile, transformers,
// pyqwest, and zstandard. It is tried after the env override and the cookbook's
// own .venv.
const cookbookPython = "/Users/tmc/.cache/localtinker-cookbook-venv/bin/python"

// qwen3MLXModel is the cached MLX checkpoint the server maps Qwen/Qwen3-8B to.
const qwen3MLXModel = "mlx-community/Qwen3-0.6B-bf16"

// TestCookbookRecipeScript runs unmodified tinker-cookbook recipes against a
// booted localtinker under rsc.io/script. It skips (rather than fails) when the
// cookbook checkout, a suitable python, or the cached MLX weights are absent.
func TestCookbookRecipeScript(t *testing.T) {
	sdk := os.Getenv("TINKER_SDK_DIR")
	if sdk == "" {
		sdk = filepath.Join(os.Getenv("HOME"), "go/src/github.com/thinking-machines-lab/tinker")
	}
	if _, err := os.Stat(filepath.Join(sdk, "src/tinker/__init__.py")); err != nil {
		t.Skipf("local Tinker SDK checkout not available: %v", err)
	}

	cookbook := os.Getenv("TINKER_COOKBOOK_DIR")
	if cookbook == "" {
		cookbook = filepath.Join(os.Getenv("HOME"), "go/src/github.com/thinking-machines-lab/tinker-cookbook")
	}
	if _, err := os.Stat(filepath.Join(cookbook, "tinker_cookbook/__init__.py")); err != nil {
		t.Skipf("local tinker-cookbook checkout not available: %v", err)
	}

	python := findPythonCookbook(t, sdk, cookbook)

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
		// Fresh HOME isolates the recipe's /tmp/tinker-examples log dirs.
		"HOME=" + t.TempDir(),
		"PATH=" + os.Getenv("PATH"),
		"PYTHONPATH=" + filepath.Join(sdk, "src") + string(os.PathListSeparator) + cookbook,
		"TINKER_API_KEY=tml-local-test",
		"TINKER_BASE_URL=http://" + addr,
		"LOCALTINKER_QWEN3_8B_MLX_BASE=" + qwen3MLXModel,
		"HF_HUB_OFFLINE=1",
		"TRANSFORMERS_OFFLINE=1",
		// HOME above would hide the real HuggingFace cache, so point HF at it
		// explicitly (honoring any HF_HOME already set) for offline loads.
		"HF_HOME=" + hfHome(),
	}
	// Pass through an explicit cache override so the offline cache is found.
	if v := os.Getenv("HUGGINGFACE_HUB_CACHE"); v != "" {
		env = append(env, "HUGGINGFACE_HUB_CACHE="+v)
	}

	// MLX recipe runs take minutes; cap each script so a hung recipe fails
	// rather than blocking the suite forever. scripttest runs each script as a
	// t.Parallel subtest, so the timeout must outlive this function: use
	// t.Cleanup, not defer, or the context cancels before any recipe runs.
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	t.Cleanup(cancel)

	scripttest.Test(t, ctx, engine, env, "testdata/recipe_*.txt")
}

// checkPythonCookbook reports whether python can import the cookbook stack and
// the cached MLX weights are present. A returned error is suitable for t.Skip.
func checkPythonCookbook(python, sdk, cookbook string) error {
	cmd := exec.Command(python, "-c", "import tinker, tinker_cookbook, chz, datasets, blobfile, transformers, pyqwest")
	cmd.Env = append(os.Environ(),
		"PYTHONPATH="+filepath.Join(sdk, "src")+string(os.PathListSeparator)+cookbook,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("import cookbook modules: %w: %s", err, bytes.TrimSpace(out))
	}

	hub := filepath.Join(hfHome(), "hub", "models--mlx-community--Qwen3-0.6B-bf16")
	if _, err := os.Stat(hub); err != nil {
		return fmt.Errorf("cached MLX weights %s not found at %s: %w", qwen3MLXModel, hub, err)
	}
	return nil
}

// findPythonCookbook resolves the python used to run cookbook recipes, probing
// the env override, the orchestrator venv, then the cookbook's own .venv. It
// skips the test (never fails) when none satisfy checkPythonCookbook.
func findPythonCookbook(t *testing.T, sdk, cookbook string) string {
	t.Helper()

	if python := os.Getenv("LOCALTINKER_COOKBOOK_PYTHON"); python != "" {
		if err := checkPythonCookbook(python, sdk, cookbook); err != nil {
			t.Skipf("cookbook dependencies not available with %s: %v", python, err)
		}
		return python
	}

	var lastErr error
	for _, python := range []string{
		cookbookPython,
		filepath.Join(cookbook, ".venv/bin/python"),
	} {
		if _, err := os.Stat(python); err != nil {
			lastErr = err
			continue
		}
		if err := checkPythonCookbook(python, sdk, cookbook); err != nil {
			lastErr = err
			continue
		}
		return python
	}
	t.Skipf("cookbook dependencies not available; set LOCALTINKER_COOKBOOK_PYTHON to a tinker-cookbook environment: %v", lastErr)
	return ""
}

// hfHome returns the HuggingFace cache root, honoring HF_HOME, else the real
// user cache under the process HOME (not the test's isolated HOME).
func hfHome() string {
	if home := os.Getenv("HF_HOME"); home != "" {
		return home
	}
	return filepath.Join(os.Getenv("HOME"), ".cache", "huggingface")
}
