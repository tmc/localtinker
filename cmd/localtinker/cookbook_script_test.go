package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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

// recipeDatasetFixture maps a recipe to the HuggingFace classic-cache dataset
// directory (under $HF_HOME/datasets) it loads offline. Recipes listed here skip
// when their fixture is absent. recipe_dpo loads datasets.load_dataset(
// "Anthropic/hh-rlhf"); a small fixture is provisioned out of band.
var recipeDatasetFixture = map[string]string{
	"recipe_dpo": "Anthropic___hh-rlhf",
}

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

	recipes, err := filepath.Glob("testdata/recipe_*.txt")
	if err != nil {
		t.Fatal(err)
	}
	if len(recipes) == 0 {
		t.Fatal("no recipe scripts in testdata")
	}

	cmds := scripttest.DefaultCmds()
	delete(cmds, "exec")
	cmds["python"] = script.Program(python, nil, 0)
	engine := &script.Engine{
		Cmds:  cmds,
		Conds: scripttest.DefaultConds(),
	}

	// localtinker serves MLX operations one at a time, so concurrent recipes
	// sharing a server would starve each other until the SDK's future waits
	// time out. Run each recipe sequentially against its own fresh server,
	// matching how a user drives a single recipe at a time.
	for _, recipe := range recipes {
		name := strings.TrimSuffix(filepath.Base(recipe), ".txt")
		t.Run(name, func(t *testing.T) {
			// Recipes that read a pre-provisioned offline dataset skip (not fail)
			// when that fixture is absent, like the cached MLX weights do.
			if fixture, ok := recipeDatasetFixture[name]; ok {
				if _, err := os.Stat(filepath.Join(hfHome(), "datasets", fixture)); err != nil {
					t.Skipf("offline dataset fixture %s not provisioned: %v", fixture, err)
				}
			}

			// scripttest.Test globs a pattern, so stage this one recipe alone
			// in a temp dir and point the glob at it.
			dir := t.TempDir()
			data, err := os.ReadFile(recipe)
			if err != nil {
				t.Fatal(err)
			}
			staged := filepath.Join(dir, filepath.Base(recipe))
			if err := os.WriteFile(staged, data, 0644); err != nil {
				t.Fatal(err)
			}

			addr := freeAddr(t)
			serverLog, stop := startServer(t, bin, addr)
			t.Cleanup(stop)
			waitHealthy(t, "http://"+addr+"/api/v1/healthz", serverLog)

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
				// HOME above would hide the real HuggingFace cache, so point HF
				// at it explicitly (honoring any HF_HOME already set).
				"HF_HOME=" + hfHome(),
			}
			// Pass through an explicit cache override so the offline cache is found.
			if v := os.Getenv("HUGGINGFACE_HUB_CACHE"); v != "" {
				env = append(env, "HUGGINGFACE_HUB_CACHE="+v)
			}

			// MLX recipe runs take minutes; cap each script so a hung recipe
			// fails rather than blocking the suite forever. scripttest runs the
			// script as a t.Parallel subtest, so the timeout must outlive this
			// function: use t.Cleanup, not defer, or it cancels before the
			// recipe runs.
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
			t.Cleanup(cancel)

			scripttest.Test(t, ctx, engine, env, filepath.Join(dir, "recipe_*.txt"))
		})
	}
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
