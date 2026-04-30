# localtinker SDK Jobs

These examples run the Thinking Machines Tinker Python SDK against an
localtinker coordinator.

Use the SDK virtualenv from the local Tinker checkout:

```sh
PYTHONPATH=/Users/tmc/go/src/github.com/thinking-machines-lab/tinker/src \
TINKER_BASE_URL=http://127.0.0.1:8080 \
TINKER_API_KEY=tml-local-test \
/Users/tmc/go/src/github.com/thinking-machines-lab/tinker/.venv/bin/python \
  ./cmd/localtinker/examples/tinker_job.py --preset short
```

Short preset:

```sh
./cmd/localtinker/examples/tinker_job.py --preset short
```

Long preset:

```sh
./cmd/localtinker/examples/tinker_job.py --preset long
```

The script prints JSON lines for each optimizer step and a final summary. The
dashboard at `http://127.0.0.1:8080/` will show the run under Run Detail and
Futures.
