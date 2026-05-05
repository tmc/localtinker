import { h, render } from 'preact';
import { useEffect, useMemo, useState } from 'preact/hooks';

const refreshMs = 2000;

function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [updatedAt, setUpdatedAt] = useState(null);
  const [path, setPath] = useState(window.location.pathname);

  async function load() {
    try {
      const res = await fetch('/api/web/dashboard', { cache: 'no-store' });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setData(await res.json());
      setError('');
      setUpdatedAt(new Date());
    } catch (err) {
      setError(err.message || String(err));
    }
  }

  useEffect(() => {
    load();
    const id = setInterval(load, refreshMs);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  function navigate(next) {
    window.history.pushState(null, '', next);
    setPath(next);
  }

  const totals = useMemo(() => summarize(data), [data]);
  const docsPage = docsRoute(path);
  return h('main', { class: 'shell' },
    h('header', { class: 'topbar' },
      h('div', null,
        h('h1', null, 'localtinker'),
        h('p', null, docsPage ? 'Local Tinker-compatible SDK documentation' : 'Coordinator and node mesh status')
      ),
      h('div', { class: 'topactions' },
        h('nav', { class: 'nav' },
          navButton('Dashboard', '/', path, navigate),
          navButton('Docs', '/docs', path, navigate),
          navButton('Quickstart', '/quickstart', path, navigate),
          navButton('SDK API', '/api', path, navigate)
        ),
        h('div', { class: 'statusline' },
          h('span', { class: error ? 'dot bad' : 'dot good' }),
          h('span', null, error || 'connected'),
          h('button', { onClick: load }, 'Refresh')
        )
      )
    ),
    docsPage
      ? h(Docs, { page: docsPage, data })
      : h(DashboardPage, { data, error, updatedAt, totals })
  );
}

function DashboardPage({ data, error, updatedAt, totals }) {
  return [
    h('section', { class: 'metrics', key: 'metrics' },
      metric('Nodes', totals.nodes),
      metric('Active leases', totals.activeLeases),
      metric('Queued', totals.queued),
      metric('Failures', totals.failures),
      metric('Models', totals.models),
      metric('Futures', totals.futures),
      metric('Artifacts', totals.artifacts),
      metric('Last loss', totals.lastLoss),
      metric('Optimizer step', totals.optimizerStep)
    ),
    error ? h('section', { class: 'notice', key: 'error' }, error) : null,
    data ? h(Dashboard, { data, updatedAt, key: 'dashboard' }) : h('section', { class: 'empty', key: 'loading' }, 'Loading dashboard...')
  ];
}

function navButton(label, href, path, navigate) {
  const current = href === '/' ? !docsRoute(path) : path === href;
  return h('button', {
    class: current ? 'active' : '',
    onClick: () => navigate(href)
  }, label);
}

function docsRoute(path) {
  switch (path) {
    case '/docs':
      return 'overview';
    case '/quickstart':
      return 'quickstart';
    case '/api':
      return 'api';
    default:
      return '';
  }
}

function Docs({ page, data }) {
  const coord = (data && data.coordinator) || {};
  const caps = coord.capabilities || {};
  const models = caps.models || [];
  return h('div', { class: 'docs' },
    h('aside', { class: 'docnav' },
      h('a', { href: '/docs' }, 'Overview'),
      h('a', { href: '/quickstart' }, 'Quickstart'),
      h('a', { href: '/api' }, 'SDK API'),
      h('a', { href: '/runs' }, 'Dashboard')
    ),
    h('article', { class: 'docbody' },
      page === 'quickstart' ? h(QuickstartDocs) :
      page === 'api' ? h(APIDocs, { models }) :
      h(OverviewDocs, { caps, models })
    )
  );
}

function OverviewDocs({ caps, models }) {
  return [
    docTitle('localtinker docs', 'Run the Tinker Python SDK against a local MLX-backed coordinator.'),
    h('section', { class: 'docsection' },
      h('h2', null, 'What it serves'),
      h('p', null, 'localtinker implements the hosted Tinker HTTP SDK surface for local training, checkpointing, sampling, and run inspection. It is intended for offline development and reproducible local experiments, not hosted authorization or fleet scheduling.'),
      h('div', { class: 'cardgrid' },
        featureCard('SDK endpoint', 'Point TINKER_BASE_URL at this server and use ServiceClient, TrainingClient, SamplingClient, and RestClient from the upstream SDK.'),
        featureCard('Local MLX backend', 'LoRA training, dense cross entropy, optimizer state, checkpoints, and sampler sessions execute through the local MLX adapter.'),
        featureCard('Operator dashboard', 'The dashboard shows nodes, queue state, futures, runs, checkpoints, artifacts, recent failures, and live training metrics.')
      )
    ),
    h('section', { class: 'docsection' },
      h('h2', null, 'Capabilities'),
      h('div', { class: 'capgrid' },
        capability('Training', true),
        capability('Sampling', true),
        capability('Prompt logprobs', true),
        capability('Top-k prompt logprobs', true),
        capability('String stops', true),
        capability('Hosted auth parity', false)
      ),
      h('p', null, `Models advertised by this coordinator: ${models.length ? models.map(m => m.model_id).join(', ') : 'not loaded yet'}.`)
    ),
    h('section', { class: 'docsection' },
      h('h2', null, 'Known limits'),
      h('ul', { class: 'checklist' },
        h('li', null, 'Checkpoint archive URLs are local HTTP download URLs. Hosted signed URL authorization is not reproduced.'),
        h('li', null, 'Hosted numeric parity varies because execution is local MLX, local tokenizer, and local model-cache dependent.'),
        h('li', null, 'Multimodal chunks and sparse TensorData are rejected with local user errors.'),
        h('li', null, 'Hosted queue cancellation semantics are approximated locally; check futures for final state.')
      )
    )
  ];
}

function QuickstartDocs() {
  return [
    docTitle('Quickstart', 'Start the coordinator, point the upstream SDK at it, then run a short local job.'),
    h('section', { class: 'docsection' },
      h('h2', null, 'Build and serve'),
      codeBlock(`go build -o /tmp/localtinker ./cmd/localtinker
/tmp/localtinker serve \\
  -addr 127.0.0.1:8080 \\
  -home /tmp/localtinker-home`)
    ),
    h('section', { class: 'docsection' },
      h('h2', null, 'Configure the SDK'),
      codeBlock(`export TINKER_SDK_DIR=$HOME/go/src/github.com/thinking-machines-lab/tinker
export PYTHONPATH=$TINKER_SDK_DIR/src
export TINKER_BASE_URL=http://127.0.0.1:8080
export TINKER_API_KEY=tml-local-test
export LOCALTINKER_SDK_PYTHON=$TINKER_SDK_DIR/.venv/bin/python`)
    ),
    h('section', { class: 'docsection' },
      h('h2', null, 'Run a training smoke'),
      codeBlock(`${'${LOCALTINKER_SDK_PYTHON:-$TINKER_SDK_DIR/.venv/bin/python}'} \\
  ./cmd/localtinker/examples/tinker_job.py --preset short`)
    ),
    h('section', { class: 'docsection' },
      h('h2', null, 'Useful pages'),
      h('table', { class: 'doctable' },
        h('tbody', null,
          docRow('/', 'Live coordinator overview'),
          docRow('/runs', 'Run summaries and recent activity'),
          docRow('/checkpoints', 'Checkpoint paths and archive state'),
          docRow('/nodes', 'Node health, load, and artifact peer labels'),
          docRow('/artifacts', 'Published artifact aliases and root hashes')
        )
      )
    )
  ];
}

function APIDocs({ models }) {
  return [
    docTitle('SDK API', 'The upstream Python SDK surface that localtinker serves locally.'),
    h('section', { class: 'docsection' },
      h('h2', null, 'ServiceClient'),
      h('p', null, 'Create a client normally after setting TINKER_BASE_URL. Server capabilities, LoRA training clients, sampling clients, REST clients, and resume-from-state flows are served locally.'),
      codeBlock(`from tinker import ServiceClient

client = ServiceClient()
training = client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
    rank=8,
)
print(training.get_info().model_id)`)
    ),
    h('section', { class: 'docsection' },
      h('h2', null, 'TrainingClient'),
      h('p', null, 'Supported training operations include forward, forward_backward, custom logprob loss, optim_step, save_weights, save_weights_and_get_sampling_client, load_state, and load_state_with_optimizer.'),
      h('table', { class: 'doctable' },
        h('tbody', null,
          docRow('forward', 'Compute loss without gradients.'),
          docRow('forward_backward', 'Compute loss and gradients for cross_entropy or custom logprob loss.'),
          docRow('optim_step', 'Apply Adam optimizer updates and record optimizer state.'),
          docRow('save_weights_and_get_sampling_client', 'Write a local tinker:// checkpoint and create a sampler session.')
        )
      )
    ),
    h('section', { class: 'docsection' },
      h('h2', null, 'SamplingClient'),
      h('p', null, 'Sampling supports max_tokens, temperature, top_k, top_p, seeds, token stop sequences, string stop sequences, generated-token logprobs, prompt logprobs, and top-k prompt logprobs.'),
      codeBlock(`params = tinker.SamplingParams(
    max_tokens=32,
    temperature=0.7,
    stop=["\\n\\n"],
)
future = sampling.sample(prompt, num_samples=1, sampling_params=params)
sample = future.result().samples[0]`)
    ),
    h('section', { class: 'docsection' },
      h('h2', null, 'REST routes'),
      h('p', null, 'RestClient-compatible routes cover sessions, training runs, checkpoints, archive URLs, publish/unpublish, TTL updates, deletes, and future lookup.'),
      h('p', null, `Current model set: ${models.length ? models.map(m => m.model_id).join(', ') : 'no advertised models in the live snapshot'}.`)
    )
  ];
}

function docTitle(title, text) {
  return h('header', { class: 'doctitle' }, h('h1', null, title), h('p', null, text));
}

function featureCard(title, text) {
  return h('section', { class: 'feature' }, h('h3', null, title), h('p', null, text));
}

function capability(label, ok) {
  return h('section', { class: `cap ${ok ? 'yes' : 'no'}` },
    h('span', null, ok ? 'Yes' : 'No'),
    h('strong', null, label)
  );
}

function codeBlock(text) {
  return h('pre', null, h('code', null, text));
}

function docRow(left, right) {
  return h('tr', { key: left }, h('td', null, h('code', null, left)), h('td', null, right));
}

function Dashboard({ data, updatedAt }) {
  const coord = data.coordinator || {};
  const mesh = data.mesh || {};
  return h('div', { class: 'layout' },
    h('section', { class: 'panel wide' },
      h(SectionHead, { title: 'Nodes', detail: `${(mesh.nodes || []).length} registered` }),
      h(NodesTable, { nodes: mesh.nodes || [] })
    ),
    h('section', { class: 'panel wide' },
      h(SectionHead, { title: 'Queue', detail: queueDetail(coord.queue || {}) }),
      h(QueueTable, { queue: coord.queue || {} })
    ),
    h('section', { class: 'panel wide' },
      h(SectionHead, { title: 'Run Detail', detail: runDetail(coord.futures || []).detail }),
      h(RunDetail, { futures: coord.futures || [], models: coord.models || [] })
    ),
    h('section', { class: 'panel' },
      h(SectionHead, { title: 'Coordinator', detail: mesh.coordinator_id || 'local' }),
      h(KeyValues, { rows: [
        ['Generated', formatTime(data.generated_at)],
        ['Updated', updatedAt ? updatedAt.toLocaleTimeString() : ''],
        ['Max request', formatBytes((coord.client_config || {}).max_request_bytes || 0)],
        ['Parallel FW/BW', String(Boolean((coord.client_config || {}).parallel_fwdbwd_chunks))]
      ] })
    ),
    h('section', { class: 'panel' },
      h(SectionHead, { title: 'Supported Models', detail: `${(((coord.capabilities || {}).models) || []).length}` }),
      h(List, { items: (((coord.capabilities || {}).models) || []).map(model =>
        `${model.model_id} · ctx ${model.context_length}`
      ) })
    ),
    h('section', { class: 'panel wide' },
      h(SectionHead, { title: 'Models', detail: `${(coord.models || []).length} loaded` }),
      h(ModelsTable, { models: coord.models || [] })
    ),
    h('section', { class: 'panel' },
      h(SectionHead, { title: 'Sessions', detail: `${(coord.sessions || []).length}` }),
      h(List, { items: (coord.sessions || []).map(s =>
        `${shortID(s.session_id)} · heartbeat ${s.heartbeat_n}`
      ) })
    ),
    h('section', { class: 'panel wide' },
      h(SectionHead, { title: 'Artifacts', detail: `${(mesh.artifacts || []).length}` }),
      h(ArtifactsTable, { artifacts: mesh.artifacts || [] })
    ),
    h('section', { class: 'panel wide' },
      h(SectionHead, { title: 'Recent Failures', detail: `${failureFutures(coord.futures || []).length}` }),
      h(FailuresTable, { futures: failureFutures(coord.futures || []).slice(0, 8) })
    ),
    h('section', { class: 'panel wide' },
      h(SectionHead, { title: 'Futures', detail: `${(coord.futures || []).length} recent` }),
      h(FuturesTable, { futures: (coord.futures || []).slice(0, 12) })
    )
  );
}

function NodesTable({ nodes }) {
  if (!nodes.length) return h(Empty, { text: 'No nodes registered yet.' });
  return h('div', { class: 'tablewrap' },
    h('table', null,
      h('thead', null, h('tr', null,
        h('th', null, 'Node'),
        h('th', null, 'State'),
        h('th', null, 'Load'),
        h('th', null, 'Memory'),
        h('th', null, 'Temp'),
        h('th', null, 'Artifacts'),
        h('th', null, 'Last command'),
        h('th', null, 'Last seen'),
        h('th', null, 'Peer')
      )),
      h('tbody', null, nodes.map(node => {
        const load = node.load || {};
        const labels = node.labels || {};
        return h('tr', { key: node.node_id },
          h('td', null, h('strong', null, node.name || node.node_id), h('span', { class: 'muted block' }, node.node_id)),
          h('td', null, h('span', { class: `pill ${node.state === 'healthy' ? 'ok' : ''}` }, node.state || 'unknown')),
          h('td', null, `${load.active_leases || 0} leases / ${load.queued_operations || 0} queued`),
          h('td', null, formatBytes(load.memory_available_bytes || 0)),
          h('td', null, load.temperature_celsius ? `${load.temperature_celsius.toFixed(1)} C` : 'n/a'),
          h('td', null, String(node.artifacts || 0)),
          h('td', null, commandLabel(labels)),
          h('td', null, relativeTime(node.last_seen_at)),
          h('td', null, labels.artifact_peer_url || labels.peer_url || 'n/a')
        );
      }))
    )
  );
}

function QueueTable({ queue }) {
  const rows = [
    ['Queued', queue.queued || 0, queue.queued_bytes || 0],
    ['Running', queue.running || 0, queue.running_bytes || 0],
    ['Complete', queue.complete || 0, queue.result_bytes || 0],
    ['User error', queue.user_error || 0, 0],
    ['System error', queue.system_error || 0, 0],
    ['Canceled', queue.canceled || 0, 0]
  ];
  return h('div', { class: 'queuegrid' }, rows.map(([label, count, bytes]) =>
    h('section', { class: 'queuebox', key: label },
      h('span', null, label),
      h('strong', null, String(count)),
      h('small', null, bytes ? formatBytes(bytes) : '')
    )
  ));
}

function ModelsTable({ models }) {
  if (!models.length) return h(Empty, { text: 'No active models.' });
  return h('div', { class: 'tablewrap' },
    h('table', null,
      h('thead', null, h('tr', null,
        h('th', null, 'Model'),
        h('th', null, 'Base'),
        h('th', null, 'LoRA'),
        h('th', null, 'Created')
      )),
      h('tbody', null, models.map(model =>
        h('tr', { key: model.id },
          h('td', null, h('strong', null, shortID(model.id)), h('span', { class: 'muted block' }, model.id)),
          h('td', null, model.base_model || 'unknown'),
          h('td', null, model.is_lora ? `rank ${model.lora_rank || 0}` : 'no'),
          h('td', null, relativeTime(model.created_at))
        )
      ))
    )
  );
}

function RunDetail({ futures, models }) {
  if (!futures.length) return h(Empty, { text: 'No run activity yet.' });
  const detail = runDetail(futures);
  return h('div', { class: 'rungrid' },
    h('section', { class: 'runbox' },
      h('span', null, 'Model'),
      h('strong', null, shortID(detail.modelID || (models[0] || {}).id)),
      h('small', null, detail.baseModel || ((models[0] || {}).base_model) || 'unknown base model')
    ),
    h('section', { class: 'runbox' },
      h('span', null, 'Latest loss'),
      h('strong', null, detail.loss == null ? 'n/a' : detail.loss.toFixed(6)),
      h('small', null, detail.lossAt ? relativeTime(detail.lossAt) : 'waiting for forward pass')
    ),
    h('section', { class: 'runbox' },
      h('span', null, 'Optimizer'),
      h('strong', null, detail.optimizerStep == null ? 'n/a' : String(detail.optimizerStep)),
      h('small', null, detail.backend || 'backend not reported')
    ),
    h('section', { class: 'runbox' },
      h('span', null, 'Gradient'),
      h('strong', null, detail.gradNorm == null ? 'n/a' : detail.gradNorm.toFixed(6)),
      h('small', null, detail.maxUpdate == null ? 'max update n/a' : `max update ${detail.maxUpdate.toFixed(6)}`)
    ),
    h('div', { class: 'activity' },
      h('h3', null, 'Recent Activity'),
      h(ActivityTable, { futures: futures.slice(0, 10) })
    )
  );
}

function ActivityTable({ futures }) {
  return h('div', { class: 'tablewrap' },
    h('table', { class: 'activitytable' },
      h('thead', null, h('tr', null,
        h('th', null, 'Operation'),
        h('th', null, 'Model'),
        h('th', null, 'Metrics'),
        h('th', null, 'When')
      )),
      h('tbody', null, futures.map(future =>
        h('tr', { key: future.id },
          h('td', null, h('span', { class: `op ${future.operation || 'unknown'}` }, operationLabel(future.operation))),
          h('td', null, shortID(future.model_id)),
          h('td', null, metricSummary(future.metrics || {}, future.error)),
          h('td', null, relativeTime(future.created_at))
        )
      ))
    )
  );
}

function FuturesTable({ futures }) {
  if (!futures.length) return h(Empty, { text: 'No futures recorded.' });
  return h('div', { class: 'tablewrap' },
    h('table', null,
      h('thead', null, h('tr', null,
        h('th', null, 'Future'),
        h('th', null, 'Operation'),
        h('th', null, 'Model'),
        h('th', null, 'State'),
        h('th', null, 'Metrics'),
        h('th', null, 'Created')
      )),
      h('tbody', null, futures.map(future =>
        h('tr', { key: future.id },
          h('td', null, shortID(future.id)),
          h('td', null, operationLabel(future.operation)),
          h('td', null, shortID(future.model_id)),
          h('td', null, h('span', { class: `pill ${future.state === 'complete' ? 'ok' : future.state}` }, future.state)),
          h('td', null, metricSummary(future.metrics || {}, future.error) || `${formatBytes(future.result_bytes || 0)} result / ${formatBytes(future.error_bytes || 0)} error`),
          h('td', null, relativeTime(future.created_at))
        )
      ))
    )
  );
}

function ArtifactsTable({ artifacts }) {
  if (!artifacts.length) return h(Empty, { text: 'No artifacts published yet.' });
  return h('div', { class: 'tablewrap' },
    h('table', null,
      h('thead', null, h('tr', null,
        h('th', null, 'Alias'),
        h('th', null, 'Kind'),
        h('th', null, 'Storage'),
        h('th', null, 'Root')
      )),
      h('tbody', null, artifacts.map(artifact =>
        h('tr', { key: artifact.root_hash },
          h('td', null, artifact.alias || 'n/a'),
          h('td', null, artifact.kind || 'artifact'),
          h('td', null, artifact.storage || 'unknown'),
          h('td', null, h('code', null, artifact.root_hash || 'n/a'))
        )
      ))
    )
  );
}

function FailuresTable({ futures }) {
  if (!futures.length) return h(Empty, { text: 'No recent failures.' });
  return h('div', { class: 'tablewrap' },
    h('table', null,
      h('thead', null, h('tr', null,
        h('th', null, 'Future'),
        h('th', null, 'Operation'),
        h('th', null, 'State'),
        h('th', null, 'Failure'),
        h('th', null, 'When')
      )),
      h('tbody', null, futures.map(future =>
        h('tr', { key: future.id },
          h('td', null, shortID(future.id)),
          h('td', null, operationLabel(future.operation)),
          h('td', null, h('span', { class: `pill ${future.state}` }, future.state || 'unknown')),
          h('td', null, future.error || `${formatBytes(future.error_bytes || 0)} error payload`),
          h('td', null, relativeTime(future.completed_at || future.created_at))
        )
      ))
    )
  );
}

function SectionHead({ title, detail }) {
  return h('div', { class: 'sectionhead' }, h('h2', null, title), h('span', null, detail));
}

function KeyValues({ rows }) {
  return h('dl', { class: 'kv' }, rows.flatMap(([k, v]) => [
    h('dt', { key: `${k}-k` }, k),
    h('dd', { key: `${k}-v` }, v)
  ]));
}

function List({ items }) {
  if (!items.length) return h(Empty, { text: 'Nothing to show yet.' });
  return h('ul', { class: 'list' }, items.map((item, i) => h('li', { key: `${item}-${i}` }, item)));
}

function Empty({ text }) {
  return h('div', { class: 'empty' }, text);
}

function metric(label, value) {
  return h('section', { class: 'metric' }, h('span', null, label), h('strong', null, value));
}

function summarize(data) {
  const coord = (data && data.coordinator) || {};
  const mesh = (data && data.mesh) || {};
  const nodes = mesh.nodes || [];
  const queue = coord.queue || {};
  const detail = runDetail(coord.futures || []);
  return {
    nodes: nodes.length,
    activeLeases: nodes.reduce((n, node) => n + (((node.load || {}).active_leases) || 0), 0),
    queued: queue.queued || 0,
    failures: (queue.user_error || 0) + (queue.system_error || 0) + (queue.canceled || 0),
    models: (coord.models || []).length,
    futures: (coord.futures || []).length,
    artifacts: (mesh.artifacts || []).length,
    supportedModels: (((coord.capabilities || {}).models) || []).length,
    lastLoss: detail.loss == null ? 'n/a' : detail.loss.toFixed(4),
    optimizerStep: detail.optimizerStep == null ? 'n/a' : detail.optimizerStep
  };
}

function queueDetail(queue) {
  return `${queue.running || 0} running / ${queue.queued || 0} queued`;
}

function failureFutures(futures) {
  return futures.filter(future =>
    future.error || future.error_bytes > 0 || ['user_error', 'system_error', 'canceled'].includes(future.state)
  );
}

function commandLabel(labels) {
  const id = labels.last_command_ack_id || '';
  const kind = labels.last_command_ack_kind || '';
  if (!id && !kind) return 'n/a';
  if (!kind) return shortID(id);
  if (!id) return operationLabel(kind);
  return `${operationLabel(kind)} ${shortID(id)}`;
}

function runDetail(futures) {
  const out = { detail: `${futures.length} events` };
  for (const future of futures) {
    const metrics = future.metrics || {};
    if (!out.modelID && future.model_id) out.modelID = future.model_id;
    if (out.loss == null && metrics['loss:mean'] != null) {
      out.loss = Number(metrics['loss:mean']);
      out.lossAt = future.created_at;
    }
    if (out.optimizerStep == null && metrics['optimizer_step:unique'] != null) {
      out.optimizerStep = Number(metrics['optimizer_step:unique']);
      out.backend = metrics['optimizer_backend:mlx'] === 1 ? 'MLX backend' : 'optimizer backend unknown';
    }
    if (out.gradNorm == null && metrics['grad_norm:mean'] != null) {
      out.gradNorm = Number(metrics['grad_norm:mean']);
    }
    if (out.maxUpdate == null && metrics['max_update:max'] != null) {
      out.maxUpdate = Number(metrics['max_update:max']);
    }
  }
  return out;
}

function operationLabel(value) {
  switch (value) {
    case 'create_model':
      return 'create model';
    case 'unload_model':
      return 'unload model';
    case 'forward':
      return 'forward';
    case 'forward_backward':
      return 'forward + backward';
    case 'optim_step':
      return 'optimizer step';
    default:
      return value || 'unknown';
  }
}

function metricSummary(metrics, error) {
  if (error) return error;
  const parts = [];
  if (metrics['loss:mean'] != null) parts.push(`loss ${formatMetric(metrics['loss:mean'])}`);
  if (metrics['tokens:sum'] != null) parts.push(`${formatMetric(metrics['tokens:sum'])} tokens`);
  if (metrics['examples:sum'] != null) parts.push(`${formatMetric(metrics['examples:sum'])} examples`);
  if (metrics['optimizer_step:unique'] != null) parts.push(`step ${formatMetric(metrics['optimizer_step:unique'])}`);
  if (metrics['grad_norm:mean'] != null) parts.push(`grad ${formatMetric(metrics['grad_norm:mean'])}`);
  if (metrics['max_update:max'] != null) parts.push(`max update ${formatMetric(metrics['max_update:max'])}`);
  if (metrics['optimizer_backend:mlx'] === 1) parts.push('MLX');
  return parts.join(' · ');
}

function formatMetric(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 'n/a';
  if (Math.abs(n) >= 100) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(2);
  return n.toFixed(4);
}

function formatBytes(value) {
  if (!value) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let n = Number(value);
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatTime(value) {
  if (!value) return 'n/a';
  return new Date(value).toLocaleString();
}

function relativeTime(value) {
  if (!value) return 'n/a';
  const delta = Date.now() - new Date(value).getTime();
  if (!Number.isFinite(delta)) return 'n/a';
  if (delta < 5000) return 'now';
  if (delta < 60000) return `${Math.round(delta / 1000)}s ago`;
  if (delta < 3600000) return `${Math.round(delta / 60000)}m ago`;
  return formatTime(value);
}

function shortID(value) {
  if (!value) return 'n/a';
  return value.length <= 18 ? value : `${value.slice(0, 10)}...${value.slice(-4)}`;
}

render(h(App), document.getElementById('app'));
