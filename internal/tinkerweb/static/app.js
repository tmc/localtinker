import { h, render } from 'preact';
import { useEffect, useMemo, useState } from 'preact/hooks';

const refreshMs = 2000;

function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [updatedAt, setUpdatedAt] = useState(null);

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

  const totals = useMemo(() => summarize(data), [data]);
  return h('main', { class: 'shell' },
    h('header', { class: 'topbar' },
      h('div', null,
        h('h1', null, 'localtinker'),
        h('p', null, 'Coordinator and node mesh status')
      ),
      h('div', { class: 'statusline' },
        h('span', { class: error ? 'dot bad' : 'dot good' }),
        h('span', null, error || 'connected'),
        h('button', { onClick: load }, 'Refresh')
      )
    ),
    h('section', { class: 'metrics' },
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
    error ? h('section', { class: 'notice' }, error) : null,
    data ? h(Dashboard, { data, updatedAt }) : h('section', { class: 'empty' }, 'Loading dashboard...')
  );
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
