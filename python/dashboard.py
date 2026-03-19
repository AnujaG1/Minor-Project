"""
dashboard.py
============
Flask dashboard — run this to see live simulation output.

Start:
  python3 dashboard.py
  Open: http://localhost:5000

The pipeline starts automatically. When you run an OMNeT++
simulation, all activity appears in real time in the browser.
"""

from flask import Flask, jsonify, render_template_string
from realtime_pipeline import RealtimePipeline

app      = Flask(__name__)
pipeline = RealtimePipeline()


# ─────────────────────────────────────────────────────────────
# REST API — polled by dashboard JS every 500ms
# ─────────────────────────────────────────────────────────────

@app.route("/api/state")
def api_state():
    return jsonify(pipeline.state.snapshot())

@app.route("/api/summary")
def api_summary():
    return jsonify(pipeline.engine.get_attack_summary())

@app.route("/api/reset", methods=["POST"])
def api_reset():
    pipeline.state.reset()
    pipeline.engine.reset()
    return jsonify({"status": "reset"})


# ─────────────────────────────────────────────────────────────
# Dashboard HTML
# ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>5G Attack Mitigation — Live Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:#0a0e1a; --panel:#111827; --border:#1e293b;
  --red:#ef4444; --red-dim:rgba(239,68,68,0.15);
  --green:#22c55e; --green-dim:rgba(34,197,94,0.15);
  --amber:#f59e0b; --amber-dim:rgba(245,158,11,0.15);
  --blue:#3b82f6; --muted:#64748b; --text:#e2e8f0;
  --mono:'Courier New',monospace;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:system-ui,sans-serif;font-size:14px;}

/* Header */
.hdr{display:flex;align-items:center;gap:10px;padding:14px 20px;
     border-bottom:1px solid var(--border);}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);
     animation:pulse 1.4s infinite;}
.dot.off{background:var(--muted);animation:none;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.hdr-title{font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;}
.hdr-right{margin-left:auto;display:flex;gap:12px;align-items:center;}
.hdr-time{font-size:12px;color:var(--muted);font-family:var(--mono);}
.btn-reset{padding:4px 12px;border:1px solid var(--border);border-radius:6px;
           background:none;color:var(--muted);cursor:pointer;font-size:12px;}
.btn-reset:hover{border-color:var(--text);color:var(--text);}

/* Metrics */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:14px 20px 0;}
.metric{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px 16px;}
.metric-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px;}
.metric-value{font-size:28px;font-weight:600;font-family:var(--mono);}
.mv-red{color:var(--red);}
.mv-green{color:var(--green);}
.mv-blue{color:var(--blue);}
.mv-amber{color:var(--amber);}

/* Main grid */
.main{display:grid;grid-template-columns:1.6fr 1fr;gap:12px;padding:12px 20px;}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 16px;}
.card-title{font-size:11px;color:var(--muted);text-transform:uppercase;
            letter-spacing:.07em;margin-bottom:12px;}
.full{grid-column:1/-1;}

/* Chart */
.chart-wrap{position:relative;width:100%;height:200px;}

/* Legend */
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;font-size:11px;color:var(--muted);}
.legend-item{display:flex;align-items:center;gap:5px;}
.legend-dot{width:10px;height:10px;border-radius:2px;flex-shrink:0;}

/* Alerts */
.alert-list{display:flex;flex-direction:column;gap:6px;max-height:220px;overflow-y:auto;}
.alert-item{padding:8px 10px;border-radius:6px;border-left:3px solid;
            font-size:11px;line-height:1.5;font-family:var(--mono);}
.a-crit{border-color:var(--red);background:var(--red-dim);color:#fca5a5;}
.a-high{border-color:var(--red);background:var(--red-dim);color:#fca5a5;}
.a-med {border-color:var(--amber);background:var(--amber-dim);color:#fcd34d;}
.a-low {border-color:var(--muted);background:#1e293b;color:var(--muted);}
.no-alert{color:var(--muted);font-size:12px;text-align:center;padding:20px 0;}

/* Intent table */
table{width:100%;border-collapse:collapse;font-size:11px;}
th{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;
   padding:5px 8px;text-align:left;border-bottom:1px solid var(--border);font-weight:400;}
td{padding:6px 8px;border-bottom:1px solid rgba(30,41,59,.5);font-family:var(--mono);}
tr:last-child td{border-bottom:none;}
tr:hover td{background:rgba(255,255,255,.02);}

.badge{display:inline-block;padding:2px 7px;border-radius:4px;
       font-size:10px;font-weight:600;font-family:system-ui;}
.b-block  {background:rgba(239,68,68,.2);color:#fca5a5;}
.b-allow  {background:rgba(34,197,94,.2);color:#86efac;}
.b-monitor{background:rgba(245,158,11,.2);color:#fcd34d;}
.b-drop   {background:rgba(239,68,68,.35);color:#fca5a5;}

.sev-CRITICAL{color:var(--red);font-weight:600;}
.sev-HIGH    {color:var(--red);}
.sev-MEDIUM  {color:var(--amber);}
.sev-LOW     {color:var(--muted);}

/* Feature bar */
.feat-bars{display:flex;flex-direction:column;gap:6px;}
.feat-row{display:flex;align-items:center;gap:8px;font-size:11px;}
.feat-name{width:80px;color:var(--muted);text-align:right;flex-shrink:0;}
.feat-bar-bg{flex:1;height:6px;background:#1e293b;border-radius:3px;}
.feat-bar{height:6px;border-radius:3px;transition:width .3s;}
.feat-val{width:38px;text-align:right;font-family:var(--mono);color:var(--muted);}
</style>
</head>
<body>

<div class="hdr">
  <div class="dot" id="sim-dot"></div>
  <span class="hdr-title">5G Attack Mitigation — Live Dashboard</span>
  <div class="hdr-right">
    <span class="hdr-time" id="clock">waiting for simulation…</span>
    <button class="btn-reset" onclick="resetDash()">Reset</button>
  </div>
</div>

<div class="metrics">
  <div class="metric">
    <div class="metric-label">Packets processed</div>
    <div class="metric-value mv-blue" id="cnt-total">0</div>
  </div>
  <div class="metric">
    <div class="metric-label">Attacks blocked</div>
    <div class="metric-value mv-red" id="cnt-attacks">0</div>
  </div>
  <div class="metric">
    <div class="metric-label">Normal flows</div>
    <div class="metric-value mv-green" id="cnt-normal">0</div>
  </div>
  <div class="metric">
    <div class="metric-label">Detection rate</div>
    <div class="metric-value mv-amber" id="cnt-rate">—</div>
  </div>
</div>

<div class="main">

  <!-- Packet rate chart -->
  <div class="card">
    <div class="card-title">Packet rate per node (pkt/s)</div>
    <div class="chart-wrap"><canvas id="rateChart"></canvas></div>
    <div class="legend" id="legend"></div>
  </div>

  <!-- Alerts -->
  <div class="card">
    <div class="card-title">Attack alerts</div>
    <div class="alert-list" id="alerts">
      <div class="no-alert">No alerts — waiting for simulation…</div>
    </div>
  </div>

  <!-- Feature breakdown for last attacked node -->
  <div class="card">
    <div class="card-title">Feature breakdown — last flagged node</div>
    <div class="feat-bars" id="feat-bars">
      <div style="color:var(--muted);font-size:12px;text-align:center;padding:20px 0">
        No attack detected yet
      </div>
    </div>
  </div>

  <!-- Intent table -->
  <div class="card">
    <div class="card-title">Recent DQN decisions</div>
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Node</th><th>Action</th>
          <th>Severity</th><th>Conf.</th>
        </tr>
      </thead>
      <tbody id="intent-table"></tbody>
    </table>
  </div>

  <!-- Full intent log -->
  <div class="card full">
    <div class="card-title">Intent engine output</div>
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Node</th><th>Action</th>
          <th>Severity</th><th>Intent</th><th>Policy</th><th>Conf.</th>
        </tr>
      </thead>
      <tbody id="intent-full"></tbody>
    </table>
  </div>

</div>

<script>
// ── Chart setup ──────────────────────────────────────────────
const ctx   = document.getElementById('rateChart').getContext('2d');
const COLS  = ['#ef4444','#f59e0b','#22c55e','#3b82f6','#a855f7','#06b6d4'];
const chart = new Chart(ctx, {
  type: 'line',
  data: { labels: [], datasets: [] },
  options: {
    responsive: true, maintainAspectRatio: false, animation: false,
    scales: {
      x: { ticks:{ color:'#64748b', maxTicksLimit:8, font:{size:10} },
           grid:{ color:'rgba(30,41,59,.8)' } },
      y: { ticks:{ color:'#64748b', font:{size:10} },
           grid:{ color:'rgba(30,41,59,.8)' }, beginAtZero:true }
    },
    plugins: { legend:{ display:false } }
  }
});

const nodeDatasets = {};
let   colorIdx     = 0;

function getDataset(node) {
  if (nodeDatasets[node] !== undefined) return nodeDatasets[node];
  const idx = chart.data.datasets.length;
  const col = COLS[colorIdx++ % COLS.length];
  chart.data.datasets.push({
    label:node, data:[], borderColor:col,
    backgroundColor:'transparent',
    pointRadius:2, tension:0.3, borderWidth:1.5
  });
  nodeDatasets[node] = idx;

  // Rebuild legend
  const lg = document.getElementById('legend');
  lg.innerHTML = chart.data.datasets.map((ds,i) =>
    `<span class="legend-item">
       <span class="legend-dot" style="background:${ds.borderColor}"></span>
       ${ds.label}
     </span>`
  ).join('');
  return idx;
}

// ── Feature bar names ────────────────────────────────────────
const FEAT_NAMES = ['f1 rate','f2 size','f3 interval',
                    'f4 jitter','f5 burst','f6 uniformity','f7 zscore'];
const FEAT_COLS  = ['#3b82f6','#3b82f6','#f59e0b',
                    '#ef4444','#ef4444','#ef4444','#ef4444'];

function renderFeats(features) {
  if (!features || features.length < 7) return;
  document.getElementById('feat-bars').innerHTML =
    FEAT_NAMES.map((n,i) => {
      const v = features[i];
      const pct = Math.round(v * 100);
      return `<div class="feat-row">
        <span class="feat-name">${n}</span>
        <div class="feat-bar-bg">
          <div class="feat-bar" style="width:${pct}%;background:${FEAT_COLS[i]}"></div>
        </div>
        <span class="feat-val">${v.toFixed(3)}</span>
      </div>`;
    }).join('');
}

// ── Poll /api/state every 500ms ──────────────────────────────
let lastAlert = null;

async function refresh() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();

    // Sim status dot
    const dot = document.getElementById('sim-dot');
    dot.className = 'dot' + (d.sim_running ? '' : ' off');

    // Clock
    const intents = d.recent_intents || [];
    if (intents.length > 0) {
      document.getElementById('clock').textContent =
        't = ' + intents[intents.length-1].sim_time.toFixed(2) + 's';
    }

    // Counters
    document.getElementById('cnt-total').textContent   = d.counters.total;
    document.getElementById('cnt-attacks').textContent = d.counters.attacks;
    document.getElementById('cnt-normal').textContent  = d.counters.normal;
    const total = d.counters.total;
    document.getElementById('cnt-rate').textContent =
      total > 0 ? (d.counters.attacks / total * 100).toFixed(1) + '%' : '—';

    // Chart
    const allTimes = new Set();
    for (const ts of Object.values(d.node_timeseries || {}))
      ts.forEach(([t]) => allTimes.add(t));
    const sorted = [...allTimes].sort((a,b)=>a-b).slice(-60);
    chart.data.labels = sorted.map(t => t.toFixed(2)+'s');

    for (const [node, ts] of Object.entries(d.node_timeseries || {})) {
      const dsIdx = getDataset(node);
      const tsMap = Object.fromEntries(ts);
      chart.data.datasets[dsIdx].data = sorted.map(t => tsMap[t] ?? null);
    }
    chart.update('none');

    // Alerts
    const alerts = d.alert_log || [];
    if (alerts.length > 0) {
      document.getElementById('alerts').innerHTML =
        alerts.slice(-8).reverse().map(a => {
          const cls = a.severity === 'CRITICAL' ? 'a-crit' :
                      a.severity === 'HIGH'     ? 'a-high' :
                      a.severity === 'MEDIUM'   ? 'a-med'  : 'a-low';
          return `<div class="alert-item ${cls}">
            <strong>[${a.severity}] ${a.action}</strong> @ t=${a.sim_time}s<br>
            ${a.policy}
          </div>`;
        }).join('');

      // Feature bars for most recent attack
      const last = alerts[alerts.length - 1];
      if (last && last !== lastAlert) {
        renderFeats(last.features);
        lastAlert = last;
      }
    }

    // Recent decisions (right panel)
    document.getElementById('intent-table').innerHTML =
      intents.slice(-8).reverse().map(i =>
        `<tr>
          <td>${i.sim_time}s</td>
          <td>${i.node}</td>
          <td><span class="badge b-${i.action_type.toLowerCase()}">${i.action_type}</span></td>
          <td class="sev-${i.severity}">${i.severity}</td>
          <td>${(i.confidence*100).toFixed(1)}%</td>
        </tr>`
      ).join('');

    // Full intent log (bottom)
    document.getElementById('intent-full').innerHTML =
      intents.slice(-15).reverse().map(i =>
        `<tr>
          <td>${i.sim_time}s</td>
          <td>${i.node}</td>
          <td><span class="badge b-${i.action_type.toLowerCase()}">${i.action_type}</span></td>
          <td class="sev-${i.severity}">${i.severity}</td>
          <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;
                     white-space:nowrap;font-family:system-ui">${i.intent}</td>
          <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;
                     white-space:nowrap;color:var(--muted)">${i.policy}</td>
          <td>${(i.confidence*100).toFixed(1)}%</td>
        </tr>`
      ).join('');

  } catch(e) {
    console.warn('Poll error:', e);
  }
}

async function resetDash() {
  await fetch('/api/reset', {method:'POST'});
  chart.data.labels = [];
  chart.data.datasets = [];
  Object.keys(nodeDatasets).forEach(k => delete nodeDatasets[k]);
  chart.update();
  colorIdx = 0;
  document.getElementById('legend').innerHTML = '';
  document.getElementById('alerts').innerHTML =
    '<div class="no-alert">Reset — waiting for simulation…</div>';
  document.getElementById('feat-bars').innerHTML =
    '<div style="color:var(--muted);font-size:12px;text-align:center;padding:20px 0">No attack detected yet</div>';
  document.getElementById('intent-table').innerHTML = '';
  document.getElementById('intent-full').innerHTML  = '';
  document.getElementById('cnt-total').textContent   = '0';
  document.getElementById('cnt-attacks').textContent = '0';
  document.getElementById('cnt-normal').textContent  = '0';
  document.getElementById('cnt-rate').textContent    = '—';
  lastAlert = null;
}

setInterval(refresh, 500);
refresh();
</script>
</body>
</html>"""


@app.route("/")
def dashboard():
    return render_template_string(HTML)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[Dashboard] Starting pipeline...")
    pipeline.start()
    print("[Dashboard] Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)