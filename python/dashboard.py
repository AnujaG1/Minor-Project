from flask import Flask, jsonify, render_template_string
from realtime_pipeline import RealtimePipeline

app      = Flask(__name__)
pipeline = RealtimePipeline()


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


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>5G IBN Attack Mitigation</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@600;800&display=swap');

:root {
  --bg:      #080c14;
  --panel:   #0d1220;
  --panel2:  #111827;
  --border:  #1e2d45;
  --red:     #ff4d4d;
  --green:   #00e676;
  --amber:   #ffab00;
  --blue:    #448aff;
  --teal:    #00bcd4;
  --purple:  #b388ff;
  --muted:   #4a6080;
  --text:    #cdd9e8;
  --mono:    'JetBrains Mono', monospace;
  --display: 'Syne', sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  font-size: 15px;
  line-height: 1.5;
}

/* ── Header ── */
.hdr {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 18px 28px;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
}
.hdr-title {
  font-family: var(--display);
  font-size: 22px;
  font-weight: 800;
  color: #fff;
  letter-spacing: .02em;
}
.hdr-sub {
  font-size: 13px;
  color: var(--muted);
}
.sim-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: var(--muted);
}
.dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 8px var(--green);
  animation: pulse 1.4s infinite;
  flex-shrink: 0;
}
.dot.off { background: var(--muted); box-shadow: none; animation: none; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
.hdr-right { margin-left: auto; display: flex; gap: 14px; align-items: center; }
.clock { font-size: 15px; color: var(--teal); }
.btn {
  padding: 7px 18px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: none;
  color: var(--muted);
  cursor: pointer;
  font-size: 14px;
  font-family: var(--mono);
  transition: all .15s;
}
.btn:hover { color: var(--text); border-color: var(--blue); }

/* ── Metric cards ── */
.metrics {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 14px;
  padding: 20px 28px 0;
}
.metric {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
}
.metric-label {
  font-size: 13px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: 10px;
}
.metric-value {
  font-size: 36px;
  font-weight: 600;
  line-height: 1;
}
.mv-red    { color: var(--red);    }
.mv-green  { color: var(--green);  }
.mv-blue   { color: var(--blue);   }
.mv-amber  { color: var(--amber);  }
.mv-teal   { color: var(--teal);   }

/* ── Main grid ── */
.main {
  display: grid;
  grid-template-columns: 1.8fr 1fr;
  gap: 14px;
  padding: 16px 28px 28px;
}
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 22px;
}
.card-title {
  font-size: 13px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: 16px;
}
.full { grid-column: 1 / -1; }

/* ── Chart ── */
.chart-wrap { position: relative; width: 100%; height: 240px; }
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin-top: 12px;
  font-size: 13px;
  color: var(--muted);
}
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-dot { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }

/* ── Alerts ── */
.alert-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 260px;
  overflow-y: auto;
}
.alert-item {
  padding: 10px 14px;
  border-radius: 8px;
  border-left: 4px solid;
  font-size: 13px;
  line-height: 1.6;
}
.a-crit, .a-high { border-color: var(--red);   background: rgba(255,77,77,.08);  color: #ffaaaa; }
.a-med            { border-color: var(--amber); background: rgba(255,171,0,.08);  color: #ffd47a; }
.a-low            { border-color: var(--teal);  background: rgba(0,188,212,.08);  color: #80deea; }
.no-data { color: var(--muted); font-size: 14px; text-align: center; padding: 24px 0; }

/* ── Action distribution ── */
.action-dist { display: flex; gap: 10px; margin-bottom: 20px; }
.action-badge {
  flex: 1;
  text-align: center;
  padding: 14px 8px;
  border-radius: 10px;
}
.ab-pass   { background: rgba(0,230,118,.08);  border: 1px solid rgba(0,230,118,.25); }
.ab-rate   { background: rgba(255,171,0,.08);  border: 1px solid rgba(255,171,0,.25); }
.ab-block  { background: rgba(255,77,77,.08);  border: 1px solid rgba(255,77,77,.25); }
.ab-val    { font-size: 30px; font-weight: 600; line-height: 1.1; }
.ab-label  { font-size: 12px; color: var(--muted); text-transform: uppercase;
             letter-spacing: .07em; margin-top: 4px; }

/* ── Tables ── */
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th {
  font-size: 12px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .07em;
  padding: 6px 10px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  font-weight: 400;
}
td {
  padding: 9px 10px;
  border-bottom: 1px solid rgba(30,45,69,.6);
  font-size: 14px;
}
tr:last-child td { border-bottom: none; }

/* ── Badges ── */
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 5px;
  font-size: 12px;
  font-weight: 600;
  font-family: var(--mono);
}
.b-block    { background: rgba(255,77,77,.2);    color: #ffaaaa; }
.b-allow    { background: rgba(0,230,118,.2);    color: #69ffb0; }
.b-throttle { background: rgba(255,171,0,.2);    color: #ffd47a; }
.b-monitor  { background: rgba(0,188,212,.2);    color: #80deea; }

.sev-CRITICAL,.sev-HIGH { color: var(--red);   }
.sev-MEDIUM             { color: var(--amber); }
.sev-LOW                { color: var(--muted); }

/* ── Feature bars ── */
.feat-bars { display: flex; flex-direction: column; gap: 8px; }
.feat-row  { display: flex; align-items: center; gap: 10px; }
.feat-name { width: 130px; color: var(--muted); text-align: right;
             flex-shrink: 0; font-size: 13px; }
.feat-bg   { flex: 1; height: 8px; background: #1a2540; border-radius: 4px; }
.feat-bar  { height: 8px; border-radius: 4px; transition: width .35s; }
.feat-val  { width: 50px; text-align: right; color: var(--text); font-size: 13px; }
</style>
</head>
<body>

<!-- Header -->
<div class="hdr">
  <div class="sim-status">
    <div class="dot" id="sim-dot"></div>
    <span id="sim-label">Waiting...</span>
  </div>
  <div>
    <div class="hdr-title">5G Attack Mitigation</div>
    <div class="hdr-sub">DDQN + Intent-Based Networking · Live Dashboard</div>
  </div>
  <div class="hdr-right">
    <span class="clock" id="clock">t = —</span>
    <button class="btn" onclick="resetDash()">Reset</button>
  </div>
</div>

<!-- Metrics -->
<div class="metrics">
  <div class="metric">
    <div class="metric-label">Total Decisions</div>
    <div class="metric-value mv-blue" id="cnt-total">0</div>
  </div>
  <div class="metric">
    <div class="metric-label">Blocked</div>
    <div class="metric-value mv-red" id="cnt-blocked">0</div>
  </div>
  <div class="metric">
    <div class="metric-label">Rate-Limited</div>
    <div class="metric-value mv-amber" id="cnt-rl">0</div>
  </div>
  <div class="metric">
    <div class="metric-label">Allowed</div>
    <div class="metric-value mv-green" id="cnt-allowed">0</div>
  </div>
  <div class="metric">
    <div class="metric-label">Block Rate</div>
    <div class="metric-value mv-teal" id="cnt-rate">—</div>
  </div>
</div>

<!-- Main content -->
<div class="main">

  <!-- Packet rate chart -->
  <div class="card">
    <div class="card-title">Packet Rate per Node (pkt/s)</div>
    <div class="chart-wrap"><canvas id="rateChart"></canvas></div>
    <div class="legend" id="legend"></div>
  </div>

  <!-- Alerts -->
  <div class="card">
    <div class="card-title">Attack Alerts</div>
    <div class="alert-list" id="alerts">
      <div class="no-data">No alerts — waiting for simulation...</div>
    </div>
  </div>

  <!-- Feature breakdown -->
  <div class="card">
    <div class="card-title">Feature Breakdown — Last Flagged Node</div>
    <div class="feat-bars" id="feat-bars">
      <div class="no-data">No attack detected yet</div>
    </div>
  </div>

  <!-- Action distribution + recent decisions -->
  <div class="card">
    <div class="card-title">Agent Action Distribution</div>
    <div class="action-dist">
      <div class="action-badge ab-pass">
        <div class="ab-val mv-green" id="dist-pass">0</div>
        <div class="ab-label">Pass</div>
      </div>
      <div class="action-badge ab-rate">
        <div class="ab-val mv-amber" id="dist-rl">0</div>
        <div class="ab-label">Rate-Limit</div>
      </div>
      <div class="action-badge ab-block">
        <div class="ab-val mv-red" id="dist-block">0</div>
        <div class="ab-label">Block</div>
      </div>
    </div>
    <div class="card-title">Recent Decisions</div>
    <table><thead>
      <tr><th>Time</th><th>Node</th><th>Action</th><th>Sev.</th><th>Conf.</th></tr>
    </thead>
    <tbody id="intent-table"></tbody></table>
  </div>

  <!-- Full intent log -->
  <div class="card full">
    <div class="card-title">Intent Engine Output</div>
    <table><thead>
      <tr><th>Time</th><th>Node</th><th>Action</th><th>Sev.</th>
          <th>Intent</th><th>Policy</th><th>Conf.</th></tr>
    </thead>
    <tbody id="intent-full"></tbody></table>
  </div>

</div>

<script>
const ctx = document.getElementById('rateChart').getContext('2d');
const COLS = ['#ff4d4d','#ffab00','#00e676','#448aff','#b388ff',
              '#00bcd4','#ff6e40','#f06292','#c6ff00','#ea80fc'];
const chart = new Chart(ctx, {
  type: 'line',
  data: { labels: [], datasets: [] },
  options: {
    responsive: true, maintainAspectRatio: false, animation: false,
    scales: {
      x: {
        ticks: { color: '#4a6080', maxTicksLimit: 8, font: { size: 12, family: "'JetBrains Mono'" } },
        grid:  { color: 'rgba(30,45,69,.6)' }
      },
      y: {
        ticks: { color: '#4a6080', font: { size: 12, family: "'JetBrains Mono'" } },
        grid:  { color: 'rgba(30,45,69,.6)' },
        beginAtZero: true
      }
    },
    plugins: { legend: { display: false } }
  }
});
const nodeDS = {}; let ci = 0;

function getDS(node) {
  if (nodeDS[node] !== undefined) return nodeDS[node];
  const col = COLS[ci++ % COLS.length];
  const idx = chart.data.datasets.length;
  chart.data.datasets.push({
    label: node, data: [],
    borderColor: col, backgroundColor: 'transparent',
    pointRadius: 2, tension: .3, borderWidth: 2
  });
  nodeDS[node] = idx;
  document.getElementById('legend').innerHTML =
    chart.data.datasets.map(d =>
      `<span class="legend-item">
         <span class="legend-dot" style="background:${d.borderColor}"></span>
         ${d.label}
       </span>`
    ).join('');
  return idx;
}

const FNAMES = [
  'f1  pkt_rate', 'f2  mean_rate', 'f3  burst_ratio',
  'f4  rate_change', 'f5  rate_trend', 'f6  flow_duration',
  'f7  activity_ratio', 'f8  cell_zscore', 'f9  consecutive', 'f10 peak_rate'
];
const FCOLS = [
  '#448aff','#448aff','#ffab00',
  '#ffab00','#b388ff','#00e676',
  '#00e676','#ff4d4d','#00bcd4','#448aff'
];

function renderFeats(f) {
  if (!f || f.length < 10) return;
  document.getElementById('feat-bars').innerHTML = FNAMES.map((n, i) => {
    const pct = Math.min(100, Math.round(f[i] * 100));
    return `<div class="feat-row">
      <span class="feat-name">${n}</span>
      <div class="feat-bg">
        <div class="feat-bar" style="width:${pct}%;background:${FCOLS[i]}"></div>
      </div>
      <span class="feat-val">${f[i].toFixed(3)}</span>
    </div>`;
  }).join('');
}

let lastAlert = null;

async function refresh() {
  try {
    const d = await fetch('/api/state').then(r => r.json());

    // Sim status
    const running = d.sim_running;
    document.getElementById('sim-dot').className = 'dot' + (running ? '' : ' off');
    document.getElementById('sim-label').textContent = running ? 'LIVE' : 'Idle';

    // Clock
    const intents = d.recent_intents || [];
    if (intents.length)
      document.getElementById('clock').textContent =
        't = ' + intents[intents.length - 1].sim_time.toFixed(1) + 's';

    // Counters
    const c      = d.counters || {};
    const total   = c.total        || 0;
    const blocked = c.blocked      || 0;
    const rateLim = c.rate_limited || 0;
    const allowed = c.allowed      || 0;
    document.getElementById('cnt-total').textContent   = total;
    document.getElementById('cnt-blocked').textContent = blocked;
    document.getElementById('cnt-rl').textContent      = rateLim;
    document.getElementById('cnt-allowed').textContent = allowed;
    document.getElementById('cnt-rate').textContent    =
      total > 0 ? (blocked / total * 100).toFixed(1) + '%' : '—';

    // Action distribution
    document.getElementById('dist-pass').textContent  = allowed;
    document.getElementById('dist-rl').textContent    = rateLim;
    document.getElementById('dist-block').textContent = blocked;

    // Chart
    const allT = new Set();
    for (const ts of Object.values(d.node_timeseries || {}))
      ts.forEach(([t]) => allT.add(t));
    const sorted = [...allT].sort((a, b) => a - b).slice(-60);
    chart.data.labels = sorted.map(t => t.toFixed(1) + 's');
    for (const [node, ts] of Object.entries(d.node_timeseries || {})) {
      const idx = getDS(node);
      const m = Object.fromEntries(ts.map(([t, r]) => [t, r]));
      chart.data.datasets[idx].data = sorted.map(t => m[t] ?? null);
    }
    chart.update('none');

    // Alerts
    const alerts = d.alert_log || [];
    if (alerts.length) {
      document.getElementById('alerts').innerHTML =
        alerts.slice(-8).reverse().map(a => {
          const sev = a.severity || 'LOW';
          const cls = (sev === 'CRITICAL' || sev === 'HIGH') ? 'a-crit'
                    : sev === 'MEDIUM' ? 'a-med' : 'a-low';
          return `<div class="alert-item ${cls}">
            <strong>[${sev}] ${a.action}</strong> &nbsp;@ t=${a.sim_time}s<br>
            ${a.policy}
          </div>`;
        }).join('');
      const last = alerts[alerts.length - 1];
      if (last && last !== lastAlert) { renderFeats(last.features); lastAlert = last; }
    }

    // Recent decisions table
    document.getElementById('intent-table').innerHTML =
      intents.slice(-8).reverse().map(i => {
        const at = (i.action_type || 'allow').toLowerCase();
        return `<tr>
          <td>${i.sim_time}s</td>
          <td>${i.node}</td>
          <td><span class="badge b-${at}">${i.action_type}</span></td>
          <td class="sev-${i.severity}">${i.severity}</td>
          <td>${(i.confidence * 100).toFixed(1)}%</td>
        </tr>`;
      }).join('');

    // Full intent log
    document.getElementById('intent-full').innerHTML =
      intents.slice(-15).reverse().map(i => {
        const at = (i.action_type || 'allow').toLowerCase();
        return `<tr>
          <td>${i.sim_time}s</td>
          <td>${i.node}</td>
          <td><span class="badge b-${at}">${i.action_type}</span></td>
          <td class="sev-${i.severity}">${i.severity}</td>
          <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
            ${i.intent}</td>
          <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;
                     white-space:nowrap;color:var(--muted)">
            ${i.policy}</td>
          <td>${(i.confidence * 100).toFixed(1)}%</td>
        </tr>`;
      }).join('');

  } catch (e) { console.warn(e); }
}

async function resetDash() {
  await fetch('/api/reset', { method: 'POST' });
  chart.data.labels = [];
  chart.data.datasets = [];
  Object.keys(nodeDS).forEach(k => delete nodeDS[k]);
  chart.update(); ci = 0;
  document.getElementById('legend').innerHTML = '';
  document.getElementById('alerts').innerHTML   = '<div class="no-data">Reset...</div>';
  document.getElementById('feat-bars').innerHTML = '<div class="no-data">No attack detected yet</div>';
  document.getElementById('intent-table').innerHTML = '';
  document.getElementById('intent-full').innerHTML  = '';
  ['cnt-total','cnt-blocked','cnt-rl','cnt-allowed']
    .forEach(id => document.getElementById(id).textContent = '0');
  ['dist-pass','dist-rl','dist-block']
    .forEach(id => document.getElementById(id).textContent = '0');
  document.getElementById('cnt-rate').textContent = '—';
  lastAlert = null;
}

setInterval(refresh, 500);
refresh();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    print("[Dashboard] Starting pipeline...")
    pipeline.start()
    print("[Dashboard] Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)