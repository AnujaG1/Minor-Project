"""
dashboard.py
Flask dashboard — live visualization of OMNeT++ simulation.

Run:  python3 dashboard.py
Open: http://localhost:5000
"""
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
<title>5G Attack Mitigation</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0a0e1a;--panel:#111827;--border:#1e293b;
  --red:#ef4444;--green:#22c55e;--amber:#f59e0b;--blue:#3b82f6;
  --muted:#64748b;--text:#e2e8f0;--mono:'Courier New',monospace;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:system-ui,sans-serif;font-size:14px;}
.hdr{display:flex;align-items:center;gap:10px;padding:14px 20px;border-bottom:1px solid var(--border);}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 1.4s infinite;}
.dot.off{background:var(--muted);animation:none;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.hdr-title{font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;}
.hdr-right{margin-left:auto;display:flex;gap:12px;align-items:center;}
.hdr-time{font-size:12px;color:var(--muted);font-family:var(--mono);}
.btn{padding:4px 12px;border:1px solid var(--border);border-radius:6px;background:none;color:var(--muted);cursor:pointer;font-size:12px;}
.btn:hover{color:var(--text);border-color:var(--text);}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:14px 20px 0;}
.metric{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px 16px;}
.metric-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px;}
.metric-value{font-size:28px;font-weight:600;font-family:var(--mono);}
.mv-red{color:var(--red);}.mv-green{color:var(--green);}
.mv-blue{color:var(--blue);}.mv-amber{color:var(--amber);}
.main{display:grid;grid-template-columns:1.6fr 1fr;gap:12px;padding:12px 20px;}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 16px;}
.card-title{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:12px;}
.full{grid-column:1/-1;}
.chart-wrap{position:relative;width:100%;height:200px;}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;font-size:11px;color:var(--muted);}
.legend-item{display:flex;align-items:center;gap:5px;}
.legend-dot{width:10px;height:10px;border-radius:2px;flex-shrink:0;}
.alert-list{display:flex;flex-direction:column;gap:6px;max-height:220px;overflow-y:auto;}
.alert-item{padding:8px 10px;border-radius:6px;border-left:3px solid;font-size:11px;line-height:1.5;font-family:var(--mono);}
.a-crit,.a-high{border-color:var(--red);background:rgba(239,68,68,.1);color:#fca5a5;}
.a-med{border-color:var(--amber);background:rgba(245,158,11,.1);color:#fcd34d;}
.no-data{color:var(--muted);font-size:12px;text-align:center;padding:20px 0;}
table{width:100%;border-collapse:collapse;font-size:11px;}
th{font-size:10px;color:var(--muted);text-transform:uppercase;padding:5px 8px;text-align:left;border-bottom:1px solid var(--border);font-weight:400;}
td{padding:6px 8px;border-bottom:1px solid rgba(30,41,59,.5);font-family:var(--mono);}
tr:last-child td{border-bottom:none;}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600;font-family:system-ui;}
.b-block{background:rgba(239,68,68,.2);color:#fca5a5;}
.b-allow{background:rgba(34,197,94,.2);color:#86efac;}
.b-monitor{background:rgba(245,158,11,.2);color:#fcd34d;}
.sev-CRITICAL,.sev-HIGH{color:var(--red);}
.sev-MEDIUM{color:var(--amber);}
.sev-LOW{color:var(--muted);}
.feat-bars{display:flex;flex-direction:column;gap:6px;}
.feat-row{display:flex;align-items:center;gap:8px;font-size:11px;}
.feat-name{width:90px;color:var(--muted);text-align:right;flex-shrink:0;}
.feat-bg{flex:1;height:6px;background:#1e293b;border-radius:3px;}
.feat-bar{height:6px;border-radius:3px;transition:width .3s;}
.feat-val{width:38px;text-align:right;font-family:var(--mono);color:var(--muted);}
</style>
</head>
<body>
<div class="hdr">
  <div class="dot" id="sim-dot"></div>
  <span class="hdr-title">5G Attack Mitigation — Live Dashboard</span>
  <div class="hdr-right">
    <span class="hdr-time" id="clock">waiting for simulation...</span>
    <button class="btn" onclick="resetDash()">Reset</button>
  </div>
</div>
<div class="metrics">
  <div class="metric"><div class="metric-label">Packets processed</div><div class="metric-value mv-blue" id="cnt-total">0</div></div>
  <div class="metric"><div class="metric-label">Attacks blocked</div><div class="metric-value mv-red" id="cnt-attacks">0</div></div>
  <div class="metric"><div class="metric-label">Normal flows</div><div class="metric-value mv-green" id="cnt-normal">0</div></div>
  <div class="metric"><div class="metric-label">Detection rate</div><div class="metric-value mv-amber" id="cnt-rate">—</div></div>
</div>
<div class="main">
  <div class="card">
    <div class="card-title">Packet rate per node (pkt/s)</div>
    <div class="chart-wrap"><canvas id="rateChart"></canvas></div>
    <div class="legend" id="legend"></div>
  </div>
  <div class="card">
    <div class="card-title">Attack alerts</div>
    <div class="alert-list" id="alerts"><div class="no-data">No alerts — waiting for simulation...</div></div>
  </div>
  <div class="card">
    <div class="card-title">Feature breakdown — last flagged node</div>
    <div class="feat-bars" id="feat-bars"><div class="no-data">No attack detected yet</div></div>
  </div>
  <div class="card">
    <div class="card-title">Recent DQN decisions</div>
    <table><thead><tr><th>Time</th><th>Node</th><th>Action</th><th>Severity</th><th>Conf.</th></tr></thead>
    <tbody id="intent-table"></tbody></table>
  </div>
  <div class="card full">
    <div class="card-title">Intent engine output</div>
    <table><thead><tr><th>Time</th><th>Node</th><th>Action</th><th>Severity</th><th>Intent</th><th>Policy</th><th>Conf.</th></tr></thead>
    <tbody id="intent-full"></tbody></table>
  </div>
</div>
<script>
const ctx=document.getElementById('rateChart').getContext('2d');
const COLS=['#ef4444','#f59e0b','#22c55e','#3b82f6','#a855f7','#06b6d4','#f97316','#ec4899'];
const chart=new Chart(ctx,{type:'line',data:{labels:[],datasets:[]},
  options:{responsive:true,maintainAspectRatio:false,animation:false,
    scales:{x:{ticks:{color:'#64748b',maxTicksLimit:8,font:{size:10}},grid:{color:'rgba(30,41,59,.8)'}},
            y:{ticks:{color:'#64748b',font:{size:10}},grid:{color:'rgba(30,41,59,.8)'},beginAtZero:true}},
    plugins:{legend:{display:false}}}});
const nodeDS={};let ci=0;
function getDS(node){
  if(nodeDS[node]!==undefined)return nodeDS[node];
  const idx=chart.data.datasets.length;
  const col=COLS[ci++%COLS.length];
  chart.data.datasets.push({label:node,data:[],borderColor:col,backgroundColor:'transparent',pointRadius:2,tension:.3,borderWidth:1.5});
  nodeDS[node]=idx;
  document.getElementById('legend').innerHTML=chart.data.datasets.map(d=>
    `<span class="legend-item"><span class="legend-dot" style="background:${d.borderColor}"></span>${d.label}</span>`).join('');
  return idx;
}
const FNAMES=['f1 rate','f2 size','f3 interval','f4 burst','f5 zscore'];
const FCOLS=['#3b82f6','#3b82f6','#f59e0b','#ef4444','#ef4444'];
function renderFeats(f){
  if(!f||f.length<5)return;
  document.getElementById('feat-bars').innerHTML=FNAMES.map((n,i)=>{
    const pct=Math.round(f[i]*100);
    return`<div class="feat-row"><span class="feat-name">${n}</span>
      <div class="feat-bg"><div class="feat-bar" style="width:${pct}%;background:${FCOLS[i]}"></div></div>
      <span class="feat-val">${f[i].toFixed(3)}</span></div>`;}).join('');}
let lastAlert=null;
async function refresh(){
  try{
    const d=await fetch('/api/state').then(r=>r.json());
    document.getElementById('sim-dot').className='dot'+(d.sim_running?'':' off');
    const intents=d.recent_intents||[];
    if(intents.length)document.getElementById('clock').textContent=
      't = '+intents[intents.length-1].sim_time.toFixed(2)+'s';
    document.getElementById('cnt-total').textContent=d.counters.total;
    document.getElementById('cnt-attacks').textContent=d.counters.attacks;
    document.getElementById('cnt-normal').textContent=d.counters.normal;
    const tot=d.counters.total;
    document.getElementById('cnt-rate').textContent=
      tot>0?(d.counters.attacks/tot*100).toFixed(1)+'%':'—';
    const allT=new Set();
    for(const ts of Object.values(d.node_timeseries||{}))ts.forEach(([t])=>allT.add(t));
    const sorted=[...allT].sort((a,b)=>a-b).slice(-60);
    chart.data.labels=sorted.map(t=>t.toFixed(2)+'s');
    for(const[node,ts]of Object.entries(d.node_timeseries||{})){
      const idx=getDS(node);const m=Object.fromEntries(ts);
      chart.data.datasets[idx].data=sorted.map(t=>m[t]??null);}
    chart.update('none');
    const alerts=d.alert_log||[];
    if(alerts.length){
      document.getElementById('alerts').innerHTML=alerts.slice(-8).reverse().map(a=>{
        const cls=a.severity==='CRITICAL'||a.severity==='HIGH'?'a-crit':'a-med';
        return`<div class="alert-item ${cls}"><strong>[${a.severity}] ${a.action}</strong> @ t=${a.sim_time}s<br>${a.policy}</div>`;}).join('');
      const last=alerts[alerts.length-1];
      if(last&&last!==lastAlert){renderFeats(last.features);lastAlert=last;}}
    document.getElementById('intent-table').innerHTML=intents.slice(-8).reverse().map(i=>
      `<tr><td>${i.sim_time}s</td><td>${i.node}</td>
       <td><span class="badge b-${i.action_type.toLowerCase()}">${i.action_type}</span></td>
       <td class="sev-${i.severity}">${i.severity}</td>
       <td>${(i.confidence*100).toFixed(1)}%</td></tr>`).join('');
    document.getElementById('intent-full').innerHTML=intents.slice(-15).reverse().map(i=>
      `<tr><td>${i.sim_time}s</td><td>${i.node}</td>
       <td><span class="badge b-${i.action_type.toLowerCase()}">${i.action_type}</span></td>
       <td class="sev-${i.severity}">${i.severity}</td>
       <td style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:system-ui">${i.intent}</td>
       <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted)">${i.policy}</td>
       <td>${(i.confidence*100).toFixed(1)}%</td></tr>`).join('');
  }catch(e){console.warn(e);}
}
async function resetDash(){
  await fetch('/api/reset',{method:'POST'});
  chart.data.labels=[];chart.data.datasets=[];
  Object.keys(nodeDS).forEach(k=>delete nodeDS[k]);
  chart.update();ci=0;
  document.getElementById('legend').innerHTML='';
  document.getElementById('alerts').innerHTML='<div class="no-data">Reset...</div>';
  document.getElementById('feat-bars').innerHTML='<div class="no-data">No attack detected yet</div>';
  document.getElementById('intent-table').innerHTML='';
  document.getElementById('intent-full').innerHTML='';
  ['cnt-total','cnt-attacks','cnt-normal'].forEach(id=>document.getElementById(id).textContent='0');
  document.getElementById('cnt-rate').textContent='—';lastAlert=null;}
setInterval(refresh,500);refresh();
</script>
</body></html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    print("[Dashboard] Starting pipeline...")
    pipeline.start()
    print("[Dashboard] Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)