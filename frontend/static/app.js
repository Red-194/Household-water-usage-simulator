/**
 * Water Leak Detection — Live Dashboard
 * Pure socket.io client. All data comes from the backend.
 */

const socket = io()

// ─────────────────────────────────────────
// STATE
// ─────────────────────────────────────────

let charts = {}
let simStart = null
let timerInt = null
let lastLeak = false
let currentSpeed = 1
let simMinutes = 0          // tracks latest sim_minutes from server for x-axis

const WINDOW_SIM_MINS = 120 // rolling window: last 120 sim-minutes shown on charts
const MAX_PTS = 600

const bufs = {
  flow:    [],
  cusum:   [],
  ifscore: [],
  normal:  [],
  anomaly: []
}

// ─────────────────────────────────────────
// INIT
// ─────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {

  initBg()
  initCharts()

  setConn(false)
  setLeakState(false)

  ;['predStat','predCNN','predFusion'].forEach(id =>
    setBadge(id, null, id === 'predFusion')
  )

  initSocketListeners()
  updateLeakUI(false)
})


// ─────────────────────────────────────────
// SOCKET
// ─────────────────────────────────────────

function initSocketListeners() {

  socket.on('connect',    () => setConn(true))
  socket.on('disconnect', () => setConn(false))

  socket.on('data_update', handleDataUpdate)

  socket.on('simulation_state', d => handleSimulationState(d.state))

  socket.on('speed_update', d => {
    currentSpeed = d.speed
    document.getElementById('speedVal').textContent = currentSpeed
  })

  socket.on('leak_status', d => {
    updateLeakUI(d.active)
    setLeakState(d.active)
  })
}


// ─────────────────────────────────────────
// SIMULATION CONTROLS
// ─────────────────────────────────────────

function startSimulation()  { socket.emit('start_simulation') }
function pauseSimulation()  { socket.emit('pause_simulation') }
function stopSimulation()   { socket.emit('stop_simulation') }

let speedEmitTimer = null

function onSpeedChange(val) {
  const v = parseFloat(val)
  currentSpeed = v
  document.getElementById('speedVal').textContent = v
  clearTimeout(speedEmitTimer)
  speedEmitTimer = setTimeout(() => socket.emit('set_speed', v), 150)
}

function handleSimulationState(state) {

  const pill     = document.getElementById('simStatusPill')
  const startBtn = document.getElementById('btnStart')
  const pauseBtn = document.getElementById('btnPause')
  const stopBtn  = document.getElementById('btnStop')

  if (state === 'running') {

    startBtn.disabled = true
    pauseBtn.disabled = false
    stopBtn.disabled  = false
    pill.className    = 'spill spill-running'
    pill.textContent  = 'RUNNING'
    startTimer()

  } else if (state === 'paused') {

    pauseBtn.disabled = true
    startBtn.disabled = false
    pill.className    = 'spill spill-paused'
    pill.textContent  = 'PAUSED'
    stopTimer()

  } else if (state === 'stopped') {

    startBtn.disabled = false
    pauseBtn.disabled = true
    stopBtn.disabled  = true
    pill.className    = 'spill spill-idle'
    pill.textContent  = 'IDLE'

    stopTimer()
    simStart    = null
    simMinutes  = 0
    document.getElementById('elapsedTime').textContent = '00:00:00'

    resetCharts()
    updateLeakUI(false)
    setLeakState(false)
  }
}


// ─────────────────────────────────────────
// BACKGROUND
// ─────────────────────────────────────────

function initBg() {

  const cv = document.getElementById('bgCanvas')
  if (!cv) return

  const ctx = cv.getContext('2d')
  let W, H

  function resize() {
    W = cv.width  = window.innerWidth
    H = cv.height = window.innerHeight
  }

  window.addEventListener('resize', resize)
  resize()

  const pts = Array.from({ length: 60 }, () => ({
    x:  Math.random() * W,
    y:  Math.random() * H,
    vx: (Math.random() - 0.5) * 0.2,
    vy: (Math.random() - 0.5) * 0.2,
    r:  Math.random() * 1.2 + 0.3
  }))

  function draw() {
    ctx.clearRect(0, 0, W, H)
    pts.forEach(p => {
      p.x += p.vx; p.y += p.vy
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0
      ctx.beginPath()
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(14,165,233,0.2)'
      ctx.fill()
    })
    requestAnimationFrame(draw)
  }

  draw()
}


// ─────────────────────────────────────────
// CHARTS
// ─────────────────────────────────────────

function simMinLabel(v) {
  const h = Math.floor(v / 60) % 24
  const m = v % 60
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`
}

function mkOpts(yMin = null, yMax = null) {
  return {
    responsive:          true,
    maintainAspectRatio: false,
    animation:           false,
    plugins:             { legend: { display: false } },
    scales: {
      x: {
        type: 'linear',
        ticks: {
          callback:     v => simMinLabel(v),
          maxTicksLimit: 6,
          color:        '#475569'
        },
        grid: { color: 'rgba(255,255,255,0.04)' }
      },
      y: { min: yMin ?? undefined, max: yMax ?? undefined }
    }
  }
}

function initCharts() {

  charts.flow = new Chart(document.getElementById('cFlow'), {
    type: 'line',
    data: { datasets: [{ data: [], borderColor: '#0ea5e9', fill: true, pointRadius: 0 }] },
    options: mkOpts(0, 15)
  })

  charts.recon = new Chart(document.getElementById('cRecon'), {
    type: 'line',
    data: {
      datasets: [
        { data: [], borderColor: '#06b6d4', fill: true,  pointRadius: 0 },
        { data: [], borderColor: '#f59e0b', borderDash: [5, 3], pointRadius: 0 }
      ]
    },
    options: mkOpts()
  })

  charts.stats = new Chart(document.getElementById('cStats'), {
    type: 'line',
    data: {
      datasets: [
        { data: [], borderColor: '#8b5cf6', fill: true,  pointRadius: 0 },
        { data: [], borderColor: '#ef4444', borderDash: [3, 3], pointRadius: 0 }
      ]
    },
    options: mkOpts(0, 1)
  })

  charts.anomaly = new Chart(document.getElementById('cAnomaly'), {
    type: 'scatter',
    data: {
      datasets: [
        { data: [], backgroundColor: '#0ea5e9', pointRadius: 3 },
        { data: [], backgroundColor: '#ef4444', pointRadius: 3 }
      ]
    },
    options: mkOpts(0, 15)
  })
}


// ─────────────────────────────────────────
// DATA HANDLER
// ─────────────────────────────────────────

function handleDataUpdate(d) {

  // x-axis is sim_minutes — speed-independent, always evenly spaced
  simMinutes = d.sim_minutes ?? simMinutes + 1

  const flow        = d.flow ?? 0
  const cusumScore  = d.level2?.score ?? 0
  const ifRaw       = d.level3?.reconstruction_error ?? 0
  const ifScore     = d.level3?.score ?? 0
  const fusionScore = d.final_score ?? 0
  const isLeak      = !!d.anomaly

  if (d.sim_time) {
    const el = document.getElementById('simTimeDisplay')
    if (el) el.textContent = d.sim_time
  }

  if(isLeak){
    handleAlert(d)
  }
  setLeakState(isLeak)
  lastLeak = isLeak

  document.getElementById('valFlow').textContent   = flow.toFixed(2)
  document.getElementById('valStat').textContent   = (cusumScore * 100).toFixed(0)
  document.getElementById('valCnn').textContent    = ifRaw.toFixed(4)
  document.getElementById('valFusion').textContent = (fusionScore * 100).toFixed(0)

  bar('barFlow',   flow / 15 * 100)
  bar('barStat',   cusumScore * 100)
  bar('barCnn',    ifScore * 100)
  bar('barFusion', fusionScore * 100)

  setBadge('predStat',   d.level2?.triggered)
  setBadge('predCNN',    d.level3?.triggered)
  setBadge('predFusion', isLeak, true)

  const x = simMinutes
  push(bufs.flow,    { x, y: flow })
  push(bufs.cusum,   { x, y: cusumScore })
  push(bufs.ifscore, { x, y: ifRaw })
  push(isLeak ? bufs.anomaly : bufs.normal, { x, y: flow })

  updateCharts()

  if (d.leak_active !== undefined) updateLeakUI(d.leak_active)
}


// ─────────────────────────────────────────
// CHART HELPERS
// ─────────────────────────────────────────

function push(buf, pt) {
  buf.push(pt)
  if (buf.length > MAX_PTS) buf.shift()
}

function updateCharts() {

  const xMax  = simMinutes
  const xMin  = Math.max(0, xMax - WINDOW_SIM_MINS)

  function setW(ch) {
    ch.options.scales.x.min = xMin
    ch.options.scales.x.max = xMax
  }

  charts.flow.data.datasets[0].data = bufs.flow
  setW(charts.flow)
  charts.flow.update('none')

  charts.recon.data.datasets[0].data = bufs.ifscore
  charts.recon.data.datasets[1].data = bufs.ifscore.map(p => ({ x: p.x, y: 0 }))
  setW(charts.recon)
  charts.recon.update('none')

  charts.stats.data.datasets[0].data = bufs.cusum
  charts.stats.data.datasets[1].data = bufs.cusum.map(p => ({ x: p.x, y: 1 }))
  setW(charts.stats)
  charts.stats.update('none')

  charts.anomaly.data.datasets[0].data = bufs.normal
  charts.anomaly.data.datasets[1].data = bufs.anomaly
  setW(charts.anomaly)
  charts.anomaly.update('none')
}

function resetCharts() {
  Object.values(bufs).forEach(b => b.length = 0)
  Object.values(charts).forEach(c => {
    c.data.datasets.forEach(ds => ds.data = [])
    c.update()
  })
}


// ─────────────────────────────────────────
// LEAK SLIDER UI
// ─────────────────────────────────────────

function onLeakIntensityChange(v) {
  document.getElementById('leakIntensityVal').textContent = parseFloat(v).toFixed(1)
}

function onLeakDurationChange(v) {
  document.getElementById('leakDurationVal').textContent = parseInt(v)
}

function onLeakModeChange(mode) {
  document.getElementById('rampRow').style.display = mode === 'ramp' ? 'block' : 'none'
}

function onLeakRampChange(v) {
  document.getElementById('leakRampVal').textContent = parseInt(v)
}


// ─────────────────────────────────────────
// LEAK CONTROLS
// ─────────────────────────────────────────

function injectLeak() {
  const intensity    = parseFloat(document.getElementById('leakIntensity').value)
  const duration     = parseInt(document.getElementById('leakDuration').value)
  const mode         = document.querySelector('input[name="leakMode"]:checked').value
  const ramp_minutes = parseInt(document.getElementById('leakRamp').value)
  socket.emit('inject_leak', { intensity, duration, mode, ramp_minutes })
}

function stopLeak() {
  socket.emit('stop_leak')
}

function updateLeakUI(active) {

  const btnInject = document.getElementById('btnInjectLeak')
  const btnStop   = document.getElementById('btnStopLeak')
  const statusEl  = document.getElementById('leakStatusPill')

  if (active) {
    btnInject.disabled   = true
    btnStop.disabled     = false
    statusEl.className   = 'spill spill-leak-active'
    statusEl.textContent = 'LEAK ACTIVE'
  } else {
    btnInject.disabled   = false
    btnStop.disabled     = true
    statusEl.className   = 'spill spill-idle'
    statusEl.textContent = 'INACTIVE'
  }
}


// ─────────────────────────────────────────
// UI HELPERS
// ─────────────────────────────────────────

function bar(id, pct) {
  const el = document.getElementById(id)
  if (el) el.style.width = Math.min(100, Math.max(0, pct)) + '%'
}

function setBadge(id, pred, isEnsemble = false) {

  const el = document.getElementById(id)
  if (!el) return

  const base = 'ai-badge' + (isEnsemble ? ' ai-badge-ensemble' : '')

  if (pred === null || pred === undefined) {
    el.className   = base + ' badge-wait'
    el.textContent = '--'
  } else if (pred) {
    el.className   = base + ' badge-leak'
    el.textContent = 'TRIGGERED'
  } else {
    el.className   = base + ' badge-normal'
    el.textContent = 'NORMAL'
  }
}

function setConn(ok) {
  const el = document.getElementById('connStatus')
  if (!el) return
  el.className = 'conn-status ' + (ok ? 'online' : 'offline')
  el.querySelector('.conn-label').textContent = ok ? 'ONLINE' : 'OFFLINE'
}

function setLeakState(isLeak) {
  const sys  = document.getElementById('systemStatus')
  const text = document.getElementById('systemStatusText')
  if (!sys || !text) return
  sys.className    = 'system-status ' + (isLeak ? 'leak' : 'normal')
  text.textContent = isLeak ? 'LEAK DETECTED' : 'NORMAL'
}


// ─────────────────────────────────────────
// ALERT
// ─────────────────────────────────────────

function handleAlert(d) {
  const box  = document.getElementById('alertBox')
  const meta = document.getElementById('alertMeta')
  meta.textContent =
    `Sim time: ${d.sim_time} | Score: ${(d.final_score * 100).toFixed(1)}% | Flow: ${d.flow.toFixed(2)}`
  box.classList.remove('hidden')
  clearTimeout(box._t)
  box._t = setTimeout(closeAlert, 8000)
}

function closeAlert() {
  document.getElementById('alertBox').classList.add('hidden')
}


// ─────────────────────────────────────────
// TIMER
// ─────────────────────────────────────────

function startTimer() {
  if (timerInt) return
  if (!simStart) simStart = Date.now()
  timerInt = setInterval(() => {
    const e = Date.now() - simStart
    const h = String(Math.floor(e / 3600000)).padStart(2, '0')
    const m = String(Math.floor((e % 3600000) / 60000)).padStart(2, '0')
    const s = String(Math.floor((e % 60000) / 1000)).padStart(2, '0')
    document.getElementById('elapsedTime').textContent = `${h}:${m}:${s}`
  }, 1000)
}

function stopTimer() {
  if (timerInt) { clearInterval(timerInt); timerInt = null }
}