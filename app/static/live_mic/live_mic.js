// Live microphone captions (Streamlit components v2, runs in the app page).
// Microphone -> AudioWorklet (ScriptProcessor fallback) -> 16 kHz PCM16, 100 ms frames ->
// WebSocket to the model gateway (/v1/audio/stream/ws) -> partial (grey) and final (black)
// captions, live latency readouts. On stop, the session (events, latencies, the audio as a
// WAV) goes back to Python as the trigger value "session", which saves it as a run.

const RATE = 16000;
const FRAME = 1600; // 100 ms at 16 kHz
const MAX_SECONDS = 600;
const WORKLET = `
class LalCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("lal-capture", LalCapture);`;

function median(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function fmt(x) {
  return x === null || x === undefined ? "—" : `${x.toFixed(2)} s`;
}

function wavBase64(chunks) {
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, RATE, true); v.setUint32(28, RATE * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, n * 2, true);
  let o = 44;
  for (const c of chunks) for (let i = 0; i < c.length; i++, o += 2) v.setInt16(o, c[i], true);
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const L = data.labels;
  const store = (window.__lalLiveMic = window.__lalLiveMic || {});
  const $ = (sel) => parentElement.querySelector(sel);
  const ui = {
    start: $(".lm-start"), stop: $(".lm-stop"), status: $(".lm-status"),
    caption: $(".lm-caption"), first: $(".lm-first"), final: $(".lm-final"),
    secs: $(".lm-secs"), utts: $(".lm-utts"), model: $(".lm-model"),
  };
  $(".lm-start-label").textContent = L.start;
  $(".lm-stop-label").textContent = L.stop;
  $(".lm-first-label").textContent = L.first;
  $(".lm-final-label").textContent = L.final;
  $(".lm-secs-label").textContent = L.seconds;
  $(".lm-utts-label").textContent = L.utterances;
  ui.model.textContent = data.model_label;

  let s = store[data.key];

  function render() {
    const busy = s && (s.phase === "connecting" || s.phase === "recording" ||
                       s.phase === "finishing");
    ui.start.disabled = busy || !data.model;
    ui.stop.disabled = !(s && s.phase === "recording");
    ui.status.className = "lm-status" + (s && s.error ? " lm-error" : "");
    ui.status.textContent = s ? (s.error || L[s.phase] || "") : L.idle;
    if (!s) return;
    const lines = s.lines.map((t) => `<div class="lm-line">${escape(t)}</div>`).join("");
    const live = s.appendOnly ? s.current : s.partial;
    const cls = s.appendOnly ? "lm-line" : "lm-partial";
    ui.caption.innerHTML = lines + (live ? `<div class="${cls}">${escape(live)}</div>` : "");
    ui.caption.scrollTop = ui.caption.scrollHeight;
    const lastF = s.firsts.length ? s.firsts[s.firsts.length - 1] : null;
    const lastZ = s.finals.length ? s.finals[s.finals.length - 1] : null;
    ui.first.textContent = `${fmt(lastF)} · ${L.median} ${fmt(median(s.firsts))}`;
    ui.final.textContent = `${fmt(lastZ)} · ${L.median} ${fmt(median(s.finals))}`;
    ui.secs.textContent = `${(s.sent / RATE).toFixed(1)} s`;
    ui.utts.textContent = String(s.utterances);
  }

  function escape(t) {
    return String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                                                  '"': "&quot;" })[c]);
  }

  function fail(message) {
    if (!s || s.phase === "done") return;
    s.error = message;
    s.phase = "failed";
    teardownAudio();
    try { s.ws && s.ws.close(); } catch (e) { /* already closed */ }
    render();
  }

  function teardownAudio() {
    if (!s) return;
    try { s.stream && s.stream.getTracks().forEach((t) => t.stop()); } catch (e) { /* gone */ }
    try { s.ctx && s.ctx.close(); } catch (e) { /* gone */ }
    s.stream = null;
    s.ctx = null;
    clearInterval(s.timer);
  }

  function sendPending(all) {
    while (s.pending.length >= FRAME || (all && s.pending.length)) {
      const n = Math.min(FRAME, s.pending.length);
      const chunk = Int16Array.from(s.pending.splice(0, n));
      s.recorded.push(chunk);
      s.sent += chunk.length;
      if (s.ws && s.ws.readyState === 1) s.ws.send(chunk.buffer);
    }
  }

  function push(x) {
    // Box-filter decimation to 16 kHz (also a crude low-pass), then 16-bit samples.
    for (let i = 0; i < x.length; i++) {
      s.acc += x[i];
      s.cnt += 1;
      s.phaseIn += 1;
      if (s.phaseIn >= s.ratio) {
        const v = Math.max(-1, Math.min(1, s.acc / s.cnt));
        s.pending.push(Math.round(v * 32767));
        s.acc = 0;
        s.cnt = 0;
        s.phaseIn -= s.ratio;
      }
    }
    sendPending(false);
    if (s.sent / RATE >= MAX_SECONDS) stop();
  }

  async function startAudio() {
    try {
      s.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false,
                 autoGainControl: false },
      });
    } catch (e) {
      const denied = e && (e.name === "NotAllowedError" || e.name === "SecurityError");
      fail(denied ? L.denied : `${L.no_mic} (${e && e.name ? e.name : e})`);
      return;
    }
    s.ctx = new AudioContext();
    s.ratio = s.ctx.sampleRate / RATE;
    const source = s.ctx.createMediaStreamSource(s.stream);
    const mute = s.ctx.createGain();
    mute.gain.value = 0;
    mute.connect(s.ctx.destination);
    if (s.ctx.audioWorklet && window.AudioWorkletNode) {
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      await s.ctx.audioWorklet.addModule(url);
      const node = new AudioWorkletNode(s.ctx, "lal-capture");
      node.port.onmessage = (e) => s && s.phase === "recording" && push(e.data);
      source.connect(node);
      node.connect(mute);
    } else {
      const node = s.ctx.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (e) =>
        s && s.phase === "recording" && push(e.inputBuffer.getChannelData(0).slice(0));
      source.connect(node);
      node.connect(mute);
    }
    s.phase = "recording";
    s.timer = setInterval(render, 250);
    render();
  }

  function onEvent(ev) {
    if (!s) return;
    if (ev.type === "ready") {
      s.appendOnly = !!ev.append_only;
      s.loadSeconds = ev.load_seconds;
      startAudio();
      return;
    }
    if (ev.type === "error") return fail(ev.error || L.closed);
    if (ev.type !== "done") s.events.push(ev);
    const shown = ev.display === undefined ? ev.text : ev.display;
    if (ev.type === "partial") s.partial = shown;
    else if (ev.type === "final") {
      if (s.appendOnly) s.current += shown;
      else if (shown.trim()) s.lines.push(shown);
      s.partial = "";
    } else if (ev.type === "speech_start") s.utterances += 1;
    else if (ev.type === "metric") (ev.kind === "first" ? s.firsts : s.finals).push(ev.latency);
    else if (ev.type === "done") {
      s.phase = "done";
      if (s.appendOnly && s.current) { s.lines.push(s.current); s.current = ""; }
      render();
      setTriggerValue("session", {
        id: s.id, model: s.model, started_at: s.startedAt, load_seconds: s.loadSeconds,
        events: s.events, done: ev, wav_base64: wavBase64(s.recorded),
      });
      return;
    }
    render();
  }

  function start() {
    if (store[data.key] && store[data.key].ws) {
      try { store[data.key].ws.close(); } catch (e) { /* closed */ }
    }
    s = store[data.key] = {
      id: `${Date.now()}`, model: data.model, startedAt: new Date().toISOString(),
      phase: "connecting", error: null, lines: [], partial: "", current: "", events: [],
      firsts: [], finals: [], utterances: 0, pending: [], recorded: [], sent: 0,
      acc: 0, cnt: 0, phaseIn: 0, ratio: 3, appendOnly: false,
    };
    render();
    let ws;
    try {
      const query = `model=${encodeURIComponent(data.model)}&s2tw=${data.s2tw ? 1 : 0}`;
      ws = new WebSocket(`${data.ws_url}?${query}`);
    } catch (e) {
      return fail(L.no_gateway);
    }
    s.ws = ws;
    ws.binaryType = "arraybuffer";
    ws.onmessage = (m) => onEvent(JSON.parse(m.data));
    ws.onerror = () => fail(L.no_gateway);
    ws.onclose = () => {
      if (s && s.ws === ws && s.phase !== "done" && s.phase !== "failed") fail(L.closed);
    };
  }

  function stop() {
    if (!s || s.phase !== "recording") return;
    s.phase = "finishing";
    teardownAudio();
    sendPending(true);
    try { s.ws.send(JSON.stringify({ type: "stop" })); } catch (e) { fail(L.closed); }
    render();
  }

  ui.start.onclick = start;
  ui.stop.onclick = stop;
  render();
  return () => {
    // Unmounted (left the page or changed model): stop listening, keep nothing running.
    if (s && (s.phase === "recording" || s.phase === "connecting")) {
      teardownAudio();
      try { s.ws && s.ws.close(); } catch (e) { /* closed */ }
      s.phase = "failed";
      s.error = L.closed;
    }
  };
}
