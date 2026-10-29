// Live voice assistant (Streamlit components v2, runs in the app page).
// Microphone (or the scripted demo) -> 16 kHz PCM16, 100 ms frames -> WebSocket to the model
// gateway (/v1/voice/agent/ws). The gateway runs streaming ASR, the front LLM, background
// tools and speech synthesis (local_llm/voice_agent.py) and sends back captions, assistant
// text, WAV audio, tool progress and timings. This file plays the audio (reporting each
// chunk's start and end), stops it at once on "barge_in", and draws the conversation, the
// background-work board and a timeline. When the session ends its summary goes back to Python
// as the trigger value "session", which saves it as a run.

const RATE = 16000;
const FRAME = 1600; // 100 ms
const MAX_SECONDS = 600;
const WORKLET = `
class LalCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("lal-va-capture", LalCapture);`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function esc(t) {
  return String(t === undefined || t === null ? "" : t).replace(
    /[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

function secs(x) {
  return x === null || x === undefined ? "—" : `${Number(x).toFixed(2)} s`;
}

function b64bytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function pcm16(b64) {
  const bytes = b64bytes(b64);
  return new Int16Array(bytes.buffer, 0, Math.floor(bytes.length / 2));
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const L = data.labels;
  const store = (window.__lalVoiceAgent = window.__lalVoiceAgent || {});
  const $ = (sel) => parentElement.querySelector(sel);
  const ui = {
    mic: $(".va-mic"), demo: $(".va-demo"), stop: $(".va-stop"), dot: $(".va-dot"),
    status: $(".va-status"), chat: $(".va-chat"), live: $(".va-live"),
    liveText: $(".va-live-text"), cards: $(".va-cards"), timeline: $(".va-timeline"),
    metrics: $(".va-metrics"),
  };
  $(".va-mic-label").textContent = L.mic;
  $(".va-demo-label").textContent = L.demo;
  $(".va-stop-label").textContent = L.stop;
  $(".va-chat-title").textContent = L.chat;
  $(".va-board-title").textContent = L.board;
  $(".va-timeline-title").textContent = L.timeline;
  $(".va-metrics-title").textContent = L.metrics;
  const mountId = `${Date.now()}-${Math.random()}`;
  let s = store[data.key];
  if (s) s.mountId = mountId;

  // -- drawing ------------------------------------------------------------------------------

  function sessionNow() {
    if (!s || s.lastT === null) return 0;
    if (s.phase === "done" || s.phase === "failed") return s.lastT;
    return s.lastT + (performance.now() - s.lastLocal) / 1000;
  }

  function render() {
    const live = s && ["connecting", "live", "finishing"].includes(s.phase);
    ui.mic.disabled = live;
    ui.demo.disabled = live || !data.demo || (!data.demo.sim && !data.demo.lines.length);
    ui.stop.disabled = !live;
    ui.dot.className = "va-dot" + (live ? (s.ready && s.llmReady ? " on" : " busy") : "");
    ui.status.className = "va-status" + (s && s.error ? " va-error" : "");
    ui.status.textContent = statusText();
    renderChat();
    renderCards();
    renderTimeline();
    renderMetrics();
  }

  function statusText() {
    if (!s) return L.idle;
    if (s.error) return s.error;
    if (s.phase === "connecting") return L.connecting;
    if (s.phase === "live") {
      if (!s.ready) return L.loading_asr;
      if (!s.llmReady) return L.loading_llm;
      if (s.demoStep) return s.demoStep;
      return s.speaking.size ? L.speaking : L.listening;
    }
    return L[s.phase] || "";
  }

  function renderChat() {
    if (!s || !s.messages.length) {
      ui.chat.innerHTML = `<div class="va-empty">${esc(L.empty)}</div>`;
      ui.live.className = "va-live";
      ui.liveText.textContent = "";
      return;
    }
    const rows = s.messages.map((m) => {
      if (m.role === "user") {
        return `<div class="va-msg user${m.partial ? " partial" : ""}">` +
          `<div class="va-bubble">${esc(m.text)}</div>` +
          `<div class="va-tags">${m.partial ? esc(L.hearing) : esc(L.you)}</div></div>`;
      }
      const tags = [];
      if (s.speaking.has(m.sid)) {
        tags.push(`<span class="va-speaking"><i></i><i></i><i></i><i></i></span>` +
                  `<span>${esc(L.speaking_tag)}</span>`);
      }
      if (m.kind === "filler") tags.push(`<span class="va-tag fill">${esc(L.filler_tag)}</span>`);
      if (m.kind === "answer") {
        tags.push(`<span class="va-tag tool"><span class="va-icon">build</span>` +
                  `${esc(L.answer_tag)}</span>`);
      }
      if (m.cut) {
        tags.push(`<span class="va-tag cut"><span class="va-icon">front_hand</span>` +
                  `${esc(L.cut_tag)}</span>`);
      }
      const cls = `va-msg asst${m.kind === "filler" ? " filler" : ""}${m.cut ? " cut" : ""}`;
      // After a barge-in, the part that was never played is shown struck through.
      const said = m.cut && m.chunks && m.chunks.length ? m.chunks.map((c) =>
        c.state === "cut" ? `<span class="va-unsaid">${esc(c.text)}</span>` : esc(c.text)).join("")
        : esc(m.text);
      return `<div class="${cls}"><div class="va-bubble">${said}</div>` +
        `<div class="va-tags">${esc(L.assistant)} ${tags.join(" ")}</div></div>`;
    });
    const stick = ui.chat.scrollHeight - ui.chat.scrollTop - ui.chat.clientHeight < 60;
    ui.chat.innerHTML = rows.join("");
    if (stick || s.phase !== "done") ui.chat.scrollTop = ui.chat.scrollHeight;
    const hearing = s.userSpeaking;
    ui.live.className = "va-live" + (hearing ? " on" : "");
    ui.liveText.textContent = hearing ? L.user_talking : (s.toolsRunning ? L.tools_running : "");
  }

  function renderCards() {
    const cards = s ? Object.values(s.cards) : [];
    if (!cards.length) {
      ui.cards.innerHTML = `<div class="va-note">${esc(L.no_tools)}</div>`;
      return;
    }
    ui.cards.innerHTML = cards.map((c) => {
      const running = c.ok === null;
      const cls = `va-card${running ? " running" : c.ok ? "" : " failed"}`;
      const icon = running ? `<span class="va-icon va-spin">progress_activity</span>`
        : `<span class="va-icon">${c.ok ? "check_circle" : "error"}</span>`;
      const cached = !running && c.ok && c.seconds !== undefined && c.seconds < 0.05;
      const took = running ? `${(sessionNow() - c.start).toFixed(1)} s…`
        : secs(c.seconds) + (cached ? ` · ${L.cached}` : "");
      const args = Object.entries(c.args || {}).map(([k, v]) => `${k}=${v}`).join(", ");
      let body = "";
      const r = c.result || {};
      if (!running && r.error) body = `<div class="va-result">${esc(r.error)}</div>`;
      else if (!running && c.name === "search_places") {
        const places = (r.places || []).slice(0, 5).map((p) =>
          `<li>${esc(p.name)} · ${esc(p.distance_m)} m` +
          `${p.opening_hours ? " · " + esc(p.opening_hours) : ""}</li>`).join("");
        body = `<div class="va-result">${esc(r.note || "")}<ol>${places}</ol></div>` +
          `<div class="va-attr">${esc(r.attribution || "")} · ${esc((r.sources || []).join(", "))}</div>`;
      } else if (!running) {
        body = `<div class="va-result">${esc(c.summary || "")}</div>` +
          (r.source ? `<div class="va-attr">${esc(r.source)}</div>` : "");
      } else {
        body = `<div class="va-result">${esc(L.tool_running)}</div>`;
      }
      return `<div class="${cls}"><div class="va-card-top">${icon}<span>${esc(c.name)}</span>` +
        `<span class="va-secs">${esc(took)}</span></div>` +
        `<span class="va-args">${esc(args)}</span>${body}</div>`;
    }).join("");
  }

  function renderTimeline() {
    if (!s || s.lastT === null) {
      ui.timeline.innerHTML = `<div class="va-note">${esc(L.no_timeline)}</div>`;
      return;
    }
    const now = sessionNow();
    const t0 = Math.max(Math.min(s.firstT === null ? now : s.firstT, now) - 0.5, 0);
    const span = Math.max(now - t0, 5);
    const pos = (t) => `${(100 * (t - t0)) / span}%`;
    const wid = (a, b) => `${Math.max((100 * (b - a)) / span, 0.4)}%`;
    const seg = (cls, a, b, title) => `<div class="va-seg ${cls}" title="${esc(title)}" ` +
      `style="left:${pos(a)};width:${wid(a, b === null ? now : b)}"></div>`;
    const lanes = [
      [L.lane_user, s.segs.user.map((x) => seg("user" + (x[1] === null ? " running" : ""),
                                                 x[0], x[1], L.lane_user))],
      [L.lane_llm, s.segs.llm.map((x) => seg("llm" + (x.end === null ? " running" : ""),
                                               x.start, x.end, x.what))],
      [L.lane_asst, s.segs.asst.map((x) => seg(`asst ${x.kind}${x.cut ? " cut" : ""}`,
                                                 x.start, x.end, x.text || ""))],
      [L.lane_tool, s.segs.tool.map((x) => seg("tool" + (x.end === null ? " running" : ""),
                                                 x.start, x.end, x.name))],
    ];
    const ticks = [];
    const step = span > 60 ? 10 : span > 25 ? 5 : 2;
    for (let t = Math.ceil(t0 / step) * step; t <= now; t += step) {
      ticks.push(`<span style="left:${pos(t)}">${t}s</span>`);
    }
    ui.timeline.innerHTML = lanes.map(([name, bars]) =>
      `<div class="va-lane"><span>${esc(name)}</span><div class="va-track">${bars.join("")}` +
      `</div></div>`).join("") + `<div class="va-axis">${ticks.join("")}</div>`;
  }

  function renderMetrics() {
    const turns = s ? Object.values(s.turns) : [];
    if (!turns.length) {
      ui.metrics.innerHTML = `<div class="va-note">${esc(L.metrics_note)}</div>`;
      return;
    }
    const head = `<tr><th>${esc(L.m_turn)}</th><th>${esc(L.m_asr)}</th><th>${esc(L.m_llm)}</th>` +
      `<th>${esc(L.m_response)}</th><th>${esc(L.m_tool)}</th><th>${esc(L.m_answer)}</th></tr>`;
    const rows = turns.map((t) => `<tr><td>${esc(t.user)}</td><td>${secs(t.asr_s)}</td>` +
      `<td>${secs(t.llm_first_s)}</td><td><b>${secs(t.response_s)}</b></td>` +
      `<td>${secs(t.tool_s)}</td><td>${secs(t.answer_s)}</td></tr>`).join("");
    ui.metrics.innerHTML = `<table>${head}${rows}</table>` +
      `<div class="va-note">${esc(L.metrics_note)}</div>`;
  }

  // -- audio out ------------------------------------------------------------------------------

  function send(obj) {
    try {
      if (s.ws && s.ws.readyState === 1) s.ws.send(JSON.stringify(obj));
    } catch (e) { /* closed */ }
  }

  function enqueueAudio(ev) {
    const bytes = b64bytes(ev.wav);
    s.decodeChain = s.decodeChain.then(() => s.ctx.decodeAudioData(bytes.buffer)).then((buf) => {
      if (s.stopped.has(ev.speech_id)) {
        send({ type: "played", speech_id: ev.speech_id, seq: ev.seq, stopped: true });
        return;
      }
      s.queue.push({ sid: ev.speech_id, seq: ev.seq, buffer: buf });
      playNext();
    }).catch(() => send({ type: "played", speech_id: ev.speech_id, seq: ev.seq }));
  }

  function chunk(sid, seq) {
    const m = s.messages.find((x) => x.role === "asst" && x.sid === sid);
    return m && m.chunks ? m.chunks.find((c) => c.seq === seq) : null;
  }

  function playNext() {
    if (s.current || !s.queue.length || !s.ctx) return;
    const item = s.queue.shift();
    const c = chunk(item.sid, item.seq);
    if (c) c.state = "playing";
    const source = s.ctx.createBufferSource();
    source.buffer = item.buffer;
    source.connect(s.ctx.destination);
    item.source = source;
    s.current = item;
    source.onended = () => {
      if (s.current !== item) return;
      s.current = null;
      const done = chunk(item.sid, item.seq);
      if (done) done.state = "played";
      send({ type: "played", speech_id: item.sid, seq: item.seq });
      playNext();
    };
    if (!s.speechStart[item.sid]) s.speechStart[item.sid] = performance.now();
    send({ type: "playing", speech_id: item.sid, seq: item.seq });
    source.start();
  }

  function bargeIn(ids) {
    ids.forEach((id) => {
      s.stopped.add(id);
      const m = s.messages.find((x) => x.role === "asst" && x.sid === id);
      (m && m.chunks ? m.chunks : []).forEach((c) => {
        if (c.state === "queued") c.state = "cut";
      });
    });
    if (s.current && s.stopped.has(s.current.sid)) {
      const item = s.current;
      s.current = null;
      item.source.onended = null;
      try { item.source.stop(); } catch (e) { /* already stopped */ }
      send({ type: "played", speech_id: item.sid, seq: item.seq, stopped: true });
    }
    s.queue = s.queue.filter((item) => {
      if (!s.stopped.has(item.sid)) return true;
      send({ type: "played", speech_id: item.sid, seq: item.seq, stopped: true });
      return false;
    });
    playNext();
  }

  // -- events from the gateway -------------------------------------------------------------------

  function asstMessage(ev) {
    let m = s.messages.find((x) => x.role === "asst" && x.sid === ev.speech_id);
    if (!m) {
      const partial = s.messages.findIndex((x) => x.role === "user" && x.partial);
      m = { role: "asst", sid: ev.speech_id, turn: ev.turn, kind: ev.kind, text: "", cut: false };
      // A reply to words still being transcribed (a cloud model answers before its transcript
      // is final) goes after them; any other reply stays above the "hearing" bubble.
      const pending = partial >= 0 ? s.messages[partial] : null;
      const answersIt = pending && pending.turn != null && ev.turn != null && ev.turn >= pending.turn;
      if (partial >= 0 && !answersIt) s.messages.splice(partial, 0, m);
      else s.messages.push(m);
    }
    return m;
  }

  function onEvent(ev) {
    if (!s) return;
    if (typeof ev.t === "number") {
      s.lastT = ev.t;
      s.lastLocal = performance.now();
    }
    if (ev.type !== "audio") s.events.push(ev);
    switch (ev.type) {
      case "ready":
        s.ready = true;
        s.asrLoad = ev.load_seconds;
        s.appendOnly = !!ev.append_only;
        if (s.mode === "mic") startMic();
        else startFeeder();
        break;
      case "llm_ready":
        s.llmReady = true;
        s.llmLoad = ev.seconds;
        break;
      case "error":
        if (!s.ready) return fail(ev.error || L.closed);
        s.error = ev.error;
        break;
      case "vad":
        if (s.firstT === null) s.firstT = ev.at;
        if (ev.state === "start") {
          s.userSpeaking = true;
          s.segs.user.push([ev.at, null]);
        } else {
          s.userSpeaking = false;
          const open = s.segs.user.find((x) => x[1] === null);
          if (open) open[1] = ev.at;
        }
        break;
      case "user_partial": {
        let m = s.messages.find((x) => x.role === "user" && x.partial);
        if (!m) { m = { role: "user", text: "", partial: true }; s.messages.push(m); }
        m.text = ev.text;
        if (ev.turn != null) m.turn = ev.turn;
        break;
      }
      case "user_final": {
        let m = s.messages.find((x) => x.role === "user" && x.partial);
        if (!m) { m = { role: "user" }; s.messages.push(m); }
        Object.assign(m, { text: ev.text, partial: false, turn: ev.turn });
        s.finals += 1;
        s.lastFinalTurn = ev.turn;
        s.idleMark = s.idleCount;
        break;
      }
      case "llm":
        if (ev.phase === "start") s.segs.llm.push({ start: ev.t, end: null, what: ev.what });
        else {
          const open = s.segs.llm.find((x) => x.end === null);
          if (open) open.end = ev.t;
        }
        break;
      case "assistant": {
        const m = asstMessage(ev);
        m.text = ev.text;
        m.kind = ev.kind;
        s.speechKind[ev.speech_id] = { kind: ev.kind, turn: ev.turn };
        break;
      }
      case "audio": {
        const m = asstMessage(ev);
        m.kind = ev.kind;
        m.chunks = m.chunks || [];
        m.chunks.push({ seq: ev.seq, text: ev.text, state: "queued" });
        s.speechKind[ev.speech_id] = { kind: ev.kind, turn: ev.turn };
        enqueueAudio(ev);
        break;
      }
      case "speaking": {
        let seg = s.segs.asst.find((x) => x.sid === ev.speech_id);
        if (ev.on) {
          s.speaking.add(ev.speech_id);
          if (!seg) {
            const info = s.speechKind[ev.speech_id] || {};
            seg = { sid: ev.speech_id, start: ev.t, end: null, kind: info.kind, cut: false };
            s.segs.asst.push(seg);
          }
        } else {
          s.speaking.delete(ev.speech_id);
          if (seg && seg.end === null) seg.end = ev.t;
        }
        break;
      }
      case "barge_in":
        bargeIn(ev.speech_ids || []);
        (ev.speech_ids || []).forEach((id) => {
          s.speaking.delete(id);
          const m = s.messages.find((x) => x.role === "asst" && x.sid === id);
          if (m) m.cut = true;
          const seg = s.segs.asst.find((x) => x.sid === id);
          if (seg) { seg.cut = true; if (seg.end === null) seg.end = ev.at; }
        });
        s.bargeIns += 1;
        break;
      case "tool_start":
        s.cards[ev.call_id] = { ...ev, ok: null };
        s.segs.tool.push({ id: ev.call_id, start: ev.start, end: null, name: ev.name });
        s.toolsRunning += 1;
        break;
      case "tool_end": {
        Object.assign(s.cards[ev.call_id] || {}, ev);
        const seg = s.segs.tool.find((x) => x.id === ev.call_id);
        if (seg) seg.end = ev.end;
        s.toolsRunning = Math.max(s.toolsRunning - 1, 0);
        break;
      }
      case "turn":
        s.turns[ev.turn] = ev;
        break;
      case "idle":
        s.idleCount += 1;
        break;
      case "sim_line":
        s.simLine = ev;
        break;
      case "done":
        finish(ev);
        return;
      default:
        break;
    }
    render();
  }

  function finish(ev) {
    s.phase = "done";
    clearInterval(s.timer);
    clearInterval(s.feedTimer);
    teardownMic();
    s.speaking.clear();
    render();
    setTriggerValue("session", {
      id: s.id, mode: s.mode, params: s.params, started_at: s.startedAt,
      asr_load_s: s.asrLoad, llm_load_s: s.llmLoad, barge_ins: s.bargeIns,
      summary: ev.summary,
      messages: s.messages.filter((m) => !m.partial).map((m) => ({
        role: m.role, text: m.text, turn: m.turn, kind: m.kind, cut: !!m.cut,
        unsaid: (m.chunks || []).filter((c) => c.state === "cut").map((c) => c.text).join(""),
      })),
    });
  }

  function fail(message) {
    if (!s || s.phase === "done") return;
    s.error = message;
    s.phase = "failed";
    clearInterval(s.timer);
    clearInterval(s.feedTimer);
    teardownMic();
    try { s.ws && s.ws.close(); } catch (e) { /* closed */ }
    render();
  }

  // -- audio in: microphone -------------------------------------------------------------------------

  function sendPcm(chunk) {
    s.sent += chunk.length;
    if (s.ws && s.ws.readyState === 1) s.ws.send(chunk.buffer);
  }

  function teardownMic() {
    if (!s) return;
    try { s.stream && s.stream.getTracks().forEach((t) => t.stop()); } catch (e) { /* gone */ }
    try { s.micNode && s.micNode.disconnect(); } catch (e) { /* gone */ }
    s.stream = null;
    s.micNode = null;
  }

  function push(x) {
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
    while (s.pending.length >= FRAME) sendPcm(Int16Array.from(s.pending.splice(0, FRAME)));
    if (s.sent / RATE >= MAX_SECONDS) stop();
  }

  async function startMic() {
    try {
      // Echo cancellation on: the assistant talks through the speakers while the mic is open.
      s.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false,
                 autoGainControl: false },
      });
    } catch (e) {
      const denied = e && (e.name === "NotAllowedError" || e.name === "SecurityError");
      fail(denied ? L.denied : `${L.no_mic} (${e && e.name ? e.name : e})`);
      return;
    }
    s.ratio = s.ctx.sampleRate / RATE;
    const source = s.ctx.createMediaStreamSource(s.stream);
    const mute = s.ctx.createGain();
    mute.gain.value = 0;
    mute.connect(s.ctx.destination);
    if (s.ctx.audioWorklet && window.AudioWorkletNode) {
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      await s.ctx.audioWorklet.addModule(url);
      const node = new AudioWorkletNode(s.ctx, "lal-va-capture");
      node.port.onmessage = (e) => s && s.phase === "live" && push(e.data);
      source.connect(node);
      node.connect(mute);
      s.micNode = node;
    } else {
      const node = s.ctx.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (e) =>
        s && s.phase === "live" && push(e.inputBuffer.getChannelData(0).slice(0));
      source.connect(node);
      node.connect(mute);
      s.micNode = node;
    }
    render();
  }

  // -- audio in: the scripted demo ----------------------------------------------------------------

  function startFeeder() {
    // A virtual microphone: one 100 ms frame per 100 ms of wall time, silence unless a demo
    // line is being "spoken" (that line is also played aloud).
    s.feedStart = performance.now();
    s.framesSent = 0;
    s.feedTimer = setInterval(() => {
      if (!s || s.phase !== "live") return;
      const due = Math.floor((performance.now() - s.feedStart) / 100);
      while (s.framesSent < due) {
        const frame = new Int16Array(FRAME);
        const line = s.feeding;
        if (line) {
          if (line.pos === 0) playLine(line);
          frame.set(line.pcm.subarray(line.pos, line.pos + FRAME));
          line.pos += FRAME;
          if (line.pos >= line.pcm.length) { s.feeding = null; line.done(); }
        }
        sendPcm(frame);
        s.framesSent += 1;
      }
    }, 25);
    runDemo();
  }

  function playLine(line) {
    const buf = s.ctx.createBuffer(1, line.pcm.length, RATE);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < line.pcm.length; i++) ch[i] = line.pcm[i] / 32768;
    const src = s.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(s.ctx.destination);
    src.start();
  }

  function speak(line) {
    return new Promise((resolve) => {
      s.feeding = { pcm: line.pcm, pos: 0, done: resolve };
    });
  }

  // The assistant is done with a line: an "idle" came after the line was spoken (a model that
  // never captions the user, or a caption that never came, still counts) and nothing of its
  // speech is left to play.
  function settled(idleBefore) {
    return s.idleCount > idleBefore && !s.current && !s.queue.length;
  }

  async function until(test, ms) {
    const end = performance.now() + ms;
    while (performance.now() < end) {
      if (!s || s.phase !== "live") return false;
      if (test()) return true;
      await sleep(50);
    }
    return false;
  }

  async function runSim() {
    // A simulated user (an LLM playing the user, local_llm/user_sim.py): after each reply the
    // gateway writes and speaks the next line from the conversation so far and the goal.
    const sim = data.demo.sim;
    await until(() => s.llmReady, 120000);
    await sleep(800);
    for (let i = 0; i < sim.max_turns; i++) {
      if (!s || s.phase !== "live") return;
      s.simLine = null;
      s.demoStep = `${L.sim_thinking} ${i + 1}/${sim.max_turns}`;
      render();
      send({ type: "sim_next", goal: sim.goal, voice: sim.voice, lang: sim.lang });
      await until(() => s.simLine, 90000);
      const line = s.simLine;
      if (!line || !line.text || !line.pcm) break;
      s.demoStep = `${L.demo_step} ${i + 1}: ${line.text}`;
      render();
      const before = s.finals;
      const idleBefore = s.idleCount;
      await speak({ text: line.text, pcm: pcm16(line.pcm) });
      await until(() => s.finals > before, 15000);
      await until(() => settled(idleBefore), 60000);
      await sleep(700);
      if (line.done) break;
    }
    s.demoStep = null;
    await sleep(500);
    stop();
  }

  async function runDemo() {
    if (data.demo.sim) return runSim();
    const lines = data.demo.lines.map((l) => ({ ...l, pcm: pcm16(l.pcm) }));
    await until(() => s.llmReady, 120000);
    await sleep(800);
    const normal = lines.filter((l) => !l.barge);
    const barge = lines.find((l) => l.barge);
    for (let i = 0; i < normal.length; i++) {
      if (!s || s.phase !== "live") return;
      s.demoStep = `${L.demo_step} ${i + 1}/${normal.length}: ${normal[i].text}`;
      render();
      const before = s.finals;
      const idleBefore = s.idleCount;
      await speak(normal[i]);
      const last = i === normal.length - 1;
      if (last && barge) {
        // Interrupt while the assistant reads out the list.
        await until(() => s.finals > before, 15000);
        const turn = s.lastFinalTurn;
        const reading = () => Object.entries(s.speechKind).some(([sid, k]) =>
          k.turn === turn && k.kind === "answer" && s.speechStart[sid] &&
          performance.now() - s.speechStart[sid] > data.demo.barge_after_ms);
        await until(() => reading() || s.idleCount > s.idleMark, 60000);
        s.demoStep = `${L.demo_barge}: ${barge.text}`;
        render();
        const bargeBefore = s.finals;
        const bargeIdle = s.idleCount;
        await speak(barge);
        await until(() => s.finals > bargeBefore, 15000);
        await until(() => settled(bargeIdle), 60000);
      } else {
        await until(() => s.finals > before, 15000);
        await until(() => settled(idleBefore), 60000);
      }
      await sleep(700);
    }
    s.demoStep = null;
    await sleep(500);
    stop();
  }

  // -- session --------------------------------------------------------------------------------------

  async function start(mode) {
    const old = store[data.key];
    if (old && old.ws) { try { old.ws.close(); } catch (e) { /* closed */ } }
    s = store[data.key] = {
      id: `${Date.now()}`, mode, params: data.params, startedAt: new Date().toISOString(),
      mountId, phase: "connecting", error: null, ready: false, llmReady: false,
      messages: [], cards: {}, turns: {}, events: [], speaking: new Set(),
      segs: { user: [], llm: [], asst: [], tool: [] }, speechKind: {}, speechStart: {},
      queue: [], current: null, stopped: new Set(), decodeChain: Promise.resolve(),
      finals: 0, idleCount: 0, idleMark: 0, lastFinalTurn: null, bargeIns: 0, toolsRunning: 0,
      userSpeaking: false, lastT: null, lastLocal: 0, firstT: null, demoStep: null,
      pending: [], sent: 0, acc: 0, cnt: 0, phaseIn: 0, ratio: 3, feeding: null,
    };
    // One AudioContext for playback (and the mic), made inside the click so it may play.
    s.ctx = new AudioContext();
    try { await s.ctx.resume(); } catch (e) { /* plays once allowed */ }
    s.timer = setInterval(() => { if (s.phase === "live") render(); }, 250);
    render();
    const p = data.params;
    const query = new URLSearchParams({
      asr: p.asr, llm: p.llm, voice: p.voice, rate: String(p.rate || 0),
      net: p.net ? "1" : "0", barge: p.barge ? "1" : "0", s2tw: "1",
      engine: p.engine || "cascade", captions: p.captions === false ? "0" : "1",
      hear: p.hear ? "1" : "0",
    });
    let ws;
    try {
      ws = new WebSocket(`${data.ws_url}?${query}`);
    } catch (e) {
      return fail(L.no_gateway);
    }
    s.ws = ws;
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { s.phase = "live"; render(); };
    ws.onmessage = (m) => onEvent(JSON.parse(m.data));
    ws.onerror = () => fail(L.no_gateway);
    ws.onclose = () => {
      if (s && s.ws === ws && s.phase !== "done" && s.phase !== "failed") fail(L.closed);
    };
  }

  function stop() {
    if (!s || (s.phase !== "live" && s.phase !== "connecting")) return;
    s.phase = "finishing";
    clearInterval(s.feedTimer);
    teardownMic();
    if (s.pending.length) sendPcm(Int16Array.from(s.pending.splice(0)));
    send({ type: "stop" });
    render();
  }

  ui.mic.onclick = () => start("mic");
  ui.demo.onclick = () => start("demo");
  ui.stop.onclick = stop;
  render();
  return () => {
    // Unmounted: when no newer mount of this component follows (a rerun mounts it again
    // at once), the page was left: stop listening and talking.
    setTimeout(() => {
      const cur = store[data.key];
      if (!cur || cur.mountId !== mountId || parentElement.isConnected) return;
      if (["connecting", "live", "finishing"].includes(cur.phase)) {
        clearInterval(cur.timer);
        clearInterval(cur.feedTimer);
        try { cur.stream && cur.stream.getTracks().forEach((t) => t.stop()); } catch (e) { /* */ }
        try { cur.ws && cur.ws.close(); } catch (e) { /* closed */ }
        try { cur.ctx && cur.ctx.close(); } catch (e) { /* closed */ }
        cur.phase = "failed";
        cur.error = L.closed;
      }
    }, 600);
  };
}
