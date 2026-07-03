/* ====================================================================
   Marvel Rivals Classifier — frontend logic
   Talks to the Python backend through window.pywebview.api.*
   ==================================================================== */

const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const SVG = {
  film: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="18" rx="2"/><line x1="7" y1="3" x2="7" y2="21"/><line x1="17" y1="3" x2="17" y2="21"/><line x1="2" y1="9" x2="7" y2="9"/><line x1="2" y1="15" x2="7" y2="15"/><line x1="17" y1="9" x2="22" y2="9"/><line x1="17" y1="15" x2="22" y2="15"/></svg>',
  x:    '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
  check:'<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
  save: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>',
};

const STAGE_MSG = {
  kills_dashes: 'Counting kills & dashes…',
  frames:       'Extracting frames…',
  classify:     'Classifying maps…',
  organize:     'Organising clips…',
};

const state = {
  files: [],
  results: [],
  stages: [],
  modelReady: false,
  fdnnAvailable: false,
  settings: {
    dash_method: 'contour', fdnn_threshold: 0.7,
    output_mode: 'rename', dash_clip_min: 2, kill_clip_min: 1, clip_pad_secs: 2.0,
  },
};

const METHOD_LABEL = { contour: 'HUD slot contour', fdnn: 'FDNN neural net', hybrid: 'Contour + FDNN (max)' };

/* ── Helpers ─────────────────────────────────────────────────────────── */
function api() {
  return (window.pywebview && window.pywebview.api) || null;
}
function showView(name) {
  ['home', 'proc', 'res', 'settings'].forEach((v) => {
    $(`#view-${v}`).hidden = (v !== name);
  });
}
let toastTimer;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg; t.hidden = false;
  requestAnimationFrame(() => t.classList.add('show'));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    t.classList.remove('show');
    setTimeout(() => (t.hidden = true), 300);
  }, 3200);
}

/* ── Status / model ──────────────────────────────────────────────────── */
async function refreshStatus() {
  const a = api(); if (!a) return;
  const s = await a.get_status();
  state.modelReady = s.model_ready;
  state.fdnnAvailable = !!s.fdnn_available;
  state.stages = s.stages || [];
  const chip = $('#modelChip'), txt = $('#modelChipText');
  chip.className = 'chip ' + (s.model_ready ? 'ok' : 'bad');
  txt.textContent = s.model_ready ? 'Model ready' : 'Model missing';
  buildStages();

  try {
    const set = await a.get_settings();
    if (set) {
      state.settings = {
        dash_method: set.dash_method || 'contour',
        fdnn_threshold: typeof set.fdnn_threshold === 'number' ? set.fdnn_threshold : 0.7,
        output_mode: set.output_mode || 'rename',
        dash_clip_min: typeof set.dash_clip_min === 'number' ? set.dash_clip_min : 2,
        kill_clip_min: typeof set.kill_clip_min === 'number' ? set.kill_clip_min : 1,
        clip_pad_secs: typeof set.clip_pad_secs === 'number' ? set.clip_pad_secs : 2.0,
      };
      if (typeof set.fdnn_available === 'boolean') state.fdnnAvailable = set.fdnn_available;
    }
  } catch (e) { /* settings optional */ }
  applySettingsToUI();
  updateProcessBtn();
}

/* ── Settings ────────────────────────────────────────────────────────── */
function applySettingsToUI() {
  // dash detector card description reflects the active method
  const dd = $('#dashDesc');
  if (dd) dd.textContent = METHOD_LABEL[state.settings.dash_method] || 'HUD slot contour';

  // method radios
  $$('input[name="dashMethod"]').forEach((r) => {
    r.checked = (r.value === state.settings.dash_method);
    const card = r.closest('.method');
    if (card) card.classList.toggle('selected', r.checked);
    // FDNN + hybrid need the model; disable when unavailable
    const needsModel = (r.value === 'fdnn' || r.value === 'hybrid');
    r.disabled = needsModel && !state.fdnnAvailable;
    if (card) card.classList.toggle('disabled', r.disabled);
  });
  $('#fdnnWarn').hidden = state.fdnnAvailable;

  // threshold slider
  const thr = state.settings.fdnn_threshold;
  $('#thrRange').value = thr;
  $('#thrValue').textContent = Number(thr).toFixed(2);
  updateThresholdVisibility();

  // output-mode radios
  $$('input[name="outputMode"]').forEach((r) => {
    r.checked = (r.value === state.settings.output_mode);
    const card = r.closest('.method');
    if (card) card.classList.toggle('selected', r.checked);
  });

  // clip options
  $('#dashClipMin').value = state.settings.dash_clip_min;
  $('#killClipMin').value = state.settings.kill_clip_min;
  $('#clipPadSecs').value = state.settings.clip_pad_secs;
  updateClipOptionsVisibility();
}
function updateThresholdVisibility() {
  const m = currentMethodSelection();
  const show = (m === 'fdnn' || m === 'hybrid');
  $('#thresholdCard').classList.toggle('disabled', !show);
}
function currentMethodSelection() {
  const r = $('input[name="dashMethod"]:checked');
  return r ? r.value : state.settings.dash_method;
}
function updateClipOptionsVisibility() {
  const show = (currentOutputModeSelection() === 'clip');
  $('#clipOptionsCard').classList.toggle('disabled', !show);
}
function currentOutputModeSelection() {
  const r = $('input[name="outputMode"]:checked');
  return r ? r.value : state.settings.output_mode;
}
function openSettings() {
  applySettingsToUI();
  showView('settings');
}
async function saveSettings() {
  const a = api(); if (!a) return;
  const method = currentMethodSelection();
  const thr = parseFloat($('#thrRange').value) || 0.7;
  const mode = currentOutputModeSelection();
  const dashClipMin = parseInt($('#dashClipMin').value, 10) || 2;
  const killClipMin = parseInt($('#killClipMin').value, 10) || 1;
  const clipPadSecs = parseFloat($('#clipPadSecs').value) || 0;
  const next = {
    dash_method: method, fdnn_threshold: thr,
    output_mode: mode, dash_clip_min: dashClipMin,
    kill_clip_min: killClipMin, clip_pad_secs: clipPadSecs,
  };
  let saved = next;
  try { saved = await a.set_settings(next); } catch (e) { /* keep local */ }
  state.settings = {
    dash_method: (saved && saved.dash_method) || method,
    fdnn_threshold: (saved && typeof saved.fdnn_threshold === 'number') ? saved.fdnn_threshold : thr,
    output_mode: (saved && saved.output_mode) || mode,
    dash_clip_min: (saved && typeof saved.dash_clip_min === 'number') ? saved.dash_clip_min : dashClipMin,
    kill_clip_min: (saved && typeof saved.kill_clip_min === 'number') ? saved.kill_clip_min : killClipMin,
    clip_pad_secs: (saved && typeof saved.clip_pad_secs === 'number') ? saved.clip_pad_secs : clipPadSecs,
  };
  applySettingsToUI();
  toast('Settings saved · ' + (METHOD_LABEL[state.settings.dash_method] || state.settings.dash_method));
  showView('home');
}
function resetSettings() {
  state.settings = {
    dash_method: 'contour', fdnn_threshold: 0.7,
    output_mode: 'rename', dash_clip_min: 2, kill_clip_min: 1, clip_pad_secs: 2.0,
  };
  applySettingsToUI();
}

/* ── Files ───────────────────────────────────────────────────────────── */
function renderFiles() {
  const wrap = $('#filelistWrap'), inner = $('#dzInner'), list = $('#filelist');
  const n = state.files.length;
  if (n === 0) {
    wrap.hidden = true; inner.hidden = false;
    list.innerHTML = '';
  } else {
    inner.hidden = true; wrap.hidden = false;
    $('#fileCount').textContent = `${n} file${n !== 1 ? 's' : ''} selected`;
    list.innerHTML = state.files.map((f) => `
      <li class="file-row">
        <span class="fic">${SVG.film}</span>
        <span class="fname" title="${esc(f.name)}">${esc(f.name)}</span>
        <button class="fx" data-name="${esc(f.name)}" aria-label="Remove">${SVG.x}</button>
      </li>`).join('');
    $$('#filelist .fx').forEach((b) =>
      b.addEventListener('click', () => removeFile(b.dataset.name)));
  }
  updateProcessBtn();
}
function setFiles(payload) { state.files = (payload && payload.files) || []; renderFiles(); }

async function pickFiles() {
  const a = api(); if (!a) return;
  setFiles(await a.pick_files());
}
async function removeFile(name) {
  const a = api(); if (!a) return;
  setFiles(await a.remove_file(name));
}
async function clearFiles() {
  const a = api(); if (!a) return;
  setFiles(await a.clear_files());
}

/* ── Detector options ────────────────────────────────────────────────── */
function getOptions() {
  return {
    map:    $('#cb-map').checked,
    kills:  $('#cb-kills').checked,
    dashes: $('#cb-dashes').checked,
    combos: $('#cb-combos').checked,
    dash_method:    state.settings.dash_method,
    fdnn_threshold: state.settings.fdnn_threshold,
  };
}
function updateProcessBtn() {
  const opt = getOptions();
  const mapNeeded = opt.map;
  const ready = state.files.length > 0 && (!mapNeeded || state.modelReady);
  $('#processBtn').disabled = !ready;
}

/* ── Processing ──────────────────────────────────────────────────────── */
function buildStages() {
  const host = $('#stages');
  host.innerHTML = state.stages.map(([key, label]) => `
    <div class="stage" data-key="${key}">
      <span class="sdot" data-key="${key}"></span>
      <span>${esc(label)}</span>
    </div>`).join('');
}
function setStage(key, status) {
  const el = $(`.stage[data-key="${key}"]`);
  if (!el) return;
  el.classList.remove('active', 'done');
  if (status === 'active') {
    el.classList.add('active');
    $('#procStage').textContent = STAGE_MSG[key] || key;
  } else if (status === 'done') {
    el.classList.add('done');
    el.querySelector('.sdot').innerHTML = SVG.check;
  }
}
function setProgress(v) {
  $('#progressFill').style.width = v + '%';
  $('#procPct').innerHTML = `${v}<span>%</span>`;
}
function logLine(msg, tag) {
  const log = $('#log');
  const span = document.createElement('span');
  span.className = 'l-line' + (tag ? ` l-${tag}` : '');
  span.textContent = msg;
  log.appendChild(span);
  log.scrollTop = log.scrollHeight;
}
function resetProc() {
  $('#log').innerHTML = '';
  setProgress(0);
  $('#procStage').textContent = 'Starting…';
  $('#procPct').classList.remove('done');
  buildStages();
}

async function startProcessing() {
  const a = api(); if (!a) return;
  resetProc();
  showView('proc');
  const r = await a.start(getOptions());
  if (!r || !r.ok) { toast(r && r.error || 'Could not start.'); showView('home'); }
}

/* Event stream from Python backend */
window.__event = function (msg) {
  switch (msg.type) {
    case 'progress': setProgress(msg.value); break;
    case 'stage':    setStage(msg.stage, msg.status); break;
    case 'log':      logLine(msg.message, msg.tag || ''); break;
    case 'done':
      setProgress(100);
      $('#procStage').textContent = 'Done!';
      state.results = msg.results || [];
      setTimeout(() => buildResults(), 650);
      break;
    case 'error':
      logLine(msg.message, 'error');
      $('#procStage').textContent = 'Error';
      break;
  }
};

/* ── Results ─────────────────────────────────────────────────────────── */
const SORTS = ['Map A→Z', 'Kills ↓', 'Kills ↑', 'Dashes ↓', 'Dashes ↑'];

function buildResults() {
  const clips = state.results;
  const n = clips.length;
  $('#resCount').textContent = `${n} clip${n !== 1 ? 's' : ''} processed`;

  const maps   = ['All', ...[...new Set(clips.map((c) => c.map))].sort()];
  const combos = ['All', ...[...new Set(clips.map((c) => c.combos || 'No Combo'))].sort()];

  fillSelect($('#fMap'), maps);
  fillSelect($('#fCombo'), combos);
  fillSelect($('#fSort'), SORTS);
  $('#fKills').value = 0;
  $('#fDashes').value = 0;

  applyFilters();
  showView('res');
}
function fillSelect(sel, vals) {
  sel.innerHTML = vals.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
}
function applyFilters() {
  const fMap = $('#fMap').value, fCombo = $('#fCombo').value;
  const minK = +$('#fKills').value || 0, minD = +$('#fDashes').value || 0;
  const fSort = $('#fSort').value;

  let out = state.results.slice();
  if (fMap && fMap !== 'All') out = out.filter((c) => c.map === fMap);
  if (fCombo && fCombo !== 'All') {
    const target = fCombo === 'No Combo' ? null : fCombo;
    out = out.filter((c) => (c.combos || null) === target);
  }
  out = out.filter((c) => c.kills >= minK && c.dashes >= minD);

  const sorters = {
    'Map A→Z': (a, b) => a.map.localeCompare(b.map),
    'Kills ↓': (a, b) => b.kills - a.kills,
    'Kills ↑': (a, b) => a.kills - b.kills,
    'Dashes ↓': (a, b) => b.dashes - a.dashes,
    'Dashes ↑': (a, b) => a.dashes - b.dashes,
  };
  if (sorters[fSort]) out.sort(sorters[fSort]);

  renderClips(out);
}
function renderClips(clips) {
  const host = $('#clips');
  if (!clips.length) {
    host.innerHTML = '<div class="empty">No clips match these filters.</div>';
    return;
  }
  host.innerHTML = clips.map((c) => `
    <div class="clip">
      <div class="clip-map" title="${esc(c.map)}">${esc(c.map)}</div>
      <div class="clip-main">
        <div class="clip-stats">
          <span class="stat">Kills <b>${c.kills}</b></span>
          <span class="stat">Dashes <b>${c.dashes}</b></span>
          ${c.combos ? `<span class="stat combo">${esc(c.combos)}</span>` : ''}
          ${c.clip_label ? `<span class="stat combo">${esc(c.clip_label)}</span>` : ''}
          <span class="stat conf">${c.conf}%</span>
        </div>
        <div class="clip-name" title="${esc(c.filename)}">${esc(c.filename)}</div>
      </div>
      <button class="ghost-btn sm clip-save" data-file="${esc(c.filename)}">${SVG.save} Save</button>
    </div>`).join('');
  $$('#clips .clip-save').forEach((b) =>
    b.addEventListener('click', () => saveClip(b.dataset.file)));
}

async function saveClip(filename) {
  const a = api(); if (!a) return;
  const r = await a.save_clip(filename);
  if (r && r.ok) toast('Saved ' + filename);
}
async function saveZip() {
  const a = api(); if (!a) return;
  const files = state.results.map((c) => c.filename);
  if (!files.length) { toast('No clips to zip.'); return; }
  const r = await a.save_zip(files);
  if (r && r.ok) toast(`Saved ${r.count} clip${r.count !== 1 ? 's' : ''} to ZIP`);
}

/* ── Utilities ───────────────────────────────────────────────────────── */
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ── Wire up ─────────────────────────────────────────────────────────── */
let _inited = false;
function init() {
  if (_inited) return;
  _inited = true;
  $('#browseBtn').addEventListener('click', pickFiles);
  $('#browseLink').addEventListener('click', pickFiles);
  $('#addMore').addEventListener('click', pickFiles);
  $('#clearFiles').addEventListener('click', clearFiles);
  $('#processBtn').addEventListener('click', startProcessing);
  $('#exitBtn').addEventListener('click', () => api() && api().close());

  ['cb-map', 'cb-kills', 'cb-dashes', 'cb-combos'].forEach((id) =>
    $('#' + id).addEventListener('change', updateProcessBtn));

  $('#processMore').addEventListener('click', () => { showView('home'); });
  $('#downloadZip').addEventListener('click', saveZip);
  $('#openFolder').addEventListener('click', () => api() && api().reveal_output());

  // Settings view
  $('#settingsBtn').addEventListener('click', openSettings);
  $('#settingsBack').addEventListener('click', () => showView('home'));
  $('#settingsSave').addEventListener('click', saveSettings);
  $('#settingsReset').addEventListener('click', resetSettings);
  const syncMethodCards = () => $$('.method').forEach((c) =>
    c.classList.toggle('selected', c.querySelector('input').checked));
  $$('input[name="dashMethod"]').forEach((r) =>
    r.addEventListener('change', () => {
      syncMethodCards();
      updateThresholdVisibility();
    }));
  $('#thrRange').addEventListener('input', (e) => {
    $('#thrValue').textContent = Number(e.target.value).toFixed(2);
  });
  $$('input[name="outputMode"]').forEach((r) =>
    r.addEventListener('change', () => {
      syncMethodCards();
      updateClipOptionsVisibility();
    }));
  ['fMap', 'fCombo', 'fSort', 'fKills', 'fDashes'].forEach((id) => {
    const el = $('#' + id);
    el.addEventListener('change', applyFilters);
    el.addEventListener('input', applyFilters);
  });

  // Drag & drop onto the dropzone (HTML5 — paths resolved by backend when available)
  const dz = $('#dropzone');
  ['dragenter', 'dragover'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); if (ev !== 'drop' && e.target !== dz && dz.contains(e.relatedTarget)) return; dz.classList.remove('drag'); }));
  dz.addEventListener('drop', async (e) => {
    const paths = [];
    if (e.dataTransfer && e.dataTransfer.files) {
      for (const f of e.dataTransfer.files) { if (f.path) paths.push(f.path); }
    }
    if (paths.length && api()) setFiles(await api().add_files(paths));
    else if (!paths.length) toast('Use Browse to add clips.');
  });

  refreshStatus();
}

if (window.pywebview) init();
else window.addEventListener('pywebviewready', init);
// Fallback if the ready event already fired or runs in a plain browser preview
window.addEventListener('DOMContentLoaded', () => { if (api()) init(); });
