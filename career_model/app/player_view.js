/* The player view (§8.3).
 *
 * This client contains no modelling logic. It reads precomputed percentiles and
 * renders them. There is no fallback computation, no threshold that encodes a
 * basketball judgment, and no place where a missing number gets filled in with
 * a plausible one — anything requiring a decision about what a number *means*
 * lives upstream in simulate/, per §8.
 *
 * The display rules it does enforce (§8.4):
 *   - no naked point estimates: every projected number carries an interval;
 *   - 80% by default, 50% and 95% as toggles;
 *   - career totals show median and band, never the mean alone;
 *   - thin projections say so in words, because most readers look at the centre
 *     line and ignore the shading.
 */
const PlayerView = (function () {
  let INDEX = null, cache = new Map();
  let state = { stat: 'pts_per100', level: 80, player: null, payload: null };

  const STAT_LABEL = k => (INDEX.stats.find(s => s.key === k) || {}).label || k;

  /* §8.5: the version stamp rides in the URL so a refit busts the browser
   * cache. Without it, a client holding a cached payload.json would pair last
   * month's projections with this month's index.json and never know — stale
   * projections silently mixed with new ones is the exact failure the spec
   * names, and an in-memory Map does not prevent it across page loads. */
  async function load(pid) {
    if (cache.has(pid)) return cache.get(pid);
    // Keyed on the build stamp, not just `model_version`: the version string is
    // a hand-maintained constant that does not change when the model is refit,
    // so on its own it would have busted nothing.  `generated` changes on every
    // precompute, which is the event that actually invalidates a payload.
    const v = encodeURIComponent(
      `${INDEX.model_version}-${INDEX.train_cutoff}-${INDEX.generated}`);
    const p = await fetch(`projections/${pid}/payload.json?v=${v}`).then(r => r.json());
    if (p.meta && p.meta.model_version && p.meta.model_version !== INDEX.model_version) {
      throw new Error(`stale payload for ${pid}: ${p.meta.model_version} ` +
                      `vs index ${INDEX.model_version}`);
    }
    // model_version is a hand-maintained constant, so a config change (e.g. the
    // Session-4 mixture->gaussian switch) leaves it unchanged.  config_fingerprint
    // hashes the enabled feature flags and DOES change, so it is the check that
    // catches a payload built under a superseded configuration.
    if (p.meta && p.meta.config_fingerprint && INDEX.config_fingerprint &&
        p.meta.config_fingerprint !== INDEX.config_fingerprint) {
      throw new Error(`stale payload for ${pid}: config ${p.meta.config_fingerprint} ` +
                      `vs index ${INDEX.config_fingerprint}`);
    }
    cache.set(pid, p);
    return p;
  }

  function init(index) { INDEX = index; }

  async function show(player) {
    const view = document.getElementById('view');
    view.innerHTML = '<div class="card"><div class="empty">Loading…</div></div>';
    let payload;
    try {
      payload = await load(player.player_id);
    } catch (e) {
      view.innerHTML = '<div class="card"><div class="empty">' +
        'Insufficient data — no projection is stored for this player.</div></div>';
      return;
    }
    state.player = player;
    state.payload = payload;
    if (!payload.seasons[state.stat]) {
      state.stat = Object.keys(payload.seasons)[0];
    }
    render();
  }

  function render() {
    const { player, payload } = state;
    const meta = payload.meta;
    const view = document.getElementById('view');
    view.innerHTML = '';

    view.appendChild(header(meta, player));

    /* No projected seasons at all: every horizon fell below the minimum of
     * surviving draws, i.e. the simulation retires him almost immediately.
     * That is a real answer and it must be said in words — a blank page reads
     * as a bug, not a finding (§8.4: thin projections say so in words). */
    if (!Object.keys(payload.seasons || {}).length) {
      const c = document.createElement('div');
      c.className = 'card';
      c.innerHTML = '<h2>Career projection</h2><div class="empty">' +
        'The simulation projects no further seasons: across the simulated ' +
        'futures, too few have him still in the league next year to describe. ' +
        'The longevity curve below is the whole story.</div>';
      view.appendChild(c);
      if (payload.survival && payload.survival.horizon.length) {
        view.appendChild(longevityCard());
      }
      return;
    }

    view.appendChild(chartCard());
    view.appendChild(perGameCard());
    view.appendChild(peakCard());
    view.appendChild(totalsCard());
    view.appendChild(longevityCard());
  }

  /* ---- 1. header ------------------------------------------------------- */
  function header(meta, player) {
    const c = document.createElement('div');
    c.className = 'card';
    const where = meta.is_rookie ? player.team : (meta.team || '—');
    const hist = meta.is_rookie
      ? 'no NBA seasons'
      : `${meta.n_history_seasons} NBA season${meta.n_history_seasons === 1 ? '' : 's'} feeding the projection`;
    c.innerHTML = `<div class="phead">
        <h2>${meta.name}</h2>
        <div class="meta">${meta.position} · age ${meta.age} · ${where} · ${hist}</div>
      </div>`;

    /* §8.1: the rookie marker is shown by default, never behind a tooltip. */
    if (meta.is_rookie) {
      const b = document.createElement('div');
      b.className = 'banner rookie';
      b.innerHTML = '<b>No NBA data.</b> This projection is drawn entirely from ' +
        'college production, combine measurements and draft position — there is no ' +
        'filtered history behind it. Expect the year-1 interval to be about as wide ' +
        'as an established player\'s year-5 interval. That is the honest width, not ' +
        'a hedge.';
      c.appendChild(b);
      if (!meta.has_college_data) {
        const b2 = document.createElement('div');
        b2.className = 'banner thin';
        b2.innerHTML = '<b>No college data on file either.</b> International, ' +
          'G-League or prep path — the prior falls back to its missing-data branch ' +
          'and the intervals widen further still.';
        c.appendChild(b2);
      }
    } else if (meta.n_history_seasons <= 2) {
      const b = document.createElement('div');
      b.className = 'banner thin';
      b.innerHTML = `<b>Thin history.</b> Only ${meta.n_history_seasons} NBA ` +
        `season${meta.n_history_seasons === 1 ? '' : 's'} of evidence, so the ` +
        `prior is still doing most of the work here and the bands are wide for ` +
        `a reason.`;
      c.appendChild(b);
    }
    if (meta.status === 'inactive') {
      const b = document.createElement('div');
      b.className = 'banner thin';
      b.innerHTML = `<b>Did not play in ${meta.last_season + 1}.</b> Last ` +
        `played in ${meta.last_season} — injury, or out of the league; the box ` +
        `score cannot tell which. The projection rolls his last filtered state ` +
        `through the missed year with no new evidence, so it resumes from ` +
        `${meta.last_season + 2} with wider bands, and P(active) already prices ` +
        `in the chance he does not return.`;
      c.appendChild(b);
    } else if (meta.status === 'retired') {
      const b = document.createElement('div');
      b.className = 'banner thin';
      b.innerHTML = `<b>Retired.</b> Last played in ${meta.last_season}. What ` +
        `follows is a counterfactual — what the model expected from his state at ` +
        `that point — not a forecast of anything that will happen.`;
      c.appendChild(b);
    }
    return c;
  }

  /* ---- 2-3. fan chart + stat selector ---------------------------------- */
  function chartCard() {
    const c = document.createElement('div');
    c.className = 'card';
    c.innerHTML = '<h2>Career projection</h2>';

    const controls = document.createElement('div');
    controls.className = 'controls';

    const sel = document.createElement('select');
    const available = INDEX.stats.filter(s => state.payload.seasons[s.key]);
    for (const s of available) {
      const o = document.createElement('option');
      o.value = s.key; o.textContent = s.label;
      if (s.key === state.stat) o.selected = true;
      sel.appendChild(o);
    }
    sel.addEventListener('change', e => { state.stat = e.target.value; render(); });

    const seg = document.createElement('span');
    seg.className = 'seg';
    for (const lv of [50, 80, 95]) {
      const b = document.createElement('button');
      b.textContent = `${lv}%`;
      if (lv === state.level) b.className = 'on';
      b.addEventListener('click', () => { state.level = lv; render(); });
      seg.appendChild(b);
    }

    const l1 = document.createElement('label'); l1.className = 'ctl'; l1.textContent = 'Stat';
    const l2 = document.createElement('label'); l2.className = 'ctl'; l2.textContent = 'Interval';
    controls.append(l1, sel, l2, seg);
    c.appendChild(controls);

    const chart = document.createElement('div');
    c.appendChild(chart);
    FanChart.render(chart, {
      proj: state.payload.seasons[state.stat],
      history: state.payload.meta.history || state.player.history || [],
      statKey: state.stat,
      statLabel: STAT_LABEL(state.stat),
      level: state.level,
    });

    const note = document.createElement('div');
    note.className = 'note';
    note.textContent = state.stat.startsWith('state:')
      ? 'State dimensions are in model units — log rate per possession for volume, ' +
        'logit for accuracy. They are what the filter actually tracks; the derived ' +
        'stats above are read off simulated box scores.'
      : 'Projected seasons are conditional on the player still being in the league ' +
        'that year. Longevity is shown separately below.';
    c.appendChild(note);
    return c;
  }

  /* ---- 3b. season-by-season per-game line ------------------------------
   *
   * The familiar box score line, one row per projected season, conditional on
   * the player being in the league that year — "if he plays, this is what the
   * season looks like". The p_active column is what tells you how likely that
   * condition is; the two must be read together, and putting them in the same
   * row is the whole reason this is a table and not five more fan charts.
   *
   * Still no naked point estimates (§8.4): the median is the primary figure and
   * the 80% band sits under it in every cell.
   */
  function perGameCard() {
    const c = document.createElement('div');
    c.className = 'card';
    c.innerHTML = '<h2>Season by season, if he plays</h2>';

    const keys = (state.payload.per_game_table || [])
      .filter(k => state.payload.seasons[k]);
    if (!keys.length) {
      c.innerHTML += '<div class="empty">No per-game projection stored.</div>';
      return c;
    }
    const SHORT = { minutes_per_game: 'MPG', pts_per_game: 'PTS', reb_per_game: 'REB',
      ast_per_game: 'AST', stl_per_game: 'STL', blk_per_game: 'BLK', games: 'GP' };
    const DP = { games: 0, minutes_per_game: 1 };

    const base = state.payload.seasons[keys[0]];
    const surv = state.payload.survival;
    /* The per-season column is P(plays that season) = career-active x shows-up
     * (deviation #4). It is strictly below the longevity card's P(still in the
     * league), by the chance of a mid-career gap. Older stores without p_play
     * fall back to p_active. */
    const pPlay = surv.p_play || surv.p_active;
    const pActive = {};
    surv.season_year.forEach((y, i) => { pActive[y] = pPlay[i]; });

    const t = document.createElement('table');
    t.innerHTML = '<thead><tr><th>Season</th><th>Age</th>' +
      keys.map(k => `<th>${SHORT[k] || k}</th>`).join('') +
      '<th>P(plays)</th></tr></thead>';
    const tb = document.createElement('tbody');

    /* Observed seasons first, in a visually distinct treatment — same rule as
       the fan chart's history/projection boundary (§8.3). */
    for (const h of (state.payload.meta.history || []).slice(-4)) {
      const tr = document.createElement('tr');
      tr.style.color = 'var(--hist)';
      tr.innerHTML = `<td>${h.season_year} <span style="opacity:.7">actual</span></td>` +
        `<td>${fmt(h.age, 0)}</td>` +
        keys.map(k => `<td>${h[k] == null ? '–' : fmt(h[k], DP[k] ?? 1)}</td>`).join('') +
        '<td>—</td>';
      tb.appendChild(tr);
    }
    if ((state.payload.meta.history || []).length) {
      const sep = document.createElement('tr');
      sep.innerHTML = `<td colspan="${keys.length + 3}" style="border-bottom:2px solid var(--proj);padding:0"></td>`;
      tb.appendChild(sep);
    }

    base.season_year.forEach((yr, i) => {
      const tr = document.createElement('tr');
      let cells = `<td>${yr}</td><td>${fmt(base.age[i], 0)}</td>`;
      for (const k of keys) {
        const s = state.payload.seasons[k];
        const dp = DP[k] ?? 1;
        cells += `<td>${fmt(s.p50[i], dp)}` +
          `<div class="band" style="padding:0;border:0;font-size:11px">` +
          `${fmt(s.p10[i], dp)}–${fmt(s.p90[i], dp)}</div></td>`;
      }
      const pa = pActive[yr];
      cells += `<td>${pa == null ? '–' : Math.round(pa * 100) + '%'}</td>`;
      tr.innerHTML = cells;
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    c.appendChild(t);

    const n = document.createElement('div');
    n.className = 'note';
    n.innerHTML =
      'Median with the 80% band beneath it, <b>conditional on him actually ' +
      'playing that season</b> — the last column, <b>P(plays)</b>, gives the ' +
      'probability of that. It is the chance his career is still going ' +
      '(the longevity curve below) times the chance he does not miss the whole ' +
      'year to injury, so it sits a little under the longevity number. ' +
      'The model projects <i>possessions</i>, not games; the split into games ' +
      'and minutes per game is a fitted derivation (R² 0.78 off possessions and ' +
      'age, plus the player\'s own durability tendency), so these rows inherit ' +
      'whatever the availability dimension gets wrong. That dimension is the ' +
      'weakest part of the model, though two corrections now target its worst ' +
      'failure: durable veterans are anchored to their own workload record, and ' +
      'a genuinely elite player\'s minutes decline on a slower, quality-adjusted ' +
      'aging curve rather than the league-average one — the reason an aging ' +
      'star\'s minutes used to collapse while his per-100 rates held.';
    c.appendChild(n);
    return c;
  }

  /* ---- 4. peak season -------------------------------------------------- */
  function peakCard() {
    const c = document.createElement('div');
    c.className = 'card';
    c.innerHTML = '<h2>Peak season</h2>';
    const rows = state.payload.peak || [];
    if (!rows.length) {
      c.innerHTML += '<div class="empty">Not enough surviving draws to describe a peak.</div>';
      return c;
    }
    const t = document.createElement('table');
    t.innerHTML = '<thead><tr><th>Stat</th><th>Peak value (median)</th>' +
      '<th>80% interval</th><th>Peak age (median)</th><th>80% interval</th></tr></thead>';
    const tb = document.createElement('tbody');
    for (const r of rows) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${STAT_LABEL(r.stat)}</td>` +
        `<td>${fmt(r.value_p50)}</td>` +
        `<td class="band">${fmt(r.value_p5)} – ${fmt(r.value_p95)}</td>` +
        `<td>${fmt(r.age_p50, 0)}</td>` +
        `<td class="band">${fmt(r.age_p5, 0)} – ${fmt(r.age_p95, 0)}</td>`;
      tb.appendChild(tr);
    }
    t.appendChild(tb);
    c.appendChild(t);

    /* §8.3.4 asks for the distribution, not a summary of it. The store keeps
       percentiles rather than draws, so the shape is drawn as a stepped density
       between stored quantiles — honest about its own resolution (five steps),
       and enough to show that peak age is broad and right-skewed rather than
       the crisp number the table's median column invites. */
    const pcts = [5, 25, 50, 75, 95];
    const headline = rows.find(r => r.stat === 'pts_per100') || rows[0];
    if (headline && headline.value_p5 != null) {
      const wrap = document.createElement('div');
      wrap.style.display = 'flex';
      wrap.style.gap = '28px';
      wrap.style.flexWrap = 'wrap';
      wrap.style.marginTop = '14px';
      for (const [keys, label] of [
        [pcts.map(q => headline[`value_p${q}`]), `Peak ${STAT_LABEL(headline.stat)} — distribution`],
        [pcts.map(q => headline[`age_p${q}`]), 'Age at peak — distribution'],
      ]) {
        if (keys.some(v => v == null)) continue;
        const box = document.createElement('div');
        box.style.flex = '1 1 320px';
        FanChart.percentileHistogram(box, pcts, keys, label);
        wrap.appendChild(box);
      }
      c.appendChild(wrap);
    }
    const n = document.createElement('div');
    n.className = 'note';
    n.textContent = 'The best single season in each simulated career, and the age it ' +
      'lands at. Read the age band, not the median — peak age is genuinely uncertain ' +
      'and a single number here would be the most misleading figure on the page.';
    c.appendChild(n);
    return c;
  }

  /* ---- 5. career totals ------------------------------------------------ */
  function totalsCard() {
    const c = document.createElement('div');
    c.className = 'card';
    c.innerHTML = '<h2>Rest-of-career totals</h2>';
    const ORDER = ['seasons', 'minutes', 'possessions', 'pts', 'reb', 'ast',
      'stl', 'blk', 'fga_3p', 'vorp'];
    const KEEP = { seasons: 'Seasons played', pts: 'Points', reb: 'Rebounds',
      ast: 'Assists', stl: 'Steals', blk: 'Blocks', possessions: 'Possessions',
      minutes: 'Minutes', vorp: 'VORP (derived)', fga_3p: 'Three-point attempts' };
    const rows = (state.payload.totals || [])
      .filter(r => KEEP[r.stat])
      .sort((a, b) => ORDER.indexOf(a.stat) - ORDER.indexOf(b.stat));
    const t = document.createElement('table');
    t.innerHTML = '<thead><tr><th>Total</th><th>Median</th><th>50%</th>' +
      '<th>80%</th><th></th></tr></thead>';
    const tb = document.createElement('tbody');
    for (const r of rows) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${KEEP[r.stat]}</td><td>${fmt(r.p50, 0)}</td>` +
        `<td class="band">${fmt(r.p25, 0)} – ${fmt(r.p75, 0)}</td>` +
        `<td class="band">${fmt(r.p10, 0)} – ${fmt(r.p90, 0)}</td><td></td>`;
      const cell = tr.lastElementChild;
      cell.appendChild(FanChart.distBar(r, r.p5, r.p95,
        { p5: 'p5', p10: 'p10', p50: 'p50', p90: 'p90', p95: 'p95' }));
      tb.appendChild(tr);
    }
    t.appendChild(tb);
    c.appendChild(t);
    const n = document.createElement('div');
    n.className = 'note';
    n.textContent = 'Cumulative over every simulated season the player survives to ' +
      'play, so the hazard model is doing visible work here: draws where he is out ' +
      'of the league in three years simply stop contributing. These distributions ' +
      'are heavily right-skewed — the mean is well above the median and is not shown, ' +
      'because it describes almost nobody.';
    c.appendChild(n);
    return c;
  }

  /* ---- 6. longevity ---------------------------------------------------- */
  function longevityCard() {
    const c = document.createElement('div');
    c.className = 'card';
    c.innerHTML = '<h2>Longevity</h2>';
    const s = state.payload.survival;
    const box = document.createElement('div');
    const W = 860, H = 200, M = { t: 14, r: 20, b: 34, l: 46 };
    const NS = 'http://www.w3.org/2000/svg';
    const el = (tag, a, txt) => {
      const n = document.createElementNS(NS, tag);
      for (const k in a) n.setAttribute(k, a[k]);
      if (txt != null) n.textContent = txt;
      return n;
    };
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}` });
    const xMin = s.season_year[0], xMax = s.season_year[s.season_year.length - 1];
    const X = v => M.l + (v - xMin) / ((xMax - xMin) || 1) * (W - M.l - M.r);
    const Y = v => H - M.b - v * (H - M.t - M.b);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      svg.appendChild(el('line', { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t),
        stroke: 'currentColor', 'stroke-opacity': 0.10 }));
      svg.appendChild(el('text', { x: M.l - 8, y: Y(t) + 4, 'text-anchor': 'end' },
        `${Math.round(t * 100)}%`));
    }
    let d = '', dArea = `M${X(xMin)},${Y(0)}`;
    s.season_year.forEach((yr, i) => {
      d += `${i ? 'L' : 'M'}${X(yr)},${Y(s.p_active[i])}`;
      dArea += `L${X(yr)},${Y(s.p_active[i])}`;
      svg.appendChild(el('text', { x: X(yr), y: H - M.b + 16, 'text-anchor': 'middle',
        'font-size': 10 }, `'${String(yr).slice(2)}`));
    });
    dArea += `L${X(xMax)},${Y(0)}Z`;
    svg.appendChild(el('path', { d: dArea, fill: 'var(--band-in)' }));
    svg.appendChild(el('path', { d, fill: 'none', stroke: 'var(--proj)', 'stroke-width': 2.4 }));
    box.appendChild(svg);
    c.appendChild(box);

    const half = s.p_active.findIndex(p => p < 0.5);
    const n = document.createElement('div');
    n.className = 'note';
    n.textContent = 'P(still in the league) — his career being active at all, from the ' +
      'hazard sub-model, coupled to the filtered state rather than to raw box score ' +
      'lines so a bad small-sample season does not inflate the exit probability by ' +
      'noise the filter already discounted. This is a career statement and sits ' +
      'above the season table’s P(plays), which also subtracts the chance of missing ' +
      'a single year to injury.' +
      (half > 0 ? ` The model puts even odds on him being out of the league by ${s.season_year[half]} (age ${s.age[half]}).` : '');
    c.appendChild(n);
    return c;
  }

  const fmt = (v, dp = 1) =>
    (v == null || !isFinite(v)) ? '–'
      : Math.abs(v) >= 10000 ? Math.round(v).toLocaleString()
      : v.toFixed(dp);

  return { init, show };
})();
