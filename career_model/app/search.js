/* Typeahead over the corpus in index.json (§8.2).
 *
 * Two things this does that a plain substring match would not.
 *
 * Accents and hyphens are folded, so "Jokic" finds Jokić and "gilgeous
 * alexander" finds Gilgeous-Alexander. Folding happens on both sides of the
 * comparison; the display string keeps its diacritics.
 *
 * Results are segmented by status. An active player, a retired player and an
 * unsigned prospect are different objects — a retired player's "projection" is
 * a counterfactual, and a prospect's is drawn entirely from the prior — and
 * §8.2 is explicit that they must not sit in one undifferentiated list.
 *
 * Every row carries a median with a band, never a bare number (§8.4).
 */
const Search = (function () {
  let players = [], onPick = null, active = -1, shown = [];

  const fold = s => (s || '')
    .normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
    .toLowerCase().replace(/[^a-z0-9]/g, '');

  const GROUPS = [
    { key: 'active',   label: 'Active players' },
    { key: 'inactive', label: 'Missed last season' },
    { key: 'prospect', label: 'Incoming draft class' },
    { key: 'retired',  label: 'Retired' },
  ];

  function score(p, q) {
    const name = p._fold, parts = p._parts;
    if (name.startsWith(q)) return 1000 - name.length;
    for (const part of parts) if (part.startsWith(q)) return 800 - name.length;
    const at = name.indexOf(q);
    if (at >= 0) return 500 - at;
    // The fuzzy fallback is gated at four characters. Below that it is not
    // fuzzy matching, it is noise: "dur" is a subsequence of "Darius Acuff Jr."
    // and every other three-letter query has a dozen accidental hits, which
    // buries the exact match the user was actually typing.
    return q.length >= 4 && subsequence(name, q) ? 100 : -1;
  }

  // Cheap fuzzy fallback: every query character in order. Catches typos and
  // partial spellings without pulling in a scoring library.
  function subsequence(hay, needle) {
    let i = 0;
    for (const ch of hay) if (ch === needle[i] && ++i === needle.length) return true;
    return i === needle.length;
  }

  function summaryHTML(p) {
    const s = p.summary;
    if (!s || s.p50 == null) return '<span class="mini">no projection</span>';
    const fmt = v => (v == null ? '–' : v.toFixed(1));
    return `<div class="mini"><b>${fmt(s.p50)}</b> pts/100` +
           `<br>${fmt(s.p10)}–${fmt(s.p90)} <span style="opacity:.7">(80%)</span></div>`;
  }

  function rowHTML(p, i) {
    const tag = `<span class="tag ${p.status}">${
      p.status === 'prospect' ? 'no NBA data' : p.status}</span>`;
    const where = p.status === 'prospect' ? p.team : (p.team || '—');
    const hist = p.status === 'prospect'
      ? 'projection from college, combine and draft position'
      : `${p.n_history_seasons} season${p.n_history_seasons === 1 ? '' : 's'} of history`;
    return `<div class="row${i === active ? ' active' : ''}" data-i="${i}">
      <div class="who">
        <div class="nm">${p.name}${tag}</div>
        <div class="sub">${p.position} · age ${p.age ?? '–'} · ${where} · ${hist}</div>
      </div>${summaryHTML(p)}</div>`;
  }

  function render(list) {
    const box = document.getElementById('results');
    if (!list.length) {
      box.innerHTML = '<div class="group-label">No match</div>' +
        '<div class="row" style="cursor:default"><div class="who">' +
        '<div class="sub">Insufficient data — this player is not in the corpus, ' +
        'and a maximally diffuse guess would be worse than nothing.</div></div></div>';
      box.classList.add('open');
      shown = [];
      return;
    }
    shown = [];
    let html = '';
    for (const g of GROUPS) {
      const members = list.filter(p => p.status === g.key);
      if (!members.length) continue;
      html += `<div class="group-label">${g.label}</div>`;
      for (const p of members) { html += rowHTML(p, shown.length); shown.push(p); }
    }
    box.innerHTML = html;
    box.classList.add('open');
  }

  function query(text) {
    const q = fold(text);
    if (q.length < 2) { close(); return; }
    const hits = [];
    for (const p of players) {
      const s = score(p, q);
      if (s > 0) hits.push([s, p]);
    }
    hits.sort((a, b) => b[0] - a[0]);
    active = -1;
    render(hits.slice(0, 40).map(h => h[1]));
  }

  function close() {
    document.getElementById('results').classList.remove('open');
    active = -1;
  }

  function init(index, pick) {
    onPick = pick;
    players = index.players.map(p => ({
      ...p, _fold: fold(p.name), _parts: (p.name || '').split(/[\s-]+/).map(fold),
    }));
    const input = document.getElementById('q');
    const box = document.getElementById('results');

    input.addEventListener('input', e => query(e.target.value));
    input.addEventListener('focus', e => { if (e.target.value) query(e.target.value); });
    input.addEventListener('keydown', e => {
      if (e.key === 'Escape') return close();
      if (!shown.length) return;
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        active = (active + (e.key === 'ArrowDown' ? 1 : shown.length - 1)) % shown.length;
        render(shown);
      } else if (e.key === 'Enter' && active >= 0) {
        e.preventDefault();
        choose(shown[active]);
      }
    });
    box.addEventListener('click', e => {
      const row = e.target.closest('.row[data-i]');
      if (row) choose(shown[+row.dataset.i]);
    });
    document.addEventListener('click', e => {
      if (!e.target.closest('.searchbox')) close();
    });
  }

  function choose(p) {
    if (!p) return;
    document.getElementById('q').value = p.name;
    close();
    location.hash = encodeURIComponent(p.player_id);
    onPick(p);
  }

  return { init, query };
})();
