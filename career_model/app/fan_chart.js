/* The career fan chart (§8.3.2, §8.4).
 *
 * Observed seasons on the left, projected seasons on the right, and a boundary
 * between them that is impossible to miss. That separation is the whole job of
 * this chart: history is a measurement and the projection is a belief, and a
 * reader who cannot tell which half they are looking at has been misled by the
 * drawing rather than informed by it.
 *
 * A note on band width with horizon. The state is a *mean-reverting* AR(1)
 * (A < 1), so its variance saturates toward a stationary bound rather than
 * growing without limit — the correct h-step covariance is
 * P_h = A^h P_0 A^h' + Σ A^j Q A^j', not P_0 + hQ. A displayed band can
 * therefore legitimately stop widening, or even narrow while a player's level
 * drifts down. There is deliberately no rule here flagging a non-widening band
 * as a bug: covariance correctness is asserted where it belongs, in
 * `tests/test_simulation.py::test_covariance_propagation`, not by eyeballing
 * the chart.
 */
const FanChart = (function () {
  const W = 860, H = 380, M = { t: 18, r: 20, b: 44, l: 60 };
  const NS = 'http://www.w3.org/2000/svg';

  const el = (tag, attrs, text) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  };

  const BANDS = {
    50: ['p25', 'p75'],
    80: ['p10', 'p90'],
    95: ['p5', 'p95'],
  };

  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1;
    const step0 = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0) || mag * 10;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
    return out;
  }

  /* There is intentionally no band-width warning here.  The state mean-reverts
   * (A < 1), so its variance saturates toward a stationary bound and the h-step
   * covariance is P_h = A^h P_0 A^h' + Σ A^j Q A^j', not P_0 + hQ.  A band that
   * stops widening -- or narrows as a player's level drifts down -- is correct,
   * not a bug.  Covariance correctness is asserted in
   * tests/test_simulation.py::test_covariance_propagation. */

  function render(node, { proj, history, statKey, statLabel, level }) {
    node.innerHTML = '';
    if (!proj || !proj.p50 || !proj.p50.length) {
      node.innerHTML = '<div class="empty">No projection for this stat.</div>';
      return;
    }
    const [loKey, hiKey] = BANDS[level] || BANDS[80];

    const hist = (history || [])
      .filter(h => h[statKey] != null && isFinite(h[statKey]))
      .map(h => ({ x: h.season_year, y: h[statKey], age: h.age }));

    const px = proj.season_year;
    const xs = hist.map(h => h.x).concat(px);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);

    const ys = [];
    for (const h of hist) ys.push(h.y);
    for (let i = 0; i < px.length; i++) {
      ys.push(proj[loKey][i], proj[hiKey][i], proj.p50[i]);
    }
    const finite = ys.filter(v => v != null && isFinite(v));
    let yMin = Math.min(...finite), yMax = Math.max(...finite);
    const pad = (yMax - yMin || 1) * 0.10;
    yMin -= pad; yMax += pad;

    const X = v => M.l + (v - xMin) / ((xMax - xMin) || 1) * (W - M.l - M.r);
    const Y = v => H - M.b - (v - yMin) / ((yMax - yMin) || 1) * (H - M.t - M.b);

    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
      'aria-label': `${statLabel}: observed seasons then projected distribution` });

    /* grid + axes */
    for (const t of niceTicks(yMin, yMax, 5)) {
      svg.appendChild(el('line', { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t),
        stroke: 'currentColor', 'stroke-opacity': 0.10 }));
      svg.appendChild(el('text', { x: M.l - 8, y: Y(t) + 4, 'text-anchor': 'end' },
        Math.abs(t) < 1 ? t.toFixed(2) : t.toFixed(1)));
    }
    const yearStep = Math.max(1, Math.round((xMax - xMin) / 9));
    for (let v = Math.ceil(xMin); v <= xMax; v += yearStep) {
      svg.appendChild(el('text', { x: X(v), y: H - M.b + 18, 'text-anchor': 'middle' },
        `'${String(v).slice(2)}`));
    }
    svg.appendChild(el('text', {
      x: M.l - 46, y: M.t + (H - M.t - M.b) / 2, 'text-anchor': 'middle',
      transform: `rotate(-90 ${M.l - 46} ${M.t + (H - M.t - M.b) / 2})`,
    }, statLabel));

    /* the history / projection boundary */
    const boundary = px[0] - 0.5;
    if (hist.length) {
      svg.appendChild(el('rect', {
        x: M.l, y: M.t, width: Math.max(0, X(boundary) - M.l), height: H - M.t - M.b,
        fill: 'currentColor', 'fill-opacity': 0.035,
      }));
      svg.appendChild(el('line', {
        x1: X(boundary), x2: X(boundary), y1: M.t, y2: H - M.b,
        stroke: 'currentColor', 'stroke-opacity': 0.45, 'stroke-dasharray': '4 4',
      }));
      svg.appendChild(el('text', { x: X(boundary) - 7, y: M.t + 12,
        'text-anchor': 'end', 'font-size': 10 }, 'observed'));
      svg.appendChild(el('text', { x: X(boundary) + 7, y: M.t + 12,
        'font-size': 10 }, 'projected'));
    }

    /* bands: always draw the wider 95 faintly behind the selected one, so the
       tail is visible without the selected band losing its emphasis */
    const area = (lo, hi) => {
      let d = '';
      for (let i = 0; i < px.length; i++) d += `${i ? 'L' : 'M'}${X(px[i])},${Y(lo[i])}`;
      for (let i = px.length - 1; i >= 0; i--) d += `L${X(px[i])},${Y(hi[i])}`;
      return d + 'Z';
    };
    if (level !== 95) {
      svg.appendChild(el('path', { d: area(proj.p5, proj.p95),
        fill: 'var(--band-out)', stroke: 'none' }));
    }
    svg.appendChild(el('path', { d: area(proj[loKey], proj[hiKey]),
      fill: 'var(--band-in)', stroke: 'none' }));

    /* median */
    let dm = '';
    for (let i = 0; i < px.length; i++) dm += `${i ? 'L' : 'M'}${X(px[i])},${Y(proj.p50[i])}`;
    svg.appendChild(el('path', { d: dm, fill: 'none', stroke: 'var(--proj)',
      'stroke-width': 2.4 }));
    for (let i = 0; i < px.length; i++) {
      svg.appendChild(el('circle', { cx: X(px[i]), cy: Y(proj.p50[i]), r: 3,
        fill: 'var(--proj)' }));
    }

    /* observed seasons — deliberately a different mark, not just a colour */
    if (hist.length) {
      let dh = '';
      for (let i = 0; i < hist.length; i++) {
        dh += `${i ? 'L' : 'M'}${X(hist[i].x)},${Y(hist[i].y)}`;
      }
      svg.appendChild(el('path', { d: dh, fill: 'none', stroke: 'var(--hist)',
        'stroke-width': 1.6 }));
      for (const h of hist) {
        svg.appendChild(el('rect', { x: X(h.x) - 3.2, y: Y(h.y) - 3.2,
          width: 6.4, height: 6.4, fill: 'var(--hist)' }));
      }
    }

    /* hover readout */
    const tip = el('g', { visibility: 'hidden' });
    const tipRect = el('rect', { rx: 4, fill: 'var(--panel)', stroke: 'var(--line)' });
    const tipText = el('text', { x: 0, y: 0, 'font-size': 11 });
    tip.appendChild(tipRect); tip.appendChild(tipText);
    const hit = el('rect', { x: M.l, y: M.t, width: W - M.l - M.r,
      height: H - M.t - M.b, fill: 'transparent' });
    hit.addEventListener('mousemove', ev => {
      const bb = svg.getBoundingClientRect();
      const xv = xMin + (ev.clientX - bb.left) / bb.width * W;
      let best = -1, bd = Infinity;
      px.forEach((v, i) => { const d = Math.abs(X(v) - xv); if (d < bd) { bd = d; best = i; } });
      if (best < 0) return;
      tipText.textContent =
        `${px[best]} · median ${proj.p50[best].toFixed(1)} · ` +
        `${level}% ${proj[loKey][best].toFixed(1)}–${proj[hiKey][best].toFixed(1)}`;
      const w = tipText.getComputedTextLength() + 14;
      const tx = Math.min(Math.max(X(px[best]) - w / 2, M.l), W - M.r - w);
      tipText.setAttribute('x', tx + 7);
      tipText.setAttribute('y', M.t + 30);
      tipRect.setAttribute('x', tx); tipRect.setAttribute('y', M.t + 17);
      tipRect.setAttribute('width', w); tipRect.setAttribute('height', 18);
      tip.setAttribute('visibility', 'visible');
    });
    hit.addEventListener('mouseleave', () => tip.setAttribute('visibility', 'hidden'));
    svg.appendChild(hit);
    svg.appendChild(tip);

    node.appendChild(svg);

    const legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML =
      `<span><span class="swatch" style="background:var(--hist)"></span>observed season</span>` +
      `<span><span class="swatch" style="background:var(--proj)"></span>projected median</span>` +
      `<span><span class="swatch" style="background:var(--band-in)"></span>${level}% interval</span>` +
      (level !== 95 ? `<span><span class="swatch" style="background:var(--band-out)"></span>95% interval</span>` : '');
    node.appendChild(legend);
  }

  /* Horizontal distribution bar: median dot plus nested bands. Used wherever a
     single number would otherwise be tempting (§8.4, no naked point estimates). */
  function distBar(row, lo, hi, keys) {
    const w = 190, h = 16;
    const X = v => (v - lo) / ((hi - lo) || 1) * w;
    const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, width: w, height: h });
    svg.appendChild(el('line', { x1: X(row[keys.p5]), x2: X(row[keys.p95]),
      y1: h / 2, y2: h / 2, stroke: 'var(--band-out)', 'stroke-width': 8 }));
    svg.appendChild(el('line', { x1: X(row[keys.p10]), x2: X(row[keys.p90]),
      y1: h / 2, y2: h / 2, stroke: 'var(--band-in)', 'stroke-width': 8 }));
    svg.appendChild(el('circle', { cx: X(row[keys.p50]), cy: h / 2, r: 3.4,
      fill: 'var(--proj)' }));
    return svg;
  }

  /* Histogram for the peak-season and career-total distributions (§8.3.4-5),
     drawn from stored percentiles as a stepped density. */
  function percentileHistogram(node, pcts, values, label) {
    node.innerHTML = '';
    const w = 400, h = 120, m = { t: 8, r: 10, b: 24, l: 10 };
    const lo = values[0], hi = values[values.length - 1];
    const X = v => m.l + (v - lo) / ((hi - lo) || 1) * (w - m.l - m.r);
    const svg = el('svg', { viewBox: `0 0 ${w} ${h}` });
    let maxD = 0;
    const cells = [];
    for (let i = 0; i < values.length - 1; i++) {
      const dx = values[i + 1] - values[i];
      const dp = (pcts[i + 1] - pcts[i]) / 100;
      const d = dx > 0 ? dp / dx : 0;
      maxD = Math.max(maxD, d);
      cells.push([values[i], values[i + 1], d]);
    }
    for (const [a, b, d] of cells) {
      const y = m.t + (1 - d / (maxD || 1)) * (h - m.t - m.b);
      svg.appendChild(el('rect', { x: X(a), y, width: Math.max(1, X(b) - X(a)),
        height: h - m.b - y, fill: 'var(--band-in)' }));
    }
    for (const v of [lo, values[Math.floor(values.length / 2)], hi]) {
      svg.appendChild(el('text', { x: X(v), y: h - 6, 'text-anchor': 'middle',
        'font-size': 10 }, v.toFixed(1)));
    }
    node.appendChild(svg);
    if (label) {
      const d = document.createElement('div');
      d.className = 'note';
      d.textContent = label;
      node.appendChild(d);
    }
  }

  return { render, distBar, percentileHistogram, BANDS };
})();
