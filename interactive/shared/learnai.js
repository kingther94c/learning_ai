/* learnai — shared runtime for the interactive pages.
 *
 * Three things live here:
 *   LA   tiny linear algebra + softmax, so pages COMPUTE their numbers live
 *        rather than animating a canned recording. Drag a slider and the matmul
 *        actually re-runs.
 *   UI   theme toggle, tooltip, sliders, data-table fallback
 *   Viz  heatmap and bar/line primitives built from plain DOM + SVG
 *
 * No build step, no dependencies, no network. Every page works from file://.
 */

// ============================================================================
// LA — the math, written to match learnai/nn/functional.py line for line
// ============================================================================

const LA = {
  /** Numerically stable softmax over a 1-D array. Subtract the max first —
   *  see the long-form explanation in learnai/nn/functional.py. */
  softmax(v, temperature = 1) {
    const t = Math.max(temperature, 1e-8);
    const scaled = v.map((x) => x / t);
    const max = Math.max(...scaled);
    const e = scaled.map((x) => Math.exp(x - max));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / sum);
  },

  /** Row-wise softmax of a 2-D array. */
  softmaxRows(M, temperature = 1) {
    return M.map((row) => LA.softmax(row, temperature));
  },

  dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  },

  /** (n x k) @ (k x m) -> (n x m) */
  matmul(A, B) {
    const n = A.length, k = B.length, m = B[0].length;
    const C = Array.from({ length: n }, () => new Array(m).fill(0));
    for (let i = 0; i < n; i++) {
      for (let p = 0; p < k; p++) {
        const a = A[i][p];
        if (a === 0) continue;
        for (let j = 0; j < m; j++) C[i][j] += a * B[p][j];
      }
    }
    return C;
  },

  transpose(A) {
    if (!A.length) return [];
    return A[0].map((_, j) => A.map((row) => row[j]));
  },

  scale(A, s) {
    return A.map((row) => row.map((x) => x * s));
  },

  /** The whole mechanism: softmax(QK^T / sqrt(d)) V.
   *  Returns { scores, weights, out } so a page can show all three stages. */
  attention(Q, K, V, { causal = false, temperature = 1, scale = null } = {}) {
    const d = Q[0].length;
    const s = scale === null ? 1 / Math.sqrt(d) : scale;
    let scores = LA.scale(LA.matmul(Q, LA.transpose(K)), s);
    if (causal) {
      scores = scores.map((row, i) => row.map((x, j) => (j > i ? -Infinity : x)));
    }
    const weights = LA.softmaxRows(scores, temperature);
    return { scores, weights, out: LA.matmul(weights, V) };
  },

  /** Shannon entropy in nats — how spread out a distribution is.
   *  0 = all mass on one item; log(n) = perfectly uniform. */
  entropy(p) {
    return -p.reduce((acc, x) => acc + (x > 1e-12 ? x * Math.log(x) : 0), 0);
  },

  cosine(a, b) {
    const na = Math.sqrt(LA.dot(a, a)), nb = Math.sqrt(LA.dot(b, b));
    return LA.dot(a, b) / (na * nb + 1e-12);
  },

  /** Deterministic pseudo-random normals — same "random" projection on reload,
   *  so a page you re-open looks like the one you left. */
  seededNormals(n, seed = 42) {
    let s = seed >>> 0;
    const rand = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
    const out = [];
    for (let i = 0; i < n; i++) {
      // Box-Muller
      const u = Math.max(rand(), 1e-9), v = rand();
      out.push(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v));
    }
    return out;
  },

  seededMatrix(rows, cols, seed = 42, scale = 1) {
    const flat = LA.seededNormals(rows * cols, seed);
    return Array.from({ length: rows }, (_, i) =>
      flat.slice(i * cols, (i + 1) * cols).map((x) => x * scale)
    );
  },

  /** Sample an index from a probability array. */
  sample(p, rand = Math.random) {
    let r = rand(), acc = 0;
    for (let i = 0; i < p.length; i++) {
      acc += p[i];
      if (r <= acc) return i;
    }
    return p.length - 1;
  },
};

// ============================================================================
// Viz — colors and marks
// ============================================================================

const Viz = {
  css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  },

  series(i) {
    return Viz.css(`--series-${(i % 8) + 1}`);
  },

  /** Sequential blue ramp, t in [0,1]. Interpolated in sRGB across the
   *  documented steps — the same ramp learnai/viz/theme.py uses. */
  seq(t) {
    const stops = ["--seq-100","--seq-200","--seq-300","--seq-400","--seq-500","--seq-600","--seq-700"]
      .map((n) => Viz.css(n));
    const c = Math.max(0, Math.min(1, t)) * (stops.length - 1);
    const i = Math.floor(c), f = c - i;
    if (i >= stops.length - 1) return stops[stops.length - 1];
    return Viz.mix(stops[i], stops[i + 1], f);
  },

  hexToRgb(hex) {
    const h = hex.replace("#", "");
    const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
    return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16));
  },

  mix(a, b, t) {
    const [r1, g1, b1] = Viz.hexToRgb(a), [r2, g2, b2] = Viz.hexToRgb(b);
    const m = (x, y) => Math.round(x + (y - x) * t);
    return `rgb(${m(r1, r2)}, ${m(g1, g2)}, ${m(b1, b2)})`;
  },

  /** Ink that stays readable on a sequential-ramp cell of intensity t. */
  inkOn(t) {
    return t > 0.55 ? "#ffffff" : Viz.css("--ink");
  },

  /**
   * Render a matrix as a heatmap of DOM cells.
   * opts: { rowLabels, colLabels, cell, format, title, max, diverging, onHover }
   * Cells carry a 2px surface gap (the spacer rule) so adjacent values stay
   * distinguishable without a border color competing with the data.
   */
  heatmap(container, M, opts = {}) {
    const {
      rowLabels = null, colLabels = null, cell = 30, format = (v) => v.toFixed(2),
      max = null, diverging = false, onHover = null, showValues = "auto",
      title = "Matrix heatmap",
    } = opts;

    const flat = M.flat().filter((v) => isFinite(v));
    const vmax = max !== null ? max : Math.max(...flat.map(Math.abs), 1e-9);
    const showNums = showValues === "auto" ? cell >= 34 : showValues;

    // The 2px surface gap keeps adjacent values distinguishable, but on a dense
    // matrix (small cells) it starts to dominate and the field reads as dots.
    // Shrink it, and drop the corner radius, once cells get small.
    const gap = cell >= 22 ? 2 : cell >= 12 ? 1 : 0;
    const radius = cell >= 22 ? 3 : cell >= 12 ? 2 : 0;

    const el = document.createElement("div");
    el.className = "hm";
    el.style.display = "grid";
    el.style.gap = `${gap}px`;
    el.style.gridTemplateColumns = `${rowLabels ? "auto " : ""}repeat(${M[0].length}, ${cell}px)`;
    el.style.width = "max-content";
    el.style.fontFamily = "var(--mono)";
    el.setAttribute("role", "grid");
    el.setAttribute("aria-label", title);

    if (colLabels) {
      if (rowLabels) el.appendChild(document.createElement("div"));
      colLabels.forEach((c) => {
        const d = document.createElement("div");
        d.textContent = c;
        d.style.cssText =
          `font-size:10px;color:var(--ink-muted);text-align:center;` +
          `writing-mode:vertical-rl;transform:rotate(180deg);height:52px;` +
          `justify-self:center;align-self:end;overflow:hidden;white-space:nowrap`;
        el.appendChild(d);
      });
    }

    M.forEach((row, i) => {
      if (rowLabels) {
        const d = document.createElement("div");
        d.textContent = rowLabels[i];
        d.style.cssText =
          "font-size:10px;color:var(--ink-muted);text-align:right;padding-right:7px;" +
          "align-self:center;max-width:88px;overflow:hidden;white-space:nowrap";
        el.appendChild(d);
      }
      row.forEach((v, j) => {
        const d = document.createElement(onHover ? "button" : "div");
        if (onHover) d.type = "button";
        const finite = isFinite(v);
        let bg, t;
        if (!finite) {
          bg = "var(--grid)"; t = 0;
        } else if (diverging) {
          t = Math.abs(v) / vmax;
          bg = Viz.mix(Viz.css("--surface"), v >= 0 ? Viz.css("--series-1") : Viz.css("--series-8"), t);
        } else {
          t = v / vmax;
          bg = Viz.seq(t);
        }
        d.style.cssText =
          `height:${cell}px;background:${bg};border-radius:${radius}px;` +
          `display:grid;place-items:center;font-size:${Math.min(11, cell / 3)}px;` +
          `color:${finite ? Viz.inkOn(diverging ? 0 : t) : "var(--ink-muted)"};` +
          `cursor:${onHover ? "pointer" : "default"};transition:outline .08s;` +
          `${onHover ? "border:0;padding:0;font-family:inherit;font-weight:400;" : ""}`;
        if (showNums) d.textContent = finite ? format(v) : "·";

        const label =
          `${rowLabels ? rowLabels[i] : "row " + i} → ` +
          `${colLabels ? colLabels[j] : "col " + j}\n${finite ? format(v) : "masked"}`;
        d.setAttribute("role", "gridcell");
        d.setAttribute("aria-label", label.replace("\n", ", "));
        d.addEventListener("mouseenter", (e) => {
          d.style.outline = `2px solid var(--ink)`;
          d.style.outlineOffset = "-1px";
          UI.tip(e, label);
          if (onHover) onHover(i, j, v);
        });
        d.addEventListener("mousemove", (e) => UI.tip(e, label));
        d.addEventListener("mouseleave", () => {
          d.style.outline = "none";
          UI.tipOff();
        });
        if (onHover) {
          d.addEventListener("focus", () => onHover(i, j, v));
          d.addEventListener("click", () => onHover(i, j, v));
        }
        el.appendChild(d);
      });
    });

    container.replaceChildren(el);
    return el;
  },

  /** Horizontal bars — the right form for comparing a handful of magnitudes
   *  (probabilities, logits) where the labels are text and need room. */
  bars(container, items, opts = {}) {
    const {
      max = null, format = (v) => v.toFixed(3), colorFn = null, height = 20,
      title = "Horizontal bar chart",
    } = opts;
    const vmax = max !== null ? max : Math.max(...items.map((d) => Math.abs(d.value)), 1e-9);

    const el = document.createElement("div");
    el.style.cssText = "display:flex;flex-direction:column;gap:4px";
    el.setAttribute("role", "list");
    el.setAttribute("aria-label", title);

    items.forEach((d, i) => {
      const row = document.createElement("div");
      row.style.cssText =
        `display:grid;grid-template-columns:minmax(48px,34%) minmax(48px,1fr) minmax(50px,auto);` +
        `gap:10px;align-items:center;font-family:var(--mono);font-size:12px`;
      row.setAttribute("role", "listitem");
      row.setAttribute("aria-label", `${d.label}: ${format(d.value)}`);

      const lab = document.createElement("div");
      lab.textContent = d.label;
      lab.style.cssText = "color:var(--ink);text-align:right;white-space:pre;overflow:hidden";

      const track = document.createElement("div");
      track.style.cssText = `height:${height}px;position:relative;background:var(--grid);border-radius:4px`;

      const fill = document.createElement("div");
      const magnitude = Math.abs(d.value);
      const w = Math.max((magnitude / vmax) * 100, magnitude > 0 ? 0.8 : 0);
      // 4px rounded data-end, square against the baseline it grows from.
      fill.style.cssText =
        `position:absolute;inset:0 auto 0 0;width:${w}%;border-radius:0 4px 4px 0;` +
        `background:${colorFn ? colorFn(d, i) : "var(--series-1)"};transition:width .18s ease-out`;
      track.appendChild(fill);

      const val = document.createElement("div");
      val.textContent = format(d.value);
      val.style.cssText = "color:var(--ink-2);font-variant-numeric:tabular-nums";

      row.append(lab, track, val);
      el.appendChild(row);
    });

    container.replaceChildren(el);
    return el;
  },

  /** Minimal SVG line chart with a crosshair. series: [{name,color,points:[[x,y]]}] */
  lineChart(container, series, opts = {}) {
    const {
      width = 640, height = 240, pad = { t: 12, r: 14, b: 44, l: 46 },
      xLabel = "", yLabel = "", format = (v) => v.toFixed(3),
      xTicks = null, title = `${yLabel || "value"} by ${xLabel || "x"}`,
    } = opts;

    const xs = series.flatMap((s) => s.points.map((p) => p[0]));
    const ys = series.flatMap((s) => s.points.map((p) => p[1]));
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const yPad = (y1 - y0) * 0.08 || 0.1;
    const sx = (v) => pad.l + ((v - x0) / (x1 - x0 || 1)) * (width - pad.l - pad.r);
    const sy = (v) =>
      height - pad.b - ((v - (y0 - yPad)) / (y1 + yPad - (y0 - yPad) || 1)) * (height - pad.t - pad.b);

    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", title);
    svg.style.cssText = "width:100%;height:auto;display:block;overflow:visible";

    // recessive grid + axis ticks
    for (let i = 0; i <= 4; i++) {
      const v = y0 - yPad + ((y1 + yPad - (y0 - yPad)) * i) / 4;
      const line = document.createElementNS(NS, "line");
      line.setAttribute("x1", pad.l); line.setAttribute("x2", width - pad.r);
      line.setAttribute("y1", sy(v)); line.setAttribute("y2", sy(v));
      line.setAttribute("stroke", "var(--grid)"); line.setAttribute("stroke-width", "1");
      svg.appendChild(line);

      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", pad.l - 7); t.setAttribute("y", sy(v) + 3.5);
      t.setAttribute("text-anchor", "end");
      t.setAttribute("fill", "var(--ink-muted)");
      t.setAttribute("font-size", "10");
      t.setAttribute("font-family", "var(--mono)");
      t.textContent = format(v);
      svg.appendChild(t);
    }

    if (xTicks) {
      xTicks.forEach(([v, label]) => {
        const t = document.createElementNS(NS, "text");
        t.setAttribute("x", sx(v));
        t.setAttribute("y", height - pad.b + 15);
        t.setAttribute("text-anchor", "middle");
        t.setAttribute("fill", "var(--ink-muted)");
        t.setAttribute("font-size", "10");
        t.setAttribute("font-family", "var(--mono)");
        t.textContent = label;
        svg.appendChild(t);
      });
    }

    series.forEach((s, i) => {
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", s.points.map((p, k) => `${k ? "L" : "M"}${sx(p[0])},${sy(p[1])}`).join(" "));
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", s.color || Viz.series(i));
      path.setAttribute("stroke-width", "2");
      path.setAttribute("stroke-linejoin", "round");
      path.setAttribute("stroke-linecap", "round");
      svg.appendChild(path);
    });

    if (xLabel) {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", width / 2); t.setAttribute("y", height - 4);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("fill", "var(--ink-muted)"); t.setAttribute("font-size", "11");
      t.textContent = xLabel;
      svg.appendChild(t);
    }
    if (yLabel) {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("transform", `rotate(-90) translate(${-height / 2} 12)`);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("fill", "var(--ink-muted)"); t.setAttribute("font-size", "11");
      t.textContent = yLabel;
      svg.appendChild(t);
    }

    container.replaceChildren(svg);
    return svg;
  },
};

// ============================================================================
// UI
// ============================================================================

const UI = {
  _tipEl: null,

  tip(evt, text) {
    if (!UI._tipEl) {
      UI._tipEl = document.createElement("div");
      UI._tipEl.id = "tip";
      document.body.appendChild(UI._tipEl);
    }
    const el = UI._tipEl;
    el.textContent = text;
    el.classList.add("on");
    // Flip the tooltip before it runs off the right/bottom edge.
    const pad = 14, w = el.offsetWidth, h = el.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    el.style.left = x + "px";
    el.style.top = y + "px";
  },

  tipOff() {
    if (UI._tipEl) UI._tipEl.classList.remove("on");
  },

  /** Wire a range input to a live value readout and a callback. */
  slider(id, onChange, format = (v) => v) {
    const input = document.getElementById(id);
    const out = document.getElementById(id + "-val");
    const fire = () => {
      const v = parseFloat(input.value);
      if (out) out.textContent = format(v);
      onChange(v);
    };
    input.addEventListener("input", fire);
    fire();
    return input;
  },

  /** A group of buttons acting as a single choice. */
  toggleGroup(containerId, onChange) {
    const box = document.getElementById(containerId);
    const buttons = [...box.querySelectorAll("button")];
    buttons.forEach((b) => {
      b.addEventListener("click", () => {
        buttons.forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
        onChange(b.dataset.value, b);
      });
    });
    const initial = buttons.find((b) => b.getAttribute("aria-pressed") === "true") || buttons[0];
    initial.setAttribute("aria-pressed", "true");
    onChange(initial.dataset.value, initial);
    return buttons;
  },

  /** The table fallback that every chart on this site is required to have. */
  table(container, headers, rows) {
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    const t = document.createElement("table");
    t.className = "data";
    const thead = document.createElement("thead");
    const hr = document.createElement("tr");
    headers.forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    const tbody = document.createElement("tbody");
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      r.forEach((c) => {
        const td = document.createElement("td");
        td.textContent = c;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    t.append(thead, tbody);
    wrap.appendChild(t);
    container.replaceChildren(wrap);
    return wrap;
  },

  legend(container, items) {
    const el = document.createElement("div");
    el.className = "legend";
    items.forEach((it) => {
      const d = document.createElement("div");
      d.className = "item";
      const s = document.createElement("span");
      s.className = "swatch";
      s.style.background = it.color;
      d.append(s, document.createTextNode(it.label));
      el.appendChild(d);
    });
    container.replaceChildren(el);
    return el;
  },

  themeToggle() {
    const btn = document.createElement("button");
    btn.id = "theme-toggle";
    btn.title = "toggle light / dark";
    btn.setAttribute("aria-label", "toggle light or dark theme");
    const stored = localStorage.getItem("learnai-theme");
    if (stored) document.documentElement.setAttribute("data-theme", stored);
    const paint = () => {
      const dark =
        document.documentElement.getAttribute("data-theme") === "dark" ||
        (!document.documentElement.getAttribute("data-theme") &&
          matchMedia("(prefers-color-scheme: dark)").matches);
      btn.textContent = dark ? "☀" : "☾";
    };
    btn.addEventListener("click", () => {
      const dark =
        document.documentElement.getAttribute("data-theme") === "dark" ||
        (!document.documentElement.getAttribute("data-theme") &&
          matchMedia("(prefers-color-scheme: dark)").matches);
      const next = dark ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("learnai-theme", next);
      paint();
      document.dispatchEvent(new CustomEvent("themechange"));
    });
    paint();
    document.body.appendChild(btn);
    return btn;
  },

  /** Optional real-model data exported by a notebook (learnai.export).
   *  Loaded via <script src="data/xxx.js"> so it works over file:// too. */
  data(key) {
    return (window.LEARNAI_DATA || {})[key] || null;
  },
};

document.addEventListener("DOMContentLoaded", () => UI.themeToggle());
