// 自绘 SVG 图表:折线 / 条形 / 置信度圆环(无外部依赖)

function fmtNum(v) {
  if (v === null || v === undefined) return "";
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

// series: [{ name, color, values: [] }]
function lineChart(svgEl, series, opts = {}) {
  const W = 600, H = 220, mL = 46, mR = 14, mT = 18, mB = 24;
  const plotW = W - mL - mR, plotH = H - mT - mB;
  const n = Math.max(...series.map((s) => s.values.length), 1);
  const all = series.flatMap((s) => s.values);
  let yMin = opts.yMin !== undefined ? opts.yMin : Math.min(...all);
  let yMax = opts.yMax !== undefined ? opts.yMax : Math.max(...all);
  if (yMax - yMin < 1e-9) { yMax = yMin + 1; }
  const X = (i) => (n <= 1 ? mL + plotW / 2 : mL + (i / (n - 1)) * plotW);
  const Y = (v) => mT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  let s = "";
  for (let t = 0; t <= 4; t++) {
    const val = yMin + ((yMax - yMin) * t) / 4;
    const yy = Y(val);
    s += `<line class="grid-line" x1="${mL}" y1="${yy.toFixed(1)}" x2="${W - mR}" y2="${yy.toFixed(1)}" />`;
    s += `<text class="axis-label" x="${mL - 8}" y="${(yy + 3).toFixed(1)}" text-anchor="end">${fmtNum(val)}</text>`;
  }
  if (n > 1) {
    s += `<text class="axis-label" x="${X(0).toFixed(1)}" y="${H - 8}" text-anchor="middle">1</text>`;
    s += `<text class="axis-label" x="${X(n - 1).toFixed(1)}" y="${H - 8}" text-anchor="middle">${n}</text>`;
  }
  series.forEach((ser) => {
    const pts = ser.values.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    s += `<polyline class="series" points="${pts}" stroke="${ser.color}" />`;
    ser.values.forEach((v, i) => {
      s += `<circle cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="2.2" fill="${ser.color}" />`;
    });
  });
  let lx = mL;
  series.forEach((ser) => {
    s += `<circle cx="${lx}" cy="${mT - 9}" r="3" fill="${ser.color}" />`;
    s += `<text class="legend-text" x="${lx + 8}" y="${mT - 5}">${ser.name}</text>`;
    lx += 8 + (ser.name.length * 6.5 + 24);
  });
  svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svgEl.innerHTML = s;
}

// values: [{ label, value, highlight }]
function barChart(svgEl, values, opts = {}) {
  const W = 600, H = 220, mL = 40, mR = 10, mT = 12, mB = 26;
  const plotW = W - mL - mR, plotH = H - mT - mB;
  const n = values.length;
  const yMax = Math.max(...values.map((v) => v.value), 1);
  const bw = plotW / n;
  let s = "";
  for (let t = 0; t <= 4; t++) {
    const val = (yMax * t) / 4;
    const yy = mT + plotH - (val / yMax) * plotH;
    s += `<line class="grid-line" x1="${mL}" y1="${yy.toFixed(1)}" x2="${W - mR}" y2="${yy.toFixed(1)}" />`;
    s += `<text class="axis-label" x="${mL - 8}" y="${(yy + 3).toFixed(1)}" text-anchor="end">${Math.round(val)}</text>`;
  }
  values.forEach((v, i) => {
    const bh = (v.value / yMax) * plotH;
    const x = mL + i * bw;
    const cls = v.highlight ? "bar tail" : "bar";
    s += `<rect class="${cls}" x="${(x + bw * 0.18).toFixed(1)}" y="${(mT + plotH - bh).toFixed(1)}" width="${(bw * 0.64).toFixed(1)}" height="${Math.max(bh, 0.5).toFixed(1)}" rx="1" />`;
  });
  svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svgEl.innerHTML = s;
}

// 置信度圆环(0..1)
function ring(percent) {
  const p = Math.max(0, Math.min(1, percent));
  const C = 2 * Math.PI * 42;
  const filled = C * p;
  const color = p >= 0.5 ? "#1B66F5" : "#B45309";
  return `<svg viewBox="0 0 100 100">
    <circle cx="50" cy="50" r="42" fill="none" stroke="#E7EAED" stroke-width="8" />
    <circle cx="50" cy="50" r="42" fill="none" stroke="${color}" stroke-width="8" stroke-linecap="round"
      stroke-dasharray="${C.toFixed(2)}" stroke-dashoffset="${(C - filled).toFixed(2)}" transform="rotate(-90 50 50)" />
    <text x="50" y="47" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="20" font-weight="500" fill="#13161A">${(p * 100).toFixed(1)}%</text>
    <text x="50" y="62" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="9" fill="#6E7681">置信度</text>
  </svg>`;
}
