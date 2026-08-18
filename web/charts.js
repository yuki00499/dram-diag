// 极简 SVG 图表：白底、浅灰网格、降饱和色系（无外部依赖）
const CHARCOAL = "#36454F";
const SLATE = "#708090";
const LIGHT_GRAY = "#D3D3D3";
const SEMANTIC_GREEN = "#2E7D32";
const SEMANTIC_RED = "#C0392B";

function fmtNum(v) {
  if (v === null || v === undefined) return "";
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

function prChart(svgEl, curves, names = {}, highlight = []) {
  const W = 640, H = 300, mL = 44, mR = 200, mT = 20, mB = 34;
  const plotW = W - mL - mR, plotH = H - mT - mB;
  const colors = ["#36454F", "#2E7D32", "#C0392B", "#708090", "#8E6C3A", "#477A8A", "#7A5A78"];
  const scaleX = value => mL + value * plotW;
  const scaleY = value => mT + (1 - value) * plotH;
  let svg = "";
  for (let tick = 0; tick <= 4; tick++) {
    const value = tick / 4;
    const x = scaleX(value), y = scaleY(value);
    svg += `<line class="grid-line" x1="${x}" y1="${mT}" x2="${x}" y2="${H - mB}" />`;
    svg += `<line class="grid-line" x1="${mL}" y1="${y}" x2="${W - mR}" y2="${y}" />`;
    svg += `<text class="axis-label" x="${x}" y="${H - 10}" text-anchor="middle">${value.toFixed(1)}</text>`;
    svg += `<text class="axis-label" x="${mL - 8}" y="${y + 3}" text-anchor="end">${value.toFixed(1)}</text>`;
  }
  svg += `<text class="axis-label" x="${mL + plotW / 2}" y="${H - 1}" text-anchor="middle">Recall</text>`;
  svg += `<text class="axis-label" transform="translate(11 ${mT + plotH / 2}) rotate(-90)" text-anchor="middle">Precision</text>`;
  Object.entries(curves || {}).forEach(([id, points], index) => {
    if (!points || !points.length) return;
    const color = colors[index % colors.length];
    const path = points.map(point => `${scaleX(point.recall).toFixed(1)},${scaleY(point.precision).toFixed(1)}`).join(" ");
    svg += `<polyline class="series pr-series" points="${path}" stroke="${color}" />`;
    const label = names[id] || `类型 ${id}`;
    const marker = highlight.includes(Number(id)) ? " · 核心" : "";
    const lx = W - mR + 12, ly = mT + 12 + index * 18;
    svg += `<line x1="${lx}" y1="${ly - 4}" x2="${lx + 18}" y2="${ly - 4}" stroke="${color}" stroke-width="2" />`;
    svg += `<text class="legend-text" x="${lx + 24}" y="${ly}">${escapeXml(label)}${marker}</text>`;
  });
  svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svgEl.innerHTML = svg;
}

function confusionHeatmap(svgEl, rows, names = {}) {
  const W = 640, rowH = 34, H = Math.max(100, 42 + rows.length * rowH);
  const mL = 240, cellW = 76, mT = 28;
  const columns = ["TN", "FP", "FN", "TP"];
  const cellColors = { TN: "#708090", FP: "#C0392B", FN: "#C97832", TP: "#2E7D32" };
  const maxValue = Math.max(1, ...rows.flatMap(row => columns.map(column => Number(row[column.toLowerCase()] || 0))));
  let svg = "";
  columns.forEach((column, index) => {
    svg += `<text class="axis-label" x="${mL + index * cellW + cellW / 2}" y="18" text-anchor="middle">${column}</text>`;
  });
  rows.forEach((row, rowIndex) => {
    const y = mT + rowIndex * rowH;
    const label = names[row.label] || `类型 ${row.label}`;
    svg += `<text class="axis-label" x="${mL - 10}" y="${y + 21}" text-anchor="end">${escapeXml(label)}</text>`;
    columns.forEach((column, columnIndex) => {
      const value = Number(row[column.toLowerCase()] || 0);
      const opacity = (0.12 + 0.78 * value / maxValue).toFixed(2);
      const x = mL + columnIndex * cellW;
      svg += `<rect x="${x + 2}" y="${y + 3}" width="${cellW - 4}" height="${rowH - 6}" rx="2" fill="${cellColors[column]}" opacity="${opacity}" />`;
      svg += `<text class="heat-value" x="${x + cellW / 2}" y="${y + 22}" text-anchor="middle">${value}</text>`;
    });
  });
  svgEl.setAttribute("viewBox", `0 0 ${mL + columns.length * cellW + 8} ${H}`);
  svgEl.innerHTML = svg;
}

// 横向条形图：类别名显示在左侧（英文（中文）长标签）
// values: [{ label, value, highlight }]
function horizontalBarChart(svgEl, values, opts = {}) {
  const W = 760, H = 250, mL = 290, mR = 56, mT = 10, mB = 8;
  const plotW = W - mL - mR, plotH = H - mT - mB;
  const n = values.length;
  const rowH = plotH / n;
  const yMax = Math.max(...values.map((v) => v.value), 1);
  let s = "";
  for (let t = 0; t <= 4; t++) {
    const val = (yMax * t) / 4;
    const xx = mL + (val / yMax) * plotW;
    s += `<line class="grid-line" x1="${xx.toFixed(1)}" y1="${mT}" x2="${xx.toFixed(1)}" y2="${H - mB}" />`;
    s += `<text class="axis-label" x="${xx.toFixed(1)}" y="${H - mB + 12}" text-anchor="middle">${Math.round(val)}</text>`;
  }
  values.forEach((v, i) => {
    const y = mT + i * rowH;
    const bh = Math.max((v.value / yMax) * plotW, 1);
    const cls = v.highlight ? "bar tail" : "bar";
    s += `<rect class="${cls}" x="${mL}" y="${(y + rowH * 0.22).toFixed(1)}" width="${bh.toFixed(1)}" height="${(rowH * 0.56).toFixed(1)}" rx="1" />`;
    s += `<text class="axis-label" x="${(mL - 8).toFixed(1)}" y="${(y + rowH * 0.5 + 3.5).toFixed(1)}" text-anchor="end">${escapeXml(v.label)}</text>`;
    s += `<text class="axis-label" x="${(mL + bh + 6).toFixed(1)}" y="${(y + rowH * 0.5 + 3.5).toFixed(1)}">${Math.round(v.value)}</text>`;
  });
  svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svgEl.innerHTML = s;
}

function escapeXml(value) {
  return String(value).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}
