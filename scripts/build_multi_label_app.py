"""生成缺陷多标签标注站（单图顺序流）并启动本地服务。

用法:
    python scripts/build_multi_label_app.py [--detach]
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dram_diag.data import load_labels

DEFECT_TYPES = [
    {"id": 1, "name": "Round Particle", "zh": "圆形颗粒、异物",
     "hint": "独立的圆形/椭圆形亮点颗粒", "color": "#C0392B"},
    {"id": 2, "name": "Elongated Particle", "zh": "细长颗粒、异物",
     "hint": "细长条状亮点颗粒", "color": "#E67E22"},
    {"id": 3, "name": "Polygonal Particle", "zh": "方形颗粒、异物",
     "hint": "方形、多边形亮点颗粒", "color": "#F1C40F"},
    {"id": 4, "name": "Scratch / Crack", "zh": "划痕、裂纹",
     "hint": "细长亮线/暗线、条带状、长条分叉连续结构", "color": "#27AE60"},
    {"id": 5, "name": "Pit / Void", "zh": "凹坑、空洞",
     "hint": "结构性孔洞、黑色小孔（形状特征明确）", "color": "#16A085"},
    {"id": 6, "name": "Dark spot", "zh": "暗点",
     "hint": "局部暗斑、反差异常（无明确孔洞结构）", "color": "#2980B9"},
    {"id": 7, "name": "Blob", "zh": "块状、块斑类",
     "hint": "方形、多边形、不规则块状亮斑", "color": "#8E44AD"},
    {"id": 8, "name": "Low-signal", "zh": "背景、低信号",
     "hint": "主体是背景纹理/晶界/边缘，无明显缺陷主体（勾选将清空其他选择）", "color": "#7F8C8D"},
    {"id": 9, "name": "Unknown", "zh": "未知",
     "hint": "存在异常但无法归入以上任何类型（勾选将清空其他选择）", "color": "#34495E"},
]
EXCLUSIVE_IDS = {8, 9}

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DRAM 缺陷多标签标注</title>
<style>
:root {
  --bg: #EDEFF2; --panel: #FFFFFF; --line: #D8DCE1; --ink: #1C2128;
  --muted: #6A7280; --blue: #1F5FA8; --blue-strong: #1558B4; --green: #2E7D46;
  --dark: #05070A; --mono: Consolas, "Courier New", monospace;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { background: var(--bg); color: var(--ink); font: 14px/1.5 "Segoe UI", "Microsoft YaHei", sans-serif; }
button { font: inherit; cursor: pointer; }
input { font: inherit; }

.toolbar {
  position: sticky; top: 0; z-index: 20; background: var(--panel);
  border-bottom: 1px solid var(--line); padding: 10px 18px;
  display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
}
.toolbar h1 { font-size: 16px; margin: 0; letter-spacing: .5px; }
.toolbar h1 .dim { color: var(--muted); font-weight: 400; }
.progress-wrap { display: flex; align-items: center; gap: 8px; min-width: 300px; }
.progress-track { width: 220px; height: 6px; background: #E2E6EA; border-radius: 3px; overflow: hidden; }
.progress-fill { height: 100%; background: var(--blue); width: 0%; transition: width .25s ease; }
.progress-text { font: 12px var(--mono); color: var(--muted); }
.btn { border: 1px solid var(--blue); border-radius: 5px; padding: 6px 14px; background: var(--panel); color: var(--blue); }
.btn:hover { background: #EFF5FC; }
.btn.primary { background: var(--blue); color: #fff; }
.btn.primary:hover { background: var(--blue-strong); }
.btn.ghost { border-color: var(--line); color: var(--ink); }
.btn.ghost:hover { background: #F2F4F6; }
#jump { border: 1px solid var(--line); border-radius: 5px; padding: 6px 8px; width: 130px; font: 13px var(--mono); outline: none; }
#jump:focus-visible { border-color: var(--blue); box-shadow: 0 0 0 2px rgba(31,95,168,.25); }

.main { display: flex; gap: 20px; padding: 18px; max-width: 1200px; margin: 0 auto; align-items: flex-start; flex-wrap: wrap; }
.image-pane { flex: 1 1 520px; background: var(--dark); border-radius: 8px; min-height: 420px; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; }
.image-pane img { max-width: 100%; max-height: 520px; object-fit: contain; cursor: zoom-in; }
.image-pane .img-name { position: absolute; left: 10px; bottom: 8px; font: 12px var(--mono); color: #8A949E; background: rgba(5,7,10,.6); padding: 2px 8px; border-radius: 4px; }
.image-pane .img-tags { position: absolute; right: 10px; top: 10px; display: flex; gap: 4px; flex-wrap: wrap; justify-content: flex-end; max-width: 60%; }
.tag-chip { font: 11px var(--mono); color: #fff; padding: 2px 8px; border-radius: 10px; }

.types-pane { flex: 1 1 380px; }
.types-pane h2 { font-size: 14px; margin: 0 0 10px; color: var(--muted); font-weight: 600; }
.type-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.type-btn {
  position: relative; display: flex; align-items: center; gap: 8px;
  border: 2px solid var(--line); border-radius: 8px; background: var(--panel);
  padding: 10px 12px 10px 10px; text-align: left; min-height: 58px;
}
.type-btn:hover { border-color: #B9C1CC; }
.type-btn .swatch { flex: none; width: 14px; height: 14px; border-radius: 3px; }
.type-btn .id { font: 700 13px var(--mono); color: var(--muted); width: 16px; }
.type-btn .names { display: flex; flex-direction: column; line-height: 1.25; }
.type-btn .en { font-weight: 600; font-size: 12.5px; }
.type-btn .zh { font-size: 11.5px; color: var(--muted); }
.type-btn .check { position: absolute; top: 6px; right: 8px; width: 16px; height: 16px; border: 1.5px solid var(--line); border-radius: 4px; font-size: 11px; line-height: 14px; text-align: center; color: #fff; }
.type-btn.selected { border-color: var(--ink); }
.type-btn.selected .check { background: var(--ink); border-color: var(--ink); }
.type-btn[data-id="8"].selected { border-color: #7F8C8D; background: #F4F6F7; }
.type-btn[data-id="9"].selected { border-color: #34495E; background: #ECEFF1; }
.tip { margin-top: 10px; font-size: 12px; color: var(--muted); background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; min-height: 40px; }

.nav { display: flex; gap: 10px; align-items: center; margin-top: 12px; flex-wrap: wrap; }
.keyhint { margin-top: 10px; font-size: 12px; color: var(--muted); }
.keyhint .kbd { font: 11px var(--mono); background: #F2F4F6; border: 1px solid var(--line); border-radius: 3px; padding: 1px 5px; }

.zoom {
  position: fixed; inset: 0; z-index: 50; display: none; background: rgba(5,7,10,.95);
  align-items: center; justify-content: center; flex-direction: column; color: #E8EAED;
}
.zoom.open { display: flex; }
.zoom img { max-width: 96vw; max-height: 88vh; }
.zoom .zoom-name { font: 13px var(--mono); margin-top: 10px; color: #8A949E; }
.zoom-close { position: absolute; top: 12px; right: 20px; background: none; border: none; color: #8A949E; font-size: 26px; }
.zoom-close:hover { color: #fff; }

.toast { position: fixed; left: 50%; bottom: 28px; transform: translateX(-50%) translateY(20px); background: #10151A; color: #E8EAED; padding: 10px 18px; border-radius: 6px; font-size: 13px; opacity: 0; pointer-events: none; transition: opacity .2s ease, transform .2s ease; z-index: 60; border: 1px solid #2A333C; }
.toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
@media (prefers-reduced-motion: reduce) { .progress-fill, .toast { transition: none; } }
</style>
</head>
<body>
<div class="toolbar">
  <h1>DRAM 缺陷多标签标注 <span class="dim" id="total-label"></span></h1>
  <div class="progress-wrap">
    <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>
    <span class="progress-text" id="progress-text"></span>
  </div>
  <input type="text" id="jump" placeholder="跳转：序号或图片名">
  <div style="flex:1"></div>
  <button class="btn ghost" onclick="loadBackup()">导入</button>
  <button class="btn ghost" onclick="clearAll()">清空</button>
  <button class="btn primary" onclick="exportLabels()">导出标注</button>
</div>

<div class="main">
  <div class="image-pane">
    <img id="main-img" alt="">
    <span class="img-name" id="img-name"></span>
    <span class="img-tags" id="img-tags"></span>
  </div>

  <div class="types-pane">
    <h2>该图包含哪些缺陷？（可多选）</h2>
    <div class="type-grid" id="type-grid"></div>
    <div class="tip" id="tip"></div>
    <div class="nav">
      <button class="btn ghost" onclick="prev()">&larr; 上一张</button>
      <button class="btn" onclick="clearSelection()">清空选择</button>
      <button class="btn primary" onclick="next()">下一张 &rarr;</button>
    </div>
    <div class="keyhint">
      <span class="kbd">1-9</span> 勾选缺陷 &nbsp;
      <span class="kbd">Enter</span>/<span class="kbd">&rarr;</span>/<span class="kbd">Space</span> 下一张 &nbsp;
      <span class="kbd">&larr;</span> 上一张 &nbsp;
      <span class="kbd">Esc</span> 取消勾选 &nbsp; 点击图片放大
    </div>
  </div>
</div>

<div class="zoom" id="zoom">
  <button class="zoom-close" onclick="closeZoom()">&times;</button>
  <img id="zoom-img" alt="">
  <div class="zoom-name" id="zoom-name"></div>
</div>

<div class="toast" id="toast"></div>

<script>
"use strict";
const STORAGE_KEY = "dram_multilabel_v1";
const TYPES = [
  {id:1, en:"Round Particle", zh:"圆形颗粒、异物", color:"#C0392B", hint:"独立的圆形/椭圆形亮点颗粒"},
  {id:2, en:"Elongated Particle", zh:"细长颗粒、异物", color:"#E67E22", hint:"细长条状亮点颗粒"},
  {id:3, en:"Polygonal Particle", zh:"方形颗粒、异物", color:"#F1C40F", hint:"方形、多边形亮点颗粒"},
  {id:4, en:"Scratch / Crack", zh:"划痕、裂纹", color:"#27AE60", hint:"细长亮线/暗线、条带状、长条分叉连续结构"},
  {id:5, en:"Pit / Void", zh:"凹坑、空洞", color:"#16A085", hint:"结构性孔洞、黑色小孔（形状特征明确）"},
  {id:6, en:"Dark spot", zh:"暗点", color:"#2980B9", hint:"局部暗斑、反差异常（无明确孔洞结构）"},
  {id:7, en:"Blob", zh:"块状、块斑类", color:"#8E44AD", hint:"方形、多边形、不规则块状亮斑"},
  {id:8, en:"Low-signal", zh:"背景、低信号", color:"#7F8C8D", hint:"主体是背景纹理/晶界/边缘，无明显缺陷主体（勾选将清空其他选择）"},
  {id:9, en:"Unknown", zh:"未知", color:"#34495E", hint:"存在异常但无法归入以上任何类型（勾选将清空其他选择）"}
];
const EXCLUSIVE = {8:true, 9:true};
let DATA = null;
let state = { labels: {}, pos: 0 };
let current = [];
let selectedIds = new Set();

const $ = id => document.getElementById(id);

function imgUrl(name) { return "/" + DATA.image_root + "/" + encodeURIComponent(name); }

function persist() { try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (e) {} }

function restore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) { state = Object.assign({ labels: {}, pos: 0 }, JSON.parse(raw)); }
    if (!state.labels || !Number.isInteger(state.pos)) state = { labels: {}, pos: 0 };
  } catch (e) { state = { labels: {}, pos: 0 }; }
}

function buildGrid() {
  const grid = $("type-grid");
  grid.innerHTML = "";
  TYPES.forEach(t => {
    const btn = document.createElement("button");
    btn.className = "type-btn";
    btn.dataset.id = t.id;
    btn.innerHTML = `<span class="swatch" style="background:${t.color}"></span>
      <span class="id">${t.id}</span>
      <span class="names"><span class="en">${t.en}</span><span class="zh">${t.zh}</span></span>
      <span class="check">&#10003;</span>`;
    btn.addEventListener("mouseenter", () => $("tip").textContent = `类型 ${t.id} ${t.en}：${t.hint}`);
    btn.addEventListener("mouseleave", () => renderTip());
    btn.addEventListener("click", () => toggleType(t.id));
    grid.appendChild(btn);
  });
}

function toggleType(id) {
  if (EXCLUSIVE[id]) {
    if (selectedIds.has(id) && selectedIds.size === 1) selectedIds.delete(id);
    else selectedIds = new Set([id]);
  } else {
    if (selectedIds.has(id)) selectedIds.delete(id);
    else { selectedIds.add(id); selectedIds.delete(8); selectedIds.delete(9); }
  }
  renderSelection();
  renderTags();
  renderTip();
  persistSelection();
}

function clearSelection() { selectedIds = new Set(); renderSelection(); renderTags(); renderTip(); persistSelection(); }

function persistSelection() {
  const name = currentName();
  if (selectedIds.size) state.labels[name] = Array.from(selectedIds).sort((a, b) => a - b);
  else delete state.labels[name];
  persist();
  updateProgress();
}

function currentName() { return DATA.images[state.pos]; }

function renderImage() {
  const name = currentName();
  $("main-img").src = imgUrl(name);
  $("img-name").textContent = `第 ${state.pos + 1} / ${DATA.total} 张 · ${name}`;
  selectedIds = new Set(state.labels[name] || []);
  renderSelection();
  renderTags();
  renderTip();
  updateProgress();
}

function renderSelection() {
  document.querySelectorAll(".type-btn").forEach(btn => {
    btn.classList.toggle("selected", selectedIds.has(Number(btn.dataset.id)));
  });
}

function renderTags() {
  const container = $("img-tags");
  container.innerHTML = "";
  if (!selectedIds.size) { container.textContent = ""; return; }
  selectedIds.forEach(id => {
    const t = TYPES.find(x => x.id === id);
    const chip = document.createElement("span");
    chip.className = "tag-chip";
    chip.style.background = t.color;
    chip.textContent = `${id} ${t.en}`;
    container.appendChild(chip);
  });
}

function renderTip() {
  if (selectedIds.size) {
    const names = Array.from(selectedIds).map(id => {
      const t = TYPES.find(x => x.id === id);
      return `${t.id}.${t.zh}`;
    });
    $("tip").textContent = "当前选择：" + names.join("、");
  } else {
    $("tip").textContent = "未选择任何类型（悬停按钮查看判断要点）";
  }
}

function updateProgress() {
  const labeled = Object.keys(state.labels).length;
  $("progress-fill").style.width = (labeled / DATA.total * 100).toFixed(1) + "%";
  $("progress-text").textContent = `位置 ${state.pos + 1}/${DATA.total} · 已标 ${labeled}/${DATA.total}`;
}

function next() {
  if (state.pos < DATA.total - 1) { state.pos += 1; persist(); renderImage(); }
  else toast("已是最后一张");
}
function prev() {
  if (state.pos > 0) { state.pos -= 1; persist(); renderImage(); }
  else toast("已是第一张");
}

function zoom() {
  const name = currentName();
  $("zoom-img").src = imgUrl(name);
  $("zoom-name").textContent = name;
  $("zoom").classList.add("open");
}
function closeZoom() { $("zoom").classList.remove("open"); }

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove("show"), 2600);
}

function exportLabels() {
  const result = {};
  const unlabeled = [];
  const unknown = [];
  const perType = {};
  TYPES.forEach(t => perType[t.id] = 0);
  DATA.images.forEach(name => {
    const labels = state.labels[name];
    if (labels && labels.length) {
      result[name] = labels;
      labels.forEach(id => perType[id] += 1);
      if (labels.includes(9)) unknown.push(name);
    } else unlabeled.push(name);
  });
  const payload = { version: 2, exported_at: new Date().toISOString(), types: TYPES.map(t => ({id: t.id, name: t.en, zh: t.zh})), labels: result, unlabeled, unknown };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "label_v2.json";
  link.click();
  const dist = TYPES.map(t => `${t.id}=${perType[t.id]}`).join(" ");
  let message = `已导出 ${Object.keys(result).length} 张标注，未标 ${unlabeled.length} 张，未知 ${unknown.length} 张`;
  toast(message);
  alert(`导出完成\n\n已标: ${Object.keys(result).length} / ${DATA.total}\n未标: ${unlabeled.length}\n未知(9): ${unknown.length}\n\n各类型分布:\n${dist}\n\n未标和未知图清单已包含在 label_v2.json 中。`);
}

function loadBackup() {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".json";
  input.onchange = () => {
    const file = input.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = JSON.parse(reader.result);
        const labels = payload.labels || {};
        state.labels = {};
        Object.entries(labels).forEach(([name, value]) => {
          const list = Array.isArray(value) ? value : String(value).split(",").map(v => Number(v.trim())).filter(v => Number.isInteger(v));
          if (list.length) state.labels[name] = list;
        });
        if (Number.isInteger(payload.pos)) state.pos = payload.pos;
        persist();
        renderImage();
        toast("已导入 " + Object.keys(state.labels).length + " 张标注");
      } catch (e) { toast("导入失败：文件格式不正确"); }
    };
    reader.readAsText(file);
  };
  input.click();
}

function clearAll() {
  if (!confirm("清空全部标注？（localStorage 中的数据也会删除）")) return;
  state = { labels: {}, pos: 0 };
  localStorage.removeItem(STORAGE_KEY);
  renderImage();
  toast("已清空");
}

$("main-img").addEventListener("click", zoom);
$("jump").addEventListener("keydown", e => {
  if (e.key === "Enter") {
    const raw = $("jump").value.trim();
    if (/^\\d+$/.test(raw)) {
      const idx = Math.min(Math.max(Number(raw) - 1, 0), DATA.total - 1);
      state.pos = idx;
    } else {
      const match = DATA.images.indexOf(raw);
      if (match >= 0) state.pos = match;
      else { toast("未找到图片: " + raw); return; }
    }
    persist();
    renderImage();
    $("jump").value = "";
  }
});

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if ($("zoom").classList.contains("open")) {
    if (e.key === "Escape") closeZoom();
    return;
  }
  const num = Number(e.key);
  if (num >= 1 && num <= 9) { toggleType(num); return; }
  if (e.key === "Enter" || e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); next(); }
  else if (e.key === "ArrowLeft") { prev(); }
  else if (e.key === "Escape") { clearSelection(); }
});

fetch("data.json").then(r => {
  if (!r.ok) throw new Error("data.json 加载失败");
  return r.json();
}).then(data => {
  DATA = data;
  restore();
  buildGrid();
  $("total-label").textContent = `${data.total} 张`;
  renderImage();
}).catch(err => {
  $("tip").textContent = "加载数据失败：" + err.message + "。请确认通过本地服务打开（127.0.0.1）。";
});
</script>
</body>
</html>
"""


def build_app(data_root, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_labels(data_root)
    image_root = Path(data_root) / "images"
    files = sorted((image_root / row["IMAGE_NAME"]).exists() for row in rows)
    names = [row["IMAGE_NAME"] for row in rows if (image_root / row["IMAGE_NAME"]).exists()]
    names = sorted(names, key=lambda n: int(n.split("_")[1].split(".")[0]))
    if len(files) != len(rows):
        raise ValueError("label.csv 与图片目录不一致")
    payload = {
        "version": 2,
        "image_root": str(image_root).replace("\\", "/"),
        "total": len(names),
        "images": names,
        "types": DEFECT_TYPES,
    }
    (out / "data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    return out, len(names)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def date_time_string(self, timestamp=None):
        try:
            return super().date_time_string(timestamp)
        except (OSError, ValueError):
            return "Thu, 01 Jan 1970 00:00:00 GMT"


def main():
    parser = argparse.ArgumentParser(description="生成并启动缺陷多标签标注站")
    parser.add_argument("--data-root", default="晶圆缺陷分类数据集")
    parser.add_argument("--out-dir", default="artifacts/labeling")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--detach", action="store_true", help="服务后台运行，关闭本窗口不影响")
    parser.add_argument("--serve", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.serve:
        server = ThreadingHTTPServer(("127.0.0.1", args.serve), QuietHandler)
        server.serve_forever()
        return

    try:
        out, total = build_app(args.data_root, args.out_dir)
        port = args.port or free_port()
        if args.detach:
            script = Path(__file__).resolve()
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            subprocess.Popen(
                [sys.executable, str(script), "--serve", str(port), "--no-browser"],
                creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                cwd=str(Path.cwd()),
            )
        else:
            server = ThreadingHTTPServer(("127.0.0.1", port), QuietHandler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        url_path = str(out / "index.html").replace("\\", "/")
        url = f"http://127.0.0.1:{port}/{url_path}"
        print(f"标注站已生成: {out / 'index.html'}（{total} 张，9 个缺陷类型）")
        print(f"本地地址: {url}")
        if args.detach:
            print("服务已在后台运行，关闭本窗口不影响。停止服务: python scripts\\stop_labeling.py")
        else:
            print("本窗口是服务进程，请保持打开，按 Ctrl+C 停止服务。")
        if not args.no_browser:
            try:
                webbrowser.open(url)
            except Exception as exc:
                print(f"自动打开浏览器失败（{exc}），请手动复制地址。")
        if not args.detach:
            server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    except Exception:
        import traceback
        log = Path("artifacts/labeling/server.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"启动失败，详细信息已写入 {log}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
