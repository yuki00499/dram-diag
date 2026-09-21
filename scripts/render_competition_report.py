import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sklearn.metrics import average_precision_score, precision_recall_curve


COLORS = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#0891B2", "#4B5563"]

FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

LABEL_NAMES = {
    1: "圆形颗粒/异物",
    2: "细长颗粒/异物",
    3: "方形颗粒/异物",
    4: "划痕/裂纹",
    5: "凹坑/空洞",
    7: "块状/块斑类",
    8: "背景/低信号",
}

NAVY = (31, 78, 121)
INK = (38, 47, 57)
MUTED = (96, 108, 120)
GRID = (226, 232, 240)
GRID_MAJOR = (203, 213, 225)
AXIS = (100, 116, 139)


def f(size, bold=False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def hex_color(value):
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))


def canvas(title, width=1000, height=640):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 20), title, fill="#111827", font=font)
    return image, draw, font


def render_cooccurrence(audit, destination):
    labels = [str(value) for value in audit["types"]]
    matrix = audit["cooccurrence_matrix"]
    maximum = max(max(row) for row in matrix)
    image, draw, font = canvas("Label co-occurrence counts", 760, 720)
    left, top, cell = 110, 90, 72
    for index, label in enumerate(labels):
        draw.text((left + index * cell + 30, top - 25), label, fill="#111827", font=font)
        draw.text((left - 30, top + index * cell + 30), label, fill="#111827", font=font)
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            intensity = value / max(1, maximum)
            color = (round(239 - 170 * intensity), round(246 - 160 * intensity), round(255 - 40 * intensity))
            box = (left + column * cell, top + row * cell,
                   left + (column + 1) * cell - 2, top + (row + 1) * cell - 2)
            draw.rectangle(box, fill=color)
            draw.text((box[0] + 8, box[1] + 28), str(value), fill="#111827", font=font)
    image.save(destination)


def render_comparison(selection, destination):
    candidates = selection["candidates"]
    image, draw, font = canvas("OOF model comparison", 900, 560)
    left, top, width, height = 100, 80, 720, 390
    draw.line((left, top + height, left + width, top + height), fill="#374151", width=2)
    names = list(candidates)
    group = width / len(names)
    for index, name in enumerate(names):
        core = candidates[name]["core_macro_f1"]
        pair = candidates[name]["mean_pair_f1"]
        x = left + index * group + 35
        for offset, value, color in ((0, core, COLORS[0]), (75, pair, COLORS[1])):
            bar_height = value * height
            draw.rectangle((x + offset, top + height - bar_height, x + offset + 55, top + height), fill=color)
            draw.text((x + offset, top + height - bar_height - 18), f"{value:.3f}", fill="#111827", font=font)
        draw.text((x, top + height + 18), name, fill="#111827", font=font)
    draw.rectangle((620, 25, 635, 40), fill=COLORS[0])
    draw.text((642, 27), "core Macro-F1", fill="#111827", font=font)
    draw.rectangle((735, 25, 750, 40), fill=COLORS[1])
    draw.text((757, 27), "pair F1", fill="#111827", font=font)
    image.save(destination)


def curve_from_scores(targets, scores):
    precision, recall, _ = precision_recall_curve(targets, scores)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "auc": float(average_precision_score(targets, scores)),
        "support": int(sum(targets)),
    }


def test_curves(predictions_path, manifest_path):
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    types = [int(value) for value in manifest["types"]]
    test_labels = {entry["image_name"]: entry["labels"] for entry in manifest["splits"]["test_known"]}
    selected = [row for row in predictions if row["image_name"] in test_labels]
    curves = {}
    for index, label in enumerate(types):
        targets = [1 if label in test_labels[row["image_name"]] else 0 for row in selected]
        scores = [row["probabilities"][str(label)] for row in selected]
        curves[label] = curve_from_scores(targets, scores)
    return curves, len(selected)


def oof_curves(predictions_path, manifest_path):
    records = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    types = [int(value) for value in manifest["types"]]
    curves = {}
    for index, label in enumerate(types):
        targets = [1 if label in row["labels"] else 0 for row in records]
        scores = [row["probabilities"][index] for row in records]
        curves[label] = curve_from_scores(targets, scores)
    return curves, len(records)


def dashed_line(draw, start, end, fill, width=2, dash=14, gap=10):
    x0, y0 = start
    x1, y1 = end
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if length == 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    position = 0.0
    while position < length:
        end_position = min(position + dash, length)
        draw.line((x0 + ux * position, y0 + uy * position,
                   x0 + ux * end_position, y0 + uy * end_position), fill=fill, width=width)
        position += dash + gap


def render_pr_chart(curves, total, destination, title, subtitle, footnote):
    width, height = 1800, 1150
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 38), title, font=f(42, True), fill=NAVY)
    draw.text((72, 100), subtitle, font=f(24), fill=MUTED)

    columns, rows = 4, 2
    left_margin, right_margin = 90, 90
    top, bottom = 190, 1020
    gap_x, gap_y = 60, 85
    cell_w = (width - left_margin - right_margin - gap_x * (columns - 1)) / columns
    cell_h = (bottom - top - gap_y * (rows - 1)) / rows

    for index, (label, curve) in enumerate(curves.items()):
        row, column = divmod(index, columns)
        x0 = left_margin + column * (cell_w + gap_x)
        y0 = top + row * (cell_h + gap_y)
        color = hex_color(COLORS[index % len(COLORS)])
        plot_left = x0 + 52
        plot_top = y0 + 62
        plot_right = x0 + cell_w - 18
        plot_bottom = y0 + cell_h - 40

        draw.text((x0, y0), f"类型 {label} · {LABEL_NAMES[int(label)]}", font=f(24, True), fill=INK)
        draw.text((x0, y0 + 33), f"PR-AUC {curve['auc']:.3f} · n={curve['support']}", font=f(20), fill=MUTED)

        for step in range(3):
            gx = plot_left + (plot_right - plot_left) * step / 2
            gy = plot_bottom - (plot_bottom - plot_top) * step / 2
            draw.line((gx, plot_top, gx, plot_bottom), fill=GRID, width=2)
            draw.line((plot_left, gy, plot_right, gy), fill=GRID, width=2)

        baseline = curve["support"] / max(1, total)
        baseline_y = plot_bottom - baseline * (plot_bottom - plot_top)
        dashed_line(draw, (plot_left, baseline_y), (plot_right, baseline_y), (168, 179, 194), 2)

        points = [(plot_left + recall * (plot_right - plot_left),
                   plot_bottom - precision * (plot_bottom - plot_top))
                  for recall, precision in zip(curve["recall"], curve["precision"])]
        if len(points) >= 2:
            draw.line(points, fill=color, width=4, joint="curve")

        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=AXIS, width=2)
        for step, text in ((0, "0.0"), (1, "0.5"), (2, "1.0")):
            gx = plot_left + (plot_right - plot_left) * step / 2
            gy = plot_bottom - (plot_bottom - plot_top) * step / 2
            draw.text((gx - 16, plot_bottom + 9), text, font=f(18), fill=MUTED)
            draw.text((plot_left - 48, gy - 11), text, font=f(18), fill=MUTED)

    if len(curves) < columns * rows:
        index = len(curves)
        row, column = divmod(index, columns)
        x0 = left_margin + column * (cell_w + gap_x)
        y0 = top + row * (cell_h + gap_y)
        draw.text((x0 + 12, y0 + 6), "图例说明", font=f(25, True), fill=NAVY)
        notes = [
            "PR-AUC：average precision",
            "n：该标签正样本支持度",
            "灰色虚线：随机基线（正样本比例）",
            "曲线颜色与各子图对应",
        ]
        for offset, line in enumerate(notes, start=1):
            draw.text((x0 + 12, y0 + 22 + offset * 44), line, font=f(21), fill=MUTED)

    draw.text((left_margin, bottom + 50), footnote, font=f(23), fill=MUTED)
    image.save(destination)


def main():
    parser = argparse.ArgumentParser(description="渲染多标签竞赛实验图表")
    parser.add_argument("--audit", default="artifacts/dram_ml_v2/audit_report.json")
    parser.add_argument("--selection", default="artifacts/dram_ml_v2/model_selection.json")
    parser.add_argument("--manifest", default="artifacts/dram_ml_v2/split_manifest.json")
    parser.add_argument("--test-predictions", default="artifacts/dram_ml_v2/test_predictions.json")
    parser.add_argument("--oof-predictions", default="runs/dram_ml_v2_weighted/oof_predictions.json")
    parser.add_argument("--out", default="artifacts/dram_ml_v2/report")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    render_cooccurrence(json.loads(Path(args.audit).read_text(encoding="utf-8")), output / "cooccurrence.png")
    render_comparison(json.loads(Path(args.selection).read_text(encoding="utf-8")), output / "model_comparison.png")

    test_predictions = Path(args.test_predictions)
    if test_predictions.exists():
        curves, total = test_curves(test_predictions, args.manifest)
        render_pr_chart(
            curves, total, output / "pr_curves.png",
            "锁定测试集逐标签 Precision-Recall 曲线",
            "115 张锁定测试图像 · 一次性评估 · PR-AUC 采用 average precision",
            "说明：稀有标签 1/7/8 在锁定测试集中支持度低，曲线仅供探索性参考。",
        )

    oof_predictions = Path(args.oof_predictions)
    if oof_predictions.exists():
        curves, total = oof_curves(oof_predictions, args.manifest)
        render_pr_chart(
            curves, total, output / "pr_curves_oof.png",
            "Development OOF 逐标签 Precision-Recall 曲线",
            "1033 张 development 图像 · 5 折 OOF 概率汇总 · PR-AUC 采用 average precision",
            "说明：OOF 概率用于逐标签阈值校准，锁定测试集不参与本图计算。",
        )

    print(json.dumps({"report_directory": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
