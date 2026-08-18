import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#0891B2", "#4B5563"]


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


def render_pr_curves(report, destination):
    curves = report["report_data"]["pr_curves"]
    image, draw, font = canvas("Final-test precision-recall curves", 900, 680)
    left, top, width, height = 90, 70, 740, 520
    draw.rectangle((left, top, left + width, top + height), outline="#9CA3AF", width=2)
    for position in range(0, 11):
        x = left + position * width / 10
        y = top + position * height / 10
        draw.line((x, top, x, top + height), fill="#E5E7EB")
        draw.line((left, y, left + width, y), fill="#E5E7EB")
    for index, (label, curve) in enumerate(curves.items()):
        points = [(left + recall * width, top + (1 - precision) * height)
                  for recall, precision in zip(curve["recall"], curve["precision"])]
        if len(points) >= 2:
            draw.line(points, fill=COLORS[index % len(COLORS)], width=3)
        draw.text((left + 15 + index * 95, top + height + 28), f"type {label}",
                  fill=COLORS[index % len(COLORS)], font=font)
    draw.text((left + width // 2, top + height + 55), "Recall", fill="#111827", font=font)
    draw.text((20, top + height // 2), "Precision", fill="#111827", font=font)
    image.save(destination)


def main():
    parser = argparse.ArgumentParser(description="渲染多标签竞赛实验图表")
    parser.add_argument("--audit", default="artifacts/dram_ml_v2/audit_report.json")
    parser.add_argument("--selection", default="artifacts/dram_ml_v2/model_selection.json")
    parser.add_argument("--test-report", default="artifacts/dram_ml_v2/test_report.json")
    parser.add_argument("--out", default="artifacts/dram_ml_v2/report")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    render_cooccurrence(json.loads(Path(args.audit).read_text(encoding="utf-8")), output / "cooccurrence.png")
    render_comparison(json.loads(Path(args.selection).read_text(encoding="utf-8")), output / "model_comparison.png")
    test_report = Path(args.test_report)
    if test_report.exists():
        render_pr_curves(json.loads(test_report.read_text(encoding="utf-8")), output / "pr_curves.png")
    print(json.dumps({"report_directory": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
