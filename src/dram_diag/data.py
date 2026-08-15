import csv
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


IMAGENET_MEAN = np.asarray([.485, .456, .406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([.229, .224, .225], dtype=np.float32)[:, None, None]


def load_labels(data_root):
    with (Path(data_root) / "label.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def letterbox(image, size=(480, 320)):
    target_width, target_height = size
    image = image.convert("L")
    scale = min(target_width / image.width, target_height / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (target_width, target_height), 0)
    canvas.paste(resized, ((target_width - width) // 2, (target_height - height) // 2))
    return canvas


def augment(image, rng=None):
    rng = rng or random
    image = image.rotate(rng.uniform(-5, 5), resample=Image.Resampling.BILINEAR, fillcolor=0)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.92, 1.08))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.92, 1.08))
    if rng.random() < .2:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(.1, .4)))
    return image


def image_array(path, size=(480, 320), training=False, normalization="imagenet", stats=None, rng=None):
    with Image.open(path) as source:
        image = letterbox(source, size)
    if training:
        image = augment(image, rng)
    gray = np.asarray(image, dtype=np.float32) / 255.0
    array = np.repeat(gray[None, ...], 3, axis=0)
    if normalization == "imagenet":
        return (array - IMAGENET_MEAN) / IMAGENET_STD
    if normalization == "dataset_gray":
        if not stats or float(stats.get("std", 0)) <= 0:
            raise ValueError("dataset_gray 需要有效的训练集统计")
        return (array - float(stats["mean"])) / float(stats["std"])
    raise ValueError(f"未知归一化方式: {normalization}")


def compute_gray_stats(image_paths, size=(480, 320)):
    total = squared = pixels = 0.0
    for path in image_paths:
        with Image.open(path) as source:
            array = np.asarray(letterbox(source, size), dtype=np.float64) / 255.0
        total += float(array.sum())
        squared += float(np.square(array).sum())
        pixels += array.size
    mean = total / pixels
    variance = max(0.0, squared / pixels - mean * mean)
    return {"mean": mean, "std": variance ** .5, "pixel_count": pixels, "source": "train_only"}
