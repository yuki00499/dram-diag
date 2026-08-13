from pathlib import Path
import csv
from PIL import Image
import numpy as np

def load_labels(data_root):
    with (Path(data_root) / "label.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def letterbox(image, size=(320, 224)):
    tw, th = size; image = image.convert("L")
    scale = min(tw / image.width, th / image.height)
    nw, nh = max(1, round(image.width*scale)), max(1, round(image.height*scale))
    resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (tw, th), 0); pad = ((tw-nw)//2, (th-nh)//2)
    canvas.paste(resized, pad); return canvas

def image_array(path, size=(320, 224)):
    arr = np.asarray(letterbox(Image.open(path), size), dtype=np.float32) / 255.0
    return np.repeat(arr[None, ...], 3, axis=0)
