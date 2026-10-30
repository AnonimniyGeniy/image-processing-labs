"""Загрузка MNIST с проверкой контрольных сумм и обработка собственных цифр."""
import base64
import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from common import read_image

MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def download_mnist(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for filename, expected in MNIST_FILES.items():
        path = root / filename
        if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == expected:
            continue
        print(f"Загрузка {filename}", flush=True)
        temp = path.with_suffix(".part")
        with urllib.request.urlopen(BASE_URL + filename, timeout=60) as src, temp.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        if hashlib.md5(temp.read_bytes()).hexdigest() != expected:
            temp.unlink(missing_ok=True)
            raise ValueError(f"Не совпала контрольная сумма {filename}")
        temp.replace(path)


def read_idx(path):
    with gzip.open(path, "rb") as f:
        header = f.read(4)
        if len(header) != 4 or header[:2] != b"\0\0" or header[2] != 8:
            raise ValueError(f"Некорректный IDX uint8: {path}")
        ndim = header[3]
        if ndim not in (1, 3):
            raise ValueError(f"Неожиданное число измерений IDX: {ndim}")
        shape = struct.unpack(">" + "I"*ndim, f.read(4*ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        if data.size != int(np.prod(shape)):
            raise ValueError(f"Некорректная длина IDX: {path}")
        return data.reshape(shape).copy()


def stratified_split(labels, validation_size=5000, seed=42):
    if validation_size < 10 or validation_size % 10:
        raise ValueError("validation_size должен быть кратен 10 и >= 10")
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = validation_size // 10
        if count >= len(indices):
            raise ValueError("Слишком большая валидационная выборка")
        val.extend(indices[:count])
        train.extend(indices[count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def normalize(images):
    return (torch.from_numpy(np.ascontiguousarray(images)).float().unsqueeze(1) / 255 - .1307) / .3081


def preprocess_digit(image):
    """Одна цифра, ровный фон: полярность → Otsu → bbox → 20x20 → центр масс.

    Метод предназначен для отдельной написанной цифры, не для страницы текста.
    Стандартный MNIST уже центрирован, к нему этот шаг не применяется.
    """
    if image.ndim == 3:
        if image.shape[2] == 4:
            # PNG с alpha: композитинг на белый фон до определения полярности.
            alpha = image[:, :, 3:4].astype(np.float32) / 255
            image = (image[:, :, :3]*alpha + 255*(1-alpha)).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2 or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("Нужно непустое uint8 изображение одной цифры")
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    ink = 255-image if np.median(border) > 127 else image.copy()
    if int(ink.max()) - int(ink.min()) < 15:
        raise ValueError("Цифра не найдена: недостаточный контраст")
    _, binary = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n <= 1:
        raise ValueError("Цифра не найдена")
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= max(3, largest*.03))
    keep = keep[keep != 0]
    mask = np.isin(labels, keep)
    yy, xx = np.nonzero(mask)
    if len(xx) < 5:
        raise ValueError("Цифра слишком мала")
    ink = np.where(mask, ink, 0).astype(np.uint8)
    crop = ink[yy.min():yy.max()+1, xx.min():xx.max()+1]
    h, w = crop.shape
    scale = 20 / max(h, w)
    small = cv2.resize(crop, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.uint8)
    sh, sw = small.shape
    top, left = (28-sh)//2, (28-sw)//2
    canvas[top:top+sh, left:left+sw] = small
    moments = cv2.moments(canvas)
    if moments["m00"]:
        dx = 13.5 - moments["m10"]/moments["m00"]
        dy = 13.5 - moments["m01"]/moments["m00"]
        canvas = cv2.warpAffine(canvas, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))
    return canvas


def load_own_digits(folder=None, json_path=None):
    images, labels, names, hashes = [], [], [], set()

    def add(image, label, name):
        if not 0 <= label <= 9:
            raise ValueError(f"Некорректная метка {label}: {name}")
        digest = hashlib.sha256(image.tobytes() + str(image.shape).encode()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Дубликат изображения в собственном тесте: {name}")
        hashes.add(digest)
        images.append(preprocess_digit(image))
        labels.append(label)
        names.append(name)

    if json_path:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if data.get("format") != "cvlabs-own-digits-v1":
            raise ValueError("Неизвестный формат набора цифр")
        for i, item in enumerate(data["samples"]):
            label = item["label"]
            if type(label) is not int:
                raise ValueError("Метка должна быть целым числом")
            raw = base64.b64decode(item["image"].split(",", 1)[-1], validate=True)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Невозможно декодировать sample {i}")
            add(img, label, f"drawn_{i:03}.png")
    elif folder:
        for label in range(10):
            for path in sorted((Path(folder)/str(label)).glob("*")):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                    add(read_image(path, cv2.IMREAD_UNCHANGED), label, str(path))
    else:
        raise ValueError("Укажите --folder или --json")
    if len(images) < 50:
        raise ValueError(f"Нужно >= 50 уникальных изображений, сейчас {len(images)}")
    if set(labels) != set(range(10)):
        raise ValueError("Для оценки 10 классов нужны примеры каждой цифры 0–9")
    return np.stack(images), np.asarray(labels, dtype=np.int64), names
