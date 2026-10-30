import json
import platform
from pathlib import Path

import cv2
import numpy as np


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_image(path, flags=cv2.IMREAD_COLOR):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    result = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if result is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return result


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Не удалось сохранить изображение: {path}")
    encoded.tofile(path)


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "opencv": cv2.__version__,
            "numpy": np.__version__, "opencv_threads": cv2.getNumThreads()}
