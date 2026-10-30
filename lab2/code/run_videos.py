"""Обработать предоставленные ролики и собственное видео. Запуск из корня проекта."""
import json
from pathlib import Path

import cv2

from common import save_json
from .main import process_video


def main():
    cv2.setNumThreads(1)
    data = Path("lab2/data")
    output = Path("lab2/results")
    config = json.loads((data / "video_rois.json").read_text(encoding="utf-8"))
    results = []
    for filename, roi in config.items():
        results.append(process_video(data / "videos" / filename,
                                     output / "provided" / Path(filename).stem,
                                     roi, "Mona Lisa"))
    save_json(output / "provided/summary.json", results)
    corners = json.loads((data / "own_video/reference_corners.json").read_text(encoding="utf-8"))
    process_video(data / "own_video/working.mp4", output / "own_video",
                  label="PyTorch book", reference_corners=corners["corners_xy"])


if __name__ == "__main__":
    main()
