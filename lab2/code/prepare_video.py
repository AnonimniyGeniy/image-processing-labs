"""Рабочая копия собственного видео: все кадры, длинная сторона <=1280."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("--output", type=Path, default=Path("lab2/data/own_video/working.mp4"))
parser.add_argument("--max-side", type=int, default=1280)
args = parser.parse_args()
if args.max_side < 64:
    parser.error("max-side должен быть >=64")
capture = cv2.VideoCapture(str(args.input))
if not capture.isOpened():
    raise ValueError(f"Не удалось открыть {args.input}")
writer = None
try:
    fps = capture.get(cv2.CAP_PROP_FPS)
    rotation = capture.get(cv2.CAP_PROP_ORIENTATION_META)
    ok, frame = capture.read()
    if not ok or fps <= 0:
        raise ValueError("В видео нет кадров либо некорректная частота кадров")
    h, w = frame.shape[:2]
    scale = min(1., args.max_side / max(h, w))
    width, height = max(2, round(w*scale)//2*2), max(2, round(h*scale)//2*2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError("Не удалось записать рабочую копию")
    count = 0
    while ok:
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(resized)
        if count == 0:
            cv2.imwrite(str(args.output.parent / "first_frame.jpg"), resized)
        count += 1
        ok, frame = capture.read()
finally:
    capture.release()
    if writer:
        writer.release()
manifest = {"source_file": args.input.name, "source_bytes": args.input.stat().st_size,
            "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "decoded_source_size_wh": [w, h], "rotation_metadata_degrees": rotation,
            "working_size_wh": [width, height], "fps": fps, "frames": count,
            "working_file": args.output.name,
            "working_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "note": "OpenCV automatic orientation; all decoded frames retained; CFR copy at reported mean FPS; audio omitted"}
(args.output.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Рабочая копия: {count} кадров, {width}x{height}, {fps:.3f} FPS", flush=True)
