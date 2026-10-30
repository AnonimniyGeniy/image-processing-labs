"""ЛР1: адаптивная бинаризация mean, библиотечная и нативная реализации."""
import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from common import environment, read_image, save_json, write_image


def validate(image, block_size, c):
    if image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Ожидается непустое одноканальное uint8-изображение")
    if not isinstance(block_size, int) or block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size должен быть нечётным целым числом >= 3")
    if not isinstance(c, int):
        raise ValueError("В этой реализации C — целое число")


def adaptive_mean_native(image, block_size=31, c=7):
    """Без библиотечных фильтров и векторных вычислений среднего.

    NumPy используется только как контейнер входа/выхода. Суммированная
    таблица, репликация границы, округление среднего и порог — циклы Python.
    Поведение соответствует OpenCV ADAPTIVE_THRESH_MEAN_C/THRESH_BINARY.
    """
    validate(image, block_size, c)
    pixels = image.tolist()
    h, w = image.shape
    radius = block_size // 2
    ph, pw = h + 2 * radius, w + 2 * radius
    integral = [[0] * (pw + 1) for _ in range(ph + 1)]
    for py in range(ph):
        source = pixels[min(max(py - radius, 0), h - 1)]
        row_sum = 0
        current, previous = integral[py + 1], integral[py]
        for px in range(pw):
            row_sum += source[min(max(px - radius, 0), w - 1)]
            current[px + 1] = previous[px + 1] + row_sum
    area = block_size * block_size
    output = []
    for y in range(h):
        top, bottom = integral[y], integral[y + block_size]
        row = []
        for x in range(w):
            total = (bottom[x + block_size] - bottom[x]
                     - top[x + block_size] + top[x])
            # Нечётная площадь: точной половины между целыми быть не может.
            local_mean = (total + area // 2) // area
            row.append(255 if pixels[y][x] > local_mean - c else 0)
        output.append(row)
    return np.array(output, dtype=np.uint8)


def adaptive_mean_opencv(image, block_size=31, c=7):
    validate(image, block_size, c)
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, block_size, c)


def document_sample(width=640, height=360):
    """Воспроизводимый синтетический документ с неоднородным освещением."""
    clean = np.full((height, width), 255, np.uint8)
    for i, text in enumerate(["COMPUTER VISION", "Adaptive threshold", "Variant 2: mean", "0123456789"]):
        cv2.putText(clean, text, (25, 65 + i * 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 15, 2, cv2.LINE_AA)
    yy, xx = np.mgrid[:height, :width]
    light = 0.3 + 0.7 * xx / max(width - 1, 1)
    light *= 1 - 0.3 * np.exp(-((xx - width * .65) ** 2 + (yy - height * .5) ** 2)
                              / (width * .18) ** 2)
    return np.clip(clean * light, 0, 255).astype(np.uint8)


def timed(fn, image, block, c, repeats):
    fn(image, block, c)  # прогрев отдельно от измерений
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(image, block, c)
        samples.append((time.perf_counter() - start) * 1000)
    return result, {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("lab1/data/document.png"))
    parser.add_argument("--output", type=Path, default=Path("lab1/results/document"))
    parser.add_argument("--block-size", type=int, default=31)
    parser.add_argument("--c", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats должен быть >= 1")
    cv2.setNumThreads(1)
    image = read_image(args.input, cv2.IMREAD_GRAYSCALE) if args.input else document_sample()
    args.output.mkdir(parents=True, exist_ok=True)
    library = adaptive_mean_opencv(image, args.block_size, args.c)
    native = adaptive_mean_native(image, args.block_size, args.c)
    write_image(args.output / "opencv.png", library)
    write_image(args.output / "native.png", native)
    write_image(args.output / "difference.png", cv2.absdiff(library, native))
    _, global_result = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    write_image(args.output / "global.png", global_result)
    cases = []
    for width, height in [(320, 180), (640, 360), (1280, 720)]:
        test_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        for block in sorted({15, args.block_size}):
            a, ta = timed(adaptive_mean_opencv, test_image, block, args.c, args.repeats)
            b, tb = timed(adaptive_mean_native, test_image, block, args.c, args.repeats)
            cases.append({"width": width, "height": height, "block_size": block,
                          "opencv": ta, "native": tb,
                          "different_pixels": int(np.count_nonzero(a != b)),
                          "speedup": tb["median_ms"] / ta["median_ms"]})
    report = {"environment": environment(), "method": "mean", "C": args.c,
              "source": str(args.input) if args.input else "generated unevenly lit document",
              "warmup_runs": 1, "repeats": args.repeats,
              "different_pixels": int(np.count_nonzero(library != native)), "benchmarks": cases}
    save_json(args.output / "metrics.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, title, img in zip(axes, ["Uneven illumination", "Global threshold = 127", "Adaptive mean"],
                              [image, global_result, native]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [f'{x["width"]}x{x["height"]}\nk={x["block_size"]}' for x in cases]
    for name in ["opencv", "native"]:
        ax.plot(labels, [x[name]["median_ms"] for x in cases], "o-", label=name)
    ax.set(yscale="log", ylabel="Median execution time (ms, log scale)",
           xlabel="Image size (pixels) and kernel size (pixels)",
           title=f"Adaptive mean: {args.repeats} timed runs after warmup")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "benchmark.png", dpi=150)
    plt.close(fig)
    print(f"Различий пикселей: {report['different_pixels']}; результаты: {args.output}")


if __name__ == "__main__":
    main()
