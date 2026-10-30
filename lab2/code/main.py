"""ЛР2: SIFT + LK-сопровождение ключевых точек + RANSAC-гомография."""
import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from common import environment, save_json, write_image


@dataclass
class Detection:
    corners: object = None
    matches: int = 0
    inliers: int = 0
    reprojection_error: object = None
    reason: str = "no_descriptors"
    method: str = "none"
    matrix: object = None
    reference_points: object = None
    frame_points: object = None

    def as_dict(self):
        return {"found": self.corners is not None, "matches": self.matches,
                "inliers": self.inliers, "reprojection_error_px": self.reprojection_error,
                "reason": self.reason, "method": self.method,
                "corners": self.corners.tolist() if self.corners is not None else None}


class PlanarTracker:
    """SIFT-переобнаружение и LK-сопровождение ключевых точек между кадрами.

    SIFT ищет исходный эталон на каждом кадре и сбрасывает накопленный дрейф.
    При недостатке совпадений LK переносит точки, но не старую рамку.
    """
    def __init__(self, reference, ratio=0.7, min_inliers=8, min_inlier_ratio=0.4):
        self.gray = self.to_gray(reference)
        self.h, self.w = self.gray.shape
        if self.w < 16 or self.h < 16:
            raise ValueError("Эталон должен быть не меньше 16x16 пикселей")
        self.detector = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.025)
        self.keypoints, self.descriptors = self.detector.detectAndCompute(self.gray, None)
        if self.descriptors is None or len(self.keypoints) < min_inliers:
            raise ValueError("В эталоне недостаточно ключевых точек. Выберите текстурный объект.")
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio, self.min_inliers = ratio, min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.corners = np.float32([[0, 0], [self.w-1, 0], [self.w-1, self.h-1], [0, self.h-1]])
        self.previous_gray = None
        self.flow_points = None
        self.reference_points = None
        self.previous_matrix = None

    @staticmethod
    def to_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    def detect(self, frame):
        gray = self.to_gray(frame)
        result = self.detect_features(gray)
        if result.corners is not None:
            result.method = "sift"
            # Новые точки выбираются только после привязки к исходному эталону.
            mask = np.zeros_like(gray)
            cv2.fillConvexPoly(mask, np.rint(result.corners).astype(np.int32), 255)
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
            points = cv2.goodFeaturesToTrack(gray, 200, .01, 4, mask=mask)
            if points is not None and len(points) >= self.min_inliers:
                self.flow_points = points
                self.reference_points = cv2.perspectiveTransform(points, np.linalg.inv(result.matrix))
            else:
                self.flow_points = result.frame_points
                self.reference_points = result.reference_points
        elif (self.previous_gray is not None and self.previous_gray.shape == gray.shape
              and self.flow_points is not None and len(self.flow_points) >= self.min_inliers):
            params = dict(winSize=(21, 21), maxLevel=3,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
            current, status, errors = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, self.flow_points, None, **params)
            if current is not None:
                backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, current, None, **params)
                if backward is not None:
                    fb_error = np.linalg.norm(backward-self.flow_points, axis=2).ravel()
                    valid = ((status.ravel() == 1) & (back_status.ravel() == 1)
                             & (fb_error < 1.5) & (errors.ravel() < 35)
                             & np.isfinite(current).all(axis=(1, 2)))
                    if int(valid.sum()) >= self.min_inliers:
                        # Малый объект даёт шумные точки: свободная 8-DOF
                        # гомография на LK способна превратить рамку в клин.
                        # Межкадровое изменение регуляризуем similarity-моделью,
                        # сохраняя перспективу последней надёжной SIFT-привязки.
                        affine, affine_mask = cv2.estimateAffinePartial2D(
                            self.flow_points[valid], current[valid], method=cv2.RANSAC,
                            ransacReprojThreshold=1.5, maxIters=2000, confidence=.995)
                        if affine is not None and affine_mask is not None:
                            scale = float(np.hypot(affine[0, 0], affine[1, 0]))
                            step = np.vstack([affine, [0., 0., 1.]])
                            if .8 < scale < 1.25:
                                flow = self.estimate(self.reference_points[valid], current[valid], gray.shape,
                                                     step @ self.previous_matrix, affine_mask)
                                if flow.corners is not None:
                                    flow.method = "lk"
                                    result = flow
                                    self.reference_points = flow.reference_points
                                    self.flow_points = flow.frame_points
        if result.corners is None:
            self.previous_gray = self.flow_points = self.reference_points = None
            self.previous_matrix = None
        else:
            self.previous_gray = gray.copy()
            self.previous_matrix = result.matrix
        return result

    def detect_features(self, gray):
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        result = Detection()
        if descriptors is None or len(descriptors) < 2:
            return result
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        candidates = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        # Один дескриптор кадра может соответствовать только одной точке эталона.
        unique = {}
        for m in sorted(candidates, key=lambda item: item.distance):
            unique.setdefault(m.trainIdx, m)
        good = list(unique.values())
        result.matches = len(good)
        result.reason = "too_few_matches"
        if len(good) < self.min_inliers:
            return result
        src = np.float32([self.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return self.estimate(src, dst, gray.shape)

    def estimate(self, src, dst, shape, matrix=None, mask=None):
        result = Detection(matches=len(src))
        if matrix is None:
            matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=2000, confidence=.995)
        result.reason = "homography_failed"
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            return result
        valid = mask.ravel().astype(bool)
        result.inliers = int(valid.sum())
        result.reason = "too_few_inliers"
        if result.inliers < self.min_inliers or result.inliers / len(src) < self.min_inlier_ratio:
            return result
        # Inliers должны покрывать площадь эталона, а не почти прямую линию.
        hull = cv2.convexHull(src[valid])
        result.reason = "insufficient_spatial_support"
        if cv2.contourArea(hull) < .01 * self.w * self.h:
            return result
        denominators = (matrix[2, :2] @ self.corners.T) + matrix[2, 2]
        result.reason = "invalid_projection"
        if np.any(np.abs(denominators) < 1e-8) or np.any(denominators * denominators[0] <= 0):
            return result
        corners = cv2.perspectiveTransform(self.corners.reshape(-1, 1, 2), matrix).reshape(4, 2)
        fh, fw = shape
        if not np.isfinite(corners).all() or np.max(np.abs(corners)) > 4 * max(fh, fw):
            return result
        area = abs(cv2.contourArea(corners))
        if not cv2.isContourConvex(corners) or not (100 < area < 2 * fh * fw):
            return result
        visible_area, _ = cv2.intersectConvexConvex(
            corners.astype(np.float32), np.float32([[0, 0], [fw-1, 0], [fw-1, fh-1], [0, fh-1]]))
        if visible_area < .2 * area:
            return result
        projected = cv2.perspectiveTransform(src[valid], matrix)
        errors = np.linalg.norm(projected - dst[valid], axis=2).ravel()
        result.reprojection_error = float(np.median(errors))
        if result.reprojection_error > 3:
            result.reason = "large_reprojection_error"
            return result
        result.corners = corners
        result.matrix = matrix
        result.reference_points = src[valid]
        result.frame_points = dst[valid]
        result.reason = "found"
        return result


def annotate(frame, detection, label, index):
    result = frame.copy()
    if detection.corners is not None:
        cv2.polylines(result, [np.rint(detection.corners).astype(np.int32)], True, (50, 220, 60), 2, cv2.LINE_AA)
        x, y = np.min(detection.corners, axis=0).astype(int)
        cv2.putText(result, label, (max(0, x), max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, .65,
                    (50, 220, 60), 2, cv2.LINE_AA)
    cv2.rectangle(result, (0, result.shape[0]-28), (result.shape[1], result.shape[0]), (20, 20, 20), -1)
    text = f"frame {index} | {detection.reason} ({detection.method}) | inliers {detection.inliers}/{detection.matches}"
    cv2.putText(result, text, (8, result.shape[0]-9), cv2.FONT_HERSHEY_SIMPLEX, .45,
                (240, 240, 240), 1, cv2.LINE_AA)
    return result


def crop_reference(first, roi):
    h, w = first.shape[:2]
    if roi is None:
        roi = [0, 0, w, h]
    x, y, width, height = roi
    if min(x, y) < 0 or min(width, height) < 16 or x+width > w or y+height > h:
        raise ValueError(f"ROI {roi} выходит за границы {w}x{h} либо слишком мал")
    return first[y:y+height, x:x+width].copy(), roi


def rectify_reference(first, corners):
    """Эталон по четырём углам плоского объекта, порядок TL, TR, BR, BL."""
    points = np.asarray(corners, dtype=np.float32)
    if points.size != 8 or not np.isfinite(points).all():
        raise ValueError("Нужны четыре конечные пары координат углов")
    points = points.reshape(4, 2)
    h, w = first.shape[:2]
    if (np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h)
            or not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 256):
        raise ValueError("Углы должны задавать выпуклый объект внутри первого кадра")
    width = round((np.linalg.norm(points[1]-points[0]) + np.linalg.norm(points[2]-points[3]))/2)
    height = round((np.linalg.norm(points[3]-points[0]) + np.linalg.norm(points[2]-points[1]))/2)
    if min(width, height) < 16:
        raise ValueError("Эталон должен быть не меньше 16x16 пикселей")
    target = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    matrix = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(first, matrix, (width, height)), points


def process_video(path, output, roi=None, label="Object", write_video=True, reference_corners=None):
    cv2.setRNGSeed(42)
    path, output = Path(path), Path(output)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")
    writer = None
    try:
        ok, first = capture.read()
        if not ok:
            raise ValueError(f"Нет декодируемых кадров: {path}")
        if roi is not None and reference_corners is not None:
            raise ValueError("Укажите либо ROI, либо четыре угла")
        quad = None
        if reference_corners is not None:
            reference, quad = rectify_reference(first, reference_corners)
            roi = list(cv2.boundingRect(quad))
        else:
            reference, roi = crop_reference(first, roi)
        tracker = PlanarTracker(reference)
        output.mkdir(parents=True, exist_ok=True)
        write_image(output / "reference.png", reference)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        height, width = first.shape[:2]
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if write_video:
            writer = cv2.VideoWriter(str(output / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))
            if not writer.isOpened():
                raise OSError("Не удалось создать MP4. Запустите с --no-video для анализа без кодирования.")
        snapshots = {int(max(expected-1, 0)*p) for p in [0, .2, .4, .6, .8, 1]}
        preview, timings, detections = [], [], []
        frame, index, start = first, 0, time.perf_counter()
        with (output / "frames.jsonl").open("w", encoding="utf-8") as stream:
            while True:
                tick = time.perf_counter()
                detection = tracker.detect(frame)
                timings.append(time.perf_counter() - tick)
                row = {"frame": index, "time_s": index/fps, **detection.as_dict()}
                detections.append(row)
                stream.write(json.dumps(row) + "\n")
                annotated = annotate(frame, detection, label, index)
                if writer:
                    writer.write(annotated)
                if index in snapshots:
                    preview.append(annotated)
                    write_image(output / f"frame_{index:04}.jpg", annotated)
                ok, frame = capture.read()
                if not ok:
                    break
                index += 1
        elapsed = time.perf_counter() - start
        if preview:
            while len(preview) % 3:
                preview.append(np.zeros_like(first))
            sheet = np.concatenate([np.concatenate(preview[i:i+3], axis=1)
                                    for i in range(0, len(preview), 3)], axis=0)
            write_image(output / "preview.jpg", sheet)
        found = [d for d in detections if d["found"]]
        summary = {"input": path.name, "roi_xywh": roi,
                   "reference_quad_xy": quad.tolist() if quad is not None else None,
                   "environment": environment(),
                   "parameters": {"detector": "SIFT", "nfeatures": 2500, "ratio": .7,
                                  "min_inliers": 8, "min_inlier_ratio": .4, "ransac_threshold_px": 3,
                                  "fallback": "pyramidal LK + incremental similarity, forward-backward <1.5px, patch error <35"},
                   "reference_keypoints": len(tracker.keypoints), "frames": len(detections),
                   "container_frame_count": expected, "fps": fps, "detected_frames": len(found),
                   "detection_rate": len(found)/len(detections),
                   "sift_frames": sum(d["method"] == "sift" for d in detections),
                   "lk_frames": sum(d["method"] == "lk" for d in detections),
                   "median_detection_ms": 1000*statistics.median(timings),
                   "processing_fps": len(detections)/sum(timings), "total_elapsed_s": elapsed,
                   "median_inliers_when_found": statistics.median([d["inliers"] for d in found]) if found else None,
                   "ground_truth_available": False,
                   "note": "Detection rate is not accuracy. No manual ground-truth corners provided."}
        save_json(output / "metrics.json", summary)
        print(f"{path.name}: найден {len(found)}/{len(detections)} кадров, "
              f"{summary['median_detection_ms']:.1f} мс/кадр", flush=True)
        return summary
    finally:
        capture.release()
        if writer:
            writer.release()




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("lab2/results/custom"))
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    reference.add_argument("--corners", nargs=8, type=float, metavar="COORD",
                           help="Углы первого кадра: x1 y1 x2 y2 x3 y3 x4 y4 (TL TR BR BL)")
    parser.add_argument("--label", default="Object")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    process_video(args.input, args.output, args.roi, args.label, not args.no_video,
                  reference_corners=args.corners)


if __name__ == "__main__":
    main()
