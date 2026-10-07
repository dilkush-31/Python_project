"""Vehicle detection with YOLO + lane-wise counting with OpenCV.

Two interchangeable backends:
  * ultralytics  -> any YOLOv8/v11 ``.pt`` (or exported) model   (pip install ultralytics)
  * opencv-dnn   -> a YOLOv5/YOLOv8 ``.onnx`` model, no PyTorch needed
The backend is picked from the model file extension (.onnx -> opencv-dnn, else ultralytics).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from . import db

# COCO class ids for road vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# PCU (passenger-car-unit) weights: a bus occupies far more road than a bike.
PCU = {"car": 1.0, "motorcycle": 0.4, "bus": 2.5, "truck": 2.0}


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    label: str
    conf: float


class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35, iou: float = 0.45, img_size: int = 640):
        self.conf, self.iou, self.img_size = conf, iou, img_size
        self.model_path = str(model_path)
        self.backend = "opencv-dnn" if self.model_path.lower().endswith(".onnx") else "ultralytics"
        if self.backend == "ultralytics":
            from ultralytics import YOLO  # imported lazily so ONNX users don't need torch

            self.model = YOLO(self.model_path)
        else:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)

    # ------------------------------------------------------------------ public
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self._detect_ultralytics(frame) if self.backend == "ultralytics" else self._detect_onnx(frame)

    # ------------------------------------------------------------ ultralytics
    def _detect_ultralytics(self, frame):
        res = self.model.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.img_size,
                                 classes=list(VEHICLE_CLASSES), verbose=False)[0]
        out = []
        for b in res.boxes:
            cls = int(b.cls[0])
            out.append(Detection(tuple(float(v) for v in b.xyxy[0]), VEHICLE_CLASSES[cls], float(b.conf[0])))
        return out

    # ------------------------------------------------------------- opencv-dnn
    def _detect_onnx(self, frame):
        h, w = frame.shape[:2]
        s = self.img_size
        scale = min(s / w, s / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        px, py = (s - nw) // 2, (s - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (s, s), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        out = np.squeeze(out)
        if out.shape[0] < out.shape[1]:  # YOLOv8 layout (84, N) -> (N, 84)
            out = out.T
        if out.shape[1] == 85:       # YOLOv5: cx, cy, w, h, objectness, 80 class scores
            scores = out[:, 4:5] * out[:, 5:]
        else:                        # YOLOv8: cx, cy, w, h, 80 class scores
            scores = out[:, 4:]
        cls_ids = scores.argmax(1)
        confs = scores.max(1)
        keep = (confs >= self.conf) & np.isin(cls_ids, list(VEHICLE_CLASSES))
        boxes, confs, cls_ids = out[keep, :4], confs[keep], cls_ids[keep]
        if len(boxes) == 0:
            return []
        xywh = []
        for cx, cy, bw, bh in boxes:
            x = (cx - bw / 2 - px) / scale
            y = (cy - bh / 2 - py) / scale
            xywh.append([float(x), float(y), float(bw / scale), float(bh / scale)])
        idx = cv2.dnn.NMSBoxes(xywh, confs.astype(float).tolist(), self.conf, self.iou)
        dets = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = xywh[i]
            dets.append(Detection((x, y, x + bw, y + bh), VEHICLE_CLASSES[int(cls_ids[i])], float(confs[i])))
        return dets


# ---------------------------------------------------------------------- lanes
def default_lanes(width: int, height: int) -> dict[str, list[list[int]]]:
    """Starter layout: split the frame into 4 regions. EDIT these polygons for your camera view."""
    w2, h2 = width // 2, height // 2
    return {
        "N": [[0, 0], [w2, 0], [w2, h2], [0, h2]],
        "E": [[w2, 0], [width, 0], [width, h2], [w2, h2]],
        "S": [[w2, h2], [width, h2], [width, height], [w2, height]],
        "W": [[0, h2], [w2, h2], [w2, height], [0, height]],
    }


def save_lanes(path, lanes, frame_size) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"frame_size": list(frame_size), "lanes": lanes}, indent=2))


def load_lanes(path) -> dict[str, np.ndarray]:
    cfg = json.loads(Path(path).read_text())
    return {name: np.array(poly, dtype=np.int32) for name, poly in cfg["lanes"].items()}


def anchor_point(box) -> tuple[float, float]:
    """Bottom-centre of the box = where the vehicle touches the road."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def count_per_lane(dets: list[Detection], lanes: dict[str, np.ndarray], weighted: bool = False) -> dict[str, float]:
    counts = {name: 0.0 for name in lanes}
    for d in dets:
        pt = anchor_point(d.box)
        for name, poly in lanes.items():
            if cv2.pointPolygonTest(poly.astype(np.float32), pt, False) >= 0:
                counts[name] += PCU[d.label] if weighted else 1
                break
    return counts


def draw_overlay(frame, dets, lanes, counts):
    out = frame.copy()
    for name, poly in lanes.items():
        cv2.polylines(out, [poly], True, (0, 255, 255), 2)
        cx, cy = poly.mean(axis=0).astype(int)
        cv2.putText(out, f"{name}: {counts[name]:g}", (int(cx) - 30, int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
    for d in dets:
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{d.label} {d.conf:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)
    return out


# --------------------------------------------------------------------- video
def process_video(source: str, lanes_path: str, db_path: str, model_path: str = "yolov8n.pt",
                  sample_every_s: float = 1.0, start: datetime | None = None, max_samples: int | None = None,
                  conf: float = 0.35, weighted: bool = False, annotated_out: str | None = None) -> int:
    """Run detection on a video, store lane counts every ``sample_every_s`` seconds. Returns rows written."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * sample_every_s)))
    lanes = load_lanes(lanes_path)
    detector = VehicleDetector(model_path, conf=conf)
    start = start or datetime.now().replace(microsecond=0)
    writer = None
    rows, frame_idx, samples = [], 0, 0
    tag = f"video:{Path(str(source)).name}"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            dets = detector.detect(frame)
            counts = count_per_lane(dets, lanes, weighted=weighted)
            ts = (start + timedelta(seconds=frame_idx / fps)).strftime("%Y-%m-%d %H:%M:%S")
            rows += [(ts, lane, int(round(c)), tag) for lane, c in counts.items()]
            if annotated_out:
                vis = draw_overlay(frame, dets, lanes, counts)
                if writer is None:
                    Path(annotated_out).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(annotated_out, cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / sample_every_s,
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            samples += 1
            if max_samples and samples >= max_samples:
                break
        frame_idx += 1
    cap.release()
    if writer:
        writer.release()
    return db.insert_counts(db_path, rows)


def detect_image(image_path: str, lanes_path: str | None, model_path: str, conf: float = 0.35, out_path: str | None = None):
    """Single-image helper (handy for testing the detector + lane polygons)."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(image_path)
    h, w = frame.shape[:2]
    lanes = load_lanes(lanes_path) if lanes_path and Path(lanes_path).exists() else {
        k: np.array(v, dtype=np.int32) for k, v in default_lanes(w, h).items()}
    dets = VehicleDetector(model_path, conf=conf).detect(frame)
    counts = count_per_lane(dets, lanes)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, draw_overlay(frame, dets, lanes, counts))
    return dets, counts
