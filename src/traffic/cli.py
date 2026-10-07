from __future__ import annotations

import argparse
from datetime import datetime

import cv2

from . import db, detector, synthetic
from .ga import GAConfig
from .pipeline import optimise


def _print_queries(db_path):
    print("\n--- Lane share of total traffic (window SUM OVER) ---")
    print(db.lane_share(db_path).to_string(index=False))
    print("\n--- Top-3 peak hours per lane (CTE + RANK OVER PARTITION) ---")
    print(db.peak_hours(db_path, 3).to_string(index=False))


def _hours(s):
    if not s:
        return None, None
    a, b = s.split("-")
    return int(a), int(b)


def main(argv=None):
    p = argparse.ArgumentParser(prog="traffic", description="YOLO traffic density + GA signal control")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init-lanes", help="write a starter lanes.json (edit the polygons for your camera)")
    s.add_argument("--width", type=int, default=1280)
    s.add_argument("--height", type=int, default=720)
    s.add_argument("--out", default="config/lanes.json")

    s = sub.add_parser("preview-lanes", help="draw lane polygons on the first frame of a video/image")
    s.add_argument("--source", required=True)
    s.add_argument("--lanes", default="config/lanes.json")
    s.add_argument("--out", default="outputs/lanes_preview.jpg")

    s = sub.add_parser("detect-image", help="run YOLO on one image and print lane counts")
    s.add_argument("--image", required=True)
    s.add_argument("--lanes", default=None)
    s.add_argument("--model", default="yolov8n.pt")
    s.add_argument("--conf", type=float, default=0.35)
    s.add_argument("--out", default="outputs/detection.jpg")

    s = sub.add_parser("detect", help="run YOLO on a video and log lane counts to SQLite")
    s.add_argument("--video", required=True)
    s.add_argument("--lanes", default="config/lanes.json")
    s.add_argument("--model", default="yolov8n.pt")
    s.add_argument("--db", default="data/traffic.db")
    s.add_argument("--every", type=float, default=1.0, help="sample every N seconds of video")
    s.add_argument("--start", default=None, help="video start time 'YYYY-MM-DD HH:MM:SS' (default: now)")
    s.add_argument("--max-samples", type=int, default=None)
    s.add_argument("--conf", type=float, default=0.35)
    s.add_argument("--weighted", action="store_true", help="count PCU (bus=2.5, bike=0.4...) instead of raw vehicles")
    s.add_argument("--annotated", default=None, help="write an annotated mp4 here")

    s = sub.add_parser("synth", help="fill the DB with SYNTHETIC counts (no video needed)")
    s.add_argument("--db", default="data/traffic.db")
    s.add_argument("--days", type=int, default=3)
    s.add_argument("--reset", action="store_true")

    s = sub.add_parser("queries", help="run the SQL analytics on the DB")
    s.add_argument("--db", default="data/traffic.db")

    for name in ("optimize", "demo"):
        s = sub.add_parser(name, help="GA signal optimisation" if name == "optimize" else "synth data + queries + optimisation")
        s.add_argument("--db", default="data/traffic.db")
        s.add_argument("--hours", default=None, help="restrict to hour window, e.g. 7-10")
        s.add_argument("--demand", type=float, default=1200.0, help="total junction demand, veh/hour")
        s.add_argument("--generations", type=int, default=40)
        s.add_argument("--pop", type=int, default=40)
        s.add_argument("--out", default="outputs")
        s.add_argument("--mlflow", action="store_true")

    a = p.parse_args(argv)

    if a.cmd == "init-lanes":
        detector.save_lanes(a.out, detector.default_lanes(a.width, a.height), (a.width, a.height))
        print(f"Wrote {a.out}. Open it and edit the polygons so they cover each approach road.")
    elif a.cmd == "preview-lanes":
        cap = cv2.VideoCapture(a.source)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"Cannot read {a.source}")
        lanes = detector.load_lanes(a.lanes)
        cv2.imwrite(a.out, detector.draw_overlay(frame, [], lanes, {k: 0 for k in lanes}))
        print(f"Saved {a.out}")
    elif a.cmd == "detect-image":
        dets, counts = detector.detect_image(a.image, a.lanes, a.model, a.conf, a.out)
        print(f"{len(dets)} vehicles: " + ", ".join(f"{d.label}({d.conf:.2f})" for d in dets))
        print("Lane counts:", counts, f"-> annotated image: {a.out}")
    elif a.cmd == "detect":
        start = datetime.strptime(a.start, "%Y-%m-%d %H:%M:%S") if a.start else None
        n = detector.process_video(a.video, a.lanes, a.db, a.model, a.every, start, a.max_samples, a.conf,
                                   a.weighted, a.annotated)
        print(f"Stored {n} rows in {a.db}")
    elif a.cmd == "synth":
        if a.reset:
            db.clear(a.db)
        print(f"Inserted {synthetic.generate(a.db, a.days)} synthetic rows into {a.db}")
    elif a.cmd == "queries":
        _print_queries(a.db)
    elif a.cmd in ("optimize", "demo"):
        if a.cmd == "demo":
            db.clear(a.db)
            print(f"Inserted {synthetic.generate(a.db)} SYNTHETIC rows (demo data, not real traffic)")
            _print_queries(a.db)
        hf, ht = _hours(a.hours if a.cmd == "optimize" else (a.hours or "7-10"))
        optimise(a.db, a.out, hf, ht, a.demand, GAConfig(pop_size=a.pop, generations=a.generations), use_mlflow=a.mlflow)


if __name__ == "__main__":
    main()
