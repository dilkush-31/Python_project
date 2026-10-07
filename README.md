# YOLO-Based Traffic Density Signal Control

Estimates **lane-wise vehicle density** from traffic video with **YOLO + OpenCV**, stores it in **SQLite**,
and uses a **Genetic Algorithm** to find signal green times that minimise vehicle delay.

```
 video ──► YOLO detect ──► lane polygons (OpenCV) ──► SQLite (density table)
                                                         │  SQL: peak hours, lane share, moving avg
                                                         ▼
                              mean density per lane ──► arrival rates (veh/s)
                                                         ▼
          Fixed timer ─┐                       4-way intersection simulator
          Proportional ├──► compared on 10 held-out traffic seeds ◄── Genetic Algorithm (green times)
          GA optimised ┘                                  │
                                                          ▼
                                   results.json + plots (+ optional MLflow tracking)
```

## Quick start

```bash
pip install -r requirements.txt

# 1) Try the whole pipeline with no video (uses SYNTHETIC counts, clearly tagged source='synthetic')
python main.py demo

# 2) Real data: describe the lanes, check them, run YOLO on your video
python main.py init-lanes --width 1280 --height 720          # writes config/lanes.json -> edit the polygons
python main.py preview-lanes --source traffic.mp4            # outputs/lanes_preview.jpg, verify polygons
python main.py detect --video traffic.mp4 --model yolov8n.pt --start "2026-10-07 08:00:00" \
                      --annotated outputs/annotated.mp4
python main.py queries                                       # SQL analytics
python main.py optimize --hours 7-10 --demand 1200 --mlflow  # GA + comparison + plots
python -m pytest -q
```

No PyTorch? Use an ONNX model with the OpenCV backend:
`curl -L -o models/yolov5n.onnx https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx`
then `--model models/yolov5n.onnx` (backend is chosen from the file extension).

## How it works

| Module | What it does |
|---|---|
| `detector.py` | YOLO (Ultralytics or OpenCV-DNN/ONNX) -> keeps car/motorcycle/bus/truck -> assigns each box to a lane by its **bottom-centre point** inside a polygon (`cv2.pointPolygonTest`). Optional PCU weighting (bus = 2.5, bike = 0.4). |
| `db.py` | SQLite `density(ts, lane, count, source)` + analytics: peak hours (`CTE + RANK() OVER PARTITION BY`), lane share (`SUM() OVER ()`), moving average (`ROWS BETWEEN`). |
| `simulator.py` | 1-second simulation of a 4-phase junction: Poisson arrivals, saturation flow 1800 veh/h, 2 s start-up loss, 4 s clearance. Metric = average delay per vehicle (queue-seconds / arrivals). |
| `ga.py` | Chromosome = 4 green times in [8, 60] s. Tournament selection, BLX-alpha crossover, Gaussian mutation, elitism. Fitness is averaged over 3 traffic seeds. |
| `pipeline.py` | SQL -> arrival rates -> GA -> evaluation on **10 unseen seeds** against two baselines. |

## Results (demo run: synthetic density, 07:00-10:59 window, 1200 veh/h)

| Strategy | Greens N/E/S/W (s) | Avg delay (s) | vs Fixed |
|---|---|---|---|
| Fixed timer (30 s each) | 30/30/30/30 | 197.4 | - |
| Proportional to demand | 53/14/43/10 | 60.2 | 69.5% lower |
| **GA optimised** | 46/16/42/12 | **53.7** | **72.8% lower** |

Sensitivity to total demand (GA vs the stronger *proportional* baseline): 600 veh/h -> 29% lower delay,
900 -> 22%, 1200 -> 11%, 1400 -> 12%. See `outputs/` for plots.

### Read this before quoting any number
* These figures come from **synthetic counts + a simulator**, not from real traffic. They prove the
  pipeline works; they are **not** a claim about a real junction.
* The big gain vs the fixed timer is mostly because the demo demand is lopsided (N and S busy), so equal
  greens over-saturate N. The honest comparison is against the proportional heuristic.
* To get numbers you can put on a resume, run `detect` on your own video, then `optimize`, and report
  *those* results.

## Limitations / future work
* Detection counts vehicles *in frame* (a density proxy), not flow in veh/h; `--demand` supplies the scale.
* The simulator ignores turning movements, pedestrians, and platooning between junctions.
* Ideas: re-optimise per hour-of-day, add a multi-objective fitness (delay + max queue), compare with
  Webster's formula, add tracking (ByteTrack) for true flow counts, adaptive control with RL.
