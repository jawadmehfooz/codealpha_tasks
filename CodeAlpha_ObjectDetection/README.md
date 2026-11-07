# CodeAlpha_ObjectDetection — Object Detection and Tracking with OpenCV + YOLOv8

Detect and track objects in a video using YOLOv8 (Ultralytics) and OpenCV.
Every detected object gets a bounding box with its class label and
confidence score, plus a **persistent tracking ID** that follows it across
frames (ByteTrack tracker).

**Author:** Jawad Mehfooz
**Program:** CodeAlpha Artificial Intelligence Internship, Oct–Nov 2026

## What it does

- Loads a video file and runs YOLOv8 (`yolov8n.pt`, the small ~6 MB model)
  on every frame.
- Draws bounding boxes with class labels and confidence scores.
- Tracks each object across frames with a persistent ID
  (`ID 3 car 0.87`), using Ultralytics' built-in ByteTrack tracker.
  If the built-in tracker is unavailable, a simple centroid tracker
  implemented in `detect.py` is used instead.
- Saves the annotated video as `output.mp4`.
- Prints a summary: frames processed, total objects detected, number of
  unique tracking IDs, and the classes seen.

## Demo video

The sample input is `data/vtest.avi` — a short traffic video from the
OpenCV samples repository (cars driving on a road, 768×576, 10 fps,
795 frames, ~8 MB), downloaded from:

`https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi`

Notes on the sample video:

- The two URLs suggested in the task brief did **not** work here:
  the Google sample-videos bucket (`ForBiggerEscapes.mp4`) now returns
  *AccessDenied* for anonymous downloads, and
  `github.com/opencv/opencv/raw/.../vtest.avi` returned a GitHub HTML page
  instead of the video file.
- The `raw.githubusercontent.com` mirror of the same OpenCV file worked
  and is the file used.
- The demo runs on a video file. There is no webcam in this VM, so
  webcam input is not used; pass any other video with `--input`.

## Installation

Python 3.12 is used. Create a virtual environment and install the
dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is minimal and loosely pinned:

- `ultralytics` — YOLOv8 (pulls in PyTorch; the first install downloads
  a few hundred MB, allow time)
- `opencv-python-headless` — video reading/writing and drawing

On the first run, the YOLOv8 nano weights (`yolov8n.pt`, ~6 MB) are
downloaded automatically by Ultralytics. The first `model.track()` call
also auto-installs the ByteTrack dependency `lap` (~2 MB) if it is
missing.

> **CPU-only machines:** plain `pip install -r requirements.txt` pulls
> PyTorch with CUDA libraries (several GB). On a machine without a GPU,
> install the CPU-only PyTorch first — it is much smaller and is all
> YOLOv8 needs here:
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

## How to run

```bash
python detect.py
```

This reads `data/vtest.avi` and writes the annotated video to
`output.mp4`, then prints a summary like (real run, 7 Nov 2026):

```
========== Detection Summary ==========
Tracker          : ByteTrack (built-in)
Frames processed : 795
Total objects detected : 6675
Unique tracking IDs    : 64
Classes seen     : bird, car, frisbee, kite, person, skis, truck
  - person: 4608
  - truck: 808
  - car: 784
  - bird: 407
  - kite: 54
  - skis: 7
  - frisbee: 7
Output video     : output.mp4
=======================================
```

Most detections are the real `person` / `car` / `truck` traffic in the
video. At the default confidence of 0.25 the small YOLOv8n model also
reports a few small distant objects as `bird`, `kite`, `frisbee` and
`skis` — those are low-confidence misclassifications of small objects,
not real birds/kites. Raising `--conf` (e.g. `0.4`) removes most of them.

Options:

```bash
python detect.py --input path/to/video.mp4 --output out.mp4
python detect.py --model yolov8n.pt --conf 0.3 --max-frames 200
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--input` | `data/vtest.avi` | Input video file |
| `--output` | `output.mp4` | Annotated output video |
| `--model` | `yolov8n.pt` | YOLO weights (e.g. `yolov8s.pt` for better accuracy) |
| `--conf` | `0.25` | Minimum detection confidence |
| `--max-frames` | _(whole video)_ | Process at most N frames (quick test) |

## What the demo shows

In `data/vtest.avi` (traffic scene), YOLOv8 detects cars (and
occasionally trucks/buses/people) driving past. Each vehicle keeps the
same `ID <n>` while it is on screen, so you can see the tracker follow
it from one side of the frame to the other. The summary counts confirm
how many frames were processed, how many detections were made in total,
and how many distinct objects were tracked.

## Project structure

```
CodeAlpha_ObjectDetection/
├── detect.py          # detection + tracking pipeline (main script)
├── requirements.txt   # ultralytics, opencv-python-headless
├── README.md          # this file
├── data/
│   └── vtest.avi      # sample traffic video (OpenCV samples)
└── output.mp4         # annotated output (created by running detect.py)
```

## Verified

Tested 7 Nov 2026 in a fresh virtual environment (Python 3.12.3):
fresh `pip install -r requirements.txt`, then `python detect.py` on
`data/vtest.avi` end to end — `output.mp4` was produced (795 frames,
768×576 @ 10 fps, 79.5 s, 18 MB) with annotated frames and tracked
objects; full run took ~2 minutes on CPU. Installed versions:
torch 2.15.0+cpu, ultralytics 8.4.204, opencv-python-headless 5.0.0.93.

Environment note: the verification venv was created at
`/home/hatch/venv-detect` (outside this project folder) instead of
`/tmp`, because `/tmp` on this VM is a 512 MB tmpfs and the
torch + ultralytics install (~1.5 GB) does not fit there; pip temp/cache
dirs were likewise pointed at `/home/hatch`.
