#!/usr/bin/env python3
"""Object Detection and Tracking with OpenCV + YOLOv8.

CodeAlpha Artificial Intelligence Internship (Oct-Nov 2026)
Author: Jawad Mehfooz

Loads a video file, runs YOLOv8 detection on every frame, draws bounding
boxes with class labels + confidence scores, and tracks objects across
frames with persistent tracking IDs (ultralytics' built-in ByteTrack
tracker; a simple centroid tracker is used as a fallback if the built-in
tracker is unavailable).

Usage:
    python detect.py                          # data/vtest.avi -> output.mp4
    python detect.py --input path/to/video.mp4 --output out.mp4
    python detect.py --model yolov8n.pt --conf 0.3 --max-frames 200

Works on a video file (no webcam is available in this VM).
"""

import argparse
import collections
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Simple centroid tracker (fallback only, used if model.track() is unusable)
# ---------------------------------------------------------------------------

class CentroidTracker:
    """Assign persistent IDs by nearest-centroid matching between frames."""

    def __init__(self, max_distance=80, max_disappeared=30):
        self.next_id = 0
        self.objects = {}      # id -> (cx, cy)
        self.disappeared = {}  # id -> frames since last seen
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """detections: list of (x1, y1, x2, y2). Returns {index: track_id}."""
        if not detections:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return {}

        centroids = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in detections]

        if not self.objects:
            for c in centroids:
                self.register(c)
            return {i: self.next_id - len(centroids) + i for i in range(len(centroids))}

        object_ids = list(self.objects)
        object_centroids = [self.objects[oid] for oid in object_ids]

        # Distance matrix between existing objects and new detections.
        dist = np.zeros((len(object_centroids), len(centroids)))
        for i, oc in enumerate(object_centroids):
            for j, c in enumerate(centroids):
                dist[i, j] = np.hypot(oc[0] - c[0], oc[1] - c[1])

        # Greedy nearest-neighbour matching (smallest distances first).
        rows = dist.min(axis=1).argsort()
        cols = dist.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()
        assignment = {}

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dist[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid] = centroids[col]
            self.disappeared[oid] = 0
            assignment[col] = oid
            used_rows.add(row)
            used_cols.add(col)

        # Existing objects with no match: count as disappeared.
        for row in set(range(len(object_ids))) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        # New detections with no match: register as new objects.
        for col in set(range(len(centroids))) - used_cols:
            self.register(centroids[col])
            assignment[col] = self.next_id - 1

        return assignment


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _color_for(class_id):
    """Deterministic BGR colour per class id."""
    rng = np.random.default_rng(class_id)
    return tuple(int(v) for v in rng.integers(0, 200, size=3))


def draw_detections(frame, detections):
    """Draw boxes + 'ID <n> <class> <conf>' labels on a frame.

    Detections are drawn in two passes (all boxes/label backgrounds first,
    then all label text) so a neighbour's label background never paints
    over another label's text. Labels that would overlap a label already
    placed on this frame are shifted upwards (or below their box if there
    is no room above), so people walking shoulder-to-shoulder stay readable.

    detections: list of dicts with keys x1, y1, x2, y2, track_id, name, conf.
    """
    frame_h = frame.shape[0]
    placed = []  # label rects already used on this frame: (x1, y1, x2, y2)
    texts = []   # (x, y, label) to draw after all backgrounds
    for d in detections:
        x1, y1, x2, y2 = (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
        color = _color_for(d["class_id"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {d['track_id']} {d['name']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx1, ly2 = x1, y1
        ly1 = ly2 - th - 8
        # Shift the label up while it overlaps a label already placed.
        while any(lx1 < px2 and lx1 + tw + 4 > px1 and ly1 < py2 and ly2 > py1
                  for px1, py1, px2, py2 in placed):
            ly1 -= th + 6
            ly2 -= th + 6
            if ly1 < 0:
                # No room above: put it just below the box instead.
                ly1 = y2 + 4
                ly2 = y2 + th + 12
                break
        ly1 = max(ly1, 0)
        ly2 = min(ly2, frame_h)
        cv2.rectangle(frame, (lx1, ly1), (lx1 + tw + 4, ly2), color, -1)
        placed.append((lx1, ly1, lx1 + tw + 4, ly2))
        texts.append((lx1 + 2, ly2 - 5, label))
    for x, y, label in texts:
        cv2.putText(frame, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Main detection + tracking pipeline
# ---------------------------------------------------------------------------

def run_detection(input_path, output_path, model_path="yolov8n.pt",
                  conf=0.25, max_frames=None):
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: could not open video: {input_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input : {input_path} ({width}x{height} @ {fps:.1f} fps)")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )
    if not writer.isOpened():
        print(f"ERROR: could not open video writer for: {output_path}",
              file=sys.stderr)
        cap.release()
        sys.exit(1)

    model = YOLO(model_path)  # downloads yolov8n.pt (~6 MB) on first run
    print(f"Model : {model_path} (classes: {len(model.names)})")

    # Try the built-in ByteTrack tracker first.
    use_builtin_tracker = True
    try:
        results = model.track(
            source=str(input_path),
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf,
            stream=True,
            verbose=False,
        )
        # Pull the first result to prove the tracker actually works.
        first = next(results)
        frame_iter = _iter_with_first(first, results)
    except Exception as exc:  # noqa: BLE001 - any tracker failure -> fallback
        print(f"WARNING: built-in tracker unavailable ({exc}).")
        print("Falling back to per-frame detection + centroid tracker.")
        use_builtin_tracker = False

    frames_processed = 0
    total_detections = 0
    unique_ids = set()
    class_counts = collections.Counter()

    if use_builtin_tracker:
        for result in frame_iter:
            frame = result.orig_img
            detections = []
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes
                ids = boxes.id
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].tolist()
                    cls_id = int(boxes.cls[i])
                    name = model.names[cls_id]
                    track_id = int(ids[i]) if ids is not None else -1
                    detections.append({
                        "x1": xyxy[0], "y1": xyxy[1],
                        "x2": xyxy[2], "y2": xyxy[3],
                        "track_id": track_id,
                        "class_id": cls_id, "name": name,
                        "conf": float(boxes.conf[i]),
                    })
                    unique_ids.add(track_id)
                    class_counts[name] += 1
            total_detections += len(detections)
            draw_detections(frame, detections)
            writer.write(frame)
            frames_processed += 1
            if frames_processed % 50 == 0:
                print(f"  ... {frames_processed} frames, "
                      f"{total_detections} detections, "
                      f"{len(unique_ids)} unique IDs")
            if max_frames and frames_processed >= max_frames:
                break
    else:
        tracker = CentroidTracker()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(frame, conf=conf, verbose=False)[0]
            raw = []
            if result.boxes is not None and len(result.boxes):
                for i in range(len(result.boxes)):
                    xyxy = result.boxes.xyxy[i].tolist()
                    cls_id = int(result.boxes.cls[i])
                    raw.append((xyxy, cls_id, float(result.boxes.conf[i])))
            assignment = tracker.update([r[0] for r in raw])
            detections = []
            for i, (xyxy, cls_id, c) in enumerate(raw):
                name = model.names[cls_id]
                track_id = assignment.get(i, -1)
                detections.append({
                    "x1": xyxy[0], "y1": xyxy[1],
                    "x2": xyxy[2], "y2": xyxy[3],
                    "track_id": track_id,
                    "class_id": cls_id, "name": name, "conf": c,
                })
                unique_ids.add(track_id)
                class_counts[name] += 1
            total_detections += len(detections)
            draw_detections(frame, detections)
            writer.write(frame)
            frames_processed += 1
            if frames_processed % 50 == 0:
                print(f"  ... {frames_processed} frames, "
                      f"{total_detections} detections, "
                      f"{len(unique_ids)} unique IDs")
            if max_frames and frames_processed >= max_frames:
                break

    cap.release()
    writer.release()

    summary = {
        "frames_processed": frames_processed,
        "total_detections": total_detections,
        "unique_ids": len(unique_ids),
        "classes_seen": sorted(class_counts),
        "class_counts": dict(class_counts),
        "tracker": "ByteTrack (built-in)" if use_builtin_tracker
                   else "centroid (fallback)",
        "output": str(output_path),
    }

    print("\n========== Detection Summary ==========")
    print(f"Tracker          : {summary['tracker']}")
    print(f"Frames processed : {frames_processed}")
    print(f"Total objects detected : {total_detections}")
    print(f"Unique tracking IDs    : {len(unique_ids)}")
    print(f"Classes seen     : {', '.join(summary['classes_seen']) or 'none'}")
    for name, count in class_counts.most_common():
        print(f"  - {name}: {count}")
    print(f"Output video     : {output_path}")
    print("=======================================")
    return summary


def _iter_with_first(first, rest):
    yield first
    yield from rest


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 object detection + tracking on a video file.")
    parser.add_argument("--input", default="data/vtest.avi",
                        help="input video file (default: data/vtest.avi)")
    parser.add_argument("--output", default="output.mp4",
                        help="annotated output video (default: output.mp4)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO weights (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence threshold (default: 0.25)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="process at most N frames (default: whole video)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: input video not found: {input_path}", file=sys.stderr)
        print("Download the sample video into data/ first (see README.md).",
              file=sys.stderr)
        sys.exit(1)

    run_detection(input_path, Path(args.output), args.model,
                  conf=args.conf, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
