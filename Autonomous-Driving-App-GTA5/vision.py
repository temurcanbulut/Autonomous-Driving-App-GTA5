"""Configurable YOLOv3 perception demo for live screens and recorded video."""

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
TRAFFIC_CLASSES = {0, 1, 2, 3, 5, 6, 7, 9, 11}
VEHICLE_CLASSES = {1, 2, 3, 5, 6, 7}


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Transform:
    scale_x: float
    scale_y: float
    left: int
    top: int
    size: int


def letterbox(frame, size):
    height, width = frame.shape[:2]
    scale = size / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = cv2.resize(
        frame, (resized_width, resized_height)
    )
    return canvas, Transform(resized_width / width, resized_height / height, left, top, size)


def decode_detections(outputs, frame_shape, transform, confidence, nms, class_count, allowed=None):
    """Vectorized decoding followed by class-aware nonmaximum suppression."""
    height, width = frame_shape[:2]
    candidates = []
    for output in outputs:
        if output.ndim != 2 or output.shape[1] != class_count + 5:
            raise ValueError("Model output and label count do not match.")
        if not len(output):
            continue
        ids = output[:, 5:].argmax(axis=1)
        # OpenCV's Darknet Region layer already multiplies class probabilities
        # by objectness. Multiplying output[:, 4] again would double-count it.
        scores = output[np.arange(len(output)), ids + 5]
        keep = np.isfinite(output).all(axis=1) & (scores >= confidence)
        if allowed is not None:
            keep &= np.isin(ids, list(allowed))
        rows, ids, scores = output[keep], ids[keep], scores[keep]
        for row, class_id, score in zip(rows, ids, scores, strict=True):
            cx, cy, box_width, box_height = row[:4] * transform.size
            x1 = max(
                0,
                min(
                    width,
                    int(np.floor((cx - box_width / 2 - transform.left) / transform.scale_x)),
                ),
            )
            y1 = max(
                0,
                min(
                    height,
                    int(np.floor((cy - box_height / 2 - transform.top) / transform.scale_y)),
                ),
            )
            x2 = max(
                0,
                min(
                    width,
                    int(np.ceil((cx + box_width / 2 - transform.left) / transform.scale_x)),
                ),
            )
            y2 = max(
                0,
                min(
                    height,
                    int(np.ceil((cy + box_height / 2 - transform.top) / transform.scale_y)),
                ),
            )
            if x2 > x1 and y2 > y1:
                candidates.append(
                    Detection(int(class_id), float(score), (x1, y1, x2 - x1, y2 - y1))
                )
    detections = []
    for class_id in sorted({item.class_id for item in candidates}):
        group = [item for item in candidates if item.class_id == class_id]
        indices = cv2.dnn.NMSBoxes(
            [item.box for item in group],
            [item.confidence for item in group],
            confidence,
            nms,
        )
        detections.extend(group[int(index)] for index in np.asarray(indices).reshape(-1))
    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def nearest_in_image(detections, frame_shape):
    """Pixel offset from bottom-center, not a physical distance estimate."""
    height, width = frame_shape[:2]
    vehicles = [item for item in detections if item.class_id in VEHICLE_CLASSES]
    if not vehicles:
        return None

    def distance(item):
        x, y, w, h = item.box
        return float(np.hypot(x + w / 2 - width / 2, y + h / 2 - height))

    closest = min(vehicles, key=distance)
    return closest, distance(closest)


class Detector:
    def __init__(self, args):
        for path in (args.config, args.weights, args.names):
            if not path.is_file():
                raise ValueError(f"Missing file: {path}. See README for model setup.")
        self.names = args.names.read_text(encoding="utf-8").splitlines()
        self.names = [name.strip() for name in self.names if name.strip()]
        counts = {
            int(count)
            for count in re.findall(
                r"^classes\s*=\s*(\d+)", args.config.read_text(encoding="utf-8"), re.M
            )
        }
        if counts != {len(self.names)}:
            raise ValueError("The label file must match the class count in every YOLO head.")
        self.net = cv2.dnn.readNetFromDarknet(str(args.config), str(args.weights))
        if args.device == "cuda":
            if cv2.cuda.getCudaEnabledDeviceCount() < 1:
                raise ValueError("CUDA is unavailable in this OpenCV build. Use --device cpu.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layers = self.net.getUnconnectedOutLayersNames()
        self.args = args

    def detect(self, frame):
        start = time.perf_counter()
        image, transform = letterbox(frame, self.args.input_size)
        blob = cv2.dnn.blobFromImage(
            image,
            1 / 255.0,
            (self.args.input_size, self.args.input_size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        inference_start = time.perf_counter()
        outputs = self.net.forward(self.layers)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        detections = decode_detections(
            outputs,
            frame.shape,
            transform,
            self.args.confidence,
            self.args.nms,
            len(self.names),
            None if self.args.all_classes else TRAFFIC_CLASSES,
        )
        return detections, inference_ms, (time.perf_counter() - start) * 1000


def frames(args):
    if args.source == "screen":
        import mss

        with mss.mss() as screen:
            if args.region:
                left, top, width, height = args.region
                region = dict(left=left, top=top, width=width, height=height)
            else:
                if not 1 <= args.monitor < len(screen.monitors):
                    raise ValueError(f"Monitor must be between 1 and {len(screen.monitors) - 1}.")
                region = screen.monitors[args.monitor]
            while True:
                yield cv2.cvtColor(np.asarray(screen.grab(region)), cv2.COLOR_BGRA2BGR)
    else:
        if not Path(args.source).is_file():
            raise ValueError(f"Video not found: {args.source}")
        capture = cv2.VideoCapture(args.source)
        try:
            if not capture.isOpened():
                raise ValueError("Video could not be opened.")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
        finally:
            capture.release()


def annotate(frame, detections, names, inference_ms, detector_ms):
    for item in detections:
        x, y, width, height = item.box
        color = (141, 228, 209) if item.class_id in VEHICLE_CLASSES else (255, 199, 112)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            frame,
            f"{names[item.class_id]} {item.confidence:.0%}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    nearest = nearest_in_image(detections, frame.shape)
    if nearest:
        item, offset = nearest
        x, y, width, height = item.box
        center = (x + width // 2, y + height // 2)
        cv2.line(frame, (frame.shape[1] // 2, frame.shape[0] - 1), center, (141, 228, 209), 1)
        cv2.putText(
            frame,
            f"Screen offset: {offset:.0f} px",
            (12, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (141, 228, 209),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"Inference {inference_ms:.0f} ms | Detector {detector_ms:.0f} ms",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def run(args):
    detector = Detector(args)
    source = frames(args)
    writer = None
    frame_count = 0
    inference_total = detector_total = 0.0
    start = time.perf_counter()
    try:
        for frame in source:
            detections, inference_ms, detector_ms = detector.detect(frame)
            inference_total += inference_ms
            detector_total += detector_ms
            annotate(frame, detections, detector.names, inference_ms, detector_ms)
            if args.output:
                if writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    codec = "mp4v" if args.output.suffix.lower() == ".mp4" else "MJPG"
                    writer = cv2.VideoWriter(
                        str(args.output),
                        cv2.VideoWriter_fourcc(*codec),
                        args.output_fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise ValueError("Output video could not be opened. Try an .avi file.")
                writer.write(frame)
            frame_count += 1
            if not args.headless:
                cv2.imshow("GTA 5 Perception", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
    if not frame_count:
        raise ValueError("No video frames were decoded.")
    elapsed = time.perf_counter() - start
    metrics = dict(
        frames=frame_count,
        elapsed_seconds=round(elapsed, 3),
        processing_fps=round(frame_count / elapsed, 2),
        mean_inference_ms=round(inference_total / frame_count, 2),
        mean_detector_ms=round(detector_total / frame_count, 2),
        input_size=args.input_size,
        confidence=args.confidence,
        nms=args.nms,
        device=args.device,
        opencv=cv2.__version__,
        note="Includes warm-up frames. Processing FPS includes capture, drawing and output; model loading is excluded. Pixel offsets are not physical distances.",
    )
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def probability(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("Use a value greater than 0 and at most 1.")
    return number


def parser():
    cli = argparse.ArgumentParser(
        description="GTA 5 object detection from a screen or video. No driving controls or metric distance estimates."
    )
    cli.add_argument("--source", default="screen", help="screen or a local video file")
    cli.add_argument("--monitor", type=int, default=1)
    cli.add_argument("--region", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    cli.add_argument("--weights", type=Path, default=BASE / "yolov3.weights")
    cli.add_argument("--config", type=Path, default=BASE / "yolov3.cfg")
    cli.add_argument("--names", type=Path, default=BASE / "coco.names")
    cli.add_argument("--input-size", type=int, choices=(320, 416, 608), default=416)
    cli.add_argument("--confidence", type=probability, default=0.5)
    cli.add_argument("--nms", type=probability, default=0.4)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    cli.add_argument("--all-classes", action="store_true")
    cli.add_argument("--headless", action="store_true")
    cli.add_argument("--max-frames", type=int, default=0, help="0 processes until quit/end of file")
    cli.add_argument("--output", type=Path, help="annotated .mp4 or .avi file (no audio)")
    cli.add_argument(
        "--output-fps",
        type=float,
        default=30,
        help="encoded playback rate; independent of processing FPS",
    )
    cli.add_argument("--metrics", type=Path, help="write run metrics as JSON")
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    if args.max_frames < 0 or not np.isfinite(args.output_fps) or args.output_fps <= 0:
        cli.error("max-frames must be nonnegative and output-fps must be positive.")
    if args.region and (args.region[2] <= 0 or args.region[3] <= 0):
        cli.error("region width and height must be positive.")
    if args.output and args.output.suffix.lower() not in (".mp4", ".avi"):
        cli.error("output must end in .mp4 or .avi")
    if args.output and args.metrics and args.output.resolve() == args.metrics.resolve():
        cli.error("output and metrics must be different files.")
    for destination in (args.output, args.metrics):
        if destination and destination.exists():
            cli.error(f"Refusing to overwrite an existing file: {destination}")
    try:
        print(json.dumps(run(args), indent=2))
    except (ValueError, OSError, cv2.error) as error:
        cli.exit(1, f"Error: {error}\n")
