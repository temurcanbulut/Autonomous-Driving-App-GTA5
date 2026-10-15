"""Replay trained semantic perception without sending any game input."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from autonomy.capture import FrameProducer
from autonomy.perception import SemanticPerception
from autonomy.tracking import Tracker
from vision import frames

PALETTE = np.array(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)


def main():
    cli = argparse.ArgumentParser(
        description="Offline GTA semantic replay. Outputs observations only; never presses keys."
    )
    cli.add_argument("--source", required=True, help="screen or a local video file")
    cli.add_argument("--monitor", type=int, default=1)
    cli.add_argument("--region", nargs=4, type=int)
    cli.add_argument("--capture-fps", type=int, default=30)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    cli.add_argument(
        "--amp",
        action="store_true",
        help="opt into autocast; benchmark before assuming it is faster",
    )
    cli.add_argument("--no-preview", action="store_true")
    cli.add_argument("--trace", type=Path)
    cli.add_argument("--max-frames", type=int, default=0)
    args = cli.parse_args()
    live = args.source == "screen"
    if not live and not Path(args.source).is_file():
        cli.error("A local video file is required.")
    if args.max_frames < 0:
        cli.error("max-frames must be nonnegative")
    if args.trace and args.trace.exists():
        cli.error("Trace exists; choose a new file.")
    perception = SemanticPerception(args.model, args.device, args.amp)
    tracker = Tracker()
    if args.region and (args.region[2] <= 0 or args.region[3] <= 0):
        cli.error("Region width and height must be positive")
    producer = (
        FrameProducer(
            frames(SimpleNamespace(source="screen", monitor=args.monitor, region=args.region)),
            args.capture_fps,
        )
        if live
        else None
    )
    capture = None if live else cv2.VideoCapture(args.source)
    trace = None
    count = 0
    try:
        trace = args.trace.open("x", encoding="utf-8") if args.trace else None
        if capture is not None and not capture.isOpened():
            cli.error("Video could not be opened")
        fps = capture.get(cv2.CAP_PROP_FPS) if capture is not None else args.capture_fps
        if not np.isfinite(fps) or fps <= 0:
            cli.error("Video timing is unavailable; track timing would be unreliable.")
        while True:
            if producer is not None:
                item = producer.get()
                if item is None:
                    if producer.slot.closed:
                        break
                    continue
                frame, captured_at = item
                if time.perf_counter() - captured_at > 0.15:
                    continue
                media_time = captured_at
            else:
                ok, frame = capture.read()
                if not ok:
                    break
                captured_at = time.perf_counter()
                media_time = count / fps
            result = perception.infer(frame, captured_at=captured_at)
            tracks = tracker.update(result.detections, media_time)
            record = dict(
                frame=count,
                media_time_s=media_time,
                timing_assumption="capture_completed_monotonic_clock"
                if live
                else "constant_video_fps",
                capture_to_result_ms=(time.perf_counter() - captured_at) * 1000,
                perception_ms=result.total_ms,
                detections=result.detections,
                tracks=[
                    dict(id=track.identifier, label=track.label, optical_ttc_s=track.optical_ttc())
                    for track in tracks
                ],
                control="DISARMED_NO_GAME_TELEMETRY",
            )
            if trace:
                trace.write(json.dumps(record) + "\n")
            count += 1
            if not args.no_preview:
                colors = PALETTE[result.mask][:, :, ::-1]
                overlay = cv2.resize(
                    colors, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
                )
                frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                cv2.putText(
                    frame,
                    f"PERCEPTION {result.total_ms:.0f} ms | INPUT DISARMED",
                    (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("GTA Assist Lab", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if args.max_frames and count >= args.max_frames:
                break
    finally:
        if capture is not None:
            capture.release()
        if producer is not None:
            producer.close()
        if trace:
            trace.close()
        if not args.no_preview:
            cv2.destroyAllWindows()
    print(
        json.dumps(
            dict(frames=count, control="no_keys_emitted", model=perception.metadata["model"])
        )
    )


if __name__ == "__main__":
    main()
