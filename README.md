# GTA 5 Perception · Object detection experiment

Real-time screen or recorded-video object detection using YOLOv3 and OpenCV. This repository continues the original GTA 5 experiment with a configurable pipeline, correct COCO labels, and reproducible run metrics.

It is a **perception demo**: there are no driving controls and no calibrated physical-distance estimates. “Screen offset” is measured in pixels from the image's bottom-center to a detected vehicle's box center. It does not tell you which vehicle is physically closest, or identify the player's own vehicle.

## Setup

Python 3.11 or newer:

```sh
git clone https://github.com/temurcanbulut/Autonomous-Driving-App-GTA5.git
cd Autonomous-Driving-App-GTA5
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux/macOS, then:

```sh
python -m pip install -r requirements.txt
```

Download [YOLOv3 weights from the original author](https://pjreddie.com/media/files/yolov3.weights) (about 237 MB) and put `yolov3.weights` alongside `yolov3.cfg` in `Autonomous-Driving-App-GTA5/`. Configuration and all 80 COCO names are included. Weights are not committed or downloaded automatically. See the [upstream model documentation](https://pjreddie.com/darknet/yolo/) for model details.

The defaults are resolved relative to the script, so it can be launched from another working directory. Custom `--weights`, `--config` and `--names` paths are also supported. The label count must match every YOLO head's class count.

## Live screen

Use a windowed/borderless GTA 5 session on monitor 1:

```sh
python Autonomous-Driving-App-GTA5/autonomous-driving-app-gta5.py
```

Press **Q** or **Escape** in the preview to quit, or **Ctrl+C** in the terminal. Screen capture may require OS permission; screen capture support depends on the OS/display server. Recorded-video mode avoids those dependencies.

Capture only the game region using global screen coordinates:

```sh
python Autonomous-Driving-App-GTA5/autonomous-driving-app-gta5.py --region 100 100 1280 720
```

## Recorded video and metrics

Process the same clip for comparable runs:

```sh
python Autonomous-Driving-App-GTA5/autonomous-driving-app-gta5.py --source gameplay.mp4 --headless --input-size 320 --max-frames 300 --output annotated.avi --metrics metrics-320.json
```

Repeat with `--input-size 416` or `608` and different output names. Reports include frame count, processing FPS, mean inference/detector time, thresholds and OpenCV version. They include warm-up frames; model loading is excluded. Capture, drawing and output are included in processing FPS. Compare runs using the same source, frame limit, device and output options. No universal FPS gain or accuracy improvement is claimed without measurements on representative footage.

Exports contain no audio. `--output-fps` controls encoded playback rate (default 30), independently of processing FPS; match it to the source video's rate if preserving timing matters. Existing output files are not overwritten. The JSON metrics summarize the run; they do not contain frame-level detection results or accuracy metrics.

## Options

| Option | Default | Purpose |
|---|---|---|
| `--source` | `screen` | Screen or local video path |
| `--monitor` | `1` | Screen monitor index |
| `--region LEFT TOP WIDTH HEIGHT` | full monitor | Capture region; overrides monitor |
| `--input-size` | `416` | 320, 416 or 608 square model input |
| `--confidence` | `0.5` | Minimum detection confidence |
| `--nms` | `0.4` | Overlap suppression threshold |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--all-classes` | off | Include all 80 classes |
| `--headless` | off | Disable preview window |
| `--max-frames` | `0` | Stop after N frames; 0 means until quit/EOF |
| `--output` | none | Annotated `.avi` or `.mp4` |
| `--output-fps` | `30` | Encoded playback FPS |
| `--metrics` | none | JSON run metrics |

Default classes: people, bicycles, cars, motorcycles, buses, trains, trucks, traffic lights and stop signs. CUDA requires a CUDA-enabled OpenCV DNN build and compatible hardware; ordinary PyPI wheels are generally used in CPU mode. A missing CUDA device produces an explicit error rather than silently changing the requested mode.

## Pipeline and limitations

1. Capture one frame or decode one video frame.
2. Letterbox it to square model input while preserving aspect ratio.
3. Run YOLOv3 using OpenCV DNN output layer names.
4. Decode class scores and restore/clamp boxes to source coordinates.
5. Suppress overlapping boxes **within each class**, then annotate and optionally export.

OpenCV's Darknet Region layer already incorporates objectness into its class scores; the decoder does not multiply it again. See the [OpenCV Region implementation](https://github.com/opencv/opencv/blob/4.x/modules/dnn/src/layers/region_layer.cpp). COCO class names/order follow the [upstream label file](https://github.com/pjreddie/darknet/blob/master/data/coco.names).

The model is trained on COCO photographs, so GTA scenes may produce misses and false positives. There is no temporal tracking, ego-vehicle separation, depth calibration, lane detection or control loop. Those require separate evaluation; changing input size also changes the speed/accuracy tradeoff.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests need no game or model weights. They cover label order, letterbox coordinate restoration, class-aware NMS, confidence handling, pixel offsets, bad model setup, video export/readback and metrics. CI runs on Windows/Linux with Python 3.11/3.13. A real model/footage smoke test is separate from this deterministic suite.

See [CHANGELOG.md](CHANGELOG.md) for the changes. Earlier source and screenshots remain in Git history.
