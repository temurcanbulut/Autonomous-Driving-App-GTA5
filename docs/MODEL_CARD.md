# GTA Assist Lab

GTA scene perception, temporal actor tracking and offline longitudinal-assistance experiments, developed in the original repository. **Research prototype: live assistance criteria remain unmet.** The code emits no real game/keyboard input. Semantic segmentation, synthetic movement diagnostics and oracle-state vehicle simulations are different measurements.

## Run locally

Python 3.11+ in a virtual environment:

```sh
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r training/requirements.txt
python Autonomous-Driving-App-GTA5/assist_lab.py --model best.pt --source gameplay.mp4 --no-preview --trace observations.jsonl
```

Choose a compatible PyTorch build through the [official installer](https://pytorch.org/get-started/locally/) for other hardware. The latest measured environment is Windows/Python 3.12, torch 2.14.1+cu130, torchvision 0.29.1+cu130 and RTX 4070 Laptop GPU (8 GB). Model weights are separate local artifacts; runtime never downloads them implicitly, checks architecture/class order and uses `weights_only=True` loading.

`--source screen --monitor 1` explicitly starts capture. A replaceable one-frame slot drops stale frames. Screen capture has not been evaluated during GTA gameplay. CUDA FP32 is the default; `--amp` is optional and its measured latency is retained separately. Capture, decode, preview and game rendering are excluded from the inference timings below.

## Expanded dataset and actual training

Source: [Playing for Data](https://download.visinf.tu-darmstadt.de/data/from_games/) by Richter, Vineet, Roth and Koltun, ECCV 2016. The original dataset is for **research/education**, including when a mirror has a different license label. Images and full weights are not committed to this repository.

All 24,966 original labels were scanned. Selected image members were fetched from official ZIP archives using verified range requests, ZIP member CRC and SHA256 hashes. Of 8,112 selected pairs, ten original image/mask size mismatches were explicitly excluded. The usable sample counts are:

| Partition | Images | Role |
|---|---:|---|
| train | 5,999 | Rare-class enriched training pool |
| val | 795 | Representative validation |
| val_challenge | 227 | Separate rare-class validation |
| test | 798 | Representative frozen test |
| test_challenge | 283 | Separate rare-class frozen test |

Training includes 4,739 person frames, 246 bicycle frames, 1,046 rider frames and 5,361 car frames with at least 100 original labelled pixels. Representative validation/test have only 5/14 bicycle-containing frames at that cutoff; the rare challenges have 29/83. Bicycle generalization therefore remains weakly supported. All 1,000 IDs from earlier experiments are excluded from every expanded partition. Official split membership and cross-partition hashes/IDs are checked; separation by unseen gameplay session is **not established**. The rare challenge distribution must not be pooled into the representative score.

```sh
python training/prepare_gta.py --root data/previous
python training/expand_gta.py --root data/expanded --previous-manifest data/previous/manifest.json
python training/verify_manifest.py --data data/expanded --previous-manifest data/previous/manifest.json --output runs/dataset-audit.json
```

The precise sample IDs, source/prepared hashes, excluded pairs and acquisition plan are in `../experiments/expanded/`. A separate 63-bit perceptual-hash screening found no train/evaluation candidate pairs at Hamming distance 4 or below. This approximate check does not prove session independence or absence of other visual similarities. Fixed seeds do not guarantee identical floating-point training across drivers and library versions.

The expanded experiments use target-centered crops, color/brightness augmentation, weighted frame sampling, class-weighted focal loss and focus-class Dice loss. A DeepLabV3/MobileNetV3 teacher completed **1 + 20 + 8 epochs** across bootstrap, continued and high-resolution stages. Run 003 was initially configured for 40 epochs and stopped after one completed epoch for training-kernel optimization; its preserved status records that interruption. The LR-ASPP/MobileNetV3 student completed **10 additional epochs** on this pool, mixing labelled supervision with frozen-teacher distillation (temperature 2, weight 0.3). Training uses replacement sampling; an epoch is not a promise that every unique frame was seen.

Checkpoint selection is `0.4 × representative-val mIoU + 0.6 × mean rare-val focus-class IoU`. Each run retains its initial checkpoint as a candidate. Candidate choice and latency criteria were written before new test predictions; the selected replay candidate is **Expanded DeepLab**. All model test results are retained. No test-driven retraining or threshold adjustment followed the final evaluation.

Run configs, frozen training scripts, full epoch logs, validation scores, checkpoint hashes and precision settings are preserved under `../experiments/run-003` through `run-006`. Earlier runs 001/002 used an exploratory 731/135/134 split from a validation-only mirror and torch 2.11.0+cu128; their old holdout scores are not directly compared with the expanded holdout. The previous model is reevaluated on the same new test images below.

## Frozen results

| Frozen model | Test mIoU | Person IoU / recall | Bicycle IoU | Car IoU | CUDA FP32 p50 / p95 |
|---|---:|---:|---:|---:|---:|
| Previous LR-ASPP | 48.7% | 56.1% / 75.9% | 10.1% | 77.7% | 18.3 / 22.2 ms |
| Expanded DeepLab | 63.5% | 68.7% / 84.0% | 55.8% | 87.6% | 29.1 / 32.9 ms |
| Expanded LR-ASPP | 58.5% | 63.5% / 81.6% | 43.3% | 84.7% | 21.4 / 31.1 ms |

IoU and recall above refer to labelled **pixels**, not collision avoidance or driving accuracy. Runtime input resolutions are stored per model (teacher 960×544; student and previous model 768×432). Ignored pixels, including much of the ego vehicle, cannot establish correct/incorrect predictions. Predicted connected components are semantic proxies, not true instances or COCO AP.

Latency uses the same first 30 validation frames, batch one and five warm-up frames, including normalization, transfers/synchronization, probabilities and components. It excludes capture/decode/game rendering and is not GTA FPS. It is one laptop run, not a multi-machine performance guarantee.

`final-*-test.json` stores per-class support, confusion matrices and component proxies at IoU 0.25/0.5. `final-*-corruptions-*.json` stores paired clean/darkened/JPEG-35/9-pixel-horizontal-blur results on 200 seeded images per partition. These synthetic corruptions are not measured weather, actual gameplay video or a YouTube-codec benchmark.

## Motion and dry braking

Temporal tracking estimates lateral motion over a time window, including at 60/240 FPS, and predicts boxes for association. `autonomy/hazards.py` requires explicit camera calibration and uses an approximate flat-ground model; it does not learn metric depth or collision probabilities. Still-frame segmentation training does not teach a driving policy.

```sh
python tools/replay_control.py --observations observations.jsonl --telemetry telemetry.jsonl --calibration camera.json --arm-dry-run --output dry-control.jsonl
python tools/replay_composites.py --data data/expanded --model best.pt --output runs/composite-diagnostic
python tools/simulate_assist.py --cases 10000 --seed 920000 --output runs/scenarios.json
```

Telemetry rows require finite `timestamp`, `speed_mps`, and explicit boolean `forward_gear`, `manual_override`; missing/invalid/stale status disarms. Calibration supplies `height_m`, `vertical_fov_degrees`, `horizon_y` on the same media clock. The brake controller bounds S leases, forces release periods and stops requesting S near zero speed under an explicit maximum-deceleration assumption. `DryRunSink` simulates lease expiry. There is no OS keyboard adapter or independent real-input watchdog.

The composite diagnostic uses five artificial cutout sequences built from validation images: crossing pedestrian, crossing bicycle, cut-in car, sidewalk actor and no inserted actor. Model predictions feed tracking/hazard/control code, but fixed speed, static background and synthetic motion are not GTA physics; no collision-success claim follows from these sequences. The corrected v2 boxes follow target-class pixels rather than rider context. The bicycle matched only 5/40 frames under the diagnostic rule. The no-inserted-actor sequence requested a brake on its first frame: the visible ego hood was treated as an external car. These failures remain unresolved; a camera-specific ego mask and actual gameplay evidence are required. Blindly cropping a generic bottom strip would conceal nearby road users and is not an accepted fix. Original v1 logs are retained separately.

An additional 10,000 oracle-state simplified-dynamics scenarios produced no reverse events or additional collisions relative to the no-assist baseline. Collisions still occurred in pedestrian, cut-in, stopped-vehicle, late-hazard and stale-sensor cohorts. This is a separate simulation with exact obstacle states, not an image-driven game evaluation.

`../experiments/final-acceptance.json` retains failed live-readiness checks. Real gameplay camera/ego-mask calibration, independently checked speed/gear signals, input/watchdog validation and perception-to-brake game scenarios remain required before actuation.

## Reproduce software checks

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m unittest discover -s tests -v
python -m bandit -r Autonomous-Driving-App-GTA5 training tools -ll
```

The local Windows run passed 57 tests, including analytic loss/gradient checks, high-frame-rate tracking, stale/unknown telemetry, finite-clock guards, reverse protection, actor-box geometry and thousands of property-generated low-speed states. Linux/Windows CI jobs are configured but remote CI status is shown on the repository; the recorded numerical results were measured locally. Python advisory scans do not cover all native/CUDA binaries.

## Original YOLO entry point

```sh
python Autonomous-Driving-App-GTA5/autonomous-driving-app-gta5.py --source gameplay.mp4 --headless --metrics metrics.json
```

The original 80-class COCO YOLOv3 demo remains available with [the author's weights](https://pjreddie.com/media/files/yolov3.weights) placed alongside its config. Its displayed offsets are pixels, not meters. It does not steer or brake.
