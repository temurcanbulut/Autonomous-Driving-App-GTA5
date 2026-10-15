# Perception-to-brake demonstration

The README animation explains the existing trained model and dry-run controller. It is a controlled visual diagnostic, not natural GTA gameplay or a new model-training run.

Three illustrated cases show a pedestrian crossing, a bicycle crossing and a vehicle entering the assumed path. Actual semantic predictions are tinted on the scene. Solid boxes are model components; the dashed box is the inserted actor. The panel shows target association, approximate path conflict and the recorded brake request. No game input is emitted.

## Source and protocol

- Images and semantic labels: Richter, Vineet, Roth & Koltun, *Playing for Data: Ground Truth from Computer Games*, ECCV 2016, [official dataset](https://download.visinf.tu-darmstadt.de/data/from_games/). Derived illustrations are used for research and education; raw source images and labels are not bundled.
- Model: the released DeepLab checkpoint, SHA256 `b242f06cb943955170a24cf9b77ad7126c9721b65cfc85ea5ee4628dd708f3f0`. Weights and training are unchanged.
- [Summary and source hashes](../experiments/full-body-illustration/summary.json): five sequences, 40 frames each at 10 Hz, static validation background and composited validation actors. A whole-body pedestrian component was selected for readability using the criteria retained in the summary. This is an illustrative selection, not a random benchmark.
- Artificial telemetry: 12 m/s, known forward gear, no driver override. Assumed camera: 1.5 m height, 60° vertical field of view, horizon at 0.4 of image height.
- Target association uses class-specific box IoU ≥ 0.15. This diagnostic threshold does not measure collision avoidance. Individual JSONL records retain detections, approximate hazard estimates, controller reasons and inference timings.

| Sequence | Visible-target frames | Target matched | Dry-brake frames |
|---|---:|---:|---:|
| Pedestrian crossing | 38 | 36 | 10 |
| Bicycle crossing | 40 | 5 | 18 |
| Vehicle cut-in | 34 | 27 | 14 |
| Sidewalk pedestrian | 40 | 40 | 1 |
| No inserted actor | 0 | 0 | 1 |

The ego hood is mistaken for a car and causes the first-frame brake in all five sequences, including the no-inserted-actor control. Brake frames therefore cannot be attributed to successful target detection. Bicycle matching remains poor. No stopping distance, collision prevention or natural temporal generalization was tested.

The original [class-box diagnostic](../experiments/composite-diagnostic-class-boxes/) and frozen 798-image perception evaluation remain unchanged. The full-body illustration uses a different pedestrian sprite, so its match count cannot establish a model improvement over that diagnostic.

## Animation provenance

`images/test-explainer.gif`: 1280 × 790, 60 frames, 20.7 seconds including explanatory holds. Every second frame from the three crossing sequences is illustrated. Labels and side panels are editorial annotations of recorded values, not a live UI or a performance recording.

GIF SHA256: `ba37e2c18dfd7a3a1c0375a6dcbbe8feef96b521fea22657d5a4475e67509611`.
