# Changes

## Unreleased

- Restore all 80 COCO labels in model order and validate them against the YOLO configuration.
- Replace the import-time screen loop with a configurable CLI and reusable detection functions.
- Add aspect-preserving letterboxing, vectorized confidence filtering and per-class nonmaximum suppression.
- Use current OpenCV output-layer names directly and handle empty outputs consistently.
- Replace the uncalibrated meter display with explicitly labeled pixel offsets. Traffic lights and hydrants no longer count as vehicles; cars are not assumed to be the player's car.
- Support recorded videos, capture regions, CPU/CUDA selection, headless processing, annotated video export and JSON run metrics.
- Add input validation, deterministic resource cleanup, model setup instructions and regression tests on Windows/Linux.

This remains a perception experiment. It does not steer a car, infer physical distance, or establish real-world driving safety.
