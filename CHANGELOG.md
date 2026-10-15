# Changelog

## Decision-pipeline demonstration — 2026-10-15

- Replace the withdrawn box-only GIF with scene overlays showing actual perception, approximate path conflicts and dry-run brake decisions.
- Retain the full-body illustration records separately from frozen evaluations, with source attribution and visible failure cases.

## Product workflows — 2026-10-15

- Added a portable replay desk built from real JSONL, with source hashes, frame controls, geometry validation and explicit missing-input states.
- Added a recorded GIF, Turkish quickstart, replay documentation and staged research roadmap.

# Changes

## Research preview — 2026-10-15

### Expanded data and temporal audit

- Acquire and verify 8,102 usable original GTA pairs with official split membership; exclude every earlier experiment ID.
- Complete DeepLab training and LR-ASPP distillation focused on pedestrian/rider/bicycle/vehicle classes; retain frozen-model comparisons and corruption results.
- Fix high-frame-rate lateral tracking, require explicit known gear/driver status, and connect observations to calibrated approximate hazards and dry-run braking.
- Add synthetic movement replay diagnostics, an independent 10,000-case scenario set, training-kernel profiling and failed live-readiness evidence.
- Extend parser/model/control tests and pin CI action revisions. No real keyboard input or gameplay validation is claimed.


### Perception and controller experiments

- Train a 19-class GTA semantic model in two real CUDA experiments and retain data partitions, validation selection, class support and confusion matrices.
- Add a semantic replay/capture lab, class-map checks, tracking helpers and a one-frame capture buffer. Live game inputs remain disabled.
- Add a bounded-pulse brake/reverse interlock, expiring dry-run leases, stale-telemetry disarming and seeded offline scenario evaluation.
- Record poor/unsupported classes, residual collisions and failed readiness gates. Do not claim end-to-end autonomous-driving accuracy.
- Benchmark the same model on CPU/CUDA, retain warm-up/scope information and retain separately measured FP32/autocast timings.

### Earlier PR revision

- Restore all 80 COCO labels in model order and validate them against the YOLO configuration.
- Replace the import-time screen loop with a configurable CLI and reusable detection functions.
- Add aspect-preserving letterboxing, vectorized confidence filtering and per-class nonmaximum suppression.
- Use current OpenCV output-layer names directly and handle empty outputs consistently.
- Replace the uncalibrated meter display with explicitly labeled pixel offsets. Traffic lights and hydrants no longer count as vehicles; cars are not assumed to be the player's car.
- Support recorded videos, capture regions, CPU/CUDA selection, headless processing, annotated video export and JSON run metrics.
- Add input validation, deterministic resource cleanup, model setup instructions and regression tests on Windows/Linux.

This remains a perception experiment. It does not steer a car, infer physical distance, or establish real-world driving safety.
