# Experiment records

These records are stored without Git line-ending conversion. Manifest and frozen training-script fingerprints refer to their original bytes, including line endings.

`run-001` and `run-002` are earlier experiments on a validation-only 1,000-image mirror with an exploratory split. Their original holdout scores must not be directly compared with the expanded test.

`run-003` through `run-006` use the expanded official-partition pool. Run 003 planned 40 epochs and was stopped after one completed epoch for kernel optimization. Runs 004/005 completed 20/8 epochs; run 006 completed ten student-distillation epochs. Check each run's status and complete log, rather than inferring completed epochs from its planned configuration.

`final-*-test.json` reevaluates all three frozen candidates on the same expanded representative and rare challenge subsets. `model-selection.json` records the validation/latency decision made before those predictions. Different input resolutions are retained per model.

`final-*-corruptions-*.json` records paired synthetic image corruptions. `final-*-benchmark.json` measures perception only. Neither is gameplay performance.

`composite-diagnostic-class-boxes` contains the corrected class-specific-box diagnostic, source-image hashes and per-frame outputs. The bicycle result is weak and the no-inserted-actor case has an ego-hood brake request. The original v1 summary is retained separately. No model weights or test thresholds were changed after final testing.

`simulation-heldout-004.json` is a separate 10,000-case oracle-state simplified-dynamics simulation. Residual collisions remain. `final-acceptance.json` records failed live-readiness criteria.

`expanded/manifest.json` and its acquisition plan contain image IDs, original/prepared hashes, split membership and excluded size-mismatch pairs. No raw dataset archive is included. Dataset and illustrative image credit: Richter, Vineet, Roth & Koltun, *Playing for Data* (ECCV 2016), research/educational use.
