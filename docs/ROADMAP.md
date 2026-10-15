# Research roadmap

The useful product today is a perception and replay lab. Live assistance stays disabled while its acceptance checks fail.

- [x] Expanded GTA semantic teacher/student training and frozen test records
- [x] Bounded capture, temporal tracks and dry-run controller with expiring S leases
- [x] Portable replay desk and downloadable trained checkpoints
- [ ] Calibrate the actual GTA camera and label an ego-vehicle mask across camera changes; measure false suppression as well as false brakes
- [ ] Collect natural temporal pedestrian, bicycle and cut-in sequences; keep recording sessions independent across training/validation/test
- [ ] Evaluate temporal bicycle/rider association and detection misses by scenario, distance and occlusion
- [ ] Remove the ego-hood false brake and measure the complete perception-to-control pipeline against no-assist baselines
- [ ] Independently verify telemetry, driver override and input watchdog in offline GTA before any live input adapter

Do not tune camera masks on held-out test outcomes or turn a box-only simulation into a gameplay accuracy claim. Additional training must record source rights, split membership, seeds, checkpoints and unsuccessful runs.
