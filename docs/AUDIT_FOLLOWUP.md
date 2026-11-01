# Review decisions — 2026-11-01

Display name: **GTA Perception Lab**. The repository URL remains compatible.

## Two distinct split protocols

The 100-frame-ID block method belongs to the earlier 1,000-image validation-only
exploration (run-001/002). Future runs of that legacy utility exclude ten IDs at
each block edge by default and record the guard band. A gap in frame IDs is only
a temporal-neighbor heuristic, not verified gameplay-session separation.

The published expanded DeepLab 63.5% mIoU uses `experiments/expanded/manifest.json`:
official train/val/test partitions, 798 representative test images and all 1,000
previous IDs excluded. The original split.mat SHA256 is recorded. It is not the
legacy block split. Original manifests, logs and scores are unchanged. Official
partitions still do not establish scene-session independence; no fresh score or
session-disjoint claim is made.

## Ego vehicle

An explicit camera-specific convex polygon can remove vehicle-class pixels
before connected-component extraction: `assist_lab.py --ego-mask camera.json`.
Schema: `{"polygon":[[0.1,0.95],[0.9,0.95],[0.9,1],[0.1,1]]}`. This is a schema
example, not validated calibration. Coordinates are normalized in the cropped
model input. Mask configuration SHA256 is recorded in every new trace.

The default is **off**. It does not remove person/rider/bicycle pixels and does
not change raw segmentation/mIoU. Synthetic pixel tests cover class boundaries.
A mask can hide real cars if the camera changes; there is no validated automatic
ego-mask, new inference, new bicycle recall, controller readiness or driving claim.
The original 200-frame replay, hood false brake and bicycle misses stay visible.

## Repository navigation

`autonomy/` and `assist_lab.py` are the current semantic-perception pipeline.
`vision.py`, `yolov3.cfg` and `coco.names` are the optional legacy YOLO demo and
compatibility frame-reader/CLI used by existing tests; they are not dead code.
They stay documented and separate from current model provenance, rather than
being deleted beneath a working entry point. [Legacy scope](LEGACY.md).

Use [experiment index](../experiments/INDEX.md) for the small summary and links.
Small immutable JSON and original `.py.txt` snapshots stay in Git so their recorded
hashes and history remain auditable; a generated index reduces navigation cost
without fabricating development history or removing failure evidence.
