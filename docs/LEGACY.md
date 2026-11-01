# Legacy YOLO compatibility

The original `Autonomous-Driving-App-GTA5/autonomous-driving-app-gta5.py` launcher
calls `vision.py`. Its optional Darknet model uses `yolov3.cfg` and `coco.names`;
no weights are bundled or downloaded automatically. `assist_lab.py` also reuses
the tested video frame reader from vision.py, but current semantic checkpoints do
not use the Darknet config/classes. `autonomy/models.py` is their architecture map.

The legacy entry point remains for compatibility and has no enabled game input.
Use the Perception Lab replay and semantic entry point for current research.
Third-party config attribution is in [licensing scope](../LICENSING.md).
