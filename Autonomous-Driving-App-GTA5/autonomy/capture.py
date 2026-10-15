"""Capture producer with one replaceable frame; no stale work queue."""

import time
from threading import Event, Thread

from .perception import LatestFrame


class FrameProducer:
    def __init__(self, source, max_fps=30):
        if not 1 <= max_fps <= 120:
            raise ValueError("Capture FPS must be between 1 and 120")
        self.source = source
        self.interval = 1 / max_fps
        self.slot = LatestFrame()
        self.stop = Event()
        self.error = None
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            for frame in self.source:
                if self.stop.is_set():
                    break
                self.slot.put(frame, time.perf_counter())
                if self.stop.wait(self.interval):
                    break
        except Exception as error:
            self.error = error
        finally:
            try:
                if hasattr(self.source, "close"):
                    self.source.close()
            except Exception as error:
                self.error = self.error or error
            finally:
                self.slot.close()

    def get(self):
        result = self.slot.get()
        if result is None and self.error is not None:
            raise RuntimeError("Frame capture failed") from self.error
        return result

    def close(self):
        self.stop.set()
        self.slot.close()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("Capture backend did not terminate in time")
