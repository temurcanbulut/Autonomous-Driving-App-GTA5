import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("replay_report", ROOT / "tools/build_replay_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class ReplayReportTests(unittest.TestCase):
    def test_bundled_failure_is_visible_in_summary(self):
        path = ROOT / "experiments/composite-diagnostic-class-boxes/no_inserted_actor.jsonl"
        trace = report.load_trace(path)
        self.assertEqual(trace["summary"]["frames"], 40)
        self.assertEqual(trace["summary"]["dry_brake_frames"], 1)
        self.assertEqual(trace["summary"]["input_evidence"], "recorded_false")
        self.assertEqual(trace["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertTrue(trace["frames"][0]["brake"])

    def test_untrusted_geometry_timing_and_flags_fail(self):
        base = dict(media_time_s=0, detections=[])
        invalid = [dict(base, media_time_s=float("nan")), dict(base, media_time_s=10**400),
                   dict(base, media_time_s=-1),
                   dict(base, decision=dict(s_pressed="false")),
                   dict(base, detections=[dict(label="car", confidence=0.9, box=[0.9, 0, 0.2, 0.1])])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            for value in invalid:
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    report.load_trace(path)
            path.write_text('{"media_time_s":2}\n{"media_time_s":1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                report.load_trace(path)

    def test_html_escapes_untrusted_context_and_has_exact_script_hash(self):
        rendered = report.render_report([], '</script><script>alert("x")</script>')
        self.assertNotIn('</script><script>alert', rendered)
        self.assertIn('\\u003c/script\\u003e', rendered)
        script = rendered.split('<script>', 1)[1].split('</script>', 1)[0]
        self.assertIn(report.digest(script), rendered)
        self.assertIn("default-src &#x27;none&#x27;", rendered)

    def test_cli_never_clobbers_existing_output(self):
        source = ROOT / "experiments/composite-diagnostic-class-boxes/no_inserted_actor.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            output.write_text("keep me", encoding="utf-8")
            result = subprocess.run([sys.executable, str(ROOT / "tools/build_replay_report.py"),
                                     str(source), "--output", str(output)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text("utf-8"), "keep me")


if __name__ == "__main__":
    unittest.main()
