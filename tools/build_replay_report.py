"""Make a portable, offline replay viewer from bounded JSONL observations.

No GPU, ML framework, server or external assets are needed to inspect the output.
The renderer never turns detections or dry-run decisions into a driving verdict.
"""

import argparse
import base64
import hashlib
import html
import json
import math
from pathlib import Path

MAX_BYTES = 16 * 1024 * 1024
MAX_FRAMES = 50000
MAX_LINE = 128 * 1024
ASSETS = Path(__file__).resolve().parent / "replay_assets"


def number(value):
    try:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def load_trace(path):
    """Project untrusted input into a small, validated display schema."""
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Trace exceeds 16 MiB")
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("Trace exceeds 16 MiB")
    rows = []
    previous_time = -1
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        if len(line) > MAX_LINE or len(rows) >= MAX_FRAMES:
            raise ValueError("Trace line/frame limit exceeded")
        try:
            record = json.loads(line)
        except (ValueError, RecursionError) as error:
            raise ValueError(f"Invalid JSON at line {line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected object at line {line_number}")
        timestamp = record.get("media_time_s")
        if not number(timestamp) or timestamp < 0 or timestamp < previous_time:
            raise ValueError("Trace timestamps must be finite, nonnegative and monotonic")
        previous_time = timestamp
        detections = record.get("detections", [])
        if not isinstance(detections, list) or len(detections) > 256:
            raise ValueError("Invalid detection list")
        found = []
        for item in detections:
            if not isinstance(item, dict):
                raise ValueError("Invalid detection")
            box = item.get("box")
            label = item.get("label")
            confidence = item.get("confidence")
            if (
                not isinstance(label, str)
                or len(label) > 80
                or not number(confidence)
                or not 0 <= confidence <= 1
                or not isinstance(box, list)
                or len(box) != 4
                or not all(number(value) and 0 <= value <= 1 for value in box)
                or box[0] + box[2] > 1.00001
                or box[1] + box[3] > 1.00001
            ):
                raise ValueError("Invalid normalized detection geometry")
            found.append(dict(label=label, confidence=confidence, box=box))
        latency = record.get("perception_ms", record.get("inference_ms"))
        if latency is not None and (not number(latency) or latency < 0):
            raise ValueError("Invalid latency")
        decision = record.get("decision", {})
        if not isinstance(decision, dict):
            raise ValueError("Invalid decision")
        brake = decision.get("s_pressed")
        if brake is not None and not isinstance(brake, bool):
            raise ValueError("Invalid dry-run brake flag")
        emitted = record.get("game_input_emitted")
        if emitted is not None and not isinstance(emitted, bool):
            raise ValueError("Invalid input-emitted flag")
        state = decision.get("state", record.get("control", "not_recorded"))
        reason = decision.get("reason", "not_recorded")
        if not all(isinstance(value, str) and len(value) <= 200 for value in (state, reason)):
            raise ValueError("Invalid control description")
        target = record.get("target_detected")
        if target is not None and not isinstance(target, bool):
            raise ValueError("Invalid diagnostic target flag")
        rows.append(
            dict(
                time=timestamp,
                detections=found,
                latency=latency,
                brake=brake,
                state=state,
                reason=reason,
                input_emitted=emitted,
                target_detected=target,
            )
        )
    if not rows:
        raise ValueError("Trace is empty")
    latencies = sorted(row["latency"] for row in rows if row["latency"] is not None)
    diagnostic = [row for row in rows if row["target_detected"] is not None]
    return dict(
        name=path.stem,
        sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
        frames=rows,
        summary=dict(
            frames=len(rows),
            span_s=rows[-1]["time"] - rows[0]["time"],
            latency_median_ms=(
                (latencies[(len(latencies) - 1) // 2] + latencies[len(latencies) // 2]) / 2
                if latencies
                else None
            ),
            dry_brake_frames=sum(row["brake"] is True for row in rows),
            diagnostic_matched=sum(row["target_detected"] is True for row in diagnostic),
            diagnostic_frames=len(diagnostic),
            input_evidence="emitted"
            if any(row["input_emitted"] is True for row in rows)
            else "recorded_false"
            if all(row["input_emitted"] is False for row in rows)
            else "not_recorded",
        ),
    )


def digest(text):
    return base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii")


def render_report(traces, context):
    payload = json.dumps(dict(context=context, traces=traces), ensure_ascii=True, allow_nan=False)
    # JSON values are data inside a script: prevent HTML parser termination.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    script = "const replay = " + payload + ";\n" + (ASSETS / "viewer.js").read_text("utf-8")
    style = (ASSETS / "viewer.css").read_text("utf-8")
    csp = (
        f"default-src 'none'; script-src 'sha256-{digest(script)}'; "
        f"style-src 'sha256-{digest(style)}'; base-uri 'none'; form-action 'none'"
    )
    template = (ASSETS / "viewer.html").read_text("utf-8")
    return template.replace("{{CSP}}", html.escape(csp, quote=True)).replace(
        "{{STYLE}}", style
    ).replace("{{SCRIPT}}", script)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("traces", nargs="+", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--context", default="User-supplied trace; source conditions unverified.")
    args = cli.parse_args()
    if not 1 <= len(args.traces) <= 8 or len(args.context) > 1000:
        cli.error("Use 1–8 traces and a context note of at most 1000 characters")
    try:
        traces = [load_trace(path) for path in args.traces]
        report = render_report(traces, args.context)
        # Fail instead of replacing an existing report.
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            output.write(report)
    except (OSError, ValueError) as error:
        cli.exit(2, f"Report not written: {error}\n")
    print(f"Wrote {args.output}: {sum(len(trace['frames']) for trace in traces)} recorded frames")


if __name__ == "__main__":
    main()
