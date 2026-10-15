# Security and experiment boundaries

The project emits observations and offline control records. It has no OS keyboard adapter. Do not connect dry-run S leases to live game input before independently validating camera geometry, telemetry, manual override and a separate watchdog. The current acceptance checks fail.

Model checkpoints are executable-framework inputs. Use the published release manifest, verify SHA256, and only load files from sources you trust. Runtime uses PyTorch weights_only loading, strict model metadata and state-dict checks; this does not make arbitrary checkpoints trustworthy.

Replay reports accept bounded JSONL and render validated values as data in a self-contained page with a hashed Content Security Policy. They make no external requests. A trace hash identifies bytes, not the truth of a recording. Do not share private screen content or file paths in diagnostics.

For a vulnerability, use GitHub private reporting if available; otherwise request a private contact route through the maintainer profile. Avoid public disclosure of credentials or personal recordings. No response-time or bounty commitment is made.
