# Replay desk

Open replay-demo.html locally for the bundled five-sequence synthetic validation diagnostic. It embeds a display projection of 200 real JSONL rows. The original records remain under experiments/composite-diagnostic-class-boxes/. The source SHA256 values cover the original bytes, not the projected display schema.

## Build your own viewer

```sh
python tools/build_replay_report.py observations.jsonl --output replay.html --context "Recorded video; constant-FPS timing"
```

Python 3.11+ standard library only. Up to eight input traces, 16 MiB and 50,000 frames per trace, 128 KiB per line and 256 detections per frame. The output must not already exist. No model, server or network connection is required to view it.

Each row needs a finite, nonnegative, monotonically nondecreasing media_time_s. Detections contain label, confidence in [0,1] and normalized [x,y,width,height] inside the image. Optional perception_ms (or inference_ms) is nonnegative. Optional decision.s_pressed and game_input_emitted are booleans. Missing flags are displayed as not recorded, never inferred as false.

assist_lab.py produces perception traces; tools/replay_composites.py produces synthetic diagnostic traces including decisions. A perception-only trace has no controller decision and the viewer says so. Labels, context and reasons are rendered as text. Source filenames and context may be sensitive; review the generated page before sharing.

## Reading the page

- The frame slider and Previous/Next move through recorded rows. The chart's horizontal axis is frame index, while the readout uses recorded timestamps.
- Next brake request skips to the next row with a recorded dry-run S request. It does not play, simulate or issue any keypress.
- Rectangles are normalized image boxes, not physical distances or verified object instances. No source video is embedded.
- Median latency uses available recorded values only. Producer scope varies: inference_ms and perception_ms are not necessarily identical operations.
- A synthetic target match is an association diagnostic. It is not recall on natural gameplay. The bundled no-actor first-frame false brake and bicycle misses remain visible.

Regenerate the bundled demo with all five original traces, in the README sequence order, and the existing synthetic-context note. The viewer style/script sources live in tools/replay_assets; output CSP hashes are computed from the final embedded bytes.
