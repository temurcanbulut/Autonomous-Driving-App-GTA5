"""Stage only explicitly allowed public browser assets. No model execution."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = [('docs/replay-demo.html', 'index.html')]

def main():
    parser = argparse.ArgumentParser(description='Stage the existing self-contained recorded GTA replay, without running a model.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', default='local-review')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        raise SystemExit('Output must be outside the repository source tree.')
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Output must be empty; existing files are never removed.')
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source_name, target_name in PUBLIC_FILES:
        source = ROOT / source_name
        if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
            raise SystemExit('Invalid public source asset.')
        payload = source.read_bytes()
        target = output / target_name
        if not target.resolve().is_relative_to(output):
            raise SystemExit('Invalid target path.')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        manifest.append({'file': target_name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
    (output / '.nojekyll').write_text('', encoding='utf-8')
    (output / 'BUILDINFO.json').write_text(json.dumps({'revision': args.revision, 'files': manifest, 'model_execution': False}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files_staged': len(manifest), 'model_execution': False}))

if __name__ == '__main__':
    main()
