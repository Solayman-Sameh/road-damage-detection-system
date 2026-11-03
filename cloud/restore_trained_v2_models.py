"""Disposable Kaggle cell: restore the three audited models/v2 downloads.

Paste after notebook library setup, or use the updated evaluation notebook,
which already includes this cell. Attach models/v2 as Kaggle input unchanged.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil

# Standalone/pasted cells default to recovery. The canonical notebook exposes
# this switch in its first settings cell so the same notebook can train anew.
REUSE_TRAINED_MODELS = globals().get('REUSE_TRAINED_MODELS', True)
if REUSE_TRAINED_MODELS:
    RUN_TRAINING = False
    RUN_WEIGHTING_TRIAL = False
    RUN_PILOT = False

RESTORE_V2_MODELS = {
    'yolov8s': {
        'run': 'v8_small-5b484e147c68', 'bytes': 22603562,
        'sha256': '5a5b96a5204dffb8371ea621b6461bc3939ec0b11fc9f6ebfbc3af38d684a3ba',
    },
    'yolo11s': {
        'run': 'v11_small-5f29b009a0a6', 'bytes': 19262042,
        'sha256': 'aed9032517943b4a5ffe30a01a74c974bc8e7e625b6b7a6e6ede1f688e1a6c63',
    },
    'yolo26m': {
        'run': 'v26_medium-9eb3c3c7ba5a', 'bytes': 44120153,
        'sha256': 'fab0e117d8871eeea47a45a98e33963a524b7751bae793c796258bf1a457c0c4',
    },
}


def restore_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restore_trained_v2_models(input_root='/kaggle/input', working_root='/kaggle/working'):
    input_root, working_root = Path(input_root), Path(working_root)
    output = working_root/'road_models_v2'
    destinations = {name: output/'runs'/spec['run']/'weights/best.pt'
                    for name, spec in RESTORE_V2_MODELS.items()}
    sources = {}
    # Existing files must match; never overwrite another experiment silently.
    for name, target in destinations.items():
        if target.exists():
            if restore_sha256(target) != RESTORE_V2_MODELS[name]['sha256']:
                raise RuntimeError(f'{name}: a different best.pt occupies {target}. Use a fresh session/output folder.')
            sources[name] = target
    # Match bytes, not folder names: the uploaded yolov8n folder holds YOLOv8s.
    sizes = {spec['bytes']: name for name, spec in RESTORE_V2_MODELS.items()}
    for root in [input_root, working_root]:
        if len(sources) == len(RESTORE_V2_MODELS):
            break
        if not root.is_dir():
            continue
        for path in root.rglob('*.pt'):
            if not path.is_file():
                continue
            name = sizes.get(path.stat().st_size)
            if name and name not in sources and restore_sha256(path) == RESTORE_V2_MODELS[name]['sha256']:
                sources[name] = path
                if len(sources) == len(RESTORE_V2_MODELS):
                    break
    missing = sorted(set(RESTORE_V2_MODELS)-set(sources))
    if missing:
        raise FileNotFoundError(
            f'Missing audited downloads: {missing}. Attach your entire models/v2 folder as Kaggle input. '
            'No training will start. Folder names may stay exactly as uploaded.')
    manifest = {}
    for name, target in destinations.items():
        source = sources[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_name('best.restore.tmp')
            shutil.copy2(source, temporary)
            if restore_sha256(temporary) != RESTORE_V2_MODELS[name]['sha256']:
                raise RuntimeError(f'Copy verification failed: {name}')
            os.replace(temporary, target)
        manifest[name] = {'source': str(source), 'restored': str(target),
                          'sha256': restore_sha256(target)}
        print(f'RESTORED {name}: {target}', flush=True)
    manifest_path = output/'restored_downloads.json'
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2))
    os.replace(temporary, manifest_path)
    print('All three trained models verified. Training is OFF. Continue with recovery registration and evaluation.', flush=True)
    return manifest


if __name__ == '__main__':
    if REUSE_TRAINED_MODELS:
        restored_v2_models = restore_trained_v2_models()
    else:
        print('Fresh-training mode: uploaded-model restoration is optional and was skipped.')
