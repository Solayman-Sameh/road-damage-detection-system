def legacy_candidates(input_root=Path('/kaggle/input')):
    """Identify round-one checkpoints from saved metadata, regardless of folder labels."""
    expected_models = ['yolov8n', 'yolo11n', 'yolo26n']
    automatic = sorted(set(p for p in Path(input_root).rglob('*.pt')
                           if p.name == 'best.pt' or p.name in {f'{m}-best.pt' for m in expected_models}))
    explicit_paths = {name: Path(path) for name, path in LEGACY_WEIGHTS.items() if path}
    summaries = {}
    for path in sorted(set(automatic) | set(explicit_paths.values())):
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model = checkpoint.get('ema')
            if model is None:
                model = checkpoint.get('model')
            args = checkpoint.get('train_args', {})
            history = checkpoint.get('train_results') or {}
            names = getattr(model, 'names', {})
            names = list(names.values()) if isinstance(names, dict) else list(names)
            summaries[path] = {
                'model': Path(str(args.get('model', ''))).stem,
                'epochs': args.get('epochs'), 'imgsz': args.get('imgsz'),
                'seed': args.get('seed'), 'names': names,
                'history_epochs': len(history.get('epoch', [])),
            }
            del checkpoint, model
        except Exception as error:
            if path in explicit_paths.values():
                raise RuntimeError(f'Cannot inspect explicit legacy checkpoint: {path}') from error
            print(f'Ignoring unreadable automatic legacy candidate: {path} ({type(error).__name__})')
    gc.collect()
    candidates = {}
    for name in expected_models:
        choices = [explicit_paths[name]] if name in explicit_paths else automatic
        verified = {}
        for path in choices:
            saved = summaries.get(path, {})
            matches = (saved.get('model') == name and saved.get('epochs') == 20
                       and saved.get('imgsz') == 640 and saved.get('seed') == SEED
                       and saved.get('names') == NAMES and saved.get('history_epochs') == 20)
            if not matches:
                if name in explicit_paths:
                    raise AssertionError(f'{name}: explicit file is not the described 20-epoch / 640 round-one model: {path}; metadata={saved}')
                continue
            verified.setdefault(file_sha(path), path)
        if not verified:
            print(name, ': no verified attached v1 checkpoint; historical row only.')
            continue
        assert len(verified) == 1, f'Multiple different verified v1 {name} checkpoints; set LEGACY_WEIGHTS explicitly.'
        checksum, path = next(iter(verified.items()))
        candidates['legacy_'+name] = {
            'trial': 'legacy_'+name, 'model': name, 'train_imgsz': 640,
            'planned_epochs': 20, 'actual_epochs': 20, 'cls_pw': 0.0,
            'checkpoint': str(path), 'checkpoint_sha256': checksum, 'run': None,
            'origin': 'round1_reevaluated', 'status': 'historical_training',
        }
    return candidates
