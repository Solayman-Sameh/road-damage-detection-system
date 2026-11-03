# Recovery helper: this cell runs after the training-engine definitions.
# Attach all three downloaded best.pt files as Kaggle input, or put them anywhere
# below /kaggle/working. Folder names do not matter for audited downloads.
AUDITED_DOWNLOADS = {
    'v8_small': (22603562, '5a5b96a5204dffb8371ea621b6461bc3939ec0b11fc9f6ebfbc3af38d684a3ba'),
    'v11_small': (19262042, 'aed9032517943b4a5ffe30a01a74c974bc8e7e625b6b7a6e6ede1f688e1a6c63'),
    'v26_medium': (44120153, 'fab0e117d8871eeea47a45a98e33963a524b7751bae793c796258bf1a457c0c4'),
}


def register_finished_download(key, source):
    spec = TRIALS[key]
    run = trial_dir(key, spec)
    source = Path(source)
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    model = checkpoint.get('ema')
    if model is None:
        model = checkpoint.get('model')
    args = checkpoint.get('train_args', {})
    assert model is not None and list(model.names.values()) == NAMES, 'Wrong class map.'
    assert Path(str(args.get('model', ''))).stem == spec['model'], 'Wrong model architecture.'
    for field in ['epochs', 'imgsz', 'batch', 'seed', 'cls_pw']:
        assert args.get(field) == spec[field], f'{key}: saved {field} differs from this experiment.'
    history = pd.DataFrame(checkpoint.get('train_results') or {})
    history.columns = history.columns.str.strip()
    assert 'epoch' in history and history.epoch.tolist() == list(range(1, spec['epochs']+1)), (
        f'{key}: best.pt is not evidence of a completed {spec["epochs"]}-epoch run. '
        'Keep its resumable checkpoint; do not mark partial training complete.')
    losses = [c for c in history if 'loss' in c and c.startswith(('train/', 'val/'))]
    assert losses and np.isfinite(history[losses].to_numpy()).all(), 'Invalid training losses.'
    assert 'metrics/mAP50-95(B)' in history and np.isfinite(history['metrics/mAP50-95(B)']).all()
    checksum = file_sha(source)
    if key in AUDITED_DOWNLOADS:
        assert checksum == AUDITED_DOWNLOADS[key][1], 'This is not the audited downloaded model.'
    assert not (OUTPUT/'frozen_test_plan.json').exists(), 'Do not alter an already frozen test plan.'
    run.mkdir(parents=True, exist_ok=True)
    weights = run/'weights'; weights.mkdir(exist_ok=True)
    destination = weights/'best.pt'
    if source.resolve() != destination.resolve():
        if destination.exists():
            assert file_sha(destination) == checksum, 'Different best.pt already occupies this run.'
        else:
            temporary = weights/'best.importing.pt'
            shutil.copy2(source, temporary)
            assert file_sha(temporary) == checksum
            os.replace(temporary, destination)
    history.to_csv(run/'results.csv', index=False)
    original_seconds = float(history['time'].iloc[-1]) if 'time' in history else None
    assert original_seconds is None or (math.isfinite(original_seconds) and original_seconds >= 0)
    atomic_json(run/'training_spec.json', {
        'train': args, 'imported': True, 'source': str(source),
        'checkpoint_sha256': checksum, 'checkpoint_ultralytics': checkpoint.get('version'),
        'current_evaluation_data_id': DATA_ID,
        'provenance_note': 'Recovered checkpoint metadata. Original data hash/environment file was not embedded; current environment describes evaluation.'})
    atomic_json(run/'class_weights.json', {
        'cls_pw': args['cls_pw'], 'weights': [1.0]*len(NAMES), 'counts': None,
        'note': 'Unweighted training; exact post-deduplication trainer counts were not embedded.'})
    (run/'timing.jsonl').write_text(json.dumps({
        'seconds': 0.0, 'original_training_seconds': original_seconds,
        'imported': True, 'exit_code': None})+'\n')
    atomic_json(run/'complete.json', {
        'status': 'imported_completed', 'actual_epochs': len(history),
        'planned_epochs': spec['epochs'], 'losses_finite': True,
        'checkpoint_sha256': checksum, 'imported': True,
        'training_ultralytics': checkpoint.get('version')})
    del checkpoint, model
    gc.collect()
    completed_status(run, spec)
    print(f'SKIP TRAINING: {key} | verified {len(history)} epochs | {destination}', flush=True)
    return run


def find_audited_download(key):
    size, checksum = AUDITED_DOWNLOADS[key]
    for root in [Path('/kaggle/working'), Path('/kaggle/input')]:
        for path in root.rglob('*.pt'):
            if path.is_file() and path.stat().st_size == size and file_sha(path) == checksum:
                return path
    return None


def train_or_reuse(key):
    spec = TRIALS[key]
    run = trial_dir(key, spec)
    if (run/'complete.json').exists():
        completed_status(run, spec)
        print(f'SKIP TRAINING: {key} | verified existing completion', flush=True)
        return run
    if key in AUDITED_DOWNLOADS:
        source = find_audited_download(key)
        if source is not None:
            return register_finished_download(key, source)
        if REQUIRE_RECOVERED_MODELS:
            raise FileNotFoundError(
                f'{key}: attach your audited best.pt as Kaggle input or copy it below /kaggle/working. '
                'Training is blocked to prevent accidentally repeating the completed model.')
    best = run/'weights/best.pt'
    resumable = run/'weights/last-resumable.pt'
    if best.exists() and not resumable.exists():
        # A completed loose best.pt can be registered; partial best.pt must not silently count as done.
        return register_finished_download(key, best)
    if RUN_TRAINING:
        assert BUDGET_PASSED, 'Run the pilot/explicit pilot-bypass cell first.'
    print(f'{"RESUME" if resumable.exists() else "TRAIN"}: {key}', flush=True)
    return run_trial(key, spec)


# Resolve all three completed models before evaluation. Missing downloads
# stop here, rather than after spending GPU time on another model.
if REQUIRE_RECOVERED_MODELS:
    for key in AUDITED_DOWNLOADS:
        spec = TRIALS[key]
        run = trial_dir(key, spec)
        assert (run/'complete.json').exists() or find_audited_download(key) is not None, (
            f'Missing {spec["model"]}: attach its downloaded best.pt before Run All.')
print('Recovery helper ready: all three completed models are required.' if REQUIRE_RECOVERED_MODELS else 'Recovery helper ready: reuse completed runs and train missing models when enabled.')
