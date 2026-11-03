"""Render the completed checkpoint audit without changing model files."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/downloaded_checkpoint_audit_2026-10-29'
records=json.loads((OUT/'checkpoints.json').read_text())
pair=json.loads((OUT/'yolo11_checkpoint_pair.json').read_text())
env=json.loads((OUT/'evaluation_environment.json').read_text())
scores=pd.read_csv(OUT/'common_1024_validation.csv')
classes=pd.read_csv(OUT/'common_1024_perclass.csv')
assert len(scores)==5
fig,axes=plt.subplots(1,2,figsize=(12,4))
scores.set_index('trial')[['map50','map50_95']].mul(100).plot.barh(ax=axes[0],color=['#286ba6','#e59934'])
axes[0].set(xlabel='Validation score (%)',xlim=(0,100),ylabel='',title='Same 973 images / 1024 / FP32 / NMS')
for r in records:
    if '/v2/' not in r['path'].replace('\\','/') or Path(r['path']).name!='best.pt':continue
    h=r['history'];axes[1].plot(h['epoch'],[x*100 for x in h['metrics/mAP50-95(B)']],label=r['model'])
axes[1].set(xlabel='Epoch',ylabel='Validation mAP50-95 (%)',title='Stored Kaggle training curves')
axes[1].legend();fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,6))
classes.pivot(index='class',columns='trial',values='ap50_95').mul(100).plot.barh(ax=ax)
ax.set(xlabel='AP50-95 (%)',xlim=(0,100),ylabel='',title='Per-class validation at common 1024 resolution')
ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'perclass_comparison.png',dpi=160);plt.close(fig)

lines=['# Downloaded checkpoint audit — 29 October 2026','',
       'All five inference checkpoints loaded and contained the expected seven SVRDD class names. '
       'All inspected floating-point state tensors were finite. Checkpoint SHA-256 hashes and full training histories are in checkpoints.json.', '',
       '## Model identity','', '| File | Actual model | History epochs | Optimizer state |',
       '|---|---|---:|---|']
for r in records:
    lines.append(f"| {Path(r['path']).relative_to(ROOT)} | {r['model']} | {r['history_epochs']} | {r['optimizer_present']} |")
lines += ['', 'YOLO11s best.pt exactly matches the EMA tensor state in epoch29.pt: '+str(pair['best_equals_periodic_ema'])+'. '
          'epoch29.pt has zero-based epoch 29, a 30-epoch history, and optimizer state. '
          'The stripped best.pt files have epoch=-1 and no optimizer, which is normal for completed inference exports.', '',
          '## Independent local validation','',
          'This is a fresh validation run, not a fresh test result. '
          'The 973-image split was reconstructed from the original archive by excluding the 27 validation images '
          'whose dHash distance to any higher-priority test image is <=3. Raw validation class support matched '
          '[576, 414, 219, 111, 393, 506, 337]. Test images were read only to reconstruct the screening hashes; '
          'no test predictions or scores were produced.', '',
          f"All models used Ultralytics {env['ultralytics']}, Torch {env['torch']} and {env['gpu']}. "
          'This differs from Kaggle Ultralytics 8.4.191 / Torch 2.11; local scores must remain separately labelled. '
          'Requested evaluation settings: 1024, batch 1, FP32, square padding, NMS, confidence 0.001, IoU 0.7, max_det 300, workers 0. '
          'Precision and recall below are the validator summaries, not fixed-confidence 0.25 counts.', '',
          '| Model | Training resolution | Epochs | mAP50 | mAP50-95 | Precision | Recall |',
          '|---|---:|---:|---:|---:|---:|---:|']
for _,r in scores.iterrows():
    lines.append(f"| {r.trial} | {r.train_imgsz} | {r.train_epochs} | {r.map50*100:.2f}% | {r.map50_95*100:.2f}% | {r.precision*100:.2f}% | {r.recall*100:.2f}% |")
lines += ['', 'The v1-to-v2 comparison changes model capacity, training resolution, epochs and batch size together. '
          'It measures the overall recipe improvement and cannot attribute the gain to one change. '
          'The v2 small-model comparison has the same nominal training settings, but one seed does not establish a statistically reliable family ranking.', '',
          '![Comparison](comparison.png)', '', '![Per-class comparison](perclass_comparison.png)', '',
          '## Notebook restart behavior','',
          'The inspected instant-sprint02-project02-road-damage-detectionv2.ipynb implements:', '',
          '- Skip a completed run only after complete.json, results.csv, finite losses and best.pt checksum verification.',
          '- Resume an interrupted matching run from weights/last-resumable.pt with optimizer state.',
          '- Recover completion metadata when the full epoch history and resumable checkpoint exist.',
          '- Train from pretrained weights when there is no recognized resumable checkpoint or verified completion.', '',
          'A standalone best.pt or arbitrary epoch29.pt is not automatically registered for recovery. '
          'Run directories depend on settings, dataset identity, package versions and worker-source hash, so changing those inputs '
          'can select a new directory. Local downloads do not make files available inside a fresh Kaggle session. '
          'No notebook settings or checkpoint files were changed during this audit.', '',
          '## Current resource screenshot','',
          'The supplied screenshot shows GPU utilization 99% and 96%, GPU memory 6.7 and 6.5 GiB of 15 GiB each, '
          'and system RAM 12.2 GiB of 30 GiB. It supports active GPU work at that instant and does not show memory exhaustion. '
          'It does not identify the earlier error or establish epoch progress on its own.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(scores.to_string(index=False))
print('REPORT',OUT/'REPORT.md')
