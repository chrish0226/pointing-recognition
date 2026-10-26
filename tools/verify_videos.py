"""Fresh full-video inference and export checks using a local input manifest.

Manifest format: {"clip_name": {"source": "path/to/video.mp4"}}.
Names and optional human annotations belong to this verifier, never the rules.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from balanced_pointing_rules import PROFILE
from realtime_rtmw3d import rule_fingerprint

KEYS = ('forward', 'left', 'right')


def metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {path}')
    result = {'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
              'fps': cap.get(cv2.CAP_PROP_FPS),
              'size': [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]}
    cap.release()
    return result


def export(name, spec, out, fingerprint):
    assert re.fullmatch(r'[A-Za-z0-9_-]+', name), name
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(spec['source']).resolve()
    video = folder / 'result.mp4'
    raw = out / '_raw' / f'{name}.mp4'
    assert not video.exists(), f'Result already exists: {video}'
    info = metadata(source)
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'}
    with (folder / 'run.log').open('w', encoding='utf-8') as log:
        command = [sys.executable, '-u', str(ROOT / 'realtime_rtmw3d.py'),
                   '--source', str(source), '--no-display', '--save', str(raw),
                   '--log', str(folder / 'events.jsonl'), '--log-frames']
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError((folder / 'run.log').read_text(encoding='utf-8')[-6000:])
    summary = json.loads((folder / 'events.summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'end_of_file' and summary['processed_frames'] == info['frames']
    assert summary['rule_profile'] == PROFILE
    for key, value in fingerprint.items():
        assert summary[key] == value, key
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(raw),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
                    '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart', str(video)],
                   check=True, capture_output=True)
    encoded = metadata(video)
    assert encoded['frames'] == info['frames'] and encoded['size'] == info['size']
    assert abs(encoded['fps'] - info['fps']) < .001
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(video),
                              '-progress', 'pipe:1', '-nostats', '-f', 'null', '-'],
                             check=True, capture_output=True, text=True)
    decoded_frames = [int(line.split('=', 1)[1]) for line in decoded.stdout.splitlines()
                      if line.startswith('frame=')]
    assert decoded_frames[-1] == info['frames']
    counts = {}; appearances = Counter(); events = []; timings = []; frame_count = 0
    for line in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['type'] == 'direction':
            current = counts.setdefault(row['track_id'], dict.fromkeys(KEYS, 0))
            current[row['direction']] += 1
            assert row['counts'] == current
            events.append({'track_id': row['track_id'], 'direction': row['direction'],
                           'time_s': row['time_s']})
        if row['type'] == 'frame':
            assert row['frame'] == frame_count
            frame_count += 1
            timings.append(row['timings']['inference_ms'])
            for person in row['people']:
                tid = person['track_id']; appearances[tid] += 1
                current = counts.get(tid, dict.fromkeys(KEYS, 0))
                assert person['state']['counts'] == current
                assert person['state']['passed'] == all(current.values())
                for ref in (person['body_reference'], person.get('orientation', {}).get('orientation_reference')):
                    if ref is not None:
                        axes = np.array([ref[key] for key in ('right', 'up', 'forward')])
                        np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-5)
    assert frame_count == info['frames']
    for person in summary['people_current_generation']:
        assert person['counts'] == counts.get(person['track_id'], dict.fromkeys(KEYS, 0))
    dominant = appearances.most_common(1)[0][0] if appearances else None
    actual = counts.get(dominant, dict.fromkeys(KEYS, 0))
    result = {'name': name, 'source': str(source), 'video': str(video), **info,
              'model_inference': 'fresh full video', 'rule_profile': PROFILE, **fingerprint,
              'h264_full_decode': True, 'frame_states_and_axes_verified': True,
              'dominant_track_id': dominant, 'track_frames': dict(appearances),
              'dominant_counts': actual, 'events': events,
              'people': [{'track_id': p['track_id'], 'counts': p['counts'],
                          'passed': p['passed'], 'passed_s': p['passed_s']}
                         for p in summary['people_current_generation']],
              'wall_processing_fps': summary['wall_processing_fps'],
              'mean_inference_ms': statistics.mean(timings)}
    (folder / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    cap = cv2.VideoCapture(str(video))
    target = [e for e in events if e['track_id'] == dominant]
    at = target[-1]['time_s'] + .1 if target else info['frames'] / info['fps'] / 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(info['frames'] - 1, round(at * info['fps'])))
    ok, image = cap.read(); cap.release()
    assert ok
    cv2.imencode('.jpg', image)[1].tofile(str(folder / 'preview.jpg'))
    print(json.dumps({'finished': name, 'frames': frame_count, 'counts': actual,
                      'events': target, 'wall_fps': summary['wall_processing_fps']}, ensure_ascii=False), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs' / PROFILE)
    args = ap.parse_args()
    specs = json.loads(args.manifest.read_text(encoding='utf-8'))
    out = args.output.resolve()
    (out / '_raw').mkdir(parents=True, exist_ok=True)
    fingerprint = rule_fingerprint()
    results = []
    for name, spec in specs.items():
        results.append(export(name, spec, out, fingerprint))
        report = {'created_at': datetime.now().astimezone().isoformat(), 'profile': PROFILE,
                  **fingerprint, 'results': results,
                  'scope': 'Known videos used for tuning/regression, not held-out accuracy measurement.'}
        (out / 'batch_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    assert rule_fingerprint() == fingerprint
    print('ALL EXPORTS AND FRAME STATES VERIFIED', flush=True)


if __name__ == '__main__':
    main()
