"""Diagnostic, video-agnostic cash persistence candidates.

Does not issue violation verdicts; model accuracy and temporal matching
must be validated on unseen videos before use in a final rule engine.
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.presence_events import analyze_presence_events

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'runs/cash_detection/v1/weights/best.pt'
OUTPUT_ROOT = ROOT / 'data/cash_auto_results'
STEP_SEC = 1.0
DETECTION_THRESHOLD = 0.10  # proposals only, not calibrated confidence
SIMILARITY_THRESHOLD = 0.75  # exploratory threshold from X-26, NOT validated
SEARCH_RADIUS = 35
MAX_SEED_AGE_SEC = 15.0
FOLLOWUP_OFFSETS_SEC = (0.0, 1.0, 3.0, 5.0)


def read_frame(cap, timestamp, fps, total):
    frame_idx = max(0, min(total - 1, round(timestamp * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def workstation_roi(frame, anchor):
    h, w = frame.shape[:2]
    ax, ay = round(float(anchor['x'])), round(float(anchor['y']))
    half_w, half_h = round(w * 0.45 / 2), round(h * 0.40 / 2)
    x1, y1 = max(0, ax-half_w), max(0, ay-half_h)
    x2, y2 = min(w, ax+half_w), min(h, ay+half_h)
    roi = frame[y1:y2, x1:x2]
    return roi.copy() if roi.size else None


def match_patch(roi, patch, reference_box):
    x1, y1, x2, y2 = reference_box
    h, w = roi.shape[:2]
    sx1 = max(0, x1-SEARCH_RADIUS)
    sy1 = max(0, y1-SEARCH_RADIUS)
    sx2 = min(w, x2+SEARCH_RADIUS)
    sy2 = min(h, y2+SEARCH_RADIUS)
    search = roi[sy1:sy2, sx1:sx2]
    ph, pw = patch.shape[:2]
    if search.shape[0] < ph or search.shape[1] < pw:
        return None
    scores = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(scores)
    if not np.isfinite(score):
        return None
    bx, by = sx1 + loc[0], sy1 + loc[1]
    return {'similarity': round(float(score), 4),
            'box': [bx, by, bx+pw, by+ph]}


def draw_and_save(roi, box, label, filepath):
    frame = roi.copy()
    if box is not None:
        a, b, c, d = map(int, box)
        cv2.rectangle(frame, (a,b), (c,d), (0,255,255), 2)
        cv2.putText(frame, label, (max(0,a), max(18,b-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
    cv2.imwrite(str(filepath), frame)


def scan_seeds(cap, fps, total, anchor, model, device, duration):
    seeds = []
    for sec in np.arange(0.0, duration, STEP_SEC):
        frame = read_frame(cap, float(sec), fps, total)
        if frame is None:
            continue
        roi = workstation_roi(frame, anchor)
        if roi is None:
            continue
        result = model.predict(roi, conf=DETECTION_THRESHOLD, imgsz=640,
                               device=device, verbose=False)[0]
        for box in result.boxes:
            label = result.names[int(box.cls.item())]
            if label != 'cash_on_desk':
                continue
            coordinates = box.xyxy[0].cpu().numpy().astype(int).tolist()
            a,b,c,d = coordinates
            if c-a < 4 or d-b < 4:
                continue
            seeds.append({'timestamp_sec': round(float(sec), 2),
                          'confidence': round(float(box.conf.item()), 4),
                          'box': coordinates})
    return seeds


def check_seed(cap, fps, total, duration, anchor, seed, away_sec, out_dir, event_index):
    seed_time = seed['timestamp_sec']
    first = read_frame(cap, seed_time, fps, total)
    if first is None:
        return None
    first_roi = workstation_roi(first, anchor)
    if first_roi is None:
        return None
    x1,y1,x2,y2 = seed['box']
    patch = first_roi[y1:y2, x1:x2].copy()
    if patch.size == 0:
        return None
    evidence = []
    path = out_dir / f'event_{event_index}_seed_{seed_time:.1f}s.jpg'
    draw_and_save(first_roi, seed['box'], 'YOLO seed', path)
    evidence.append(str(path.relative_to(ROOT)))
    checks = []
    for offset in FOLLOWUP_OFFSETS_SEC:
        when = round(away_sec + offset, 2)
        if when >= duration:
            continue
        frame = read_frame(cap, when, fps, total)
        if frame is None:
            continue
        roi = workstation_roi(frame, anchor)
        if roi is None:
            continue
        result = match_patch(roi, patch, seed['box'])
        if result is None:
            continue
        bx,by,_,_ = result['box']
        shift = math.hypot(bx-x1, by-y1)
        result['shift_px'] = round(shift, 2)
        # Similarity alone can match an unrelated nearby object.
        result['supports_same_stationary_patch'] = (
            result['similarity'] >= SIMILARITY_THRESHOLD and shift <= 15
        )
        result['timestamp_sec'] = when
        path = out_dir / f'event_{event_index}_{when:.2f}s.jpg'
        draw_and_save(roi, result['box'],
                      f"MATCH {result['similarity']:.2f}", path)
        result['evidence_frame'] = str(path.relative_to(ROOT))
        checks.append(result)
        evidence.append(result['evidence_frame'])
    matches = sum(x['supports_same_stationary_patch'] for x in checks)
    return {
        'rule_id': 'CH-02',
        'subtype': 'unattended_cash',
        'candidate_only': True,
        'confirmed_violation': False,
        'cash_seed': seed,
        'cashier_away_timestamp_sec': round(away_sec, 2),
        'matches': matches,
        'checks': checks,
        'temporal_support': matches >= 2,
        'evidence_frames': evidence,
        'limitations': ('Low-confidence cash model and template matching can '
                        'produce false positives. Not a confirmed violation.')
    }


def main():
    if len(sys.argv) != 2:
        print('Usage: python detect_cash_events.py data/videos/X-26.mov')
        sys.exit(1)
    video = Path(sys.argv[1]).resolve()
    if not video.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(f'Check paths: {video} and {MODEL_PATH}')
    print('\n=== AUTOMATIC CASH EVENTS (CANDIDATES ONLY) ===')
    presence = analyze_presence_events(video)
    anchor = presence.get('workstation_anchor')
    if anchor is None:
        print('No workstation anchor; no cash analysis possible.')
        return
    events = presence.get('events', [])
    away_events = [e for e in events
                   if isinstance(e, dict)
                   and e.get('type') == 'CASHIER_AWAY_FROM_WORKSTATION']
    print('Anchor:', anchor)
    print('Away events:', len(away_events))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot read {video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        raise RuntimeError('Invalid FPS/frame count')
    duration = total/fps
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('Device:', device, '| Duration:', round(duration,1), 'sec')
    model = YOLO(str(MODEL_PATH))
    out_dir = OUTPUT_ROOT / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        seeds = scan_seeds(cap, fps, total, anchor, model, device, duration)
        print('YOLO cash_on_desk proposals:', len(seeds))
        print('First seeds:', seeds[:8])
        candidates = []
        for event_idx, event in enumerate(away_events, 1):
            away = float(event.get('start_sec', event.get('timestamp_sec', -1)))
            eligible = [s for s in seeds if 0 <= away-s['timestamp_sec'] <= MAX_SEED_AGE_SEC]
            eligible.sort(key=lambda s: (-s['confidence'], abs(away-s['timestamp_sec'])))
            if not eligible:
                print(f'Away {away:.2f}s: no cash seed before departure')
                continue
            for seed in eligible[:3]:
                candidate = check_seed(cap, fps, total, duration, anchor,
                                       seed, away, out_dir, event_idx)
                if candidate:
                    candidates.append(candidate)
                    print(f"Away {away:.2f}s | seed {seed['timestamp_sec']:.1f}s "
                          f"conf={seed['confidence']:.3f} | "
                          f"matches={candidate['matches']}/{len(candidate['checks'])} "
                          f"| support={candidate['temporal_support']}")
                    if candidate['temporal_support']:
                        break
    finally:
        cap.release()
    result = {
        'video': video.name,
        'analysis_type': 'diagnostic_cash_temporal_candidates',
        'confirmed_violation': False,
        'workstation_anchor': anchor,
        'away_events': away_events,
        'yolo_cash_on_desk_proposals': seeds,
        'rule_candidates': candidates,
        'limitations': ['Prototype, not validated on unseen videos',
                        'A single low-confidence seed and template similarity do not prove cash is present',
                        'No cash proposals does not imply no cash was present']
    }
    outfile = out_dir / 'auto_cash_events.json'
    with outfile.open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print('Candidate pairs:', len(candidates))
    print('Saved:', outfile)
    print('DONE')


if __name__ == '__main__':
    main()
