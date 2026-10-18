"""Conservative local YOLO object-to-process signals for cash workstation videos.

Generic COCO objects are only *proposals*: a handbag/backpack is not proof of
an official collection bag; a cup alone is not a proved rule breach. All
collection/workspace outputs remain candidate_only=True.
"""
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / 'yolov8n.pt'
OUTPUT_ROOT = PROJECT_ROOT / 'data' / 'cash_auto_results'
BAG_CLASSES = {'handbag', 'backpack', 'suitcase'}
DRINK_CLASSES = {'cup', 'bottle', 'wine glass'}
FOOD_CLASSES = {'sandwich', 'cake', 'donut', 'pizza', 'apple', 'banana', 'orange'}
PHONE_CLASSES = {'cell phone'}
WORKSTATION_CLASSES = {'laptop', 'keyboard', 'mouse', 'tv'}
OBJECT_CLASSES = BAG_CLASSES | DRINK_CLASSES | FOOD_CLASSES | PHONE_CLASSES | WORKSTATION_CLASSES

# Safety gate for generic COCO bag labels. NOT a calibrated confidence score
# or a substitute for training on actual collection bags and bank equipment.
# The original model accepts raw bag labels at 0.17; on X-15 the desk device
# was mislabelled as handbag with scores below 0.37.
MIN_BAG_CONF_FOR_REVIEW = min(1.0, max(0.0,
    float(os.getenv('FORTE_BAG_REVIEW_MIN_CONF', '0.55'))))
MIN_BAG_STRONG_FRAMES = max(3,
    int(os.getenv('FORTE_BAG_REVIEW_MIN_FRAMES', '3')))


def _near(a, b, radius=0.09):
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= radius


def _track_bags(records):
    """Associate stationary bag proposals by screen location, independent of label.

    Detection data is not 'tracking' with identity; the proximity clusters are
    interpreted only as recurring proposals of an object in the same area.
    """
    tracks = []
    for record in records:
        t = record['timestamp_sec']
        for det in record['detections']:
            if det['class'] not in BAG_CLASSES:
                continue
            x1, y1, x2, y2 = det['box_norm']
            center = ((x1+x2)/2.0, (y1+y2)/2.0)
            options = [track for track in tracks
                       if _near(track['center'], center)
                       and t not in track['timestamps']]
            if options:
                tr = min(options, key=lambda tr: math.dist(tr['center'], center))
                n = len(tr['observations'])
                tr['center'] = [(tr['center'][0]*n + center[0])/(n+1),
                                (tr['center'][1]*n + center[1])/(n+1)]
            else:
                tr = {'center': list(center), 'timestamps': set(), 'observations': []}
                tracks.append(tr)
            tr['timestamps'].add(t)
            tr['observations'].append({'time': t, 'detection': det})
    return tracks




def _track_objects(records, classes, radius=0.10):
    """Associate generic repeated proposals spatially, not by verified identity."""
    tracks = []
    for record in records:
        t = record['timestamp_sec']
        for det in record['detections']:
            if det['class'] not in classes:
                continue
            x1, y1, x2, y2 = det['box_norm']
            center = ((x1+x2)/2.0, (y1+y2)/2.0)
            possible = [tr for tr in tracks if t not in tr['times']
                        and _near(tr['center'], center, radius)]
            if possible:
                tr = min(possible, key=lambda z: math.dist(z['center'], center))
                n = len(tr['observations'])
                tr['center'] = [(tr['center'][0]*n+center[0])/(n+1),
                                (tr['center'][1]*n+center[1])/(n+1)]
            else:
                tr = {'center': list(center), 'times': set(), 'observations': []}
                tracks.append(tr)
            tr['times'].add(t)
            tr['observations'].append({'time': t, 'detection': det})
    return tracks


def _qualified_observations(track, min_conf, min_frames, min_span_sec):
    strong = sorted([ob for ob in track['observations']
                     if ob['detection']['confidence'] >= min_conf],
                    key=lambda x: x['time'])
    if len({ob['time'] for ob in strong}) < min_frames:
        return []
    if strong[-1]['time'] - strong[0]['time'] < min_span_sec:
        return []
    return strong


def _near_cashier_area(center, presence, frame_width, frame_height, radius=0.44):
    """Very coarse association to cashier's screen location; NOT cash proximity."""
    anchor = (presence or {}).get('workstation_anchor') or {}
    try:
        x = float(anchor['x'])/frame_width
        y = float(anchor['y'])/frame_height
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return _near(center, (x,y), radius)


def _screen_departure_candidate(tracks, away_events, presence, width, height):
    """Candidate for visual review, never proof a session was unlocked."""
    for away in sorted(away_events, key=lambda e: e.get('start_sec', 0)):
        try:
            start = float(away.get('start_sec', away.get('timestamp_sec')))
            duration = float(away.get('duration_sec') or 0)
            end = float(away.get('end_sec') or (start+duration))
        except (TypeError, ValueError):
            continue
        if max(duration, end-start) < 3.0:
            continue
        for track in tracks:
            if not _near_cashier_area(track['center'], presence, width, height):
                continue
            strong = _qualified_observations(track, 0.42, 2, 1.0)
            near = [ob for ob in strong if start-8 <= ob['time'] <= end+3]
            if len(near) >= 2:
                return {'timestamp_sec': round(start, 2),
                        'equipment_observation': min(near, key=lambda x: abs(x['time']-start)),
                        'detected_classes': sorted({ob['detection']['class'] for ob in near}),
                        'away_duration_sec': round(max(duration, end-start), 2)}
    return None


def _qualify_bag_track(observations):
    """Decide if a generic bag track is even usable as a REVIEW proposal.

    Multiple weak predictions for a static piece of equipment are NOT
    independent evidence that the equipment is a bag. Require several
    sufficiently strong observations before adding BAG signals to timeline.

    Returns (strong_observations, rejection_reason). Pure for unit testing.
    This is a conservative screening gate, not bag identity verification.
    """
    obs = sorted(observations, key=lambda z: z['time'])
    if len({x['time'] for x in obs}) < 4 or obs[-1]['time'] - obs[0]['time'] < 3:
        return [], 'insufficient_temporal_support'
    strong = [x for x in obs
              if float(x['detection'].get('confidence', 0)) >= MIN_BAG_CONF_FOR_REVIEW]
    if len({x['time'] for x in strong}) < MIN_BAG_STRONG_FRAMES:
        return [], 'weak_generic_bag_classification'
    return strong, None


def _bag_departure_overlap(strong_obs, away_events):
    """Only strong observations can establish before/after temporal context."""
    for away in away_events:
        try:
            start = float(away.get('start_sec', away.get('timestamp_sec')))
            end = float(away.get('end_sec') or
                        start + float(away.get('duration_sec') or 0))
        except (TypeError, ValueError):
            continue
        if any(x['time'] <= start for x in strong_obs) and any(
            start + 2 <= x['time'] <= end + 2 for x in strong_obs
        ):
            return {'start_sec': start, 'duration_sec': max(0, end - start)}
    return None


def _write_evidence(cap, fps, total, t, bbox, label, output_path):
    import cv2
    frame_index = max(0, min(total-1, round(t*fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (20,215,255), 3)
    cv2.putText(frame, label, (max(0,x1), max(25,y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (0,0,0), 4)
    cv2.putText(frame, label, (max(0,x1), max(25,y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (20,215,255), 2)
    if not cv2.imwrite(str(output_path), frame):
        return None
    return str(output_path.relative_to(PROJECT_ROOT))


def analyze_workspace_objects(video_path, presence=None):
    """Produce events and review-only candidates using a LOCAL generic model.

    Requires a pre-downloaded `yolov8n.pt` in project root. Does not download
    weights or call any remote inference. Errors are represented as status,
    never mistaken for absence of objects.
    """
    empty = {'events': [], 'rule_candidates': [], 'detections_by_class': {},
             'model': 'yolov8n.pt / generic COCO', 'model_limitations':
             ['Generic bag cannot be identified as an official collection bag',
              'A cup/food proposal does not establish a cash-discipline violation',
              'Workspace/cash proximity is not yet verified geometrically',
              'Visible equipment does not show that a workstation is unlocked']}
    if not MODEL_PATH.is_file():
        return dict(empty, status='skipped', error='Missing local yolov8n.pt. '
                    'Copy the model downloaded by forte_object_probe.py to the project root.')
    import cv2
    try:
        from ultralytics import YOLO
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return dict(empty, status='failed', error='Unable to open video')
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or total <= 0:
                return dict(empty, status='failed', error='Invalid video frame metadata')
            duration = total / fps
            max_frames = max(1, int(os.getenv('FORTE_OBJECT_MAX_FRAMES', '100')))
            interval = max(1.0, duration / max_frames)
            model = YOLO(str(MODEL_PATH))
            records = []
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
            counts = {}
            n = 0
            while n * interval < duration and n < max_frames:
                t = round(n*interval, 3)
                n += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        max(0, min(total-1, round(t*fps))))
                ok, frame = cap.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                prediction = model.predict(frame, conf=0.17, imgsz=640,
                                           device='cpu', verbose=False, max_det=80)[0]
                detections = []
                for b in prediction.boxes:
                    name = model.names[int(b.cls.item())]
                    if name not in OBJECT_CLASSES:
                        continue
                    conf = float(b.conf.item())
                    if name in DRINK_CLASSES | FOOD_CLASSES and conf < 0.30:
                        continue
                    if name in PHONE_CLASSES and conf < 0.45:
                        continue
                    if name in WORKSTATION_CLASSES and conf < 0.42:
                        continue
                    box = [round(float(x), 1) for x in b.xyxy[0].tolist()]
                    detections.append({'class': name, 'confidence': round(conf, 4),
                                       'box': box, 'box_norm': [round(box[0]/width,4),
                                       round(box[1]/height,4), round(box[2]/width,4),
                                       round(box[3]/height,4)]})
                    counts[name] = counts.get(name, 0) + 1
                records.append({'timestamp_sec': t, 'detections': detections})

            # Evidence stays under the existing public evidence mount.
            output_dir = OUTPUT_ROOT / Path(video_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)
            signals, candidate_specs = [], []
            bags = _track_bags(records)
            away_events = [e for e in (presence or {}).get('events', [])
                           if isinstance(e, dict) and
                           e.get('type') == 'CASHIER_AWAY_FROM_WORKSTATION']
            suppressed_bag_tracks = []
            for idx, track in enumerate(bags, 1):
                obs = sorted(track['observations'], key=lambda z: z['time'])
                strong, reject_reason = _qualify_bag_track(obs)
                if reject_reason:
                    # Keep diagnostics, but DO NOT show BAG as an observed
                    # event or link this unreliable identity to COL-03.
                    suppressed_bag_tracks.append({
                        'track_index': idx,
                        'reason': reject_reason,
                        'raw_observations': len({o['time'] for o in obs}),
                        'max_generic_confidence': round(max(
                            float(o['detection'].get('confidence', 0)) for o in obs
                        ), 4),
                    })
                    continue

                overlapping_away = _bag_departure_overlap(strong, away_events)
                # Evidence uses strong observations only; weak printer-like
                # boxes cannot become a misleading yellow BAG label.
                evidence_obs = [strong[0], strong[-1]]
                if overlapping_away:
                    start = overlapping_away['start_sec']
                    before = [o for o in strong if o['time'] <= start]
                    after = [o for o in strong if o['time'] >= start + 2]
                    if before and after:
                        evidence_obs = [before[-1], after[0]]
                evidence = []
                for j, ob in enumerate(evidence_obs):
                    p = output_dir / ('object_bag_%d_%d.jpg' % (idx, j))
                    ev = _write_evidence(cap, fps, total, ob['time'],
                                         ob['detection']['box'],
                                         'GENERIC BAG? REVIEW', p)
                    if ev:
                        evidence.append(ev)
                event = {'type': 'BAG_STATIONARY_PROPOSAL',
                         'timestamp_sec': strong[0]['time'],
                         'duration_sec': round(strong[-1]['time']-strong[0]['time'],2),
                         'observed_frames': len({o['time'] for o in strong}),
                         'object_classes': sorted({o['detection']['class'] for o in strong}),
                         'confidence_max': max(o['detection']['confidence'] for o in strong),
                         'object_identity_confirmed': False,
                         'object_identity_status': 'generic_coco_unverified',
                         'cashier_departure_overlap': bool(overlapping_away),
                         'evidence_frames': evidence, 'source': 'generic_object_detector',
                         'candidate_only': True}
                signals.append(event)
                if overlapping_away:
                    candidate_specs.append({'subtype': 'bag_unattended',
                        'timestamp_sec': overlapping_away['start_sec'],
                        'evidence_frames': evidence,
                        'trigger_signal': 'BAG_STATIONARY_PROPOSAL',
                        'missing_context': ['OBJECT_IDENTITY_VERIFICATION',
                                            'OFFICIAL_COLLECTION_BAG_IDENTIFICATION',
                                            'COLLECTION_OPERATION_CONFIRMED',
                                            'BAG_CUSTODY_VERIFICATION'],
                        'extra_description': (
                          'Repeated high-score generic bag proposals overlap with '
                          'cashier departure, but bag identity is not verified. '
                          'May be ordinary equipment: REVIEW ONLY.')})
            # Food/drink: retain the FIRST observation of an associated object,
            # not just the frame with the highest YOLO score. Limit to two clusters.
            food_tracks = _track_objects(records, DRINK_CLASSES | FOOD_CLASSES)
            food_tracks.sort(key=lambda tr: (-len(tr['times']),
                           -max(ob['detection']['confidence'] for ob in tr['observations'])))
            for idx, track in enumerate(food_tracks[:2], 1):
                observations = sorted(track['observations'], key=lambda z: z['time'])
                first = observations[0]
                peak = max(observations, key=lambda z: z['detection']['confidence'])
                evidence = []
                for j, ob in enumerate([first, peak]):
                    if j and ob['time'] == first['time']:
                        continue
                    ev = _write_evidence(cap, fps, total, ob['time'],
                        ob['detection']['box'],
                        'FOOD/DRINK? REVIEW', output_dir/('object_drink_%d_%d.jpg' % (idx, j)))
                    if ev:
                        evidence.append(ev)
                signals.append({'type': 'FOOD_DRINK_OBJECT_PROPOSAL',
                    'timestamp_sec': first['time'],
                    'peak_timestamp_sec': peak['time'],
                    'object_class': peak['detection']['class'],
                    'confidence': peak['detection']['confidence'],
                    'sampled_frames_with_food_drink': len(track['times']),
                    'evidence_frames': evidence, 'source': 'generic_object_detector',
                    'candidate_only': True})
                candidate_specs.append({'subtype': 'food_or_drink_near_cash',
                    'timestamp_sec': first['time'], 'evidence_frames': evidence,
                    'trigger_signal': 'FOOD_DRINK_OBJECT_PROPOSAL',
                    'missing_context': ['CASH_WORKSPACE_PROXIMITY',
                                        'INTERNAL_BANK_POLICY_VALIDATION'],
                    'extra_description': ('Generic %s proposal first seen at %.1fs; '
                        'highest model score %.2f at %.1fs (%d sampled frames). '
                        'Cash proximity and policy breach NOT confirmed.'
                        % (peak['detection']['class'], first['time'],
                           peak['detection']['confidence'], peak['time'], len(track['times'])))})

            # Personal phones: only stable, strong proposals near cashier area.
            # Do not turn into a binary violation without workstation/cash policy.
            phone_tracks = _track_objects(records, PHONE_CLASSES)
            for idx, track in enumerate(phone_tracks, 1):
                if not _near_cashier_area(track['center'], presence, width, height):
                    continue
                strong = _qualified_observations(track, 0.50, 2, 1.0)
                if not strong:
                    continue
                first = strong[0]
                peak = max(strong, key=lambda z: z['detection']['confidence'])
                ev = _write_evidence(cap, fps, total, first['time'],
                    first['detection']['box'], 'PHONE? REVIEW',
                    output_dir/('object_phone_%d.jpg' % idx))
                evidence = [ev] if ev else []
                signals.append({'type': 'PERSONAL_DEVICE_PROPOSAL',
                    'timestamp_sec': first['time'], 'confidence': peak['detection']['confidence'],
                    'observed_frames': len({ob['time'] for ob in strong}),
                    'evidence_frames': evidence, 'source': 'generic_object_detector',
                    'candidate_only': True})
                candidate_specs.append({'subtype': 'phone_in_cash_zone',
                    'timestamp_sec': first['time'], 'evidence_frames': evidence,
                    'trigger_signal': 'PERSONAL_DEVICE_PROPOSAL',
                    'missing_context': ['DEVICE_IDENTITY_AND_POLICY', 'CASH_WORKSPACE_PROXIMITY'],
                    'extra_description': ('Repeated generic phone proposals near the cashier '
                        'screen position. Phone identity and cash zone NOT confirmed.')})
                break  # only one unverified mobile-device alert per video

            # Screen/keyboard visually present during cashier absence: WS-01
            # candidate only. Image CANNOT prove an unlocked, active session.
            screen_tracks = _track_objects(records, WORKSTATION_CLASSES)
            screen_event = _screen_departure_candidate(
                screen_tracks, away_events, presence, width, height)
            if screen_event:
                ob = screen_event['equipment_observation']
                ev = _write_evidence(cap, fps, total, ob['time'],
                    ob['detection']['box'], 'WORKSTATION? REVIEW',
                    output_dir/'object_workstation_1.jpg')
                evidence = [ev] if ev else []
                signals.append({'type': 'WORKSTATION_UNATTENDED_PROPOSAL',
                    'timestamp_sec': screen_event['timestamp_sec'],
                    'away_duration_sec': screen_event['away_duration_sec'],
                    'equipment_classes': screen_event['detected_classes'],
                    'evidence_frames': evidence, 'source': 'multi_event_fusion',
                    'candidate_only': True})
                candidate_specs.append({'subtype': 'unattended_active_workstation',
                    'timestamp_sec': screen_event['timestamp_sec'],
                    'evidence_frames': evidence,
                    'trigger_signal': 'WORKSTATION_UNATTENDED_PROPOSAL',
                    'missing_context': ['WORKSTATION_SCREEN_ACTIVE_UNLOCKED',
                                        'CASHIER_IDENTITY_AND_ROLE',
                                        'INTERNAL_BANK_POLICY_VALIDATION'],
                    'extra_description': ('Workstation equipment was visually observed '
                        'around a cashier-away interval (%.1fs). Screen/session state '
                        'and unauthorized access NOT proven.'
                        % screen_event['away_duration_sec'])})
            details = {'video': Path(video_path).name,
                'status': 'ok', 'model': str(MODEL_PATH.name),
                'sample_interval_sec': round(interval,3), 'sampled_frames': len(records),
                'detections_by_class': counts, 'events': signals,
                'candidate_specs': candidate_specs,
                'suppressed_unverified_bag_tracks': suppressed_bag_tracks,
                'bag_review_gate': {'min_confidence': MIN_BAG_CONF_FOR_REVIEW,
                                    'min_high_confidence_frames': MIN_BAG_STRONG_FRAMES},
                'limitations': empty['model_limitations']}
            output_json = output_dir / 'object_events.json'
            output_json.write_text(json.dumps(details, ensure_ascii=False, indent=2),
                                   encoding='utf-8')
            return dict(empty, status='ok', error=None, events=signals,
                        rule_candidates=candidate_specs, detections_by_class=counts,
                        suppressed_unverified_bag_tracks=suppressed_bag_tracks,
                        evidence_path=str(output_json), sampled_frames=len(records))
        finally:
            cap.release()
    except Exception as ex:
        return dict(empty, status='failed', error=type(ex).__name__+': '+str(ex))
