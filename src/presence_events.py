from pathlib import Path
from collections import defaultdict
import os

import cv2
import numpy as np

from src.detector import detect_people
from src.tracker import SimpleTracker
from src.auto_roles import AutoRoleAssigner


# ============================================================
# SETTINGS
# ============================================================

FRAME_STEP = 5

MIN_CUSTOMER_DURATION_SEC = 1.0
MIN_WORKSTATION_AWAY_DURATION_SEC = 1.0

# Если кассир просто потерялся detector-ом на 1–2 сек,
# это не считаем настоящим исчезновением.
MIN_NOT_VISIBLE_DURATION_SEC = 3.0

MIN_MULTIPLE_CUSTOMERS_DURATION_SEC = 1.0


# ============================================================
# TRACK STITCHING
# ============================================================

# Насколько близко дополнительный track должен находиться
# к обычной позиции cashier.
STITCH_DISTANCE_THRESHOLD = 0.16

# Если два track существуют одновременно дольше этого времени,
# это почти наверняка два разных человека.
STITCH_MAX_OVERLAP_SEC = 0.5

# Максимальный gap между двумя фрагментами одного человека.
STITCH_MAX_GAP_SEC = 6.0

MIN_STITCH_TRACK_POINTS = 10


# ============================================================
# WORKSTATION
# ============================================================

# Более чувствительный threshold.
# Кассир не обязан исчезнуть из кадра:
# достаточно реально отойти от рабочего места.
WORKSTATION_AWAY_THRESHOLD = 0.045

# Away is not merely a body-box center shift: a seated cashier may lean,
# turn, or stand without leaving the workstation. Conservative geometric gate.
# These thresholds are heuristic, NOT calibrated on held-out surveillance videos.
FOOT_AWAY_THRESHOLD = float(os.getenv("FORTE_FOOT_AWAY_THRESHOLD", "0.065"))
STATION_IOU_KEEP_THRESHOLD = float(os.getenv("FORTE_STATION_IOU_KEEP_THRESHOLD", "0.35"))
STRONG_FOOT_AWAY_THRESHOLD = float(os.getenv("FORTE_STRONG_FOOT_AWAY_THRESHOLD", "0.12"))


# ============================================================
# HELPERS
# ============================================================

def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def normalized_distance(
    point_a,
    point_b,
    frame_width,
    frame_height
):

    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]

    distance = (
        dx * dx
        +
        dy * dy
    ) ** 0.5


    diagonal = (
        frame_width * frame_width
        +
        frame_height * frame_height
    ) ** 0.5


    if diagonal <= 0:
        return 0.0


    return distance / diagonal


def box_floor_point(box):
    """Bottom-center as a camera-space proxy for standing/sitting location."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def box_iou(a, b):
    """Intersection-over-union; normalizes coordinates and handles invalid boxes."""
    if a is None or b is None:
        return 0.0
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def guarded_cashier_away(box, center, center_anchor, floor_anchor,
                         reference_box, width, height):
    """Require independent positional evidence beyond a shifted body-box center.

    A conservative hypothesis only. Not proof that a station is unattended.
    """
    if center_anchor is None or floor_anchor is None or reference_box is None:
        return False
    if normalized_distance(center, center_anchor, width, height) <= WORKSTATION_AWAY_THRESHOLD:
        return False
    floor_distance = normalized_distance(box_floor_point(box), floor_anchor, width, height)
    if floor_distance < FOOT_AWAY_THRESHOLD:
        return False
    overlap = box_iou(box, reference_box)
    if overlap >= STATION_IOU_KEEP_THRESHOLD and floor_distance < STRONG_FOOT_AWAY_THRESHOLD:
        return False
    return True


def calculate_overlap_sec(
    start_a,
    end_a,
    start_b,
    end_b
):

    start = max(
        start_a,
        start_b
    )

    end = min(
        end_a,
        end_b
    )

    return max(
        0.0,
        end - start
    )


def calculate_gap_sec(
    start_a,
    end_a,
    start_b,
    end_b
):

    # A finishes before B
    if end_a < start_b:

        return (
            start_b
            -
            end_a
        )


    # B finishes before A
    if end_b < start_a:

        return (
            start_a
            -
            end_b
        )


    # Tracks overlap
    return 0.0


def _intervals_from_boolean_timeline(
    timeline,
    min_duration_sec=1.0
):

    intervals = []

    start_time = None
    last_active_time = None


    for timestamp, active in timeline:

        if active:

            if start_time is None:
                start_time = timestamp

            last_active_time = timestamp

        else:

            if start_time is None:
                continue


            end_time = (
                last_active_time
                if last_active_time is not None
                else timestamp
            )


            duration = (
                end_time
                -
                start_time
            )


            if duration >= min_duration_sec:

                intervals.append({
                    "start_sec":
                        round(
                            start_time,
                            2
                        ),

                    "end_sec":
                        round(
                            end_time,
                            2
                        ),

                    "duration_sec":
                        round(
                            duration,
                            2
                        )
                })


            start_time = None
            last_active_time = None


    # Событие продолжается до конца видео
    if (
        start_time is not None
        and
        last_active_time is not None
    ):

        duration = (
            last_active_time
            -
            start_time
        )


        if duration >= min_duration_sec:

            intervals.append({
                "start_sec":
                    round(
                        start_time,
                        2
                    ),

                "end_sec":
                    round(
                        last_active_time,
                        2
                    ),

                "duration_sec":
                    round(
                        duration,
                        2
                    )
            })


    return intervals


# ============================================================
# MAIN
# ============================================================

def analyze_presence_events(
    video_path
):

    video_path = Path(
        video_path
    )


    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )


    # ========================================================
    # MODELS
    # ========================================================

    tracker = SimpleTracker(
        max_distance=120,
        max_missing=8
    )


    role_assigner = AutoRoleAssigner(
        min_track_frames=20,
        min_presence_ratio=0.10
    )


    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open {video_path}"
        )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    if fps <= 0:

        cap.release()

        raise RuntimeError(
            "Invalid FPS"
        )


    frame_number = 0

    observations = []


    track_centers = defaultdict(
        list
    )
    track_boxes = defaultdict(list)


    track_times = defaultdict(
        list
    )


    # ========================================================
    # PASS 1
    # PERSON DETECTION + TRACKING
    # ========================================================

    while True:

        ok, frame = cap.read()

        if not ok:
            break


        if (
            frame_number
            %
            FRAME_STEP
            != 0
        ):

            frame_number += 1
            continue


        timestamp = (
            frame_number
            /
            fps
        )


        detections = detect_people(
            frame,
            threshold=0.5
        )


        tracks = tracker.update(
            detections
        )


        frame_tracks = {}


        for track in tracks:

            person_id = (
                track["id"]
            )

            box = (
                track["box"]
            )


            center = box_center(
                box
            )


            role_assigner.update(
                person_id,
                box
            )


            track_centers[
                person_id
            ].append(
                center
            )
            track_boxes[person_id].append(tuple(float(v) for v in box))


            track_times[
                person_id
            ].append(
                timestamp
            )


            frame_tracks[
                person_id
            ] = {
                "box":
                    box,

                "center":
                    center
            }


        observations.append({
            "timestamp_sec":
                timestamp,

            "tracks":
                frame_tracks
        })


        frame_number += 1


    cap.release()


    # ========================================================
    # ORIGINAL ROLES
    # ========================================================

    original_role_map = (
        role_assigner
        .assign_roles()
    )


    initial_cashier_ids = [
        person_id

        for person_id, role
        in original_role_map.items()

        if role == "CASHIER"
    ]


    # ========================================================
    # TRACK TIME INTERVALS
    # ========================================================

    track_intervals = {}


    for person_id, times in (
        track_times.items()
    ):

        if not times:
            continue


        track_intervals[
            person_id
        ] = {
            "start":
                min(times),

            "end":
                max(times)
        }


    # ========================================================
    # CASHIER WORKSTATION ANCHOR
    # ========================================================

    cashier_anchor_points = []


    for cashier_id in initial_cashier_ids:

        points = (
            track_centers.get(
                cashier_id,
                []
            )
        )


        if not points:
            continue


        # Используем первую половину
        # основного cashier-track.
        count = max(
            1,
            len(points) // 2
        )


        cashier_anchor_points.extend(
            points[:count]
        )


    if cashier_anchor_points:

        cashier_anchor = (
            float(
                np.median([
                    point[0]
                    for point
                    in cashier_anchor_points
                ])
            ),

            float(
                np.median([
                    point[1]
                    for point
                    in cashier_anchor_points
                ])
            )
        )

    else:

        cashier_anchor = None


    # A separate spatial reference for departure detection: reference box +
    # floor point estimated from the same early cashier observations. The old
    # center anchor is kept unchanged for all existing downstream cash logic.
    floor_anchor = None
    workstation_reference_box = None
    stable_boxes = []
    if cashier_anchor is not None:
        for cashier_id in initial_cashier_ids:
            boxes = track_boxes.get(cashier_id, [])
            first_half = boxes[:max(1, len(boxes)//2)]
            for b in first_half:
                if normalized_distance(box_center(b), cashier_anchor,
                                       frame_width, frame_height) <= 0.075:
                    stable_boxes.append(b)
        if stable_boxes:
            floors = [box_floor_point(b) for b in stable_boxes]
            floor_anchor = (float(np.median([p[0] for p in floors])),
                            float(np.median([p[1] for p in floors])))
            workstation_reference_box = tuple(
                float(np.median([b[i] for b in stable_boxes])) for i in range(4)
            )

    mode = os.getenv("FORTE_PRESENCE_AWAY_MODE", "guarded").strip().lower()
    if mode not in ("guarded", "legacy"):
        raise ValueError("FORTE_PRESENCE_AWAY_MODE must be guarded or legacy")

    # ========================================================
    # CASHIER TRACK STITCHING
    # ========================================================

    cashier_ids = set(
        initial_cashier_ids
    )


    if cashier_anchor is not None:

        for person_id, centers in (
            track_centers.items()
        ):

            # Уже cashier
            if person_id in cashier_ids:
                continue


            original_role = (
                original_role_map.get(
                    person_id
                )
            )


            # IGNORE никогда не превращаем
            # обратно в человека.
            if original_role == "IGNORE":
                continue


            if (
                len(centers)
                <
                MIN_STITCH_TRACK_POINTS
            ):
                continue


            candidate_interval = (
                track_intervals.get(
                    person_id
                )
            )


            if candidate_interval is None:
                continue


            # =================================================
            # TEMPORAL CHECK
            #
            # Если candidate существует одновременно
            # с настоящим cashier,
            # значит это другой человек.
            # =================================================

            temporal_match = False


            for cashier_id in (
                initial_cashier_ids
            ):

                cashier_interval = (
                    track_intervals.get(
                        cashier_id
                    )
                )


                if cashier_interval is None:
                    continue


                overlap = (
                    calculate_overlap_sec(
                        candidate_interval[
                            "start"
                        ],

                        candidate_interval[
                            "end"
                        ],

                        cashier_interval[
                            "start"
                        ],

                        cashier_interval[
                            "end"
                        ]
                    )
                )


                gap = (
                    calculate_gap_sec(
                        candidate_interval[
                            "start"
                        ],

                        candidate_interval[
                            "end"
                        ],

                        cashier_interval[
                            "start"
                        ],

                        cashier_interval[
                            "end"
                        ]
                    )
                )


                # Два условия:
                #
                # 1. почти не пересекаются
                # 2. разрыв не слишком большой
                if (
                    overlap
                    <=
                    STITCH_MAX_OVERLAP_SEC

                    and

                    gap
                    <=
                    STITCH_MAX_GAP_SEC
                ):

                    temporal_match = True
                    break


            if not temporal_match:
                continue


            # =================================================
            # SPATIAL CHECK
            # =================================================

            distances = [
                normalized_distance(
                    center,
                    cashier_anchor,
                    frame_width,
                    frame_height
                )

                for center in centers
            ]


            median_distance = float(
                np.median(
                    distances
                )
            )


            near_ratio = (
                sum(
                    distance
                    <
                    STITCH_DISTANCE_THRESHOLD

                    for distance
                    in distances
                )
                /
                len(
                    distances
                )
            )


            if (
                near_ratio >= 0.50
                and
                median_distance
                <
                STITCH_DISTANCE_THRESHOLD
            ):

                cashier_ids.add(
                    person_id
                )


    # ========================================================
    # FINAL ROLES
    # ========================================================

    role_map = {}


    for person_id, original_role in (
        original_role_map.items()
    ):

        if original_role == "IGNORE":

            role_map[
                person_id
            ] = "IGNORE"


        elif person_id in cashier_ids:

            role_map[
                person_id
            ] = "CASHIER"


        else:

            role_map[
                person_id
            ] = "CUSTOMER"


    cashier_ids = [
        person_id

        for person_id, role
        in role_map.items()

        if role == "CASHIER"
    ]


    customer_ids = [
        person_id

        for person_id, role
        in role_map.items()

        if role == "CUSTOMER"
    ]


    # ========================================================
    # TIMELINES
    # ========================================================

    customer_present_timeline = []

    cashier_present_timeline = []

    cashier_away_from_workstation_timeline = []
    cashier_away_legacy_timeline = []
    cashier_away_guarded_timeline = []

    cashier_not_visible_timeline = []

    cashier_away_with_customer_timeline = []

    multiple_customers_timeline = []


    cashier_seen_once = False


    for observation in observations:

        timestamp = (
            observation[
                "timestamp_sec"
            ]
        )


        tracks = (
            observation[
                "tracks"
            ]
        )


        present_cashier_ids = [
            person_id

            for person_id
            in cashier_ids

            if person_id in tracks
        ]


        present_customer_ids = [
            person_id

            for person_id
            in customer_ids

            if person_id in tracks
        ]


        cashier_present = (
            len(
                present_cashier_ids
            )
            >= 1
        )


        customer_present = (
            len(
                present_customer_ids
            )
            >= 1
        )


        multiple_customers = (
            len(
                present_customer_ids
            )
            >= 2
        )


        if cashier_present:

            cashier_seen_once = True


        # ====================================================
        # NOT VISIBLE
        # ====================================================

        cashier_not_visible = (
            cashier_seen_once
            and
            not cashier_present
        )


        # ====================================================
        # AWAY FROM WORKSTATION
        #
        # ВАЖНО:
        # здесь cashier должен быть ВИДЕН.
        #
        # Если detector потерял человека,
        # это отдельный CASHIER_NOT_VISIBLE.
        # ====================================================

        # Compare legacy center-distance rule and conservative box-footprint
        # rule in ONE detector pass. Change only which event becomes active;
        # include both timelines as diagnostic intervals in the output.
        legacy_away = False
        guarded_away = False
        if cashier_present and cashier_anchor is not None:
            distances = [
                normalized_distance(tracks[pid]["center"], cashier_anchor,
                                    frame_width, frame_height)
                for pid in present_cashier_ids
            ]
            legacy_away = bool(distances and min(distances) > WORKSTATION_AWAY_THRESHOLD)
            guarded_results = [
                guarded_cashier_away(
                    tracks[pid]["box"], tracks[pid]["center"],
                    cashier_anchor, floor_anchor, workstation_reference_box,
                    frame_width, frame_height)
                for pid in present_cashier_ids
            ]
            # One cashier still at desk -> do not flag the workstation as empty.
            guarded_away = bool(guarded_results and all(guarded_results))

        cashier_away_from_workstation = (
            legacy_away if mode == "legacy" else guarded_away
        )
        cashier_away_legacy_timeline.append((timestamp, legacy_away))
        cashier_away_guarded_timeline.append((timestamp, guarded_away))


        cashier_away_with_customer = (
            cashier_away_from_workstation
            and
            customer_present
        )


        # ====================================================
        # SAVE
        # ====================================================

        customer_present_timeline.append(
            (
                timestamp,
                customer_present
            )
        )


        cashier_present_timeline.append(
            (
                timestamp,
                cashier_present
            )
        )


        cashier_away_from_workstation_timeline.append(
            (
                timestamp,
                cashier_away_from_workstation
            )
        )


        cashier_not_visible_timeline.append(
            (
                timestamp,
                cashier_not_visible
            )
        )


        cashier_away_with_customer_timeline.append(
            (
                timestamp,
                cashier_away_with_customer
            )
        )


        multiple_customers_timeline.append(
            (
                timestamp,
                multiple_customers
            )
        )


    # ========================================================
    # EVENTS
    # ========================================================

    events = []


    # CUSTOMER_PRESENT
    intervals = (
        _intervals_from_boolean_timeline(
            customer_present_timeline,
            MIN_CUSTOMER_DURATION_SEC
        )
    )


    for interval in intervals:

        events.append({
            "type":
                "CUSTOMER_PRESENT",

            **interval
        })


    # CASHIER_PRESENT
    intervals = (
        _intervals_from_boolean_timeline(
            cashier_present_timeline,
            1.0
        )
    )


    for interval in intervals:

        events.append({
            "type":
                "CASHIER_PRESENT",

            **interval
        })


    # CASHIER_AWAY_FROM_WORKSTATION
    intervals = (
        _intervals_from_boolean_timeline(
            cashier_away_from_workstation_timeline,
            MIN_WORKSTATION_AWAY_DURATION_SEC
        )
    )


    for interval in intervals:

        events.append({
            "type":
                "CASHIER_AWAY_FROM_WORKSTATION",

            **interval
        })


    # CASHIER_NOT_VISIBLE
    intervals = (
        _intervals_from_boolean_timeline(
            cashier_not_visible_timeline,
            MIN_NOT_VISIBLE_DURATION_SEC
        )
    )


    for interval in intervals:

        events.append({
            "type":
                "CASHIER_NOT_VISIBLE",

            **interval
        })


    # CASHIER_AWAY_WHILE_CUSTOMER_PRESENT
    intervals = (
        _intervals_from_boolean_timeline(
            cashier_away_with_customer_timeline,
            MIN_WORKSTATION_AWAY_DURATION_SEC
        )
    )


    for interval in intervals:

        events.append({
            "type":
                "CASHIER_AWAY_WHILE_CUSTOMER_PRESENT",

            **interval
        })


    # MULTIPLE_CUSTOMERS_PRESENT
    intervals = (
        _intervals_from_boolean_timeline(
            multiple_customers_timeline,
            MIN_MULTIPLE_CUSTOMERS_DURATION_SEC
        )
    )


    for interval in intervals:

        events.append({
            "type":
                "MULTIPLE_CUSTOMERS_PRESENT",

            **interval
        })


    # ========================================================
    # SORT
    # ========================================================

    events.sort(
        key=lambda event: (
            event[
                "start_sec"
            ],
            event[
                "type"
            ]
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    return {
        "video":
            video_path.name,

        "roles":
            {
                str(person_id):
                    role

                for person_id, role
                in role_map.items()
            },

        "cashier_ids":
            sorted(
                cashier_ids
            ),

        "customer_ids":
            sorted(
                customer_ids
            ),

        "workstation_anchor":
            (
                {
                    "x":
                        round(
                            cashier_anchor[0],
                            2
                        ),

                    "y":
                        round(
                            cashier_anchor[1],
                            2
                        )
                }

                if cashier_anchor is not None

                else None
            ),

        "events":
            events,
        "presence_diagnostics": {
            "away_mode": mode,
            "legacy_away_intervals": _intervals_from_boolean_timeline(
                cashier_away_legacy_timeline, MIN_WORKSTATION_AWAY_DURATION_SEC
            ),
            "guarded_away_intervals": _intervals_from_boolean_timeline(
                cashier_away_guarded_timeline, MIN_WORKSTATION_AWAY_DURATION_SEC
            ),
            "floor_anchor": (
                {"x": round(floor_anchor[0], 2), "y": round(floor_anchor[1], 2)}
                if floor_anchor is not None else None
            ),
            "reference_box": (
                [round(v, 2) for v in workstation_reference_box]
                if workstation_reference_box is not None else None
            ),
            "guarded_thresholds": {
                "foot_distance_fraction_of_diagonal": FOOT_AWAY_THRESHOLD,
                "box_iou_keep_at_workstation": STATION_IOU_KEEP_THRESHOLD,
                "strong_foot_distance_fraction": STRONG_FOOT_AWAY_THRESHOLD,
            },
            "limitations": (
                "Heuristic geometry only; cannot establish active workstation "
                "state or absence of a cashier from a camera's blind spot."
            ),
        }
    }