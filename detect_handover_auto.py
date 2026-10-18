import sys
import math
import json
from pathlib import Path

import cv2

from src.detector import detect_people
from src.tracker import SimpleTracker
from src.auto_roles import AutoRoleAssigner
from src.pose_detector import detect_poses
from src.handover import HandoverDetector


if len(sys.argv) < 2:
    print(
        "Usage: python detect_handover_auto.py "
        "data/videos/X-1.mov"
    )
    sys.exit(1)


VIDEO_PATH = Path(sys.argv[1])

FRAME_STEP = 5


OUTPUT_DIR = Path(
    "data/handover_auto_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


EVIDENCE_DIR = (
    OUTPUT_DIR /
    VIDEO_PATH.stem
)

EVIDENCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_JSON = (
    OUTPUT_DIR /
    f"{VIDEO_PATH.stem}_handover.json"
)


# ============================================================
# IOU
# ============================================================

def iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    w = max(
        0,
        x2 - x1
    )

    h = max(
        0,
        y2 - y1
    )

    intersection = (
        w * h
    )

    area_a = (
        max(0, ax2 - ax1)
        *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        *
        max(0, by2 - by1)
    )

    union = (
        area_a
        +
        area_b
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        / union
    )


# ============================================================
# MATCH POSE -> TRACK ID
# ============================================================

def match_poses_to_tracks(
    tracks,
    poses,
    min_iou=0.15
):

    candidates = []


    for track_index, track in enumerate(
        tracks
    ):

        for pose_index, pose in enumerate(
            poses
        ):

            score = iou(
                track["box"],
                pose["box"]
            )

            if score >= min_iou:

                candidates.append(
                    (
                        score,
                        track_index,
                        pose_index
                    )
                )


    candidates.sort(
        reverse=True,
        key=lambda x: x[0]
    )


    used_tracks = set()
    used_poses = set()

    result = {}


    for (
        score,
        track_index,
        pose_index
    ) in candidates:

        if track_index in used_tracks:
            continue

        if pose_index in used_poses:
            continue


        track = tracks[
            track_index
        ]

        pose = poses[
            pose_index
        ]


        result[
            track["id"]
        ] = pose


        used_tracks.add(
            track_index
        )

        used_poses.add(
            pose_index
        )


    return result


# ============================================================
# WRISTS
# ============================================================

def wrists(pose):

    result = []

    if pose is None:
        return result


    if (
        pose["left_wrist"]
        is not None
    ):

        result.append(
            pose["left_wrist"]
        )


    if (
        pose["right_wrist"]
        is not None
    ):

        result.append(
            pose["right_wrist"]
        )


    return result


# ============================================================
# CLOSEST WRIST PAIR
# ============================================================

def closest_wrist_pair(
    pose_a,
    pose_b
):

    points_a = wrists(
        pose_a
    )

    points_b = wrists(
        pose_b
    )


    if (
        not points_a
        or
        not points_b
    ):
        return None


    best = None


    for point_a in points_a:

        for point_b in points_b:

            distance = math.dist(
                point_a,
                point_b
            )


            if (
                best is None
                or
                distance
                <
                best["distance"]
            ):

                best = {
                    "distance":
                        distance,

                    "point_a":
                        point_a,

                    "point_b":
                        point_b
                }


    return best


# ============================================================
# INTERACTION CORRIDOR
# ============================================================

def interaction_corridor_ok(
    cashier_box,
    customer_box,
    cashier_wrist,
    customer_wrist,
    max_corridor_ratio=0.35
):
    """
    Проверяем, что midpoint рук
    находится между cashier и customer,
    а не сбоку около принтера / оборудования.
    """

    # ------------------------------
    # Centers of people
    # ------------------------------

    cashier_center_x = (
        cashier_box[0]
        +
        cashier_box[2]
    ) / 2

    cashier_center_y = (
        cashier_box[1]
        +
        cashier_box[3]
    ) / 2


    customer_center_x = (
        customer_box[0]
        +
        customer_box[2]
    ) / 2

    customer_center_y = (
        customer_box[1]
        +
        customer_box[3]
    ) / 2


    # ------------------------------
    # Midpoint between wrists
    # ------------------------------

    midpoint_x = (
        cashier_wrist[0]
        +
        customer_wrist[0]
    ) / 2

    midpoint_y = (
        cashier_wrist[1]
        +
        customer_wrist[1]
    ) / 2


    # ------------------------------
    # Vector cashier -> customer
    # ------------------------------

    dx = (
        customer_center_x
        -
        cashier_center_x
    )

    dy = (
        customer_center_y
        -
        cashier_center_y
    )


    segment_sq = (
        dx * dx
        +
        dy * dy
    )


    if segment_sq <= 1:
        return (
            False,
            999.0,
            0.0
        )


    # ------------------------------
    # Projection of wrist midpoint
    # onto person-person line
    # ------------------------------

    projection = (
        (
            (
                midpoint_x
                -
                cashier_center_x
            )
            *
            dx
        )
        +
        (
            (
                midpoint_y
                -
                cashier_center_y
            )
            *
            dy
        )
    ) / segment_sq


    # Контакт должен находиться
    # между людьми.
    if (
        projection < 0.15
        or
        projection > 0.85
    ):

        return (
            False,
            999.0,
            projection
        )


    closest_x = (
        cashier_center_x
        +
        projection
        *
        dx
    )

    closest_y = (
        cashier_center_y
        +
        projection
        *
        dy
    )


    perpendicular_distance = (
        math.dist(
            (
                midpoint_x,
                midpoint_y
            ),
            (
                closest_x,
                closest_y
            )
        )
    )


    people_distance = (
        math.dist(
            (
                cashier_center_x,
                cashier_center_y
            ),
            (
                customer_center_x,
                customer_center_y
            )
        )
    )


    corridor_ratio = (
        perpendicular_distance
        /
        max(
            people_distance,
            1.0
        )
    )


    ok = (
        corridor_ratio
        <=
        max_corridor_ratio
    )


    return (
        ok,
        corridor_ratio,
        projection
    )


# ============================================================
# PASS 1
# TRACKING + AUTOMATIC ROLES
# ============================================================

print()
print(
    "=== PASS 1: TRACKING + AUTO ROLES ==="
)
print()


tracker = SimpleTracker(
    max_distance=120,
    max_missing=8
)


role_assigner = (
    AutoRoleAssigner(
        min_track_frames=20,
        min_presence_ratio=0.10
    )
)


cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open {VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


frame_number = 0

observations = []


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


    frame_tracks = []


    for track in tracks:

        person_id = (
            track["id"]
        )

        box = (
            track["box"]
        )


        role_assigner.update(
            person_id,
            box
        )


        frame_tracks.append({
            "id":
                person_id,

            "box": [
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3])
            ]
        })


    observations.append({
        "timestamp_sec":
            timestamp,

        "frame_number":
            frame_number,

        "tracks":
            frame_tracks
    })


    frame_number += 1


cap.release()


# ============================================================
# FINAL ROLES
# ============================================================

role_map = (
    role_assigner
    .assign_roles()
)


print(
    "FINAL ROLES:"
)


for (
    person_id,
    role
) in role_map.items():

    data = (
        role_assigner
        .stats[person_id]
    )


    print(
        f"ID {person_id} -> {role}"
        f" | frames={data['frames']}"
    )


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


print()


if not cashier_ids:

    print(
        "No CASHIER found."
    )


if not customer_ids:

    print(
        "No CUSTOMER found."
    )

    print(
        "Handover detection "
        "is not applicable."
    )


# ============================================================
# CREATE HANDOVER DETECTORS
# ============================================================

handover_detectors = {}


for cashier_id in cashier_ids:

    for customer_id in customer_ids:

        handover_detectors[
            (
                cashier_id,
                customer_id
            )
        ] = HandoverDetector()


# ============================================================
# PASS 2
# POSE + HANDOVER
# ============================================================

events = []


if handover_detectors:

    print(
        "=== PASS 2: HANDOVER DETECTION ==="
    )

    print()


    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )


    for observation in observations:

        timestamp = (
            observation[
                "timestamp_sec"
            ]
        )

        frame_number = (
            observation[
                "frame_number"
            ]
        )


        # ----------------------------------
        # Keep useful tracks only
        # ----------------------------------

        valid_tracks = []


        for track in observation[
            "tracks"
        ]:

            role = role_map.get(
                track["id"],
                "IGNORE"
            )


            if role in (
                "CASHIER",
                "CUSTOMER"
            ):

                valid_tracks.append(
                    track
                )


        present_ids = {
            track["id"]
            for track
            in valid_tracks
        }


        # ----------------------------------
        # Check whether a pair exists
        # ----------------------------------

        has_pair = False


        for (
            cashier_id,
            customer_id
        ) in handover_detectors:

            if (
                cashier_id
                in present_ids
                and
                customer_id
                in present_ids
            ):

                has_pair = True
                break


        if not has_pair:
            continue


        # ----------------------------------
        # Load frame
        # ----------------------------------

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )


        ok, frame = cap.read()

        if not ok:
            continue


        # ----------------------------------
        # Pose
        # ----------------------------------

        poses = detect_poses(
            frame,
            score_threshold=0.7
        )


        matched = (
            match_poses_to_tracks(
                valid_tracks,
                poses
            )
        )


        # ----------------------------------
        # Each cashier/customer pair
        # ----------------------------------

        for (
            cashier_id,
            customer_id
        ), detector in (
            handover_detectors.items()
        ):

            cashier_pose = (
                matched.get(
                    cashier_id
                )
            )

            customer_pose = (
                matched.get(
                    customer_id
                )
            )


            if (
                cashier_pose is None
                or
                customer_pose is None
            ):
                continue


            # ----------------------------------
            # Closest wrist pair
            # ----------------------------------

            pair = closest_wrist_pair(
                cashier_pose,
                customer_pose
            )


            if pair is None:
                continue


            distance_px = (
                pair["distance"]
            )


            cashier_wrist = (
                pair["point_a"]
            )


            customer_wrist = (
                pair["point_b"]
            )


            # ----------------------------------
            # Find cashier/customer tracks
            # ----------------------------------

            cashier_track = next(
                (
                    track
                    for track
                    in valid_tracks
                    if (
                        track["id"]
                        ==
                        cashier_id
                    )
                ),
                None
            )


            customer_track = next(
                (
                    track
                    for track
                    in valid_tracks
                    if (
                        track["id"]
                        ==
                        customer_id
                    )
                ),
                None
            )


            if (
                cashier_track is None
                or
                customer_track is None
            ):
                continue


            # ----------------------------------
            # Interaction corridor check
            # ----------------------------------

            (
                corridor_ok,
                corridor_ratio,
                projection
            ) = interaction_corridor_ok(
                cashier_track["box"],
                customer_track["box"],
                cashier_wrist,
                customer_wrist,
                max_corridor_ratio=0.35
            )


            # Если руки близко,
            # но находятся сбоку —
            # например возле принтера,
            # пропускаем.
            if not corridor_ok:
                continue


            # ----------------------------------
            # Normalized distance
            # ----------------------------------

            distance_norm = (
                distance_px
                /
                width
            )


            # ----------------------------------
            # Temporal handover detector
            # ----------------------------------

            event = detector.update(
                timestamp,
                distance_norm
            )


            if event is None:
                continue


            event[
                "cashier_id"
            ] = cashier_id


            event[
                "customer_id"
            ] = customer_id


            event[
                "corridor_ratio"
            ] = round(
                corridor_ratio,
                4
            )


            event[
                "projection"
            ] = round(
                projection,
                4
            )


            events.append(
                event
            )


    cap.release()


# ============================================================
# EVIDENCE FRAMES
# ============================================================

if events:

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )


    for index, event in enumerate(
        events,
        start=1
    ):

        evidence = []


        moments = {
            "start":
                event[
                    "timestamp_sec"
                ],

            "peak":
                event[
                    "peak_timestamp_sec"
                ],

            "end":
                event[
                    "end_timestamp_sec"
                ]
        }


        for (
            name,
            timestamp
        ) in moments.items():

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp * 1000
            )


            ok, frame = cap.read()

            if not ok:
                continue


            cv2.putText(
                frame,
                (
                    f"HANDOVER "
                    f"{timestamp:.2f}s"
                ),
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )


            filename = (
                EVIDENCE_DIR
                /
                f"event_{index}_{name}.jpg"
            )


            cv2.imwrite(
                str(filename),
                frame
            )


            evidence.append(
                str(filename)
            )


        event[
            "evidence_frames"
        ] = evidence


    cap.release()


# ============================================================
# PRINT RESULT
# ============================================================

print()
print(
    "=== HANDOVER EVENTS ==="
)
print()


if not events:

    print(
        "No handover detected."
    )


else:

    for event in events:

        print(
            f'{event["timestamp_sec"]:.2f}s'
            f' -> '
            f'{event["end_timestamp_sec"]:.2f}s'
        )

        print(
            f'  peak: '
            f'{event["peak_timestamp_sec"]:.2f}s'
        )

        print(
            f'  min distance: '
            f'{event["min_distance_norm"]:.4f}'
        )

        print(
            f'  corridor ratio: '
            f'{event["corridor_ratio"]:.4f}'
        )

        print(
            f'  CASHIER ID '
            f'{event["cashier_id"]}'
            f' -> '
            f'CUSTOMER ID '
            f'{event["customer_id"]}'
        )

        print()


# ============================================================
# SAVE JSON
# ============================================================

result = {
    "video":
        VIDEO_PATH.name,

    "roles":
        {
            str(person_id):
                role

            for person_id, role
            in role_map.items()
        },

    "handover_events":
        events
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    "SAVED:",
    OUTPUT_JSON
)

print("DONE")