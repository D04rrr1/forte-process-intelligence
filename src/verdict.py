from pathlib import Path

import cv2
import joblib
import numpy as np

from src.video_features import (
    extract_video_embedding
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


MODEL_PATH = (
    PROJECT_ROOT
    /
    "models"
    /
    "final_verdict.joblib"
)


_model_bundle = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_verdict_model():

    global _model_bundle

    if _model_bundle is None:

        if not MODEL_PATH.exists():

            raise RuntimeError(
                "Final verdict model not found. "
                "Run train_final_classifier.py first."
            )


        _model_bundle = (
            joblib.load(
                MODEL_PATH
            )
        )


    return _model_bundle


# ============================================================
# EMBEDDING NORMALIZATION
# ============================================================

def normalize_embedding(
    embedding
):

    embedding = np.asarray(
        embedding,
        dtype=np.float64
    )


    embedding = np.nan_to_num(
        embedding,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    norm = np.linalg.norm(
        embedding
    )


    if norm > 0:

        embedding = (
            embedding
            /
            norm
        )


    return embedding


# ============================================================
# VIDEO DURATION
# ============================================================

def get_video_duration(
    video_path
):

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

    frame_count = cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )


    cap.release()


    if fps <= 0:
        raise RuntimeError(
            "Invalid video FPS."
        )


    return (
        frame_count
        /
        fps
    )


# ============================================================
# WINDOWS
# ============================================================

def build_windows(
    duration,
    window_sec,
    stride_sec
):

    if duration <= window_sec:

        return [
            (
                0.0,
                duration
            )
        ]


    windows = []

    start = 0.0


    while (
        start
        <
        duration
    ):

        end = min(
            duration,
            start + window_sec
        )


        windows.append(
            (
                start,
                end
            )
        )


        if end >= duration:
            break


        start += (
            stride_sec
        )


    return windows


# ============================================================
# ANALYZE VERDICT
# ============================================================

def analyze_verdict(
    video_path,
    threshold=None
):

    bundle = (
        load_verdict_model()
    )


    classifier = bundle[
        "model"
    ]


    window_sec = float(
        bundle.get(
            "window_sec",
            6.0
        )
    )


    stride_sec = float(
        bundle.get(
            "stride_sec",
            3.0
        )
    )


    if threshold is None:

        threshold = float(
            bundle.get(
                "threshold",
                0.50
            )
        )


    video_path = Path(
        video_path
    )


    duration = (
        get_video_duration(
            video_path
        )
    )


    windows = build_windows(
        duration,
        window_sec,
        stride_sec
    )


    results = []


    print()
    print(
        "=== VERDICT WINDOWS ==="
    )


    for index, (
        start_sec,
        end_sec
    ) in enumerate(
        windows,
        start=1
    ):

        embedding = (
            extract_video_embedding(
                video_path,
                start_sec=start_sec,
                end_sec=end_sec
            )
        )


        embedding = (
            normalize_embedding(
                embedding
            )
        )


        probability = float(
            classifier.predict_proba(
                embedding.reshape(
                    1,
                    -1
                )
            )[0, 1]
        )


        midpoint = (
            start_sec
            +
            end_sec
        ) / 2


        item = {
            "window_index":
                index,

            "start_sec":
                round(
                    start_sec,
                    2
                ),

            "end_sec":
                round(
                    end_sec,
                    2
                ),

            "midpoint_sec":
                round(
                    midpoint,
                    2
                ),

            "violation_score":
                round(
                    probability,
                    4
                )
        }


        results.append(
            item
        )


        print(
            f"{start_sec:.1f}"
            f"-"
            f"{end_sec:.1f}s"
            f" -> "
            f"{probability:.4f}"
        )


    # ========================================================
    # BEST WINDOW
    # ========================================================

    best = max(
        results,
        key=lambda item:
            item[
                "violation_score"
            ]
    )


    score = float(
        best[
            "violation_score"
        ]
    )


    if score >= threshold:

        verdict = (
            "violation"
        )

    else:

        verdict = (
            "no_violation"
        )


    return {
        "verdict":
            verdict,

        "violation_score":
            round(
                score,
                4
            ),

        "threshold":
            threshold,

        "candidate_timestamp_sec":
            best[
                "midpoint_sec"
            ],

        "best_window":
            best,

        "windows":
            results
    }