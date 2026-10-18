import json
import subprocess
import sys
import time
from pathlib import Path


from src.presence_events import (
    analyze_presence_events
)

from src.rule_engine import (
    RuleEngine
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


HANDOVER_RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "handover_auto_results"
)


# ============================================================
# RULE ENGINE
# ============================================================

RULE_ENGINE = RuleEngine()


# ============================================================
# HANDOVER DETECTOR
# ============================================================

def run_handover_detector(
    video_path
):
    """
    Existing automatic handover detector.

    HANDOVER != VIOLATION.

    Handover is only a visual event/signal.
    """

    video_path = Path(
        video_path
    )


    script = (
        PROJECT_ROOT
        / "detect_handover_auto.py"
    )


    command = [
        sys.executable,
        str(script),
        str(video_path)
    ]


    process = subprocess.run(
        command,
        cwd=str(
            PROJECT_ROOT
        ),
        capture_output=True,
        text=True
    )


    if process.returncode != 0:

        return {
            "status":
                "failed",

            "error":
                process.stderr,

            "roles":
                {},

            "events":
                []
        }


    result_file = (
        HANDOVER_RESULTS_DIR
        /
        f"{video_path.stem}_handover.json"
    )


    if not result_file.exists():

        return {
            "status":
                "failed",

            "error":
                "Handover result JSON not found.",

            "roles":
                {},

            "events":
                []
        }


    with open(
        result_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    return {
        "status":
            "ok",

        "error":
            None,

        "roles":
            data.get(
                "roles",
                {}
            ),

        "events":
            data.get(
                "handover_events",
                []
            )
    }


# ============================================================
# PRESENCE DETECTOR
# ============================================================

def run_presence_detector(
    video_path
):
    """
    Person / role / presence / workstation events.

    Uses:
    - person detection
    - tracking
    - automatic roles
    - workstation-distance logic
    """

    try:

        result = (
            analyze_presence_events(
                video_path
            )
        )


        return {
            "status":
                "ok",

            "error":
                None,

            "roles":
                result.get(
                    "roles",
                    {}
                ),

            "events":
                result.get(
                    "events",
                    []
                ),

            "workstation_anchor":
                result.get(
                    "workstation_anchor"
                )
        }


    except Exception as exc:

        return {
            "status":
                "failed",

            "error":
                str(exc),

            "roles":
                {},

            "events":
                [],

            "workstation_anchor":
                None
        }


# ============================================================
# NORMALIZE HANDOVER EVENTS
# ============================================================

def normalize_handover_events(
    handover_events
):

    signals = []


    for event in handover_events:

        timestamp = (
            event.get(
                "timestamp_sec"
            )
        )


        signals.append({

            "type":
                "HANDOVER_EVENT",

            "timestamp_sec":
                timestamp,

            "start_sec":
                timestamp,

            "peak_timestamp_sec":
                event.get(
                    "peak_timestamp_sec"
                ),

            "end_timestamp_sec":
                event.get(
                    "end_timestamp_sec"
                ),

            "cashier_id":
                event.get(
                    "cashier_id"
                ),

            "customer_id":
                event.get(
                    "customer_id"
                ),

            "min_distance_norm":
                event.get(
                    "min_distance_norm"
                ),

            "evidence_frames":
                event.get(
                    "evidence_frames",
                    []
                ),

            "source":
                "handover_detector"
        })


    return signals


# ============================================================
# NORMALIZE PRESENCE EVENTS
# ============================================================

def normalize_presence_events(
    presence_events
):

    signals = []


    for event in presence_events:

        start_sec = (
            event.get(
                "start_sec"
            )
        )


        signals.append({

            "type":
                event.get(
                    "type"
                ),

            "timestamp_sec":
                start_sec,

            "start_sec":
                start_sec,

            "end_sec":
                event.get(
                    "end_sec"
                ),

            "duration_sec":
                event.get(
                    "duration_sec"
                ),

            "evidence_frames":
                [],

            "source":
                "presence_detector"
        })


    return signals


# ============================================================
# SORT EVENT TIMELINE
# ============================================================

def sort_signals(
    signals
):

    def timestamp_of(
        signal
    ):

        value = signal.get(
            "timestamp_sec"
        )


        if value is None:

            value = signal.get(
                "start_sec"
            )


        if value is None:
            return 999999999.0


        try:
            return float(
                value
            )

        except Exception:
            return 999999999.0


    return sorted(
        signals,
        key=timestamp_of
    )


# ============================================================
# RULE CANDIDATES
# ============================================================

def build_rule_candidates(
    signals
):
    """
    IMPORTANT:

    These are NOT confirmed violations yet.

    Presence alone is not enough for many banking rules.

    Examples:

    CASHIER_AWAY
    does NOT automatically mean
    unattended_cash.

    We still need:
        CASH_VISIBLE

    MULTIPLE_CUSTOMERS
    does NOT automatically mean
    two active transactions.

    We still need:
        transaction context.

    Therefore these results are
    candidate rules only.
    """

    candidates = []


    for signal in signals:

        event_type = (
            signal.get(
                "type"
            )
        )


        timestamp = (
            signal.get(
                "timestamp_sec"
            )
        )


        # ====================================================
        # CANDIDATE:
        # UNATTENDED CASH
        # ====================================================

        if (
            event_type
            ==
            "CASHIER_AWAY_FROM_WORKSTATION"
        ):

            candidate = (
                RULE_ENGINE
                .create_rule_result(
                    subtype=
                        "unattended_cash",

                    timestamp_sec=
                        timestamp,

                    evidence_frames=
                        signal.get(
                            "evidence_frames",
                            []
                        ),

                    extra_description=(
                        "AI detected that the cashier "
                        "moved away from the workstation. "
                        "Cash visibility has not yet been "
                        "confirmed."
                    )
                )
            )


            if (
                candidate.get(
                    "status"
                )
                !=
                "unknown_rule"
            ):

                # --------------------------------------------
                # IMPORTANT:
                # DO NOT promote to violation yet.
                # --------------------------------------------

                candidate[
                    "candidate_only"
                ] = True


                candidate[
                    "raises_violation_verdict"
                ] = False


                candidate[
                    "missing_context"
                ] = [
                    "CASH_VISIBLE"
                ]


                candidate[
                    "trigger_signal"
                ] = event_type


                candidates.append(
                    candidate
                )


        # ====================================================
        # CANDIDATE:
        # CASHIER LEFT DURING SERVICE
        # ====================================================

        elif (
            event_type
            ==
            "CASHIER_AWAY_WHILE_CUSTOMER_PRESENT"
        ):

            candidate = (
                RULE_ENGINE
                .create_rule_result(
                    subtype=
                        "cashier_left_during_service",

                    timestamp_sec=
                        timestamp,

                    evidence_frames=
                        signal.get(
                            "evidence_frames",
                            []
                        ),

                    extra_description=(
                        "Customer remained visible while "
                        "the cashier moved away. "
                        "The active transaction state "
                        "has not yet been confirmed."
                    )
                )
            )


            if (
                candidate.get(
                    "status"
                )
                !=
                "unknown_rule"
            ):

                candidate[
                    "candidate_only"
                ] = True


                candidate[
                    "raises_violation_verdict"
                ] = False


                candidate[
                    "missing_context"
                ] = [
                    "TRANSACTION_ACTIVE"
                ]


                candidate[
                    "trigger_signal"
                ] = event_type


                candidates.append(
                    candidate
                )


        # ====================================================
        # CANDIDATE:
        # TWO ACTIVE TRANSACTIONS
        # ====================================================

        elif (
            event_type
            ==
            "MULTIPLE_CUSTOMERS_PRESENT"
        ):

            candidate = (
                RULE_ENGINE
                .create_rule_result(
                    subtype=
                        "two_active_transactions",

                    timestamp_sec=
                        timestamp,

                    evidence_frames=
                        signal.get(
                            "evidence_frames",
                            []
                        ),

                    extra_description=(
                        "Multiple customers were visible. "
                        "Presence alone does not prove "
                        "parallel active transactions."
                    )
                )
            )


            if (
                candidate.get(
                    "status"
                )
                !=
                "unknown_rule"
            ):

                candidate[
                    "candidate_only"
                ] = True


                candidate[
                    "raises_violation_verdict"
                ] = False


                candidate[
                    "missing_context"
                ] = [
                    "TRANSACTION_ACTIVE_CUSTOMER_1",
                    "TRANSACTION_ACTIVE_CUSTOMER_2"
                ]


                candidate[
                    "trigger_signal"
                ] = event_type


                candidates.append(
                    candidate
                )


    return candidates


# ============================================================
# CONFIRMED VIOLATIONS
# ============================================================

def build_confirmed_violations(
    signals
):
    """
    Пока здесь пусто.

    Здесь будут настоящие composite rules.

    Например:

    CASH_VISIBLE
        +
    CASHIER_AWAY_FROM_WORKSTATION
        +
    duration > threshold

        -> CH-02 unattended_cash


    Пока CASH_VISIBLE detector не подключен,
    мы сознательно НЕ поднимаем violation.
    """

    violations = []


    # ========================================================
    # NEXT IMPLEMENTATION:
    #
    # cash_events = ...
    #
    # if overlap(
    #     CASH_VISIBLE,
    #     CASHIER_AWAY_FROM_WORKSTATION
    # ):
    #
    #     violation = RULE_ENGINE.create_rule_result(...)
    #
    #     violations.append(violation)
    #
    # ========================================================


    return violations


# ============================================================
# MAIN PIPELINE
# ============================================================

def analyze_video(
    video_path
):
    """
    Main AI pipeline.

    API and Streamlit call ONLY this function.

    Architecture:

    VIDEO
        |
        +--> presence / roles
        |
        +--> handover
        |
        v
    unified event timeline
        |
        v
    rule candidates
        |
        v
    confirmed composite rules
        |
        v
    final verdict
    """

    started = time.time()


    video_path = Path(
        video_path
    )


    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )


    # ========================================================
    # AI BLOCK 1
    # PRESENCE / ROLES
    # ========================================================

    presence_started = (
        time.time()
    )


    presence = (
        run_presence_detector(
            video_path
        )
    )


    presence_time = (
        time.time()
        -
        presence_started
    )


    # ========================================================
    # AI BLOCK 2
    # HANDOVER
    # ========================================================

    handover_started = (
        time.time()
    )


    handover = (
        run_handover_detector(
            video_path
        )
    )


    handover_time = (
        time.time()
        -
        handover_started
    )


    # ========================================================
    # UNIFIED SIGNALS
    # ========================================================

    signals = []


    signals.extend(
        normalize_presence_events(
            presence.get(
                "events",
                []
            )
        )
    )


    signals.extend(
        normalize_handover_events(
            handover.get(
                "events",
                []
            )
        )
    )


    signals = sort_signals(
        signals
    )


    # ========================================================
    # RULE ENGINE
    # ========================================================

    rule_candidates = (
        build_rule_candidates(
            signals
        )
    )


    violations = (
        build_confirmed_violations(
            signals
        )
    )


    # ========================================================
    # FINAL VERDICT
    # ========================================================

    if violations:

        verdict = (
            "violation"
        )

    else:

        verdict = (
            "no_violation"
        )


    # ========================================================
    # PRIMARY ROLES
    # ========================================================

    # Presence detector uses the newer
    # role / track-stitching logic,
    # therefore use it as the main role output.

    roles = presence.get(
        "roles",
        {}
    )


    # Fallback if presence failed.

    if not roles:

        roles = handover.get(
            "roles",
            {}
        )


    # ========================================================
    # PROCESSING TIME
    # ========================================================

    elapsed = (
        time.time()
        -
        started
    )


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "video":
            video_path.name,


        # ----------------------------------------------------
        # FINAL OUTPUT
        # ----------------------------------------------------

        "verdict":
            verdict,

        "violations":
            violations,


        # ----------------------------------------------------
        # AI EVENT TIMELINE
        # ----------------------------------------------------

        "signals":
            signals,


        # ----------------------------------------------------
        # RULE ENGINE
        # ----------------------------------------------------

        "rule_candidates":
            rule_candidates,


        # ----------------------------------------------------
        # ROLES
        # ----------------------------------------------------

        "roles":
            roles,


        "roles_by_detector": {

            "presence":
                presence.get(
                    "roles",
                    {}
                ),

            "handover":
                handover.get(
                    "roles",
                    {}
                )
        },


        # ----------------------------------------------------
        # SCENE INFORMATION
        # ----------------------------------------------------

        "scene": {

            "workstation_anchor":
                presence.get(
                    "workstation_anchor"
                )
        },


        # ----------------------------------------------------
        # DETECTOR STATUS
        # ----------------------------------------------------

        "detectors": {

            "presence": {

                "status":
                    presence.get(
                        "status"
                    ),

                "error":
                    presence.get(
                        "error"
                    ),

                "processing_time_sec":
                    round(
                        presence_time,
                        2
                    )
            },


            "handover": {

                "status":
                    handover.get(
                        "status"
                    ),

                "error":
                    handover.get(
                        "error"
                    ),

                "processing_time_sec":
                    round(
                        handover_time,
                        2
                    )
            },


            "rule_engine": {

                "status":
                    "ok",

                "rules_loaded":
                    RULE_ENGINE.count_rules()
            }
        },


        # ----------------------------------------------------
        # VERSION
        # ----------------------------------------------------

        "analysis_version":
            "visual_process_mining_v1",


        # ----------------------------------------------------
        # TOTAL TIME
        # ----------------------------------------------------

        "processing_time_sec":
            round(
                elapsed,
                2
            )
    }


    return result