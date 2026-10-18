"""Forte Process Intelligence -- unified, local video analysis pipeline.

Outputs API-compatible binary verdict plus explicit diagnostics / review flags.
Cash alerts are PROVISIONAL and must be reviewed by a human.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.presence_events import analyze_presence_events
from src.rule_engine import RuleEngine
from src.object_events import analyze_workspace_objects

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDOVER_RESULTS_DIR = PROJECT_ROOT / "data" / "handover_auto_results"
RULE_ENGINE = RuleEngine()

# Operational safeguards for a 5-minute API budget. Configure via environment.
HANDOVER_MAX_VIDEO_SEC = float(os.getenv("FORTE_HANDOVER_MAX_VIDEO_SEC", "35"))
HANDOVER_TIMEOUT_SEC = int(os.getenv("FORTE_HANDOVER_TIMEOUT_SEC", "170"))

# These are provisional screening thresholds, not validated probabilities.
MIN_CASH_SEED_CONF = 0.30
MIN_AWAY_DURATION_SEC = 2.0
MIN_STABLE_MATCHES = 3


def run_presence_detector(video_path):
    try:
        data = analyze_presence_events(video_path)
        return {
            "status": "ok", "error": None,
            "roles": data.get("roles", {}),
            "events": data.get("events", []),
            "workstation_anchor": data.get("workstation_anchor"),
        }
    except Exception as exc:
        return {
            "status": "failed", "error": str(exc),
            "roles": {}, "events": [], "workstation_anchor": None,
        }


def video_duration_sec(video_path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return total / fps if fps > 0 and total > 0 else None
    finally:
        cap.release()


def run_handover_detector(video_path, presence, duration):
    """Handover is an observation, NOT proof of a violation."""
    script = PROJECT_ROOT / "detect_handover_auto.py"
    empty = {"roles": {}, "events": []}

    if not script.exists():
        return dict(empty, status="failed", error="detect_handover_auto.py missing")

    if duration and duration > HANDOVER_MAX_VIDEO_SEC:
        return dict(
            empty, status="skipped",
            error=(f"Handover skipped on {duration:.1f}s video to meet "
                   f"time budget (limit={HANDOVER_MAX_VIDEO_SEC:.0f}s)"),
        )

    if presence.get("status") == "ok":
        roles = presence.get("roles") or {}
        has_customer_role = any(value == "CUSTOMER" for value in roles.values())
        has_customer_event = any(
            event.get("type") == "CUSTOMER_PRESENT"
            for event in presence.get("events", [])
            if isinstance(event, dict)
        )
        if roles and not has_customer_role and not has_customer_event:
            return dict(empty, status="skipped", error="No customer role detected")

    try:
        process = subprocess.run(
            [sys.executable, str(script), str(video_path)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            timeout=HANDOVER_TIMEOUT_SEC,
        )
        if process.returncode != 0:
            return dict(
                empty, status="failed",
                error=(process.stderr or process.stdout or "Handover failed")[-1200:],
            )
        result_file = HANDOVER_RESULTS_DIR / f"{video_path.stem}_handover.json"
        if not result_file.exists():
            return dict(empty, status="failed", error="Handover JSON not found")
        with result_file.open(encoding="utf-8") as f:
            data = json.load(f)
        return {
            "status": "ok", "error": None,
            "roles": data.get("roles", {}),
            "events": data.get("handover_events", []),
        }
    except subprocess.TimeoutExpired:
        return dict(empty, status="skipped", error="Handover runtime exceeded budget")
    except Exception as exc:
        return dict(empty, status="failed", error=str(exc))


def run_cash_detector(video_path, presence):
    """Reuse presence output. No second person-detection pass.

    Uses the exact helper functions already tested in detect_cash_events.py.
    The saved file remains diagnostic and never independently claims violation.
    """
    empty = {
        "away_events": [], "yolo_cash_on_desk_proposals": [],
        "rule_candidates": [], "evidence_path": None,
    }
    if presence.get("status") != "ok":
        return dict(empty, status="skipped", error="Presence detector unavailable")
    anchor = presence.get("workstation_anchor")
    if not anchor:
        return dict(empty, status="skipped", error="Workstation anchor unavailable")

    try:
        import cv2
        import torch
        from ultralytics import YOLO
        from detect_cash_events import (
            MODEL_PATH, OUTPUT_ROOT, MAX_SEED_AGE_SEC,
            scan_seeds, check_seed,
        )

        if not MODEL_PATH.exists():
            return dict(empty, status="failed", error=f"Missing model: {MODEL_PATH}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return dict(empty, status="failed", error="Cannot open video")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or total <= 0:
                return dict(empty, status="failed", error="Invalid video FPS/frame count")

            duration = total / fps
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = YOLO(str(MODEL_PATH))
            output_dir = OUTPUT_ROOT / video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            seeds = scan_seeds(cap, fps, total, anchor, model, device, duration)
            away_events = [
                e for e in presence.get("events", [])
                if isinstance(e, dict)
                and e.get("type") == "CASHIER_AWAY_FROM_WORKSTATION"
            ]
            candidates = []
            for event_idx, away_event in enumerate(away_events, 1):
                away_at = float(away_event.get(
                    "start_sec", away_event.get("timestamp_sec", -1)
                ))
                eligible = [
                    seed for seed in seeds
                    if 0 <= away_at - float(seed["timestamp_sec"]) <= MAX_SEED_AGE_SEC
                ]
                eligible.sort(key=lambda seed: (
                    -seed["confidence"], abs(away_at - seed["timestamp_sec"])
                ))
                for seed_idx, seed in enumerate(eligible[:3], 1):
                    # Unique evidence names for each attempt (avoids overwrites).
                    unique_idx = event_idx * 10 + seed_idx
                    candidate = check_seed(
                        cap, fps, total, duration, anchor,
                        seed, away_at, output_dir, unique_idx,
                    )
                    if candidate:
                        candidate["cashier_away_duration_sec"] = float(
                            away_event.get("duration_sec") or 0
                        )
                        candidates.append(candidate)
                        if candidate.get("temporal_support"):
                            break

            result = {
                "video": video_path.name,
                "analysis_type": "diagnostic_cash_temporal_candidates",
                "confirmed_violation": False,
                "workstation_anchor": anchor,
                "away_events": away_events,
                "yolo_cash_on_desk_proposals": seeds,
                "rule_candidates": candidates,
                "limitations": [
                    "Cash model and temporal thresholds are not calibrated on unseen video",
                    "Temporal template similarity alone does not prove cash identity",
                ],
            }
            output_path = output_dir / "auto_cash_events.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return dict(result, status="ok", error=None, evidence_path=str(output_path))
        finally:
            cap.release()
    except Exception as exc:
        return dict(empty, status="failed", error=f"{type(exc).__name__}: {exc}")


def normalize_presence_events(events):
    return [
        {
            "type": e.get("type"),
            "timestamp_sec": e.get("start_sec", e.get("timestamp_sec")),
            "start_sec": e.get("start_sec"),
            "end_sec": e.get("end_sec"),
            "duration_sec": e.get("duration_sec"),
            "evidence_frames": [],
            "source": "presence_detector",
        }
        for e in events if isinstance(e, dict)
    ]


def normalize_handover_events(events):
    return [
        {
            "type": "HANDOVER_EVENT",
            "timestamp_sec": e.get("timestamp_sec"),
            "start_sec": e.get("timestamp_sec"),
            "peak_timestamp_sec": e.get("peak_timestamp_sec"),
            "end_timestamp_sec": e.get("end_timestamp_sec"),
            "cashier_id": e.get("cashier_id"),
            "customer_id": e.get("customer_id"),
            "min_distance_norm": e.get("min_distance_norm"),
            "evidence_frames": e.get("evidence_frames", []),
            "source": "handover_detector",
        }
        for e in events if isinstance(e, dict)
    ]


def normalize_cash_events(cash):
    signals = []
    for seed in cash.get("yolo_cash_on_desk_proposals", []):
        signals.append({
            "type": "CASH_ON_DESK_PROPOSAL",
            "timestamp_sec": seed.get("timestamp_sec"),
            "cash_confidence": seed.get("confidence"),
            "box": seed.get("box"),
            "source": "cash_yolo",
            "evidence_frames": [],
        })
    for candidate in cash.get("rule_candidates", []):
        signals.append({
            "type": "CASH_PERSISTENCE_CANDIDATE",
            "timestamp_sec": candidate.get("cashier_away_timestamp_sec"),
            "cash_seed_timestamp_sec": (candidate.get("cash_seed") or {}).get("timestamp_sec"),
            "matches": candidate.get("matches", 0),
            "temporal_support": bool(candidate.get("temporal_support")),
            "evidence_frames": candidate.get("evidence_frames", []),
            "source": "cash_temporal_detector",
        })
    return signals


def sort_signals(signals):
    def when(e):
        try:
            return float(e.get("timestamp_sec"))
        except (ValueError, TypeError):
            return float("inf")
    return sorted(signals, key=when)


def select_evidence(candidate):
    """Prefer seed, departure, and latest after-departure frame (max 3)."""
    original = candidate.get("evidence_frames", [])
    if not original:
        return []
    if len(original) <= 3:
        return original
    return [original[0], original[1], original[-1]]


def is_high_support(candidate):
    """Conservative provisional screening, NOT validated legal confirmation."""
    if not candidate.get("temporal_support"):
        return False
    if float((candidate.get("cash_seed") or {}).get("confidence") or 0) < MIN_CASH_SEED_CONF:
        return False
    if float(candidate.get("cashier_away_duration_sec") or 0) < MIN_AWAY_DURATION_SEC:
        return False
    if int(candidate.get("matches") or 0) < MIN_STABLE_MATCHES:
        return False
    away = float(candidate.get("cashier_away_timestamp_sec") or 0)
    post_departure = [
        check for check in candidate.get("checks", [])
        if check.get("supports_same_stationary_patch")
        and float(check.get("timestamp_sec") or 0) >= away + 2.0
    ]
    return bool(post_departure)



def infer_operation_context(signals, roles=None, detector_statuses=None):
    """Conservative observed-activity recognizer (not a trained classifier).

    Our handover detector observes hand proximity, not direction or cash identity.
    For that reason, acceptance / disbursement must remain UNKNOWN.
    """
    roles = roles or {}
    detector_statuses = detector_statuses or {}
    types = [event.get("type") for event in signals if isinstance(event, dict)]
    observed = set(types)
    has_customer = (
        "CUSTOMER_PRESENT" in observed
        or "CASHIER_AWAY_WHILE_CUSTOMER_PRESENT" in observed
        or any(str(role).upper() == "CUSTOMER" for role in roles.values())
    )
    has_handover = "HANDOVER_EVENT" in observed
    has_cash_proposal = "CASH_ON_DESK_PROPOSAL" in observed
    has_cash_persistence = any(
        bool(event.get("temporal_support"))
        for event in signals
        if event.get("type") == "CASH_PERSISTENCE_CANDIDATE"
    )
    has_departure = "CASHIER_AWAY_FROM_WORKSTATION" in observed

    # Only correlate candidate cash and a handover when close in time.
    def times(event_type):
        result = []
        for event in signals:
            if event.get("type") != event_type:
                continue
            try:
                result.append(float(event.get("timestamp_sec")))
            except (ValueError, TypeError):
                pass
        return result

    handover_times = times("HANDOVER_EVENT")
    cash_times = times("CASH_ON_DESK_PROPOSAL")
    cash_near_handover = bool(
        handover_times and cash_times
        and any(abs(h - c) <= 6.0 for h in handover_times for c in cash_times)
    )

    if has_handover and cash_near_handover:
        activity = "cash_related_handover_candidate"
        possibilities = ["cash_acceptance", "cash_disbursement", "other_exchange"]
    elif has_handover:
        activity = "person_to_person_handover_observed"
        possibilities = ["cash_acceptance", "cash_disbursement", "document_exchange"]
    elif has_cash_proposal and has_departure:
        activity = "cash_at_workstation_with_cashier_absence_candidate"
        possibilities = ["cash_handling", "cash_storage", "unknown"]
    elif has_cash_proposal:
        activity = "cash_at_workstation_candidate"
        possibilities = ["cash_acceptance", "cash_disbursement", "cash_counting"]
    elif has_customer:
        activity = "customer_presence_observed"
        possibilities = ["customer_service", "unknown"]
    elif has_departure:
        activity = "cashier_departure_observed"
        possibilities = ["unknown"]
    else:
        activity = "unclassified_scene"
        possibilities = ["unknown"]

    return {
        "operation_type": "unknown",
        "operation_recognition_status": "insufficient_directional_or_procedural_evidence",
        "observed_activity": activity,
        "possible_operations_not_confirmed": possibilities,
        "observation_flags": {
            "customer_present": has_customer,
            "handover_proximity_event": has_handover,
            "cash_on_desk_yolo_proposal": has_cash_proposal,
            "cash_temporal_persistence_candidate": has_cash_persistence,
            "cashier_departure_observed": has_departure,
            "cash_proposal_near_handover": cash_near_handover,
        },
        "event_counts": {
            kind: types.count(kind) for kind in sorted(set(types)) if kind
        },
        "detector_statuses": detector_statuses,
        "method": "deterministic_interpretation_of_visual_signals",
        "trained_operation_classifier": False,
        "requires_review": True,
        "reason": (
            "Hand proximity does not establish cash transfer direction or "
            "transaction type; counting, identity verification, and collection "
            "procedures are not currently recognized reliably."
        ),
    }


def build_rule_candidates(signals, cash_candidates=None):
    candidates = []
    specs = {
        "CASHIER_AWAY_FROM_WORKSTATION": (
            "unattended_cash", ["CASH_ON_DESK_REMAINS_AFTER_DEPARTURE"],
            "Cashier left the workstation; cash persistence must be verified."
        ),
        "CASHIER_AWAY_WHILE_CUSTOMER_PRESENT": (
            "cashier_left_during_service", ["TRANSACTION_ACTIVE"],
            "A customer was present; active transaction is not verified."
        ),
        "MULTIPLE_CUSTOMERS_PRESENT": (
            "two_active_transactions",
            ["TRANSACTION_ACTIVE_CUSTOMER_1", "TRANSACTION_ACTIVE_CUSTOMER_2"],
            "Multiple customers alone do not prove simultaneous transactions."
        ),
    }
    for signal in signals:
        spec = specs.get(signal.get("type"))
        if not spec:
            continue
        subtype, missing, explanation = spec
        item = RULE_ENGINE.create_rule_result(
            subtype=subtype,
            timestamp_sec=signal.get("timestamp_sec"),
            evidence_frames=signal.get("evidence_frames", []),
            extra_description=explanation,
        )
        if item.get("status") == "unknown_rule":
            continue
        item.update({
            "candidate_only": True,
            "raises_violation_verdict": False,
            "missing_context": missing,
            "trigger_signal": signal.get("type"),
        })
        candidates.append(item)

    for cash in cash_candidates or []:
        item = RULE_ENGINE.create_rule_result(
            subtype="unattended_cash",
            timestamp_sec=cash.get("cashier_away_timestamp_sec"),
            evidence_frames=select_evidence(cash),
            extra_description=(
                "Automatic cash-object proposal and temporal matching near "
                "cashier departure; screening result, requires review."
            ),
        )
        if item.get("status") == "unknown_rule":
            continue
        item.update({
            "candidate_only": True,
            "raises_violation_verdict": False,
            "missing_context": [] if is_high_support(cash) else ["STRONG_TEMPORAL_EVIDENCE"],
            "trigger_signal": "CASH_PERSISTENCE_CANDIDATE",
            "temporal_support": bool(cash.get("temporal_support")),
            "matched_checks": cash.get("matches", 0),
            "cash_seed_confidence": (cash.get("cash_seed") or {}).get("confidence"),
        })
        candidates.append(item)
    return candidates


def build_confirmed_violations(signals, cash_candidates=None):
    """AI alerts for binary API; findings explicitly require manual review."""
    violations = []
    for cash in cash_candidates or []:
        if not is_high_support(cash):
            continue
        item = RULE_ENGINE.create_rule_result(
            subtype="unattended_cash",
            timestamp_sec=cash.get("cashier_away_timestamp_sec"),
            evidence_frames=select_evidence(cash),
            extra_description=(
                "AI flagged cash left on the workstation during cashier "
                "absence based on object detection and temporal consistency. "
                "Requires human and regulatory verification."
            ),
        )
        if item.get("status") == "unknown_rule" or not item.get("raises_violation_verdict"):
            continue
        item.update({
            "decision_status": "ai_alert_requires_review",
            "requires_review": True,
            "regulatory_check_required": True,
            "detection_method": "YOLO cash + stationary-patch check + cashier tracking",
            "cash_seed_confidence": (cash.get("cash_seed") or {}).get("confidence"),
            "matched_checks": cash.get("matches", 0),
            "cashier_away_duration_sec": cash.get("cashier_away_duration_sec"),
            "candidate_only": False,
        })
        violations.append(item)
    return violations



def build_workspace_object_candidates(candidate_specs):
    """Map generic object proposals to catalog entries, never to binary verdicts."""
    results = []
    for spec in candidate_specs or []:
        item = RULE_ENGINE.create_rule_result(
            subtype=spec.get("subtype"),
            timestamp_sec=spec.get("timestamp_sec"),
            evidence_frames=spec.get("evidence_frames") or [],
            extra_description=spec.get("extra_description"),
        )
        if item.get("status") == "unknown_rule":
            continue
        item.update({
            "candidate_only": True,
            "raises_violation_verdict": False,
            "requires_review": True,
            "decision_status": "visual_object_proposal_requires_review",
            "missing_context": spec.get("missing_context") or [],
            "trigger_signal": spec.get("trigger_signal"),
            "detection_method": "generic YOLO COCO object detector; unverified context",
        })
        results.append(item)
    return results


def analyze_video(video_path):
    """Entry point used by both FastAPI and Streamlit."""
    started = time.time()
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    duration = video_duration_sec(video_path)

    t0 = time.time()
    presence = run_presence_detector(video_path)
    presence_time = time.time() - t0

    t0 = time.time()
    cash = run_cash_detector(video_path, presence)
    cash_time = time.time() - t0

    t0 = time.time()
    handover = run_handover_detector(video_path, presence, duration)
    handover_time = time.time() - t0

    t0 = time.time()
    workspace_objects = analyze_workspace_objects(video_path, presence)
    workspace_time = time.time() - t0

    signals = sort_signals(
        normalize_presence_events(presence.get("events", []))
        + normalize_handover_events(handover.get("events", []))
        + normalize_cash_events(cash)
        + workspace_objects.get("events", [])
    )
    roles = presence.get("roles") or handover.get("roles", {})
    operation_context = infer_operation_context(
        signals,
        roles=roles,
        detector_statuses={
            "presence": presence.get("status"),
            "cash": cash.get("status"),
            "handover": handover.get("status"),
        },
    )
    rule_applicability = RULE_ENGINE.describe_rule_applicability(
        operation_context, cash_status=cash.get("status")
    )
    candidates = (
        build_rule_candidates(signals, cash.get("rule_candidates", []))
        + build_workspace_object_candidates(workspace_objects.get("rule_candidates", []))
    )
    violations = build_confirmed_violations(signals, cash.get("rule_candidates", []))

    warnings = []
    for key, output in (("presence", presence), ("cash", cash),
                        ("handover", handover), ("workspace_objects", workspace_objects)):
        if output.get("status") != "ok":
            warnings.append(f"{key}: {output.get('status')} -- {output.get('error')}")
    if violations:
        warnings.append("AI alerts are provisional, require human/regulatory review")

    result = {
        "video": video_path.name,
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
        "signals": signals,
        "rule_candidates": candidates,
        "object_observations": {
            "status": workspace_objects.get("status"),
            "detections_by_class": workspace_objects.get("detections_by_class", {}),
            "review_candidates": len(workspace_objects.get("rule_candidates", [])),
            "note": "Generic object proposals only; not proven regulatory violations.",
        },
        "operation_context": operation_context,
        "rule_applicability": rule_applicability,
        "verdict_scope_note": (
            "no_violation only means no implemented automatic alert triggered; "
            "it does not certify cash-discipline compliance."
        ),
        "roles": roles,
        "roles_by_detector": {
            "presence": presence.get("roles", {}),
            "handover": handover.get("roles", {}),
        },
        "scene": {
            "workstation_anchor": presence.get("workstation_anchor"),
        },
        "detectors": {
            "presence": {"status": presence.get("status"), "error": presence.get("error"),
                         "processing_time_sec": round(presence_time, 2)},
            "cash": {"status": cash.get("status"), "error": cash.get("error"),
                     "processing_time_sec": round(cash_time, 2),
                     "proposals": len(cash.get("yolo_cash_on_desk_proposals", [])),
                     "candidate_pairs": len(cash.get("rule_candidates", [])),
                     "result_json": cash.get("evidence_path")},
            "handover": {"status": handover.get("status"), "error": handover.get("error"),
                         "processing_time_sec": round(handover_time, 2)},
            "workspace_objects": {"status": workspace_objects.get("status"),
                                   "error": workspace_objects.get("error"),
                                   "processing_time_sec": round(workspace_time, 2),
                                   "sampled_frames": workspace_objects.get("sampled_frames", 0),
                                   "detections_by_class": workspace_objects.get("detections_by_class", {}),
                                   "result_json": workspace_objects.get("evidence_path")},
            "rule_engine": {"status": "ok", "rules_loaded": RULE_ENGINE.count_rules(),
                            "actually_screened_rule_ids": rule_applicability["actually_screened_rule_ids"]},
            "operation_recognition": {"status": "partial", "operation_type": operation_context["operation_type"],
                                      "method": operation_context["method"]},
        },
        "analysis_status": "complete" if all(
            x.get("status") == "ok" for x in (presence, cash, handover, workspace_objects)
        ) else "partial",
        "requires_review": bool(violations or candidates or warnings),
        "review_candidate_rule_ids": sorted({c.get("rule_id") for c in candidates if c.get("rule_id")}),
        "screening_status": ("ai_alert_requires_review" if violations else
                            "visual_review_candidates" if candidates else
                            "no_implemented_alert"),
        "analysis_warnings": warnings,
        "analysis_version": "visual_process_mining_v3_6_multi_event_review",
        "processing_time_sec": round(time.time() - started, 2),
    }
    return result
