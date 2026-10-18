import json
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RULES_PATH = (
    PROJECT_ROOT
    / "rules"
    / "rules.json"
)


# Taxonomy for this visual prototype. This is applicability metadata,
# NOT a claim that all listed rules are automatically evaluated.
GLOBAL_PARENT_CLASSES = {
    "PHYSICAL_SECURITY", "WORKSTATION_SECURITY",
    "WORKSPACE_RISK", "UNKNOWN_ANOMALY",
}
GLOBAL_RULE_IDS = {"CH-02", "CH-04", "CH-05"}

# Operation requirements for rules in the current 44-rule catalog.
# Rule lists are intentionally conservative; specialized rules MUST NOT be
# marked evaluated merely because their operation is recognized.
RULE_OPERATION_REQUIREMENTS = {
    "CH-01": ["cash_acceptance"],
    "CH-03": ["customer_service"],
    "CH-06": ["customer_service"],
    "CS-01": ["cash_counting"],
    "CS-02": ["cash_acceptance"],
    "CS-03": ["cash_disbursement"],
    "CS-04": ["cash_counting"],
    "CS-05": ["cash_counting"],
    "CS-06": ["cash_counting"],
    "CS-07": ["cash_counting"],
    "CS-08": ["cash_disbursement", "currency_exchange"],
}


class RuleEngine:

    def __init__(self):

        if not RULES_PATH.exists():
            raise RuntimeError(
                f"Rules file not found: {RULES_PATH}"
            )

        with open(
            RULES_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            self.catalog = json.load(f)

        self.rules = (
            self.catalog.get(
                "rules",
                []
            )
        )

        self.acts = (
            self.catalog.get(
                "acts",
                {}
            )
        )

        self.rules_by_id = {}

        self.rules_by_subtype = {}

        for rule in self.rules:

            rule_id = rule.get(
                "rule_id"
            )

            subtype = rule.get(
                "subtype"
            )

            if rule_id:
                self.rules_by_id[
                    rule_id
                ] = rule

            if subtype:
                self.rules_by_subtype[
                    subtype
                ] = rule


    # ========================================================
    # INFO
    # ========================================================

    def count_rules(self):

        return len(
            self.rules
        )


    def get_rule_by_id(
        self,
        rule_id
    ):

        return self.rules_by_id.get(
            rule_id
        )


    def get_rule_by_subtype(
        self,
        subtype
    ):

        return self.rules_by_subtype.get(
            subtype
        )



    # ========================================================
    # OPERATION-AWARE RULE APPLICABILITY
    # ========================================================

    def get_rule_scope(self, rule):
        """Scope is not execution status. Unrecognized catalog groups fail closed."""
        explicit = rule.get("scope")
        if explicit in ("global", "operation_specific"):
            return explicit
        if (rule.get("parent_class") in GLOBAL_PARENT_CLASSES
                or rule.get("rule_id") in GLOBAL_RULE_IDS):
            return "global"
        return "operation_specific"

    def get_required_operations(self, rule):
        """Operations relevant to a rule, without asserting visual detectability."""
        explicit = rule.get("operation_types")
        if isinstance(explicit, list) and explicit:
            return [str(item) for item in explicit if item]
        if self.get_rule_scope(rule) == "global":
            return []
        rule_id = rule.get("rule_id", "")
        if rule_id in RULE_OPERATION_REQUIREMENTS:
            return list(RULE_OPERATION_REQUIREMENTS[rule_id])
        if rule_id.startswith("ID-"):
            return ["cash_acceptance", "cash_disbursement"]
        if rule_id.startswith("CO-"):
            return ["customer_service"]
        if rule_id.startswith("COL-"):
            return ["collection"]
        # Empty here means operation taxonomy is not yet specified.
        return []

    def describe_rule_applicability(self, operation_context, cash_status=None):
        """Audit catalog coverage without pretending unimplemented checks ran."""
        operation_type = (operation_context or {}).get("operation_type", "unknown")
        evaluated = []
        global_rule_ids = []
        specialized_rule_ids = []
        pending = []
        details = []
        for rule in self.rules:
            rid = rule.get("rule_id")
            if not rid:
                continue
            scope = self.get_rule_scope(rule)
            requirements = self.get_required_operations(rule)
            if scope == "global":
                global_rule_ids.append(rid)
            else:
                specialized_rule_ids.append(rid)

            # CH-02 is the ONLY rule backed by an implemented multi-signal
            # verdict detector in the supplied pipeline.
            if rid == "CH-02" and cash_status == "ok":
                state = "screened_by_implemented_detector"
                evaluated.append(rid)
            elif rid == "CH-02":
                state = "detector_unavailable"
                pending.append(rid)
            elif scope == "global":
                state = "requires_additional_visual_detector"
                pending.append(rid)
            elif operation_type == "unknown":
                state = "awaiting_operation_classification_and_visual_detector"
                pending.append(rid)
            elif requirements and operation_type not in requirements:
                state = "different_operation_context"
            else:
                state = "operation_matches_but_visual_detector_not_implemented"
                pending.append(rid)

            details.append({
                "rule_id": rid,
                "scope": scope,
                "required_operation_types": requirements,
                "evaluation_status": state,
            })

        return {
            "catalog_rule_count": self.count_rules(),
            "global_rule_ids": global_rule_ids,
            "operation_specific_rule_ids": specialized_rule_ids,
            "actually_screened_rule_ids": evaluated,
            "not_fully_evaluated_rule_ids": pending,
            "rules": details,
            "disclaimer": (
                "Catalog coverage is not model coverage: only rules listed in "
                "actually_screened_rule_ids have implemented visual screening. "
                "Other rules require additional detectors and, where relevant, "
                "reliable operation recognition."
            ),
        }

    # ========================================================
    # NORMATIVE SOURCES
    # ========================================================

    def build_normative_sources(
        self,
        rule
    ):

        result = []

        for norm in rule.get(
            "norms",
            []
        ):

            act_id = norm.get(
                "act"
            )

            clause = norm.get(
                "clause"
            )

            act = self.acts.get(
                act_id,
                {}
            )

            result.append({
                "act_id":
                    act_id,

                "title":
                    act.get(
                        "title"
                    ),

                "short":
                    act.get(
                        "short"
                    ),

                "clause":
                    clause,

                "url":
                    act.get(
                        "url"
                    ),

                "role":
                    norm.get(
                        "role"
                    )
            })

        return result


    # ========================================================
    # CREATE VIOLATION OBJECT
    # ========================================================

    def create_rule_result(
        self,
        subtype,
        timestamp_sec,
        evidence_frames=None,
        confidence=None,
        extra_description=None
    ):

        rule = (
            self.get_rule_by_subtype(
                subtype
            )
        )

        if rule is None:

            return {
                "status":
                    "unknown_rule",

                "subtype":
                    subtype,

                "timestamp_sec":
                    timestamp_sec
            }


        emit_policy = rule.get(
            "emit_policy",
            "review_only"
        )


        description = rule.get(
            "explanation",
            ""
        )


        if extra_description:

            description = (
                description
                +
                " "
                +
                extra_description
            )


        result = {
            "rule_id":
                rule.get(
                    "rule_id"
                ),

            "parent_class":
                rule.get(
                    "parent_class"
                ),

            "subtype":
                rule.get(
                    "subtype"
                ),

            "type":
                rule.get(
                    "violation_type"
                ),

            "timestamp_sec":
                timestamp_sec,

            "description":
                description,

            "severity":
                rule.get(
                    "severity"
                ),

            "emit_policy":
                emit_policy,

            "norm_confidence":
                rule.get(
                    "norm_confidence"
                ),

            "norm_status":
                rule.get(
                    "norm_status"
                ),

            "false_positive_guards":
                rule.get(
                    "false_positive_guards"
                ),

            "normative_sources":
                self.build_normative_sources(
                    rule
                ),

            "evidence_frames":
                evidence_frames or [],

            "rule_scope":
                self.get_rule_scope(rule),

            "required_operation_types":
                self.get_required_operations(rule),
        }


        if confidence is not None:

            result[
                "confidence"
            ] = confidence


        # ----------------------------------------------
        # Решение, может ли правило само поднять verdict
        # ----------------------------------------------

        result[
            "raises_violation_verdict"
        ] = (
            emit_policy
            in (
                "violation",
                "violation_with_disclaimer"
            )
        )


        result[
            "requires_review"
        ] = (
            emit_policy
            ==
            "review_only"
        )


        return result