import math


class AutoRoleAssigner:
    def __init__(
        self,
        min_track_frames=20,
        min_presence_ratio=0.10
    ):
        self.stats = {}

        self.min_track_frames = min_track_frames
        self.min_presence_ratio = min_presence_ratio

        # Diagnostics. Fallback is provisional, not verified human identity.
        self.assignment_method = "unassigned"
        self.assignment_confidence = "unknown"
        self.assignment_reason = None


    def update(self, person_id, box):

        x1, y1, x2, y2 = box

        center = (
            (x1 + x2) / 2,
            (y1 + y2) / 2
        )


        if person_id not in self.stats:

            self.stats[person_id] = {
                "frames": 0,
                "last_center": None,
                "movement": 0.0
            }


        data = self.stats[
            person_id
        ]


        data["frames"] += 1


        if data["last_center"] is not None:

            data["movement"] += math.dist(
                data["last_center"],
                center
            )


        data["last_center"] = center


    def assign_roles(self):

        self.assignment_method = "unassigned"
        self.assignment_confidence = "unknown"
        self.assignment_reason = None

        if not self.stats:
            return {}


        max_frames = max(
            data["frames"]
            for data
            in self.stats.values()
        )


        valid_ids = []


        for person_id, data in self.stats.items():

            presence_ratio = (
                data["frames"]
                / max_frames
            )


            if (
                data["frames"]
                >= self.min_track_frames
                and
                presence_ratio
                >= self.min_presence_ratio
            ):

                valid_ids.append(
                    person_id
                )


        # ----------------------------
        # Noise / tiny tracks
        # ----------------------------

        roles = {
            person_id: "IGNORE"
            for person_id
            in self.stats
        }


        if not valid_ids:
            # Only when the standard policy rejected ALL tracks.
            # A frequent CCTV problem is one long stable cashier track split
            # into shorter fragments (e.g., 18 + 9 + 8 instead of >=20).
            # This fallback is deliberately narrow and is not a proof of role.
            ordered = sorted(
                self.stats.items(),
                key=lambda pair: pair[1]["frames"],
                reverse=True,
            )
            if len(ordered) < 2:
                self.assignment_reason = "insufficient_comparison_tracks"
                return roles

            lead_id, lead = ordered[0]
            runner_frames = ordered[1][1]["frames"]
            lead_frames = lead["frames"]
            min_frames = max(12, math.ceil(self.min_track_frames * 0.70))
            avg_movement = lead["movement"] / max(1, lead_frames - 1)

            # Evidence: enough frames, clearly dominant track, limited motion.
            if (
                lead_frames >= min_frames
                and lead_frames >= 1.60 * max(1, runner_frames)
                and avg_movement <= 3.0
            ):
                roles[lead_id] = "CASHIER"
                self.assignment_method = "dominant_stable_track_fallback"
                self.assignment_confidence = "provisional"
                self.assignment_reason = (
                    f"{lead_frames} frames, runner={runner_frames}, "
                    f"mean_motion={avg_movement:.2f}px; no normal valid tracks"
                )
            else:
                self.assignment_reason = (
                    f"fallback_rejected: frames={lead_frames}/{min_frames}, "
                    f"runner={runner_frames}, mean_motion={avg_movement:.2f}px"
                )
            return roles


        # ----------------------------
        # Find CASHIER
        # ----------------------------

        cashier_scores = {}


        for person_id in valid_ids:

            data = self.stats[
                person_id
            ]


            presence = (
                data["frames"]
                / max_frames
            )


            avg_movement = (
                data["movement"]
                /
                max(
                    data["frames"] - 1,
                    1
                )
            )


            cashier_score = (
                presence
                -
                0.01
                *
                avg_movement
            )


            cashier_scores[
                person_id
            ] = cashier_score


        cashier_id = max(
            cashier_scores,
            key=cashier_scores.get
        )


        roles[
            cashier_id
        ] = "CASHIER"

        self.assignment_method = "standard"
        self.assignment_confidence = "standard_heuristic"


        # ----------------------------
        # Remaining stable people
        # ----------------------------

        for person_id in valid_ids:

            if person_id == cashier_id:
                continue

            roles[
                person_id
            ] = "CUSTOMER"


        return roles