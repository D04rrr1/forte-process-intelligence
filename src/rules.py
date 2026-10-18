class RuleEngine:
    def __init__(
        self,
        role_map,
        restricted_confirm_sec=1.0,
        missing_reset_sec=1.0
    ):
        self.role_map = role_map

        self.restricted_confirm_sec = (
            restricted_confirm_sec
        )

        self.missing_reset_sec = (
            missing_reset_sec
        )

        # Состояние потенциального нарушения
        self.restricted_state = {}

        # Когда человека последний раз видели
        self.last_seen = {}

    def update(
        self,
        timestamp,
        tracks
    ):
        """
        tracks:

        [
            {
                "id": 1,
                "zones": ["CASHIER_ZONE"]
            },
            {
                "id": 2,
                "zones": ["CUSTOMER_ZONE"]
            }
        ]
        """

        violations = []

        present_ids = set()

        for track in tracks:

            person_id = track["id"]

            zones = set(
                track.get(
                    "zones",
                    []
                )
            )

            present_ids.add(
                person_id
            )

            self.last_seen[
                person_id
            ] = timestamp

            role = self.role_map.get(
                person_id,
                "UNKNOWN"
            )

            # Кассир имеет право находиться
            # в служебных областях
            if role == "CASHIER":
                self.restricted_state.pop(
                    person_id,
                    None
                )
                continue

            # --------------------------------
            # Проверяем restricted zone
            # --------------------------------

            restricted_zone = None

            if "SECURE_ZONE" in zones:
                restricted_zone = (
                    "SECURE_ZONE"
                )

            elif "CASHIER_ZONE" in zones:
                restricted_zone = (
                    "CASHIER_ZONE"
                )

            # Человек находится
            # в запрещенной зоне
            if restricted_zone is not None:

                state = (
                    self.restricted_state
                    .get(person_id)
                )

                # Только начал находиться
                # в этой зоне
                if (
                    state is None
                    or
                    state["zone"]
                    != restricted_zone
                ):

                    self.restricted_state[
                        person_id
                    ] = {
                        "zone":
                            restricted_zone,

                        "start":
                            timestamp,

                        "emitted":
                            False
                    }

                    continue

                duration = (
                    timestamp -
                    state["start"]
                )

                # Не реагируем на случайный
                # заход bbox на границу
                if (
                    duration
                    >=
                    self.restricted_confirm_sec
                    and
                    not state["emitted"]
                ):

                    violation = {
                        "timestamp_sec":
                            round(
                                state["start"],
                                2
                            ),

                        "detected_sec":
                            round(
                                timestamp,
                                2
                            ),

                        "person_id":
                            person_id,

                        "role":
                            role,

                        "type":
                            "UNAUTHORIZED_ACCESS",

                        "parent_class":
                            "ACCESS_CONTROL",

                        "zone":
                            restricted_zone,

                        "description":
                            (
                                f"{role} remained "
                                f"in {restricted_zone} "
                                f"for {duration:.1f}s."
                            ),

                        "confidence_source":
                            "rule"
                    }

                    violations.append(
                        violation
                    )

                    state["emitted"] = True

            else:

                # Человек вышел
                # из restricted area
                self.restricted_state.pop(
                    person_id,
                    None
                )

        # --------------------------------
        # Очистка пропавших tracks
        # --------------------------------

        for person_id in list(
            self.restricted_state.keys()
        ):

            last_seen = self.last_seen.get(
                person_id,
                timestamp
            )

            if (
                timestamp - last_seen
                >
                self.missing_reset_sec
            ):

                self.restricted_state.pop(
                    person_id,
                    None
                )

        return violations