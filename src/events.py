class EventDetector:
    def __init__(self, confirm_frames=3):
        """
        confirm_frames:
        сколько последовательных измерений нужно,
        чтобы подтвердить ENTER/EXIT.

        При FRAME_STEP=5 и видео ~30 FPS:
        3 измерения ≈ 0.5 секунды.
        """

        self.confirm_frames = confirm_frames

        # Подтвержденные зоны человека
        self.stable_zones = {}

        # Счетчики потенциального входа
        self.enter_counts = {}

        # Счетчики потенциального выхода
        self.exit_counts = {}

    def _init_person(self, person_id, current_zones):
        self.stable_zones[person_id] = set(current_zones)
        self.enter_counts[person_id] = {}
        self.exit_counts[person_id] = {}

    def update(self, timestamp, tracks_with_zones):
        events = []

        for item in tracks_with_zones:

            person_id = item["id"]

            current_zones = set(
                item.get("zones", [])
            )

            # Первый раз увидели человека
            if person_id not in self.stable_zones:
                self._init_person(
                    person_id,
                    current_zones
                )
                continue

            stable = self.stable_zones[person_id]

            enter_counts = self.enter_counts[person_id]
            exit_counts = self.exit_counts[person_id]

            # ==================================
            # CHECK ENTER
            # ==================================

            possible_enters = (
                current_zones - stable
            )

            for zone in possible_enters:

                enter_counts[zone] = (
                    enter_counts.get(zone, 0) + 1
                )

                if (
                    enter_counts[zone]
                    >= self.confirm_frames
                ):

                    stable.add(zone)

                    events.append({
                        "timestamp_sec": timestamp,
                        "person_id": person_id,
                        "event": "ENTER_ZONE",
                        "zone": zone
                    })

                    enter_counts[zone] = 0

            # Если потенциальный вход исчез,
            # сбрасываем счетчик
            for zone in list(
                enter_counts.keys()
            ):

                if zone not in current_zones:
                    enter_counts[zone] = 0

            # ==================================
            # CHECK EXIT
            # ==================================

            possible_exits = (
                stable - current_zones
            )

            for zone in possible_exits:

                exit_counts[zone] = (
                    exit_counts.get(zone, 0) + 1
                )

                if (
                    exit_counts[zone]
                    >= self.confirm_frames
                ):

                    stable.remove(zone)

                    events.append({
                        "timestamp_sec": timestamp,
                        "person_id": person_id,
                        "event": "EXIT_ZONE",
                        "zone": zone
                    })

                    exit_counts[zone] = 0

            # Если снова появился в зоне,
            # значит выхода не было
            for zone in list(
                exit_counts.keys()
            ):

                if zone in current_zones:
                    exit_counts[zone] = 0

        return events