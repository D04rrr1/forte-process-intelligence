class HandoverDetector:
    def __init__(
        self,
        approach_threshold=0.10,
        contact_threshold=0.06,
        release_threshold=0.09,
        approach_window_sec=3.0,
        max_event_sec=4.0,
        min_contact_samples=3
    ):
        self.approach_threshold = approach_threshold
        self.contact_threshold = contact_threshold
        self.release_threshold = release_threshold

        self.approach_window_sec = approach_window_sec
        self.max_event_sec = max_event_sec
        self.min_contact_samples = min_contact_samples

        self.state = "IDLE"

        self.last_far_time = None

        self.contact_start = None
        self.contact_samples = 0

        self.min_distance = None
        self.min_distance_time = None

    def reset(self):
        self.state = "IDLE"

        self.contact_start = None
        self.contact_samples = 0

        self.min_distance = None
        self.min_distance_time = None

    def update(
        self,
        timestamp,
        distance_norm
    ):
        """
        Returns HANDOVER_EVENT when pattern:

        far
        -> close/contact
        -> far again

        is completed.
        """

        # ----------------------------------
        # IDLE
        # ----------------------------------

        if self.state == "IDLE":

            # Руки достаточно далеко
            if (
                distance_norm
                >= self.approach_threshold
            ):
                self.last_far_time = timestamp

                return None

            # Начинается контакт
            if (
                distance_norm
                <= self.contact_threshold
            ):

                # Перед этим руки должны были
                # быть далеко друг от друга
                if self.last_far_time is None:
                    return None

                time_since_far = (
                    timestamp
                    - self.last_far_time
                )

                if (
                    time_since_far
                    > self.approach_window_sec
                ):
                    return None

                self.state = "CONTACT"

                self.contact_start = timestamp
                self.contact_samples = 1

                self.min_distance = (
                    distance_norm
                )

                self.min_distance_time = (
                    timestamp
                )

                return None

        # ----------------------------------
        # CONTACT
        # ----------------------------------

        if self.state == "CONTACT":

            duration = (
                timestamp
                - self.contact_start
            )

            # Обновляем минимальную дистанцию
            if (
                self.min_distance is None
                or
                distance_norm
                < self.min_distance
            ):
                self.min_distance = (
                    distance_norm
                )

                self.min_distance_time = (
                    timestamp
                )

            # Всё ещё близкий контакт
            if (
                distance_norm
                <= self.contact_threshold
            ):
                self.contact_samples += 1

            # Контакт слишком долгий —
            # скорее всего это не handover
            if (
                duration
                > self.max_event_sec
            ):
                self.reset()

                if (
                    distance_norm
                    >= self.approach_threshold
                ):
                    self.last_far_time = (
                        timestamp
                    )

                return None

            # ----------------------------------
            # SEPARATION
            # ----------------------------------

            if (
                distance_norm
                >= self.release_threshold
            ):

                enough_contact = (
                    self.contact_samples
                    >= self.min_contact_samples
                )

                if enough_contact:

                    event = {
                        "event":
                            "HANDOVER_EVENT",

                        "timestamp_sec":
                            round(
                                self.contact_start,
                                2
                            ),

                        "peak_timestamp_sec":
                            round(
                                self.min_distance_time,
                                2
                            ),

                        "end_timestamp_sec":
                            round(
                                timestamp,
                                2
                            ),

                        "min_distance_norm":
                            round(
                                self.min_distance,
                                4
                            ),

                        "contact_samples":
                            self.contact_samples
                    }

                    self.reset()

                    self.last_far_time = (
                        timestamp
                    )

                    return event

                self.reset()

                self.last_far_time = (
                    timestamp
                )

        return None