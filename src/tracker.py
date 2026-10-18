import math


class SimpleTracker:
    def __init__(
        self,
        max_distance=120,
        max_missing=10
    ):
        self.next_id = 1

        self.tracks = {}

        self.max_distance = max_distance
        self.max_missing = max_missing


    def _center(self, box):
        x1, y1, x2, y2 = box

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        return cx, cy


    def update(self, detections):
        """
        detections:

        [
            {
                "box": [x1, y1, x2, y2],
                "score": 0.95
            }
        ]

        Возвращает:

        [
            {
                "id": 1,
                "box": [...],
                "score": ...
            }
        ]
        """

        detection_centers = [
            self._center(
                detection["box"]
            )
            for detection in detections
        ]


        # Все существующие tracks
        # сначала считаем пропущенными.

        for track_id in self.tracks:
            self.tracks[
                track_id
            ]["missing"] += 1


        assigned_tracks = set()
        assigned_detections = set()


        # -----------------------------
        # MATCH EXISTING TRACKS
        # -----------------------------

        candidates = []

        for track_id, track in self.tracks.items():

            tx, ty = track["center"]

            for detection_index, center in enumerate(
                detection_centers
            ):

                dx, dy = center

                distance = math.sqrt(
                    (tx - dx) ** 2 +
                    (ty - dy) ** 2
                )

                candidates.append(
                    (
                        distance,
                        track_id,
                        detection_index
                    )
                )


        candidates.sort(
            key=lambda x: x[0]
        )


        for (
            distance,
            track_id,
            detection_index
        ) in candidates:

            if distance > self.max_distance:
                continue

            if track_id in assigned_tracks:
                continue

            if detection_index in assigned_detections:
                continue


            detection = detections[
                detection_index
            ]

            center = detection_centers[
                detection_index
            ]


            self.tracks[
                track_id
            ] = {
                "center": center,
                "box": detection["box"],
                "score": detection["score"],
                "missing": 0
            }


            assigned_tracks.add(
                track_id
            )

            assigned_detections.add(
                detection_index
            )


        # -----------------------------
        # NEW PEOPLE
        # -----------------------------

        for detection_index, detection in enumerate(
            detections
        ):

            if detection_index in assigned_detections:
                continue


            track_id = self.next_id

            self.next_id += 1


            self.tracks[
                track_id
            ] = {
                "center": detection_centers[
                    detection_index
                ],
                "box": detection["box"],
                "score": detection["score"],
                "missing": 0
            }


            assigned_tracks.add(
                track_id
            )


        # -----------------------------
        # DELETE LOST TRACKS
        # -----------------------------

        to_delete = []

        for track_id, track in self.tracks.items():

            if (
                track["missing"] >
                self.max_missing
            ):
                to_delete.append(
                    track_id
                )


        for track_id in to_delete:

            del self.tracks[
                track_id
            ]


        # -----------------------------
        # OUTPUT ACTIVE TRACKS
        # -----------------------------

        result = []

        for track_id, track in self.tracks.items():

            if track["missing"] != 0:
                continue


            result.append({
                "id": track_id,
                "box": track["box"],
                "score": track["score"],
                "center": track["center"]
            })


        return result