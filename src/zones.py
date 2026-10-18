import json

import cv2
import numpy as np


class ZoneManager:
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.zones = json.load(f)

    def get_polygon(
        self,
        zone_name,
        width,
        height
    ):
        normalized = self.zones[zone_name]

        points = []

        for x, y in normalized:
            points.append([
                int(x * width),
                int(y * height)
            ])

        return np.array(
            points,
            dtype=np.int32
        )

    def point_inside(
        self,
        zone_name,
        point,
        width,
        height
    ):
        polygon = self.get_polygon(
            zone_name,
            width,
            height
        )

        result = cv2.pointPolygonTest(
            polygon,
            point,
            False
        )

        return result >= 0

    def zones_for_point(
        self,
        point,
        width,
        height
    ):
        result = []

        for zone_name in self.zones:
            if self.point_inside(
                zone_name,
                point,
                width,
                height
            ):
                result.append(
                    zone_name
                )

        return result