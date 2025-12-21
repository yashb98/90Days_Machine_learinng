import math
from dataclasses import dataclass


@dataclass
class GeoPose:
    lat: float
    lng: float
    heading: float = 0.0


class LocationService:
    """
    Handles geospatial calculations to find relevant memories near the user.
    """

    def calculate_distance(self, pose1: GeoPose, pose2: GeoPose) -> float:
        """
        Haversine formula to calculate distance in meters between two points.
        """
        R = 6371000  # Radius of Earth in meters
        phi1 = math.radians(pose1.lat)
        phi2 = math.radians(pose2.lat)
        delta_phi = math.radians(pose2.lat - pose1.lat)
        delta_lambda = math.radians(pose2.lng - pose1.lng)

        a = (math.sin(delta_phi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) *
             math.sin(delta_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def is_within_radius(self, current: GeoPose, target: GeoPose, radius_meters: float) -> bool:
        return self.calculate_distance(current, target) <= radius_meters
