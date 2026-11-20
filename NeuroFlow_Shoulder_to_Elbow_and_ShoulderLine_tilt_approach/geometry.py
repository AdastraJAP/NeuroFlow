"""Pure geometry. Coordinates must be PIXELS (or other isotropic units)."""
import math


def arm_elevation(shoulder, elbow):
    """2D angle (0..180 deg) of shoulder->elbow relative to image DOWN.

    0 = upper arm points down, 90 = horizontal, 180 = up.
    This is a camera-plane estimate, not a clinical shoulder ROM measurement.
    """
    dx = elbow[0] - shoulder[0]
    dy = elbow[1] - shoulder[1]
    if not all(math.isfinite(v) for v in (*shoulder, *elbow)):
        return None
    if math.hypot(dx, dy) < 2.0:
        return None
    return math.degrees(math.atan2(abs(dx), dy))


def shoulder_line_tilt(right_shoulder, left_shoulder):
    """Undirected shoulder line tilt vs image horizontal, in [-90,90) deg."""
    dx = left_shoulder[0] - right_shoulder[0]
    dy = left_shoulder[1] - right_shoulder[1]
    if not all(math.isfinite(v) for v in (*right_shoulder, *left_shoulder)):
        return None
    if math.hypot(dx, dy) < 2.0:
        return None
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 90.0) % 180.0 - 90.0


def tilt_change(current, baseline):
    """Smallest undirected line-angle change in degrees, [-90,90)."""
    if current is None or baseline is None:
        return None
    return (current - baseline + 90.0) % 180.0 - 90.0
