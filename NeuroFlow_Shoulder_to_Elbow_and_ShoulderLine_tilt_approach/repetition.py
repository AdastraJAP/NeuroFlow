"""Detect one shoulder-abduction repetition from measured frame times."""
from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite
from statistics import median

REST_MAX_DEG = 25
REST_SPREAD_DEG = 5
START_DELTA_DEG = 15
RETURN_DELTA_DEG = 7
REST_DWELL_MS = 500
START_DWELL_MS = 180
RETURN_DWELL_MS = 400
MAX_FRAME_GAP_MS = 250
START_TIMEOUT_MS = 8000


@dataclass(frozen=True)
class Repetition:
    outcome: str
    start_ms: int | None = None
    end_ms: int | None = None
    segment: tuple = ()
    termination_reason: str = 'capture_ended'
    rest_baseline_deg: float | None = None


class RepetitionDetector:
    def __init__(self, timeout_s):
        self.timeout_ms = timeout_s * 1000
        self.first_ms = self.last_ms = None
        self.rest = []
        self.baseline = None
        self.candidate = []
        self.active = []
        self.return_index = None
        self.result = None

    @property
    def instruction(self):
        if self.active:
            return 'Raise, then lower and hold still'
        if self.baseline is None:
            return 'Hold your arm down and still'
        return 'Ready - raise your arm once'

    def _finish(self, outcome, reason, end_ms=None, segment=()):
        start_ms = self.active[0][0] if self.active else None
        self.result = Repetition(outcome, start_ms, end_ms, segment, reason, self.baseline)
        return self.result

    def finish(self):
        if self.result is not None:
            return self.result
        if self.baseline is None:
            return self._finish('unable_to_judge', 'capture_ended')
        if self.active:
            return self._finish('incomplete_return', 'capture_ended')
        return self._finish('no_start', 'capture_ended')

    def push(self, timestamp_ms, angle):
        if self.result is not None:
            return self.result
        if not isfinite(timestamp_ms) or (self.last_ms is not None and timestamp_ms <= self.last_ms):
            return self._finish('unable_to_judge', 'invalid_timestamp')
        if self.first_ms is None:
            self.first_ms = timestamp_ms
        gap = self.last_ms is not None and timestamp_ms - self.last_ms > MAX_FRAME_GAP_MS
        self.last_ms = timestamp_ms
        if gap or angle is None or not isfinite(angle):
            if self.baseline is not None:
                return self._finish('unable_to_judge', 'frame_gap' if gap else 'tracking_lost')
            self.rest.clear()
        if self.active and timestamp_ms - self.active[0][0] >= self.timeout_ms:
            return self._finish('incomplete_return', 'movement_timeout')
        if not self.active and timestamp_ms - self.first_ms >= START_TIMEOUT_MS:
            if self.baseline is None:
                return self._finish('unable_to_judge', 'rest_baseline_timeout')
            return self._finish('no_start', 'start_timeout')
        if gap or angle is None or not isfinite(angle):
            return None

        sample = (timestamp_ms, angle)
        if self.baseline is None:
            if angle > REST_MAX_DEG:
                self.rest.clear()
                return None
            self.rest.append(sample)
            values = [value for _, value in self.rest]
            if max(values) - min(values) > REST_SPREAD_DEG:
                self.rest = [sample]
            elif timestamp_ms - self.rest[0][0] >= REST_DWELL_MS:
                self.baseline = median(values)
        elif not self.active:
            if angle - self.baseline >= START_DELTA_DEG:
                self.candidate.append(sample)
                if timestamp_ms - self.candidate[0][0] >= START_DWELL_MS:
                    self.active, self.candidate = self.candidate, []
            else:
                self.candidate.clear()
        else:
            self.active.append(sample)
            if abs(angle - self.baseline) > RETURN_DELTA_DEG:
                self.return_index = None
            else:
                if self.return_index is None:
                    self.return_index = len(self.active) - 1
                end_ms = self.active[self.return_index][0]
                if timestamp_ms - end_ms >= RETURN_DWELL_MS:
                    return self._finish(
                        'complete', 'confirmed_return', end_ms, tuple(self.active[:self.return_index + 1]))
        return None


def detect_repetition(samples, timeout_s):
    detector = RepetitionDetector(timeout_s)
    for timestamp_ms, angle in samples:
        result = detector.push(timestamp_ms, angle)
        if result is not None:
            return result
    return detector.finish()


def resample(segment, count=101):
    """Interpolate at equally spaced fractions of the measured movement time."""
    timestamps, angles = zip(*segment)
    first, last = timestamps[0], timestamps[-1]
    values = [angles[0]]
    for index in range(1, count - 1):
        timestamp = first + (last - first) * index / (count - 1)
        right = bisect_right(timestamps, timestamp)
        left = right - 1
        weight = (timestamp - timestamps[left]) / (timestamps[right] - timestamps[left])
        values.append(angles[left] + weight * (angles[right] - angles[left]))
    return [*values, angles[-1]]
