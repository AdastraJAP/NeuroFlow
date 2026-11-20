"""Clock-driven, hands-free countdown + recording timer (no camera imports)."""
import math


class SoloTimer:
    def __init__(self, countdown_s=5, duration_s=10):
        if countdown_s < 0 or (duration_s is not None and duration_s <= 0):
            raise ValueError('Invalid countdown or duration')
        self.countdown_s = float(countdown_s)
        self.duration_s = float(duration_s) if duration_s is not None else None
        self.state = 'ready'
        self.countdown_started_at = None
        self.record_started_at = None

    def arm(self, now):
        if self.state not in ('ready', 'finished'):
            raise ValueError('Already counting down or recording')
        self.countdown_started_at = float(now)
        self.record_started_at = None
        self.state = 'countdown'

    def tick(self, now):
        """Returns 'started', 'finished', or None on each frame."""
        now = float(now)
        if self.state == 'countdown' and now - self.countdown_started_at >= self.countdown_s:
            self.state = 'recording'
            self.record_started_at = now
            return 'started'
        if (self.state == 'recording' and self.duration_s is not None
                and now - self.record_started_at >= self.duration_s):
            self.state = 'finished'
            return 'finished'
        return None

    def countdown_left(self, now):
        if self.state != 'countdown':
            return 0
        return max(0, math.ceil(self.countdown_s - (float(now) - self.countdown_started_at)))

    def elapsed(self, now):
        if self.record_started_at is None:
            return 0.0
        return max(0.0, float(now) - self.record_started_at)

    def cancel(self):
        self.state = 'cancelled'
