"""Synchronized, independent arm cycles for seated bilateral capture.

This is a capture detector, not a bilateral clinical score or DTW template.
The first confirmed arm onset starts the shared 25-second movement clock.
"""
from dataclasses import dataclass
from math import isfinite

from repetition import MAX_FRAME_GAP_MS, RepetitionDetector, START_TIMEOUT_MS

ARMS = ('left', 'right')


@dataclass(frozen=True)
class BilateralRepetition:
    outcome: str
    termination_reason: str
    start_ms: int | None
    end_ms: int | None
    arm_results: dict
    peaks_deg: dict
    peak_times_ms: dict

    def metadata(self):
        starts = {arm: result.start_ms if result else None
                  for arm, result in self.arm_results.items()}
        ends = {arm: result.end_ms if result else None
                for arm, result in self.arm_results.items()}
        return {
            'detector_status': self.outcome,
            'termination_reason': self.termination_reason,
            'repetition_start_ms': self.start_ms,
            'repetition_end_ms': self.end_ms,
            'arm_start_ms': starts,
            'arm_end_ms': ends,
            'arm_outcomes': {arm: result.outcome if result else None
                             for arm, result in self.arm_results.items()},
            'arm_rest_baseline_deg': {arm: detector.rest_baseline_deg if detector else None
                                      for arm, detector in self.arm_results.items()},
            'arm_peak_deg': self.peaks_deg,
            'arm_peak_ms': self.peak_times_ms,
            'onset_difference_left_minus_right_ms': (
                starts['left'] - starts['right']
                if starts['left'] is not None and starts['right'] is not None else None),
            'peak_time_difference_left_minus_right_ms': (
                self.peak_times_ms['left'] - self.peak_times_ms['right']
                if all(self.peak_times_ms[arm] is not None for arm in ARMS) else None),
        }


class BilateralRepetitionDetector:
    """Two unilateral cycles with one capture clock and arm-specific loss handling.

    A missing arm sample is retained in the saved frame. A gap up to 250 ms
    can recover; a longer gap after rest calibration is unable to judge.
    Neither arm's completion substitutes for the other's confirmed return.
    """
    def __init__(self, timeout_s, restart_startup_gap=True, reliable_baseline=True):
        self.timeout_ms = timeout_s * 1000
        self.restart_startup_gap = restart_startup_gap
        self.reliable_baseline = reliable_baseline
        self.startup_gap_restarts = 0
        self.startup_gap_reset = False
        self.arms = {arm: RepetitionDetector(timeout_s, rolling_rest_window=reliable_baseline)
                     for arm in ARMS}
        self.last_valid_ms = {arm: None for arm in ARMS}
        self.peaks_deg = {arm: None for arm in ARMS}
        self.peak_times_ms = {arm: None for arm in ARMS}
        self.first_ms = self.last_ms = None
        self.result = None

    @property
    def state(self):
        if self.result:
            return self.result.outcome
        return ' | '.join(f'{arm}: {self.arms[arm].state}' for arm in ARMS)

    @property
    def instruction(self):
        if self.result:
            return f'{self.result.outcome}: {self.result.termination_reason}'
        if self.reliable_baseline and any(d.baseline is None for d in self.arms.values()):
            return 'WAIT - hold BOTH arms down until BOTH READY'
        if any(detector.active for detector in self.arms.values()):
            return 'Lower both arms to rest and hold still'
        if any(detector.baseline is None for detector in self.arms.values()):
            return 'Hold both arms down and still'
        return 'BOTH READY - raise both arms once'

    def _finish(self, outcome, reason, end_ms=None):
        results = {arm: detector.result or detector.finish()
                   for arm, detector in self.arms.items()}
        if self.reliable_baseline:
            # Use the detected movement interval, including the sustained
            # onset candidate; exclude all pre-onset and post-return frames.
            for arm, result in results.items():
                samples = (result.segment if result.outcome == 'complete' else
                           tuple(self.arms[arm].active))
                if samples:
                    peak_ms, peak_angle = max(samples, key=lambda sample: sample[1])
                    self.peaks_deg[arm] = peak_angle
                    self.peak_times_ms[arm] = peak_ms
                else:
                    self.peaks_deg[arm] = self.peak_times_ms[arm] = None
        starts = [result.start_ms for result in results.values() if result.start_ms is not None]
        self.result = BilateralRepetition(outcome, reason, min(starts) if starts else None,
                                          end_ms, results, self.peaks_deg.copy(),
                                          self.peak_times_ms.copy())
        return self.result

    def finish(self):
        if self.result:
            return self.result
        if any(d.baseline is None for d in self.arms.values()):
            return self._finish('unable_to_judge', 'capture_ended')
        if any(d.active or (d.result and d.result.outcome == 'complete') for d in self.arms.values()):
            return self._finish('incomplete_return', 'capture_ended')
        return self._finish('no_start', 'capture_ended')

    def push(self, timestamp_ms, left_angle, right_angle):
        if self.result:
            return self.result
        self.startup_gap_reset = False
        if (not isinstance(timestamp_ms, (int, float)) or not isfinite(timestamp_ms)
                or (self.last_ms is not None and timestamp_ms <= self.last_ms)):
            return self._finish('unable_to_judge', 'invalid_timestamp')
        if self.first_ms is None:
            self.first_ms = timestamp_ms
        if self.last_ms is not None and timestamp_ms - self.last_ms > MAX_FRAME_GAP_MS:
            movement_started = any(d.active or d.result for d in self.arms.values())
            if not self.restart_startup_gap or movement_started:
                return self._finish('unable_to_judge', 'frame_gap')
            if self.startup_gap_restarts:
                return self._finish('unable_to_judge', 'startup_frame_gap')
            # No movement was established. Invalidate both incomplete rest
            # baselines and start fresh from this actual captured frame/time.
            self.arms = {arm: RepetitionDetector(self.timeout_ms / 1000,
                                                  rolling_rest_window=self.reliable_baseline)
                         for arm in ARMS}
            self.last_valid_ms = {arm: None for arm in ARMS}
            self.peaks_deg = {arm: None for arm in ARMS}
            self.peak_times_ms = {arm: None for arm in ARMS}
            self.first_ms = timestamp_ms
            self.last_ms = None
            self.startup_gap_restarts += 1
            self.startup_gap_reset = True
        self.last_ms = timestamp_ms
        starts = [d.active[0][0] for d in self.arms.values() if d.active]
        if starts and timestamp_ms - min(starts) >= self.timeout_ms:
            return self._finish('incomplete_return', 'movement_timeout')
        if timestamp_ms - self.first_ms >= START_TIMEOUT_MS:
            if any(d.baseline is None for d in self.arms.values()):
                return self._finish('unable_to_judge', 'rest_baseline_timeout')
            if not starts:
                return self._finish('no_start', 'start_timeout')

        for arm, angle in (('left', left_angle), ('right', right_angle)):
            detector = self.arms[arm]
            if detector.result and detector.result.outcome == 'complete':
                continue
            valid = angle is not None and isinstance(angle, (int, float)) and isfinite(angle)
            if not valid and detector.baseline is not None:
                last = self.last_valid_ms[arm]
                if last is not None and timestamp_ms - last > MAX_FRAME_GAP_MS:
                    return self._finish('unable_to_judge', f'{arm}_tracking_lost')
                continue
            if valid:
                self.last_valid_ms[arm] = timestamp_ms
                if not self.reliable_baseline and (self.peaks_deg[arm] is None or angle > self.peaks_deg[arm]):
                    self.peaks_deg[arm] = angle
                    self.peak_times_ms[arm] = timestamp_ms
            result = detector.push(timestamp_ms, angle)
            if result and result.outcome == 'unable_to_judge':
                return self._finish('unable_to_judge', f'{arm}_{result.termination_reason}')
            if result and result.outcome == 'no_start':
                return self._finish('incomplete_return' if any(d.active for d in self.arms.values())
                                    else 'no_start', f'{arm}_no_start')
            if result and result.outcome == 'incomplete_return':
                return self._finish('incomplete_return', f'{arm}_{result.termination_reason}')
        if (self.reliable_baseline and any(d.active for d in self.arms.values())
                and any(d.baseline is None for d in self.arms.values())):
            return self._finish('unable_to_judge', 'movement_before_both_ready')
        if all(d.result and d.result.outcome == 'complete' for d in self.arms.values()):
            return self._finish('complete', 'both_confirmed_return',
                                max(d.result.end_ms for d in self.arms.values()))
        return None
