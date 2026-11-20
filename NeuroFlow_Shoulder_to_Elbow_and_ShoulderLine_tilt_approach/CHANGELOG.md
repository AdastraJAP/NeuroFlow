# Changes in this solo seated/standing edition

Based on the earlier professor-update package (hip-free angle, shoulder tilt, quality checks, timestamps, and DTW):

- **Hands-free recording**: press SPACE once, wait a configurable 5-second countdown, then automatically record for 10 seconds and stop. No helper needed.
- **Hands-free live demo**: choose C/I/U expected label, press SPACE once, wait for countdown, perform one repetition, and receive an automatically calculated result.
- **Two independent conditions**: seated and standing have separate reference videos, practice/test clips, DTW thresholds, angle traces, reviews, logs and calibration data.
- **Menu 14** switches conditions; **menu 15** changes countdown (3–20s) and recording duration (6–30s).
- The initial DTW threshold is **null for both positions**; no fabricated calibration value is included.
- Every clip metadata JSON identifies its condition, arm, timing, angle definition, and reference; scoring rejects incompatible clips and reference metadata.
- Preserves the original scripts and the pretrained MediaPipe model. Older hip-based results remain archived and are not used.

Do not merge this folder with the previous version: settings.json has a new schema.

## Menu polish

- Place the option-selection prompt on a separate line below **0 Exit**.
- Remove “SOLO” from menu, launcher, and webcam window titles. The timed recording functionality is unchanged.
