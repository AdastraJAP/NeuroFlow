"""
Camera diagnostic. Run this if a script says the frames are black.

It tries every camera index and backend, reports which combination actually
returns a picture, and saves a test photo from the best one.

Run:  python diagnose_camera.py
"""

import time

import cv2

BACKENDS = [
    ("default", 0),
    ("MSMF", getattr(cv2, "CAP_MSMF", 1400)),
    ("DSHOW", getattr(cv2, "CAP_DSHOW", 700)),
]


def probe(index, name, api):
    cap = cv2.VideoCapture(index) if api == 0 else cv2.VideoCapture(index, api)
    if not cap.isOpened():
        cap.release()
        return None, "not opened"

    best, frame = 0.0, None
    for _ in range(15):
        ok, f = cap.read()
        if ok and f is not None:
            m = float(f.mean())
            if m > best:
                best, frame = m, f
        time.sleep(0.03)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame is None:
        return None, "opened but no frames"
    if best <= 1.0:
        return None, f"black frames (mean {best:.2f})"
    return frame, f"OK  {w}x{h}  mean brightness {best:.1f}"


def main():
    print("Probing cameras. Do not have Zoom/Teams/Camera app open.\n")
    winners = []
    for index in (0, 1, 2):
        for name, api in BACKENDS:
            frame, note = probe(index, name, api)
            print(f"  index {index}  {name:<8} {note}")
            if frame is not None:
                winners.append((index, name, frame))
        print()

    if not winners:
        print("No working camera found.")
        print("  1. Close every other app that might use the webcam.")
        print("  2. Settings > Privacy & security > Camera:")
        print("     turn ON 'Camera access' and 'Let desktop apps access your camera'.")
        print("  3. If your laptop has a physical camera shutter or an Fn key")
        print("     that disables the webcam, check it.")
        print("  4. Reboot and run this again.")
        return

    index, name, frame = winners[0]
    cv2.imwrite("camera_test.jpg", frame)
    print(f"Working: index {index} via {name}")
    print("Saved camera_test.jpg - open it and check you can see yourself.")
    if index != 0:
        print(f"\nNOTE: your camera is index {index}, not 0.")
        print("Tell Claude this so the scripts can be updated.")


if __name__ == "__main__":
    main()
