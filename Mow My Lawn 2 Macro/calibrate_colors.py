"""
calibrate_colors.py - Interactive HSV Color Picker & Screen Calibrator for MML2

Run this utility while Mow My Lawn 2 is open on your screen.
Click anywhere on the preview window (e.g., on a piece of Grill Explosion food,
a boss red telegraph ring, or a patch of grass) to print the exact HSV color range
you can paste into `config.json`.
"""

import cv2
import mss
import numpy as np


def main():
    print("=" * 65)
    print("  MML2 INTERACTIVE HSV COLOR CALIBRATOR")
    print("=" * 65)
    print("  - Left-Click anywhere in the preview window to sample a pixel's")
    print("    exact HSV values and get ready-to-paste bounds for config.json.")
    print("  - Press [SPACE] to freeze/unfreeze the current frame (great for")
    print("    capturing fast boss attacks or food explosions!).")
    print("  - Press [Q] or [ESC] to exit.")
    print("=" * 65)

    sct = mss.mss()
    monitor = sct.monitors[1]
    frozen = False
    current_bgr = None
    current_hsv = None
    scale = 0.55

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and current_hsv is not None:
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            orig_x = min(max(0, orig_x), current_hsv.shape[1] - 1)
            orig_y = min(max(0, orig_y), current_hsv.shape[0] - 1)

            h, s, v = [int(c) for c in current_hsv[orig_y, orig_x]]
            lower = [max(0, h - 10), max(30, s - 45), max(30, v - 45)]
            upper = [min(180, h + 10), 255, 255]
            print(f"\n[Sampled @ ({orig_x}, {orig_y})] HSV = [{h}, {s}, {v}]")
            print(f'  Suggested "hsv_lower": {lower},')
            print(f'  Suggested "hsv_upper": {upper}')

    win_name = "MML2 Color Calibrator (Click to sample HSV, SPACE to freeze, Q to quit)"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        if not frozen or current_bgr is None:
            raw = sct.grab(monitor)
            current_bgr = cv2.cvtColor(np.array(raw, dtype=np.uint8), cv2.COLOR_BGRA2BGR)
            current_hsv = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2HSV)

        preview = cv2.resize(
            current_bgr,
            (int(current_bgr.shape[1] * scale), int(current_bgr.shape[0] * scale)),
        )
        status = "FROZEN (Press SPACE to resume)" if frozen else "LIVE (Press SPACE to freeze frame)"
        cv2.putText(preview, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win_name, preview)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key == ord(" "):
            frozen = not frozen

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
