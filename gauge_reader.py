#!/usr/bin/env python3
"""Analog pressure-gauge needle reader (prototype v1).

Pipeline:
  1. Detect the dial face (Hough circle) or use a stored calibration.
  2. Find the needle as the darkest radial line fanning out from the center.
  3. Convert needle angle to an engineering value using per-equipment
     calibration (two reference angles -> min/max values).

Calibration model per equipment ID (calibration.json):
  {"equipment_id": {
      "cx": <dial center x fraction>, "cy": ..., "radius": <fraction of min dim>,
      "angle_min": <deg>, "angle_max": <deg>,
      "value_min": <float>, "value_max": <float>, "unit": "psi" }}

Angles are in dial degrees: 0 = pointing straight up, increasing clockwise
(gauge convention).

Usage:
  python3 gauge_reader.py read <image> <equipment_id> [--calib calibration.json]
  python3 gauge_reader.py calibrate ...   (see README)
"""
import json
import math
import sys

import cv2
import numpy as np

CALIB_FILE = "calibration.json"


def load_calib(path=CALIB_FILE):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def detect_dial(gray):
    """Find the dial face circle; return (cx, cy, radius) or None."""
    blur = cv2.medianBlur(gray, 5)
    h, w = blur.shape
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(h, w) / 2,
        param1=100, param2=60,
        minRadius=int(min(h, w) * 0.30),
        maxRadius=int(min(h, w) * 0.52),
    )
    if circles is None:
        # Fallback: assume the dial fills the frame, centered.
        return w / 2.0, h / 2.0, min(h, w) * 0.45
    x, y, r = circles[0][0]
    return float(x), float(y), float(r)


def _radial_profile(gray, cx, cy, r_inner, r_outer, angles):
    """For each candidate angle, mean darkness along the radial segment."""
    h, w = gray.shape
    scores = np.empty(len(angles), dtype=np.float64)
    n_samples = 60
    for i, a in enumerate(angles):
        rad = math.radians(a)
        dx, dy = math.sin(rad), -math.cos(rad)  # 0 deg = up, clockwise
        ts = np.linspace(r_inner, r_outer, n_samples)
        xs = np.clip((cx + ts * dx).astype(int), 0, w - 1)
        ys = np.clip((cy + ts * dy).astype(int), 0, h - 1)
        vals = gray[ys, xs].astype(np.float64)
        # Darkness score: low mean + low variance = solid dark line (needle),
        # not a shadow or tick mark.
        scores[i] = (255.0 - vals.mean()) + 0.5 * (255.0 - vals.std())
    return scores


def needle_angle(gray, cx, cy, radius, coarse_step=2.0, refine_step=0.25):
    """Return the needle angle in dial degrees (0 = up, clockwise)."""
    # Restrict to the dial face; ignore bezel and glass reflections at edge.
    mask = np.zeros_like(gray)
    cv2.circle(mask, (int(cx), int(cy)), int(radius * 0.98), 255, -1)
    masked = gray.copy()
    masked[mask == 0] = 200  # don't let dark corners bias the search

    r_inner = radius * 0.12
    r_outer = radius * 0.86

    angles = np.arange(0, 360, coarse_step)
    scores = _radial_profile(masked, cx, cy, r_inner, r_outer, angles)
    best = float(angles[int(np.argmax(scores))])

    # Sub-degree refinement around the peak.
    angles2 = np.arange(best - coarse_step * 1.5, best + coarse_step * 1.5 + 1e-9,
                       refine_step)
    angles2 %= 360.0
    scores2 = _radial_profile(masked, cx, cy, r_inner, r_outer, angles2)
    best2 = float(angles2[int(np.argmax(scores2))])
    return best2


def angle_to_value(angle, cal):
    """Linear interpolation between calibrated min/max dial angles."""
    a_min, a_max = cal["angle_min"], cal["angle_max"]
    v_min, v_max = cal["value_min"], cal["value_max"]
    span = (a_max - a_min) % 360.0
    rel = (angle - a_min) % 360.0
    frac = max(0.0, min(1.0, rel / span))
    return v_min + frac * (v_max - v_min)


def read_image(path, equipment_id=None, calib=None):
    """Read a gauge photo. Returns dict with angle and (optionally) value."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cal = (calib or {}).get(equipment_id or "", {})
    if {"cx", "cy", "radius"} <= set(cal):
        h, w = gray.shape
        cx, cy = cal["cx"] * w, cal["cy"] * h
        radius = cal["radius"] * min(h, w)
    else:
        cx, cy, radius = detect_dial(gray)

    angle = needle_angle(gray, cx, cy, radius)
    out = {"equipment_id": equipment_id, "needle_angle_deg": round(angle, 2)}
    if cal and "angle_min" in cal and "angle_max" in cal:
        out["value"] = round(angle_to_value(angle, cal), 2)
        out["unit"] = cal.get("unit", "")
    return out


def main(argv):
    if len(argv) < 4 or argv[1] != "read":
        print(__doc__)
        return 2
    path, equipment_id = argv[2], argv[3]
    calib_path = CALIB_FILE
    if "--calib" in argv:
        calib_path = argv[argv.index("--calib") + 1]
    calib = load_calib(calib_path)
    result = read_image(path, equipment_id, calib)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
