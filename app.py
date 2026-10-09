#!/usr/bin/env python3
"""Boiler gauge inspection server (v1).

- GET  /e/<equip_id>        : equipment inspection page (mobile)
- GET  /api/equipment/<id> : equipment info + gauge list JSON
- POST /api/read           : gauge photo upload → automatic needle reading {angle, value, unit}
- POST /api/log            : save measurement
- GET  /api/export         : download measurement log CSV
- GET  /health             : status check
"""
import csv
import io
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from gauge_reader import detect_dial, needle_angle, angle_to_value  # noqa: E402

import cv2
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB

EQUIPMENT = json.loads((BASE / "equipment.json").read_text())
CALIB = json.loads((BASE / "calibration.json").read_text())
LOG_CSV = Path(__file__).parent / "data" / "readings.csv"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
LOG_FIELDS = ["time", "system", "equip_id", "equip_name", "gauge_id",
              "gauge_name", "state", "value", "unit", "angle_deg",
              "manual", "worker", "photo"]


def find_equipment(equip_id):
    for sys in EQUIPMENT["systems"]:
        for grp in ("boilers", "pumps"):
            for dev in sys[grp]:
                if dev["id"] == equip_id:
                    return sys, dev
    return None, None


def read_gauge_bytes(data: bytes, gauge_id: str):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the photo")
    # 큰 사진은 축소 (처리 속도 향상 — 판독 정확도에 영향 없음)
    h0, w0 = img.shape[:2]
    if max(h0, w0) > 1280:
        s = 1280 / max(h0, w0)
        img = cv2.resize(img, (int(w0 * s), int(h0 * s)),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cal = CALIB.get(gauge_id, {})
    if {"cx", "cy", "radius"} <= set(cal):
        h, w = gray.shape
        cx, cy = cal["cx"] * w, cal["cy"] * h
        radius = cal["radius"] * min(h, w)
    else:
        cx, cy, radius = detect_dial(gray)
    angle = needle_angle(gray, cx, cy, radius)
    out = {"angle_deg": round(angle, 2)}
    if "angle_min" in cal and "angle_max" in cal:
        out["value"] = round(angle_to_value(angle, cal), 2)
        out["unit"] = cal.get("unit", "")
    else:
        out["value"] = None
        out["unit"] = ""
        out["needs_calibration"] = True
    return out


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    items = []
    for sys in EQUIPMENT["systems"]:
        for grp in ("boilers", "pumps"):
            for dev in sys[grp]:
                items.append({"system": sys["name"], "id": dev["id"],
                              "name": dev["name"],
                              "kind": "Boiler" if grp == "boilers" else "Pump"})
    return render_template("index.html", items=items)


@app.get("/e/<equip_id>")
def equipment_page(equip_id):
    sys, dev = find_equipment(equip_id)
    if dev is None:
        return f"Equipment {equip_id} not found", 404
    return render_template("equip.html", sys=sys, dev=dev, equip_id=equip_id)


@app.get("/api/equipment/<equip_id>")
def api_equipment(equip_id):
    sys, dev = find_equipment(equip_id)
    if dev is None:
        return jsonify({"error": "not found"}), 404
    gauges = []
    for g in dev["gauges"]:
        cal = CALIB.get(g["id"], {})
        gauges.append({
            "id": g["id"], "name": g["name"],
            "item": g.get("item", ""), "unit": g.get("unit", "") or cal.get("unit", ""),
            "check_running": g.get("check_running", ""),
            "check_stopped": g.get("check_stopped", ""),
            "calibrated": "angle_min" in cal,
        })
    return jsonify({"system": sys["name"], "id": dev["id"], "name": dev["name"],
                    "gauges": gauges})


@app.post("/api/read")
def api_read():
    gauge_id = request.form.get("gauge_id", "")
    photo = request.files.get("photo")
    if not photo:
        return jsonify({"error": "No photo uploaded"}), 400
    data = photo.read()
    # 원본 저장 (이력용)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{gauge_id}.jpg"
    (UPLOAD_DIR / fname).write_bytes(data)
    try:
        result = read_gauge_bytes(data, gauge_id)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500
    result["photo"] = fname
    return jsonify(result)


@app.post("/api/log")
def api_log():
    body = request.get_json(force=True)
    sys, dev = find_equipment(body.get("equip_id", ""))
    gauge = next((g for g in (dev["gauges"] if dev else [])
                  if g["id"] == body.get("gauge_id")), None)
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system": sys["name"] if sys else "",
        "equip_id": body.get("equip_id", ""),
        "equip_name": dev["name"] if dev else "",
        "gauge_id": body.get("gauge_id", ""),
        "gauge_name": gauge["name"] if gauge else "",
        "state": body.get("state", ""),
        "value": body.get("value", ""),
        "unit": body.get("unit", ""),
        "angle_deg": body.get("angle_deg", ""),
        "manual": "Y" if body.get("manual") else "",
        "worker": body.get("worker", ""),
        "photo": body.get("photo", ""),
    }
    new_file = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)
    return jsonify({"ok": True})


@app.get("/api/export")
def api_export():
    if not LOG_CSV.exists():
        return "No records yet", 404
    return send_file(LOG_CSV, as_attachment=True,
                     download_name="gauge_readings.csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
