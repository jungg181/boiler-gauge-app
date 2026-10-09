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
import threading
import urllib.request
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, session

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from gauge_reader import detect_dial, needle_angle, angle_to_value  # noqa: E402

import cv2
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "ChangeMe123")
SHEETS_URL = os.environ.get("SHEETS_URL", "")  # Google Apps Script web app URL


def sheets_post(payload, timeout=12):
    """Best-effort POST to the Google Sheets webhook. Never raises."""
    if not SHEETS_URL:
        return None
    try:
        req = urllib.request.Request(
            SHEETS_URL, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"sheets_post failed: {e}", flush=True)
        return None


def sheets_get_config(timeout=15):
    """Fetch the persisted settings from Google Sheets. Returns dict or None."""
    if not SHEETS_URL:
        return None
    try:
        with urllib.request.urlopen(SHEETS_URL + "?action=getConfig",
                                    timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("config")
    except Exception as e:  # noqa: BLE001
        print(f"sheets_get_config failed: {e}", flush=True)
        return None


def sheets_post_async(payload):
    threading.Thread(target=sheets_post, args=(payload,), daemon=True).start()

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


@app.get("/api/qr")
def api_qr():
    """Return a QR code PNG for the given text (used for gauge labels)."""
    import qrcode
    from qrcode.constants import ERROR_CORRECT_H
    text = request.args.get("text", "")
    if not text or len(text) > 500:
        return "bad request", 400
    qr = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_H,
                       box_size=12, border=2)
    qr.add_data(text)
    qr.make(fit=True)
    buf = io.BytesIO()
    qr.make_image(fill_color="black", back_color="white").save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    items = []
    boilers = []
    pumps = []
    for sys in EQUIPMENT["systems"]:
        for grp in ("boilers", "pumps"):
            for dev in sys[grp]:
                items.append({"system": sys["name"], "id": dev["id"],
                              "name": dev["name"],
                              "kind": "Boiler" if grp == "boilers" else "Pump"})
                if grp == "boilers":
                    boilers.append({"system": sys["name"], "id": dev["id"],
                                    "name": dev["name"]})
                else:
                    pumps.append({"system": sys["name"], "id": dev["id"],
                                  "name": dev["name"],
                                  "boilers": dev.get("boilers", [])})
    return render_template("index.html", items=items, boilers=boilers,
                           pumps=pumps)


def _equip_payload(sys, dev, kind):
    gauges = []
    for g in dev.get("gauges", []):
        cal = CALIB.get(g["id"], {})
        gauges.append({
            "id": g["id"], "name": g["name"],
            "item": g.get("item", ""),
            "unit": g.get("unit", "") or cal.get("unit", ""),
            "calibrated": "angle_min" in cal,
        })
    return {"system": sys["name"], "id": dev["id"], "name": dev["name"],
            "kind": kind, "gauges": gauges}


@app.get("/inspect")
def inspect_page():
    return render_template("inspect.html")


@app.get("/api/inspect")
def api_inspect():
    ids = [b.strip() for b in request.args.get("boilers", "").split(",")
           if b.strip()]
    pump_id_set = {p["id"] for s in EQUIPMENT["systems"] for p in s["pumps"]}
    items, seen = [], set()
    for bid in ids:  # selected boilers first
        sys, dev = find_equipment(bid)
        if dev and bid not in seen and bid not in pump_id_set:
            items.append(_equip_payload(sys, dev, "Boiler"))
            seen.add(bid)
    if "pumps" in request.args:
        # explicit pump selection from the home screen
        pump_ids = [p.strip() for p in request.args.get("pumps", "").split(",")
                    if p.strip()]
        for pid in pump_ids:
            if pid in pump_id_set and pid not in seen:
                sys, dev = find_equipment(pid)
                items.append(_equip_payload(sys, dev, "Pump"))
                seen.add(pid)
    else:
        # auto: pumps connected to the selected boilers (deduplicated)
        for bid in ids:
            for sys in EQUIPMENT["systems"]:
                for p in sys["pumps"]:
                    if bid in p.get("boilers", []) and p["id"] not in seen:
                        items.append(_equip_payload(sys, p, "Pump"))
                        seen.add(p["id"])
    return jsonify({"boilers": ids, "items": items})


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
    # persist to Google Sheets (best-effort, background)
    sheets_post_async({"action": "reading", "reading": row})
    return jsonify({"ok": True})


@app.get("/api/export")
def api_export():
    if not LOG_CSV.exists():
        return "No records yet", 404
    return send_file(LOG_CSV, as_attachment=True,
                     download_name="gauge_readings.csv")


# ---------------- Admin (settings) ----------------

def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("admin"):
            return jsonify({"error": "login required"}), 401
        return fn(*args, **kwargs)
    return wrapper


@app.get("/admin")
def admin_page():
    return render_template("admin.html")


@app.post("/api/admin/login")
def admin_login():
    body = request.get_json(force=True, silent=True) or {}
    if body.get("password") == ADMIN_PASSWORD:
        session["admin"] = True
        return jsonify({"ok": True})
    return jsonify({"error": "Wrong password"}), 403


@app.post("/api/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return jsonify({"ok": True})


@app.get("/api/admin/config")
@admin_required
def admin_config():
    return jsonify({"equipment": EQUIPMENT, "calibration": CALIB})


def _validate_config(eq, cal):
    if not isinstance(eq, dict) or not isinstance(eq.get("systems"), list):
        return "bad equipment structure"
    if not isinstance(cal, dict):
        return "bad calibration structure"
    seen = set()
    for sys in eq["systems"]:
        for grp in ("boilers", "pumps"):
            for dev in sys.get(grp, []):
                if not dev.get("id") or not dev.get("name"):
                    return "equipment needs id and name"
                for g in dev.get("gauges", []):
                    if not g.get("id"):
                        return "gauge needs id"
                    if g["id"] in seen:
                        return f'duplicate gauge id: {g["id"]}'
                    seen.add(g["id"])
    return None


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@app.post("/api/admin/save")
@admin_required
def admin_save():
    global EQUIPMENT, CALIB
    body = request.get_json(force=True, silent=True) or {}
    eq, cal = body.get("equipment"), body.get("calibration")
    err = _validate_config(eq, cal)
    if err:
        return jsonify({"error": err}), 400
    eq_path = BASE / "equipment.json"
    cal_path = BASE / "calibration.json"
    if eq_path.exists():
        _atomic_write(eq_path.with_suffix(".json.bak"),
                      eq_path.read_text(encoding="utf-8"))
    if cal_path.exists():
        _atomic_write(cal_path.with_suffix(".json.bak"),
                      cal_path.read_text(encoding="utf-8"))
    _atomic_write(eq_path, json.dumps(eq, indent=2, ensure_ascii=False))
    _atomic_write(cal_path, json.dumps(cal, indent=2, ensure_ascii=False))
    EQUIPMENT, CALIB = eq, cal
    # persist settings to Google Sheets (best-effort, background)
    sheets_post_async({"action": "saveConfig",
                       "config": {"equipment": eq, "calibration": cal}})
    return jsonify({"ok": True})


@app.get("/api/admin/backup")
@admin_required
def admin_backup():
    blob = json.dumps({"equipment": EQUIPMENT, "calibration": CALIB},
                      indent=2, ensure_ascii=False).encode("utf-8")
    return send_file(io.BytesIO(blob), as_attachment=True,
                     download_name="gauge_config_backup.json",
                     mimetype="application/json")


# Restore settings from Google Sheets on startup (survives server wipes).
# Local disk files remain the fallback when Sheets is unreachable/empty.
if SHEETS_URL:
    _remote = sheets_get_config()
    if isinstance(_remote, dict) and isinstance(_remote.get("equipment"), dict):
        EQUIPMENT = _remote["equipment"]
        CALIB = _remote.get("calibration", {}) or {}
        try:
            _atomic_write(BASE / "equipment.json",
                          json.dumps(EQUIPMENT, indent=2, ensure_ascii=False))
            _atomic_write(BASE / "calibration.json",
                          json.dumps(CALIB, indent=2, ensure_ascii=False))
            print("config restored from Google Sheets", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"config cache write failed: {e}", flush=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
