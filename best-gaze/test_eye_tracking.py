"""
Eye tracking test with calibration, target dots, and a reading test.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import pyautogui
import math
import time
import os
import json
import numpy as np

pyautogui.FAILSAFE = False

try:
    from Quartz import CGEventCreateMouseEvent, CGEventPost, kCGEventMouseMoved, kCGHIDEventTap

    def move_mouse(x, y):
        e = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), 0)
        CGEventPost(kCGHIDEventTap, e)
except ImportError:
    def move_mouse(x, y):
        pyautogui.moveTo(int(x), int(y))

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR, "face_landmarker.task")
CALIB_PATH = os.path.join(DIR, "calibration.json")

screen_w, screen_h = pyautogui.size()
EDGE = 8
HEAD_W_X = 0.3
HEAD_W_Y = 0.75

NOSE = 4
L_IRIS, R_IRIS = 468, 473
L_EYE_H, R_EYE_H = (33, 133), (362, 263)


class OneEuro:
    def __init__(self, fc_min=1.5, beta=0.01, d_cutoff=1.0):
        self.fc_min, self.beta, self.d_cutoff = fc_min, beta, d_cutoff
        self.x_prev = self.t_prev = None
        self.dx_prev = 0.0

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        te = t - self.t_prev
        if te <= 1e-6:
            return self.x_prev
        dx = (x - self.x_prev) / te
        a_d = self._a(te, self.d_cutoff)
        dx = a_d * dx + (1 - a_d) * self.dx_prev
        a = self._a(te, self.fc_min + self.beta * abs(dx))
        out = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = out, dx, t
        return out

    @staticmethod
    def _a(te, fc):
        tau = 1.0 / (2 * math.pi * fc)
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        self.x_prev = self.t_prev = None
        self.dx_prev = 0.0


filt_x = OneEuro(fc_min=3.0, beta=0.05)
filt_y = OneEuro(fc_min=3.0, beta=0.05)
DEAD_ZONE = 1.5
prev_sx, prev_sy = float(screen_w) / 2, float(screen_h) / 2


def get_nose_and_iris(lms):
    nx, ny = lms[NOSE].x, lms[NOSE].y
    rs = []
    for ir, (hl, hr) in [(L_IRIS, L_EYE_H), (R_IRIS, R_EYE_H)]:
        x0, x1 = min(lms[hl].x, lms[hr].x), max(lms[hl].x, lms[hr].x)
        eye_w = x1 - x0
        if eye_w < 1e-6:
            continue
        ix = (lms[ir].x - x0) / eye_w
        corner_mid_y = (lms[hl].y + lms[hr].y) / 2
        iy = (lms[ir].y - corner_mid_y) / eye_w
        rs.append((ix, iy))
    if not rs:
        return None
    return nx, ny, sum(r[0] for r in rs) / len(rs), sum(r[1] for r in rs) / len(rs)


def clamp(x, y):
    return max(EDGE, min(screen_w - EDGE, x)), max(EDGE, min(screen_h - EDGE, y))


def map_to_screen(nose_x, nose_y, iris_x, iris_y, cal):
    nrx = cal["nose_x_max"] - cal["nose_x_min"]
    nry = cal["nose_y_max"] - cal["nose_y_min"]
    irx = cal["iris_x_max"] - cal["iris_x_min"]
    iry = cal["iris_y_max"] - cal["iris_y_min"]
    hx = (nose_x - cal["nose_x_min"]) / nrx if nrx > 1e-6 else 0.5
    hy = (nose_y - cal["nose_y_min"]) / nry if nry > 1e-6 else 0.5
    ix = (iris_x - cal["iris_x_min"]) / irx if irx > 1e-6 else 0.5
    iy = (iris_y - cal["iris_y_min"]) / iry if iry > 1e-6 else 0.5
    bx = hx * HEAD_W_X + ix * (1.0 - HEAD_W_X)
    by = hy * HEAD_W_Y + iy * (1.0 - HEAD_W_Y)
    return clamp(bx * screen_w, by * screen_h)


def move_cursor(feat, cal):
    global prev_sx, prev_sy
    tgt_x, tgt_y = map_to_screen(*feat, cal)
    now = time.time()
    sx, sy = filt_x(tgt_x, now), filt_y(tgt_y, now)
    if math.dist((sx, sy), (prev_sx, prev_sy)) > DEAD_ZONE:
        prev_sx, prev_sy = sx, sy
        sx, sy = clamp(sx, sy)
        move_mouse(sx, sy)


# ── Camera preview with landmarks ────────────────────────────────
CAM_W, CAM_H = 320, 240

LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
ALL_IRIS = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]

cam_visible = True


def show_cam_preview(frame, lms=None):
    """Show a small camera window with eye/iris landmarks drawn."""
    if not cam_visible:
        return
    h, w = frame.shape[:2]
    preview = cv2.resize(frame, (CAM_W, CAM_H))
    sx, sy = CAM_W / w, CAM_H / h

    if lms is not None:
        for idx in LEFT_EYE_CONTOUR + RIGHT_EYE_CONTOUR:
            cx, cy = int(lms[idx].x * w * sx), int(lms[idx].y * h * sy)
            cv2.circle(preview, (cx, cy), 1, (255, 160, 50), -1)

        for idx in ALL_IRIS:
            cx, cy = int(lms[idx].x * w * sx), int(lms[idx].y * h * sy)
            cv2.circle(preview, (cx, cy), 2, (0, 255, 0), -1)

        for idx in [L_IRIS, R_IRIS]:
            cx, cy = int(lms[idx].x * w * sx), int(lms[idx].y * h * sy)
            cv2.circle(preview, (cx, cy), 8, (0, 255, 0), 1)

        cv2.putText(preview, "FACE OK", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1)

        avg_bright = np.mean(cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY))
        if avg_bright < 60:
            light_txt, light_col = "LOW LIGHT", (0, 80, 255)
        elif avg_bright > 200:
            light_txt, light_col = "TOO BRIGHT", (0, 200, 255)
        else:
            light_txt, light_col = "GOOD LIGHT", (0, 220, 100)
        cv2.putText(preview, light_txt, (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, light_col, 1)
    else:
        cv2.putText(preview, "NO FACE", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.putText(preview, "V = toggle cam", (CAM_W - 130, CAM_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 100), 1)

    cv2.imshow("Camera", preview)
    cv2.moveWindow("Camera", 10, 10)


# ── Live iris data overlay ───────────────────────────────────────
iris_y_history = []


def draw_iris_data(board, feat, cal):
    """Live iris data overlay: bars + Y-movement chart."""
    px, py_ = screen_w - 250, 10
    pw, ph = 235, 155

    roi = board[py_:py_ + ph, px:px + pw]
    roi[:] = (roi.astype(np.int16) * 3 // 10).clip(0, 255).astype(np.uint8)
    cv2.rectangle(board, (px, py_), (px + pw, py_ + ph), (45, 45, 60), 1)
    cv2.putText(board, "LIVE IRIS DATA", (px + 8, py_ + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 180), 1)

    if feat is None:
        cv2.putText(board, "NO FACE", (px + 8, py_ + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 200), 1)
        return

    _, _, ix, iy = feat
    iris_y_history.append(iy)
    if len(iris_y_history) > 150:
        iris_y_history.pop(0)

    bx, bw = px + 8, pw - 16
    irx_r = cal["iris_x_max"] - cal["iris_x_min"]
    iry_r = cal["iris_y_max"] - cal["iris_y_min"]
    bar_w = bw - 58

    ix_n = max(0.0, min(1.0, (ix - cal["iris_x_min"]) / irx_r)) if irx_r > 1e-6 else 0.5
    cv2.putText(board, f"X {ix:.4f}", (bx, py_ + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 140, 160), 1)
    bx2 = bx + 58
    cv2.rectangle(board, (bx2, py_ + 26), (bx2 + bar_w, py_ + 36), (30, 30, 40), -1)
    cv2.rectangle(board, (bx2, py_ + 26), (bx2 + max(1, int(bar_w * ix_n)), py_ + 36),
                  (0, 200, 160), -1)

    iy_n = max(0.0, min(1.0, (iy - cal["iris_y_min"]) / iry_r)) if iry_r > 1e-6 else 0.5
    cv2.putText(board, f"Y {iy:.4f}", (bx, py_ + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 140, 160), 1)
    cv2.rectangle(board, (bx2, py_ + 43), (bx2 + bar_w, py_ + 53), (30, 30, 40), -1)
    cv2.rectangle(board, (bx2, py_ + 43), (bx2 + max(1, int(bar_w * iy_n)), py_ + 53),
                  (0, 160, 255), -1)

    if len(iris_y_history) > 2:
        ch_x, ch_y, ch_w, ch_h = bx, py_ + 62, bw, 60
        cv2.rectangle(board, (ch_x, ch_y), (ch_x + ch_w, ch_y + ch_h), (22, 22, 32), -1)
        cv2.putText(board, "Y MOVEMENT", (ch_x + 2, ch_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, (70, 70, 90), 1)
        mid_y = ch_y + ch_h // 2
        cv2.line(board, (ch_x, mid_y), (ch_x + ch_w, mid_y), (40, 40, 50), 1)

        ymin, ymax = cal["iris_y_min"], cal["iris_y_max"]
        yr = ymax - ymin if ymax - ymin > 1e-6 else 1.0
        pts = []
        n = len(iris_y_history)
        for i, v in enumerate(iris_y_history):
            cx = ch_x + int(i / max(1, n - 1) * ch_w)
            norm = max(0.0, min(1.0, (v - ymin) / yr))
            cy = ch_y + ch_h - 2 - int(norm * (ch_h - 14))
            pts.append((cx, cy))
        for i in range(len(pts) - 1):
            cv2.line(board, pts[i], pts[i + 1], (0, 160, 255), 1)

    cv2.putText(board, f"Y range: {iry_r:.4f}  X range: {irx_r:.4f}",
                (bx, py_ + ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.22, (65, 65, 85), 1)


# ── Calibration (same as main.py) ────────────────────────────────
CALIB_DOTS = [
    (0.50, 0.50, "CENTER"),
    (0.10, 0.10, "TOP-LEFT"),
    (0.90, 0.10, "TOP-RIGHT"),
    (0.50, 0.05, "TOP-CENTER"),
    (0.10, 0.90, "BOTTOM-LEFT"),
    (0.90, 0.90, "BOTTOM-RIGHT"),
    (0.50, 0.95, "BOTTOM-CENTER"),
]


def collect_at_point(lmk, cam, t0, sx, sy, label, idx, total):
    global cam_visible
    while True:
        ok, frame = cam.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int((time.time() - t0) * 1000)
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        lms = res.face_landmarks[0] if res.face_landmarks else None
        show_cam_preview(frame, lms)

        board = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        board[:] = (15, 15, 20)
        for j, (px, py, _) in enumerate(CALIB_DOTS):
            pos = (int(px * screen_w), int(py * screen_h))
            if j == idx:
                r = int(22 + 7 * math.sin(time.time() * 4))
                cv2.circle(board, pos, r, (0, 190, 255), 2)
                cv2.circle(board, pos, 5, (0, 190, 255), -1)
            elif j < idx:
                cv2.circle(board, pos, 8, (0, 90, 0), -1)
            else:
                cv2.circle(board, pos, 4, (40, 40, 50), 1)
        cv2.putText(board, f"{label}  ({idx+1}/{total})",
                    (screen_w // 2 - 80, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 190, 255), 2)
        cv2.putText(board, "Look at the dot.  Press SPACE.",
                    (screen_w // 2 - 200, screen_h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 110, 130), 1)
        cv2.imshow("Test", board)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            break
        if k == ord("q"):
            return None
        if k == ord("v"):
            cam_visible = not cam_visible
            if not cam_visible:
                cv2.destroyWindow("Camera")

    cd = time.time() + 1.5
    while time.time() < cd:
        ok, frame = cam.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int((time.time() - t0) * 1000)
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        lms = res.face_landmarks[0] if res.face_landmarks else None
        show_cam_preview(frame, lms)

        left = cd - time.time()
        board = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        board[:] = (15, 15, 20)
        cv2.circle(board, (sx, sy), 26, (0, 230, 255), 2)
        cv2.circle(board, (sx, sy), 5, (0, 230, 255), -1)
        cv2.putText(board, f"Hold still... {left:.0f}",
                    (screen_w // 2 - 100, screen_h // 2 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2)
        cv2.imshow("Test", board)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("v"):
            cam_visible = not cam_visible
            if not cam_visible:
                cv2.destroyWindow("Camera")

    samples = []
    ce = time.time() + 2.0
    while time.time() < ce:
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int((time.time() - t0) * 1000)
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        lms = res.face_landmarks[0] if res.face_landmarks else None
        show_cam_preview(frame, lms)

        if res.face_landmarks:
            f = get_nose_and_iris(res.face_landmarks[0])
            if f:
                samples.append(f)
        pct = (time.time() - (ce - 2.0)) / 2.0
        board = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        board[:] = (15, 15, 20)
        cv2.circle(board, (sx, sy), 26, (50, 220, 100), 2)
        cv2.circle(board, (sx, sy), 5, (50, 220, 100), -1)
        bw = 260
        bx = screen_w // 2 - bw // 2
        cv2.rectangle(board, (bx, screen_h - 48),
                      (bx + int(bw * pct), screen_h - 38), (50, 220, 100), -1)
        cv2.rectangle(board, (bx, screen_h - 48),
                      (bx + bw, screen_h - 38), (50, 50, 60), 1)
        cv2.imshow("Test", board)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("v"):
            cam_visible = not cam_visible
            if not cam_visible:
                cv2.destroyWindow("Camera")
    return samples


def run_calibration(lmk, cam, t0):
    cv2.namedWindow("Test", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    pd = {}
    for i, (nx, ny, label) in enumerate(CALIB_DOTS):
        sx, sy = int(nx * screen_w), int(ny * screen_h)
        s = collect_at_point(lmk, cam, t0, sx, sy, label, i, len(CALIB_DOTS))
        if s is None:
            return None
        if len(s) < 8:
            print(f"  {label}: too few samples")
            return None
        avg = tuple(sum(v[k] for v in s) / len(s) for k in range(4))
        pd[label] = avg
        print(f"  {label}: OK ({len(s)} samples)")

    lnx = np.mean([pd["TOP-LEFT"][0], pd["BOTTOM-LEFT"][0]])
    rnx = np.mean([pd["TOP-RIGHT"][0], pd["BOTTOM-RIGHT"][0]])
    tny = np.mean([pd["TOP-LEFT"][1], pd["TOP-RIGHT"][1], pd["TOP-CENTER"][1]])
    bny = np.mean([pd["BOTTOM-LEFT"][1], pd["BOTTOM-RIGHT"][1], pd["BOTTOM-CENTER"][1]])
    lix = np.mean([pd["TOP-LEFT"][2], pd["BOTTOM-LEFT"][2]])
    rix = np.mean([pd["TOP-RIGHT"][2], pd["BOTTOM-RIGHT"][2]])
    tiy = np.mean([pd["TOP-LEFT"][3], pd["TOP-RIGHT"][3], pd["TOP-CENTER"][3]])
    biy = np.mean([pd["BOTTOM-LEFT"][3], pd["BOTTOM-RIGHT"][3], pd["BOTTOM-CENTER"][3]])

    nrx, nry = rnx - lnx, bny - tny
    irx, iry = rix - lix, biy - tiy

    if abs(irx) < 0.001 and abs(iry) < 0.0005:
        print("  ERROR: iris barely moved. Make sure eyes are visible.")
        return None

    p = 0.15
    nose_ok_x = abs(nrx) >= 0.003
    nose_ok_y = abs(nry) >= 0.003
    center_nx = (lnx + rnx) / 2
    center_ny = (tny + bny) / 2

    iris_y_center = (tiy + biy) / 2
    cal = {
        "nose_x_min": (lnx - nrx * p) if nose_ok_x else (center_nx - 0.15),
        "nose_x_max": (rnx + nrx * p) if nose_ok_x else (center_nx + 0.15),
        "nose_y_min": (tny - nry * p) if nose_ok_y else (center_ny - 0.10),
        "nose_y_max": (bny + nry * p) if nose_ok_y else (center_ny + 0.10),
        "iris_x_min": lix - irx * p if abs(irx) > .001 else 0.3,
        "iris_x_max": rix + irx * p if abs(irx) > .001 else 0.7,
        "iris_y_min": tiy - abs(iry) * p if abs(iry) > .0005 else (iris_y_center - 0.08),
        "iris_y_max": biy + abs(iry) * p if abs(iry) > .0005 else (iris_y_center + 0.08),
    }
    print(f"  Iris Y range: {tiy:.4f} -> {biy:.4f} (delta={iry:.4f})")
    if not nose_ok_x or not nose_ok_y:
        print("  NOTE: head barely moved -- using iris-only calibration.")
    with open(CALIB_PATH, "w") as f:
        json.dump(cal, f)
    print("  Calibration saved!")
    return cal


# ── Dynamic calibration (follow moving dot) ──────────────────────
def run_dynamic_calibration(lmk, cam, t0, cal):
    """User follows a moving dot for ~15s. Captures live iris range and
    merges it into the static calibration for much better vertical tracking."""
    global cam_visible
    duration = 15.0
    start = time.time()
    obs_ix, obs_iy = [], []

    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            break

        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int((time.time() - t0) * 1000)
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        lms = res.face_landmarks[0] if res.face_landmarks else None
        show_cam_preview(frame, lms)

        cur_feat = None
        if res.face_landmarks:
            feat = get_nose_and_iris(res.face_landmarks[0])
            if feat:
                cur_feat = feat
                obs_ix.append(feat[2])
                obs_iy.append(feat[3])

        t = elapsed / duration
        if t < 0.20:
            p = t / 0.20
            dot_x = 0.1 + 0.8 * p
            dot_y = 0.5
            phase = "HORIZONTAL"
        elif t < 0.65:
            p = (t - 0.20) / 0.45
            dot_x = 0.5
            dot_y = 0.5 + 0.43 * math.sin(4 * math.pi * p)
            phase = "VERTICAL"
        else:
            p = (t - 0.65) / 0.35
            dot_x = 0.5 + 0.35 * math.sin(2 * math.pi * p)
            dot_y = 0.5 + 0.40 * math.sin(4 * math.pi * p)
            phase = "COMBINED"

        dx, dy = int(dot_x * screen_w), int(dot_y * screen_h)

        board = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        board[:] = (15, 15, 20)

        cv2.circle(board, (dx, dy), 22, (0, 200, 180), 2)
        cv2.circle(board, (dx, dy), 4, (0, 240, 210), -1)

        cv2.putText(board, "DYNAMIC CALIBRATION", (screen_w // 2 - 165, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 180), 2)
        cv2.putText(board, f"Follow the dot with your eyes   [{phase}]",
                    (screen_w // 2 - 220, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (110, 110, 130), 1)

        draw_iris_data(board, cur_feat, cal)

        pw_ = 300
        px_ = screen_w // 2 - pw_ // 2
        cv2.rectangle(board, (px_, screen_h - 45),
                      (px_ + int(pw_ * t), screen_h - 35), (0, 200, 180), -1)
        cv2.rectangle(board, (px_, screen_h - 45),
                      (px_ + pw_, screen_h - 35), (50, 50, 60), 1)
        remaining = max(0, duration - elapsed)
        cv2.putText(board, f"{remaining:.0f}s left   Q = skip   V = cam",
                    (screen_w // 2 - 140, screen_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 110), 1)

        cv2.imshow("Test", board)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("v"):
            cam_visible = not cam_visible
            if not cam_visible:
                cv2.destroyWindow("Camera")

    if len(obs_iy) < 30:
        print("  Dynamic cal: too few samples, keeping static calibration.")
        return cal

    diy_min = float(np.percentile(obs_iy, 3))
    diy_max = float(np.percentile(obs_iy, 97))
    dix_min = float(np.percentile(obs_ix, 3))
    dix_max = float(np.percentile(obs_ix, 97))

    pad = 0.10
    d_iry = diy_max - diy_min
    d_irx = dix_max - dix_min

    new_iy_min = min(cal["iris_y_min"], diy_min - d_iry * pad)
    new_iy_max = max(cal["iris_y_max"], diy_max + d_iry * pad)
    new_ix_min = min(cal["iris_x_min"], dix_min - d_irx * pad)
    new_ix_max = max(cal["iris_x_max"], dix_max + d_irx * pad)

    print(f"  Dynamic X: [{new_ix_min:.4f}, {new_ix_max:.4f}]  (was [{cal['iris_x_min']:.4f}, {cal['iris_x_max']:.4f}])")
    print(f"  Dynamic Y: [{new_iy_min:.4f}, {new_iy_max:.4f}]  (was [{cal['iris_y_min']:.4f}, {cal['iris_y_max']:.4f}])")

    cal["iris_y_min"] = new_iy_min
    cal["iris_y_max"] = new_iy_max
    cal["iris_x_min"] = new_ix_min
    cal["iris_x_max"] = new_ix_max

    with open(CALIB_PATH, "w") as f:
        json.dump(cal, f)
    print("  Dynamic calibration merged and saved!")
    return cal


# ── Reading text for the reading test ────────────────────────────
READING_LINES = [
    "The development of artificial intelligence has transformed",
    "the way we interact with technology on a daily basis.",
    "From voice assistants to autonomous vehicles, AI systems",
    "are becoming deeply integrated into modern society.",
    "",
    "Machine learning, a subset of AI, enables computers to",
    "learn from data without being explicitly programmed.",
    "Neural networks, inspired by the human brain, can now",
    "recognize images, translate languages, and even compose",
    "music with remarkable accuracy and creativity.",
    "",
    "Eye tracking technology uses cameras and infrared sensors",
    "to determine where a person is looking on a screen.",
    "This has applications in accessibility, gaming, market",
    "research, and human-computer interaction design.",
    "",
    "The cursor on your screen should be following along as",
    "you read each line of this text from left to right,",
    "then dropping down to the next line naturally.",
    "If it tracks your reading smoothly, the calibration",
    "is working correctly. Press SPACE when done reading.",
]


# ── Target dots ──────────────────────────────────────────────────
TARGETS = [
    (screen_w // 2, screen_h // 2, "CENTER"),
    (screen_w // 4, screen_h // 4, "TOP-LEFT"),
    (3 * screen_w // 4, screen_h // 4, "TOP-RIGHT"),
    (screen_w // 4, 3 * screen_h // 4, "BOT-LEFT"),
    (3 * screen_w // 4, 3 * screen_h // 4, "BOT-RIGHT"),
    (screen_w // 4, screen_h // 2, "LEFT"),
    (3 * screen_w // 4, screen_h // 2, "RIGHT"),
]


def main():
    global prev_sx, prev_sy, cam_visible

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    lmk = vision.FaceLandmarker.create_from_options(opts)
    t0 = time.time()

    print("\n" + "=" * 60)
    print("  EYE TRACKING TEST")
    print("  Static Cal -> Dynamic Cal -> Target Dots -> Reading Test")
    print("  TIP: just look at each dot with your eyes")
    print("  Press V at any time to toggle the camera preview")
    print("=" * 60)

    # ── Phase 1a: Static calibration ─────────────────────────────
    cal = run_calibration(lmk, cam, t0)
    if not cal:
        print("  Aborted.")
        lmk.close(); cam.release(); return

    # ── Phase 1b: Dynamic calibration ────────────────────────────
    print("\n  DYNAMIC CALIBRATION: follow the moving dot with your eyes.\n")
    iris_y_history.clear()
    cal = run_dynamic_calibration(lmk, cam, t0, cal)
    if not cal:
        print("  Aborted.")
        lmk.close(); cam.release(); return

    # ── Phase 2: Target dots ─────────────────────────────────────
    tidx, hits, total = 0, 0, 0
    print("\n  TARGET TEST: look at each dot, press SPACE to score.\n")

    while tidx < len(TARGETS):
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int((time.time() - t0) * 1000)
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        lms = res.face_landmarks[0] if res.face_landmarks else None
        show_cam_preview(frame, lms)

        cur_feat = None
        if res.face_landmarks:
            feat = get_nose_and_iris(res.face_landmarks[0])
            if feat:
                cur_feat = feat
                move_cursor(feat, cal)

        gx, gy, name = TARGETS[tidx]
        board = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        board[:] = (15, 15, 20)
        for j, (jx, jy, nm) in enumerate(TARGETS):
            if j == tidx:
                pulse = int(24 + 7 * math.sin(time.time() * 4))
                cv2.circle(board, (jx, jy), pulse, (0, 190, 255), 2)
                cv2.circle(board, (jx, jy), 5, (0, 190, 255), -1)
                cv2.putText(board, nm, (jx - 40, jy - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 190, 255), 1)
            elif j < tidx:
                cv2.circle(board, (jx, jy), 8, (0, 80, 0), -1)
            else:
                cv2.circle(board, (jx, jy), 5, (40, 40, 50), 1)

        cv2.putText(board, f"Target {tidx+1}/{len(TARGETS)}",
                    (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 160), 1)
        cv2.putText(board, f"Hits: {hits}/{total}",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 120), 1)
        cv2.putText(board, "SPACE = score     V = cam     Q = quit",
                    (screen_w // 2 - 210, screen_h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 110), 1)
        draw_iris_data(board, cur_feat, cal)
        cv2.imshow("Test", board)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("v"):
            cam_visible = not cam_visible
            if not cam_visible:
                cv2.destroyWindow("Camera")
        elif k == ord(" "):
            mx, my = pyautogui.position()
            dist = math.dist((mx, my), (gx, gy))
            total += 1
            if dist < 200:
                hits += 1
                print(f"  {name}: HIT  ({dist:.0f}px)")
            else:
                print(f"  {name}: MISS ({dist:.0f}px)")
            tidx += 1
            filt_x.reset(); filt_y.reset()
            prev_sx, prev_sy = float(screen_w) / 2, float(screen_h) / 2

    if total > 0:
        print(f"\n  Targets: {hits}/{total} ({hits/total*100:.0f}%)\n")

    # ── Phase 3: Reading test ────────────────────────────────────
    print("  READING TEST: read the text naturally. Cursor should follow.")
    print("  Press SPACE when done reading.\n")
    filt_x.reset(); filt_y.reset()
    prev_sx, prev_sy = float(screen_w) / 2, float(screen_h) / 2

    reading = True
    while reading:
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int((time.time() - t0) * 1000)
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        lms = res.face_landmarks[0] if res.face_landmarks else None
        show_cam_preview(frame, lms)

        cur_feat = None
        if res.face_landmarks:
            feat = get_nose_and_iris(res.face_landmarks[0])
            if feat:
                cur_feat = feat
                move_cursor(feat, cal)

        board = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        board[:] = (15, 15, 20)

        # Header
        cv2.putText(board, "READING TEST", (screen_w // 2 - 100, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 255), 2)
        cv2.line(board, (100, 65), (screen_w - 100, 65), (35, 35, 50), 1)

        # Render text
        margin_left = 120
        margin_top = 100
        line_height = 38
        for i, line in enumerate(READING_LINES):
            y = margin_top + i * line_height
            if y > screen_h - 80:
                break
            if line == "":
                continue
            color = (210, 210, 225)
            cv2.putText(board, line, (margin_left, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)

        # Footer
        cv2.line(board, (100, screen_h - 60), (screen_w - 100, screen_h - 60), (35, 35, 50), 1)
        cv2.putText(board, "Read naturally. Cursor should follow.  SPACE = done   V = cam   Q = quit",
                    (screen_w // 2 - 380, screen_h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 110), 1)
        draw_iris_data(board, cur_feat, cal)

        cv2.imshow("Test", board)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("v"):
            cam_visible = not cam_visible
            if not cam_visible:
                cv2.destroyWindow("Camera")
        elif k == ord(" ") or k == ord("q"):
            reading = False

    cv2.destroyAllWindows()
    lmk.close()
    cam.release()

    print("=" * 60)
    if total > 0:
        pct = hits / total * 100
        print(f"  Targets: {hits}/{total} ({pct:.0f}%)")
        if pct >= 60:
            print("  PASS -- run: python main.py")
        else:
            print("  Recalibrate -- look directly at each dot with your eyes.")
    print("  Reading test complete. Check if cursor tracked your reading.")
    print("=" * 60)


if __name__ == "__main__":
    main()
