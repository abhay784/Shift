"""
Eye-controlled cursor.

Head + iris blend (both calibrated per-axis, weighted 55/45).
1-Euro filter for buttery smooth cursor with zero stick drift.

Keys:  R = recalibrate   L = lock/unlock gaze   Q = quit
       [ / ] = adjust head/eye blend
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import pyautogui
import math
import time
import os
import json
import threading
import numpy as np

try:
    import speech_recognition as sr
    import sounddevice as sd
    VOICE_OK = True
except ImportError:
    VOICE_OK = False

pyautogui.FAILSAFE = False

# Use Quartz for near-zero-latency mouse movement on macOS
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

NOSE = 4
L_IRIS, R_IRIS = 468, 473
L_EYE_H, R_EYE_H = (33, 133), (362, 263)

# Per-axis blend: head barely moves vertically on a laptop, so lean on iris for Y
HEAD_W_X = 0.3
HEAD_W_Y = 0.0
DEAD_ZONE = 1.5

BLINK_EAR = 0.19
BLINK_NEEDED = 3
blink_ctr = 0

gaze_locked = False
lock_x, lock_y = screen_w // 2, screen_h // 2
running = True


# ── 1-Euro Filter (kills jitter, zero lag on fast moves) ─────────
class OneEuro:
    def __init__(self, fc_min=1.5, beta=0.01, d_cutoff=1.0):
        self.fc_min = fc_min
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        te = t - self.t_prev
        if te <= 1e-6:
            return self.x_prev
        dx = (x - self.x_prev) / te
        a_d = self._alpha(te, self.d_cutoff)
        dx = a_d * dx + (1 - a_d) * self.dx_prev
        fc = self.fc_min + self.beta * abs(dx)
        a = self._alpha(te, fc)
        out = a * x + (1 - a) * self.x_prev
        self.x_prev = out
        self.dx_prev = dx
        self.t_prev = t
        return out

    @staticmethod
    def _alpha(te, fc):
        tau = 1.0 / (2 * math.pi * fc)
        return 1.0 / (1.0 + tau / te)

    def reset(self, x=None, t=None):
        self.x_prev = x
        self.t_prev = t
        self.dx_prev = 0.0


filt_x = OneEuro(fc_min=3.0, beta=0.05)
filt_y = OneEuro(fc_min=3.0, beta=0.05)


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


def ear(lms, t, b, l, r):
    v = math.dist((lms[t].x, lms[t].y), (lms[b].x, lms[b].y))
    h = math.dist((lms[l].x, lms[l].y), (lms[r].x, lms[r].y))
    return v / h if h > 0 else 0


def clamp(x, y):
    return max(EDGE, min(screen_w - EDGE, x)), max(EDGE, min(screen_h - EDGE, y))


# ── Calibration ──────────────────────────────────────────────────
CALIB_DOTS = [
    (0.50, 0.50, "CENTER"),
    (0.10, 0.10, "TOP-LEFT"),
    (0.90, 0.10, "TOP-RIGHT"),
    (0.50, 0.05, "TOP-CENTER"),
    (0.10, 0.90, "BOTTOM-LEFT"),
    (0.90, 0.90, "BOTTOM-RIGHT"),
    (0.50, 0.95, "BOTTOM-CENTER"),
]


def save_cal(cal):
    with open(CALIB_PATH, "w") as f:
        json.dump(cal, f)


def load_cal():
    if not os.path.exists(CALIB_PATH):
        return None
    with open(CALIB_PATH) as f:
        return json.load(f)


def collect_at_point(lmk, cam, t0, sx, sy, label, idx, total):
    cv2.namedWindow("Cal", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Cal", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ok, frame = cam.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int((time.time() - t0) * 1000)
        lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

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

        cv2.putText(board, f"{label}", (screen_w // 2 - 60, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 190, 255), 2)
        cv2.putText(board, f"{idx+1} / {total}",
                    (screen_w // 2 - 25, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 120), 1)
        cv2.putText(board, "Look at the dot.  Press SPACE.",
                    (screen_w // 2 - 200, screen_h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 110, 130), 1)
        cv2.imshow("Cal", board)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            break
        if k == ord("q"):
            return None

    cd_end = time.time() + 1.5
    while time.time() < cd_end:
        ok, frame = cam.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int((time.time() - t0) * 1000)
        lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        left = cd_end - time.time()
        board = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        board[:] = (15, 15, 20)
        cv2.circle(board, (sx, sy), 26, (0, 230, 255), 2)
        cv2.circle(board, (sx, sy), 5, (0, 230, 255), -1)
        cv2.putText(board, f"Hold still... {left:.0f}",
                    (screen_w // 2 - 100, screen_h // 2 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2)
        cv2.imshow("Cal", board)
        cv2.waitKey(1)

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
        cv2.imshow("Cal", board)
        cv2.waitKey(1)
    return samples


def run_calibration(lmk, cam, t0):
    point_data = {}
    for i, (nx, ny, label) in enumerate(CALIB_DOTS):
        sx, sy = int(nx * screen_w), int(ny * screen_h)
        samples = collect_at_point(lmk, cam, t0, sx, sy, label, i, len(CALIB_DOTS))
        if samples is None:
            cv2.destroyWindow("Cal")
            return None
        if len(samples) < 8:
            print(f"  {label}: too few samples, retry")
            cv2.destroyWindow("Cal")
            return None
        avg = tuple(sum(s[k] for s in samples) / len(samples) for k in range(4))
        point_data[label] = avg
        print(f"  {label}: OK ({len(samples)} samples)")

    cv2.destroyWindow("Cal")

    # Head ranges
    lnx = np.mean([point_data["TOP-LEFT"][0], point_data["BOTTOM-LEFT"][0]])
    rnx = np.mean([point_data["TOP-RIGHT"][0], point_data["BOTTOM-RIGHT"][0]])
    tny = np.mean([point_data["TOP-LEFT"][1], point_data["TOP-RIGHT"][1], point_data["TOP-CENTER"][1]])
    bny = np.mean([point_data["BOTTOM-LEFT"][1], point_data["BOTTOM-RIGHT"][1], point_data["BOTTOM-CENTER"][1]])

    lix = np.mean([point_data["TOP-LEFT"][2], point_data["BOTTOM-LEFT"][2]])
    rix = np.mean([point_data["TOP-RIGHT"][2], point_data["BOTTOM-RIGHT"][2]])
    tiy = np.mean([point_data["TOP-LEFT"][3], point_data["TOP-RIGHT"][3], point_data["TOP-CENTER"][3]])
    biy = np.mean([point_data["BOTTOM-LEFT"][3], point_data["BOTTOM-RIGHT"][3], point_data["BOTTOM-CENTER"][3]])

    nrx, nry = rnx - lnx, bny - tny
    irx, iry = rix - lix, biy - tiy

    if abs(irx) < 0.001 and abs(iry) < 0.0005:
        print("  ERROR: iris barely moved. Make sure eyes are visible.")
        return None

    pad = 0.15
    nose_ok_x = abs(nrx) >= 0.003
    nose_ok_y = abs(nry) >= 0.003
    center_nx = (lnx + rnx) / 2
    center_ny = (tny + bny) / 2

    iris_y_center = (tiy + biy) / 2
    cal = {
        "nose_x_min": (lnx - nrx * pad) if nose_ok_x else (center_nx - 0.15),
        "nose_x_max": (rnx + nrx * pad) if nose_ok_x else (center_nx + 0.15),
        "nose_y_min": (tny - nry * pad) if nose_ok_y else (center_ny - 0.10),
        "nose_y_max": (bny + nry * pad) if nose_ok_y else (center_ny + 0.10),
        "iris_x_min": lix - irx * pad if abs(irx) > 0.001 else 0.3,
        "iris_x_max": rix + irx * pad if abs(irx) > 0.001 else 0.7,
        "iris_y_min": tiy - abs(iry) * pad if abs(iry) > 0.0005 else (iris_y_center - 0.08),
        "iris_y_max": biy + abs(iry) * pad if abs(iry) > 0.0005 else (iris_y_center + 0.08),
    }
    print(f"  Iris Y range: {tiy:.4f} -> {biy:.4f} (delta={iry:.4f})")
    if not nose_ok_x or not nose_ok_y:
        print("  NOTE: head barely moved -- using iris-only calibration.")
    save_cal(cal)
    print("  Calibration saved!")
    return cal


def run_dynamic_calibration(lmk, cam, t0, cal):
    """Follow moving dot for ~15s to capture real iris range."""
    cv2.namedWindow("DynCal", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("DynCal", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
        if res.face_landmarks:
            feat = get_nose_and_iris(res.face_landmarks[0])
            if feat:
                obs_ix.append(feat[2])
                obs_iy.append(feat[3])

        t = elapsed / duration
        if t < 0.20:
            p = t / 0.20
            dot_x, dot_y = 0.1 + 0.8 * p, 0.5
        elif t < 0.65:
            p = (t - 0.20) / 0.45
            dot_x = 0.5
            dot_y = 0.5 + 0.43 * math.sin(4 * math.pi * p)
        else:
            p = (t - 0.65) / 0.35
            dot_x = 0.5 + 0.35 * math.sin(2 * math.pi * p)
            dot_y = 0.5 + 0.40 * math.sin(4 * math.pi * p)

        dx, dy = int(dot_x * screen_w), int(dot_y * screen_h)
        board = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        board[:] = (15, 15, 20)
        cv2.circle(board, (dx, dy), 22, (0, 200, 180), 2)
        cv2.circle(board, (dx, dy), 4, (0, 240, 210), -1)
        cv2.putText(board, "DYNAMIC CALIBRATION  -  follow the dot",
                    (screen_w // 2 - 250, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 180), 2)
        pw_ = 300
        px_ = screen_w // 2 - pw_ // 2
        cv2.rectangle(board, (px_, screen_h - 45),
                      (px_ + int(pw_ * t), screen_h - 35), (0, 200, 180), -1)
        cv2.rectangle(board, (px_, screen_h - 45),
                      (px_ + pw_, screen_h - 35), (50, 50, 60), 1)
        cv2.putText(board, f"{max(0, duration - elapsed):.0f}s   Q = skip",
                    (screen_w // 2 - 60, screen_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 110), 1)
        cv2.imshow("DynCal", board)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

    cv2.destroyWindow("DynCal")

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

    cal["iris_y_min"] = min(cal["iris_y_min"], diy_min - d_iry * pad)
    cal["iris_y_max"] = max(cal["iris_y_max"], diy_max + d_iry * pad)
    cal["iris_x_min"] = min(cal["iris_x_min"], dix_min - d_irx * pad)
    cal["iris_x_max"] = max(cal["iris_x_max"], dix_max + d_irx * pad)

    print(f"  Dynamic X: [{cal['iris_x_min']:.4f}, {cal['iris_x_max']:.4f}]")
    print(f"  Dynamic Y: [{cal['iris_y_min']:.4f}, {cal['iris_y_max']:.4f}]")

    save_cal(cal)
    print("  Dynamic calibration merged and saved!")
    return cal


def map_to_screen(nose_x, nose_y, iris_x, iris_y, cal):
    """Blend head and iris, both normalized 0-1 per-axis."""
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


# ── Wake word ────────────────────────────────────────────────────
def wake_word_loop():
    global gaze_locked, lock_x, lock_y, running
    if not VOICE_OK:
        return
    rec = sr.Recognizer()
    rec.energy_threshold = 400
    print("  Voice: say 'Hey Agent' to lock / 'release' to unlock")
    while running:
        try:
            a = sd.rec(int(2.5 * 16000), samplerate=16000, channels=1, dtype="int16")
            sd.wait()
            if not running:
                break
            audio = sr.AudioData(a.tobytes(), 16000, 2)
            txt = rec.recognize_google(audio).lower()
            if not gaze_locked and ("hey agent" in txt or "hey, agent" in txt):
                gaze_locked = True
                lock_x, lock_y = pyautogui.position()
                print(f"\n  LOCKED at ({lock_x}, {lock_y})")
            elif gaze_locked and ("release" in txt or "unlock" in txt):
                gaze_locked = False
                print("  UNLOCKED")
        except (sr.UnknownValueError, sr.RequestError):
            pass
        except Exception:
            time.sleep(1)


def make_panel(status, locked):
    p = np.zeros((55, 400, 3), dtype=np.uint8)
    p[:] = (25, 25, 30)
    c = (50, 230, 100) if status == "TRACKING" else (60, 60, 230)
    cv2.putText(p, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    if locked:
        cv2.putText(p, "LOCKED", (150, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.putText(p, f"H/E  X:{HEAD_W_X:.0%}/{1-HEAD_W_X:.0%}  Y:{HEAD_W_Y:.0%}/{1-HEAD_W_Y:.0%}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (90, 90, 110), 1)
    cv2.putText(p, "R=recal  L=lock  Q=quit", (230, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 100), 1)
    return p


def main():
    global gaze_locked, lock_x, lock_y, running, blink_ctr, HEAD_W_X, HEAD_W_Y, filt_x, filt_y

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Cannot open webcam")
        return

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    lmk = vision.FaceLandmarker.create_from_options(opts)
    t0 = time.time()

    cal = load_cal()
    if cal and "iris_x_min" in cal:
        print("  Loaded saved calibration. Press R to redo.")
    else:
        cal = run_calibration(lmk, cam, t0)
        if not cal:
            lmk.close(); cam.release(); return
        print("\n  DYNAMIC CALIBRATION: follow the dot with your eyes.\n")
        cal = run_dynamic_calibration(lmk, cam, t0, cal)

    if VOICE_OK:
        threading.Thread(target=wake_word_loop, daemon=True).start()

    cv2.namedWindow("Eye Tracker", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Eye Tracker", screen_w - 380, 10)
    print("\n  TRACKING ACTIVE\n")

    prev_sx, prev_sy = screen_w / 2, screen_h / 2

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int((time.time() - t0) * 1000)
            res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            status = "NO FACE"
            if res.face_landmarks:
                lms = res.face_landmarks[0]
                feat = get_nose_and_iris(lms)
                if feat:
                    status = "TRACKING"
                    tgt_x, tgt_y = map_to_screen(*feat, cal)

                    if not gaze_locked:
                        now = time.time()
                        sx = filt_x(tgt_x, now)
                        sy = filt_y(tgt_y, now)

                        if math.dist((sx, sy), (prev_sx, prev_sy)) > DEAD_ZONE:
                            prev_sx, prev_sy = sx, sy
                            sx, sy = clamp(sx, sy)
                            move_mouse(sx, sy)
                    else:
                        move_mouse(lock_x, lock_y)

                    le = ear(lms, 159, 145, 33, 133)
                    re = ear(lms, 386, 374, 362, 263)
                    ae = (le + re) / 2
                    if ae < BLINK_EAR:
                        blink_ctr += 1
                    else:
                        if blink_ctr >= BLINK_NEEDED:
                            pyautogui.click()
                        blink_ctr = 0

            cv2.imshow("Eye Tracker", make_panel(status, gaze_locked))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                new = run_calibration(lmk, cam, t0)
                if new:
                    new = run_dynamic_calibration(lmk, cam, t0, new)
                    cal = new
                    filt_x.reset(); filt_y.reset()
                    prev_sx, prev_sy = screen_w / 2, screen_h / 2
            elif key == ord("l"):
                if gaze_locked:
                    gaze_locked = False
                    print("  UNLOCKED")
                else:
                    gaze_locked = True
                    lock_x, lock_y = pyautogui.position()
                    print(f"  LOCKED at ({lock_x}, {lock_y})")
            elif key == ord("]"):
                HEAD_W_X = min(0.9, HEAD_W_X + 0.05)
                HEAD_W_Y = min(0.9, HEAD_W_Y + 0.05)
                print(f"  X: Head {HEAD_W_X:.0%}  Y: Head {HEAD_W_Y:.0%}")
            elif key == ord("["):
                HEAD_W_X = max(0.1, HEAD_W_X - 0.05)
                HEAD_W_Y = max(0.1, HEAD_W_Y - 0.05)
                print(f"  X: Head {HEAD_W_X:.0%}  Y: Head {HEAD_W_Y:.0%}")

    except KeyboardInterrupt:
        pass

    running = False
    lmk.close(); cam.release(); cv2.destroyAllWindows()
    print("  Done.")


if __name__ == "__main__":
    main()
