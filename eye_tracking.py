"""
Eye tracking control (Python + OpenCV + MediaPipe) - port of eye_tracking.html

Setup:  pip install opencv-python mediapipe==0.10.14 numpy
Run:    python eye_tracking.py

Keys:   C = calibrate (9 dots)   F = fullscreen   D = toggle dwell   Q / Esc = quit

How it works
  1. Iris position inside each eye (plus nose position) = eye features.
  2. 9-dot calibration fits a linear map: features -> screen position.
  3. Gaze moves the cursor. Stare at a target (dwell) or hold a ~0.3 s blink to pop it.
Tips: sit an arm's length away, good front light, keep your head fairly still,
      recalibrate if you move.
"""
import random
import time

import cv2
import mediapipe as mp
import numpy as np

# ---------- Settings (tweak these) ----------
W, H = 960, 540              # game canvas size
CAM_W, CAM_H = 640, 480
SMOOTH = 0.25                # 0.05 = very smooth/laggy, 1 = raw/jittery
BLINK_ON, BLINK_OFF = 0.6, 0.3   # blink score thresholds (hysteresis)
BLINK_HOLD = 0.3             # seconds the eyes must stay closed to fire
EAR_OPEN, EAR_CLOSED = 0.28, 0.10  # eye-aspect-ratio range -> blink score 0..1
DWELL_TIME = 0.9             # seconds of staring to pop a target
GRID = (0.1, 0.5, 0.9)
POINTS = [(x, y) for y in GRID for x in GRID]   # 3x3 calibration grid
WAIT, COLLECT = 0.9, 0.8     # seconds: settle on dot, then record

mp_face = mp.solutions.face_mesh


# ---------------- Eye features ----------------
def _ratio(p, a, b):
    return (p - a) / ((b - a) or 1e-6)


def features(lm):
    """[1, horizontal gaze, vertical gaze, nose x, nose y] from landmarks."""
    hr = _ratio(lm[468].x, lm[33].x, lm[133].x)    # right eye: outer -> inner
    hl = _ratio(lm[473].x, lm[362].x, lm[263].x)   # left eye: inner -> outer
    vr = _ratio(lm[468].y, lm[159].y, lm[145].y)   # upper -> lower eyelid
    vl = _ratio(lm[473].y, lm[386].y, lm[374].y)
    return np.array([1.0, (hr + hl) / 2, (vr + vl) / 2, lm[1].x, lm[1].y])


def blink_score(lm):
    """0 = eyes open, 1 = closed (from eye aspect ratio)."""
    def ear(top, bottom, outer, inner):
        v = np.hypot(lm[top].x - lm[bottom].x, lm[top].y - lm[bottom].y)
        h = np.hypot(lm[outer].x - lm[inner].x, lm[outer].y - lm[inner].y) + 1e-6
        return v / h
    e = (ear(159, 145, 33, 133) + ear(386, 374, 362, 263)) / 2
    return float(np.clip((EAR_OPEN - e) / (EAR_OPEN - EAR_CLOSED), 0, 1))


# ---------------- Regression ----------------
def fit(data):
    """data: list of (features, x, y). Returns (predict_fn, mean_error)."""
    X = np.array([d[0] for d in data])
    Y = np.array([[d[1], d[2]] for d in data])
    ridge = np.diag([0, 1e-4, 1e-4, 1e-4, 1e-4])           # no penalty on bias
    Wt = np.linalg.solve(X.T @ X + ridge, X.T @ Y)

    def predict(f):
        return np.clip(f @ Wt, 0, 1)

    err = float(np.mean(np.linalg.norm(X @ Wt - Y, axis=1)))
    return predict, err


# ---------------- Blink trigger with hysteresis + hold ----------------
class Trigger:
    def __init__(self):
        self.on, self.since = False, None

    def update(self, v, on_at, off_at, hold):
        """True only on the frame the trigger turns on."""
        now = time.time()
        if not self.on:
            if v > on_at:
                if self.since is None:
                    self.since = now
                if now - self.since >= hold:
                    self.on = True
                    return True
            else:
                self.since = None
            return False
        if v < off_at:
            self.on, self.since = False, None
        return False


# ---------------- Calibration ----------------
class Calibration:
    def __init__(self):
        self.active, self.i, self.t0 = False, 0, 0.0
        self.samples, self.data = [], []

    def start(self):
        self.active, self.i, self.t0 = True, 0, time.time()
        self.samples, self.data = [], []

    def update(self, feat):
        """Returns (finished, model, error_or_None)."""
        e = time.time() - self.t0
        if WAIT < e < WAIT + COLLECT and feat is not None:
            self.samples.append(feat)
        if e >= WAIT + COLLECT:
            if len(self.samples) >= 5:
                avg = np.mean(self.samples, axis=0)
                self.data.append((avg, *POINTS[self.i]))
            self.samples, self.i, self.t0 = [], self.i + 1, time.time()
            if self.i >= len(POINTS):
                self.active = False
                if len(self.data) >= 6:
                    model, err = fit(self.data)
                    return True, model, err
                return True, None, None
        return False, None, None


# ---------------- Drawing helpers ----------------
def center_text(img, s, y, scale=0.9, color=(230, 237, 243)):
    (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(img, s, ((W - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera.")

    mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,   # iris landmarks
                            min_detection_confidence=0.6,
                            min_tracking_confidence=0.6)

    win = "Eye Tracking Control"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    gaze = np.array([0.5, 0.5])        # smoothed screen position (0..1)
    model, status = None, "Press C to calibrate"
    cal, trig = Calibration(), Trigger()
    targets, score = [], 0
    last_spawn, flash = time.time(), 0.0
    use_dwell, fullscreen = True, False
    prev_t = time.time()
    feat, seen = None, False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # unflipped; calibration absorbs mirroring
        now = time.time()
        dt, prev_t = min(now - prev_t, 0.05), now

        # ---------- tracking ----------
        fired = False
        seen = bool(res.multi_face_landmarks)
        if seen:
            lm = res.multi_face_landmarks[0].landmark
            blink = blink_score(lm)
            if blink < 0.35:                       # ignore frames while eyes are closing
                feat = features(lm)
                if model is not None:
                    p = model(feat)
                    gaze += SMOOTH * (p - gaze)
            fired = trig.update(blink, BLINK_ON, BLINK_OFF, BLINK_HOLD)
        else:
            feat = None

        # ---------- calibration ----------
        canvas = np.full((H, W, 3), (10, 7, 5), np.uint8)
        if cal.active:
            finished, new_model, err = cal.update(feat)
            if finished:
                if new_model is not None:
                    model = new_model
                    status = f"Calibrated (avg error {err * 100:.1f}% of screen)"
                else:
                    status = "Calibration failed: face not seen enough. Press C to retry."
            else:
                px, py = int(POINTS[cal.i][0] * W), int(POINTS[cal.i][1] * H)
                e = now - cal.t0
                pulse = int(10 + 4 * np.sin(now * 8))
                cv2.circle(canvas, (px, py), pulse + 8, (255, 182, 63), 2)
                cv2.circle(canvas, (px, py), 6 if e > WAIT else pulse,
                           (90, 90, 255) if e > WAIT else (255, 182, 63), -1)
                center_text(canvas, f"Look at the dot ({cal.i + 1}/{len(POINTS)})", H // 2 - 20)
        else:
            # ---------- game ----------
            if model is not None and seen:
                if now - last_spawn > 1.1:
                    last_spawn = now
                    targets.append({"x": random.randint(80, W - 80), "y": random.randint(80, H - 80),
                                    "life": 4.0, "r": 44, "dwell": 0.0})
                gx, gy = gaze[0] * W, gaze[1] * H
                for t in targets:
                    on = np.hypot(t["x"] - gx, t["y"] - gy) < t["r"] + 20
                    t["dwell"] = t["dwell"] + dt if on else max(0.0, t["dwell"] - dt * 2)
                    t["life"] -= dt
                if use_dwell:
                    for t in targets:
                        if t["dwell"] > DWELL_TIME:
                            targets.remove(t)
                            score += 1
                            break
                if fired:
                    flash = 0.12
                    for t in targets:
                        if np.hypot(t["x"] - gx, t["y"] - gy) < t["r"] + 20:
                            targets.remove(t)
                            score += 1
                            break
                targets = [t for t in targets if t["life"] > 0]
                flash = max(0.0, flash - dt)

            for t in targets:
                cv2.circle(canvas, (t["x"], t["y"]), t["r"], (76, 138, 31), -1)
                arc = int(min(1.0, t["dwell"] / DWELL_TIME) * 360)
                if arc > 0:
                    cv2.ellipse(canvas, (t["x"], t["y"]), (t["r"] + 7, t["r"] + 7),
                                0, -90, -90 + arc, (176, 255, 125), 4)

            if model is not None and seen:
                cx, cy = int(gaze[0] * W), int(gaze[1] * H)
                col = (90, 90, 255) if flash > 0 else (255, 182, 63)
                cv2.circle(canvas, (cx, cy), 14, col, 2)
                cv2.circle(canvas, (cx, cy), 3, col, -1)

            cv2.putText(canvas, f"Score: {score}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (243, 237, 230), 2, cv2.LINE_AA)
            if model is None:
                center_text(canvas, "Press C to calibrate (F for fullscreen first)", H // 2)
            elif not seen:
                center_text(canvas, "PAUSED - face lost", H // 2)

        # camera preview (mirrored) + status bar
        small = cv2.flip(cv2.resize(frame, (160, 120)), 1)
        canvas[H - 130:H - 10, W - 170:W - 10] = small
        cv2.putText(canvas, status, (20, H - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (184, 167, 154), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"[C] calibrate  [F] fullscreen  [D] dwell: {'on' if use_dwell else 'off'}  [Q] quit",
                    (20, H - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (184, 167, 154), 1, cv2.LINE_AA)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("c"):
            if seen or True:
                cal.start()
                model = None
                status = "Calibrating..."
        elif key == ord("d"):
            use_dwell = not use_dwell
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
