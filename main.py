import os
import sys
if sys.platform == "linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from collections import deque, Counter
import time
from threading import Thread, Lock

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "HSPR_ConvNextLarge_Aug_CB.pt")
INPUT_H, INPUT_W = 320, 240

ZOOM = 1.0

WINDOW_SIZE = 50
WIN_RATIO = 0.75
MIN_CONFIDENCE = 0.40
SILHOUETTE_DARKEST_PCT = 23  # treat the darkest N% of pixels as shadow
MIN_CONTOUR_AREA = 2000
MAX_PUPPETS = 1
TRAIL_LENGTH = 30
MOTION_WINDOW = 12
WALK_THRESH = 10           # right-edge must move this many avg px/frame
WALK_FAST_THRESH = 22
MOTION_DIR_THRESH = 0.80
JUMP_THRESH = 26
JUMP_MIN_PHASE_FRAMES = 3
JUMP_MIN_SWING = 55
STILL_DELAY_FRAMES = 75  # ~2.5s at 30fps before reporting "still"
# max pixel distance between bbox centers across frames to consider same puppet
TRACK_MAX_DIST = 150
TRACK_STALE_FRAMES = 15

CLASS_NAMES = [
    "bird", "chicken", "cow", "crab", "deer",
    "dog", "elephant", "moose", "panther", "rabbit", "snail",
]
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# distinct colors per puppet slot (BGR)
PUPPET_COLORS = [
    (0, 255, 0),
    (255, 100, 0),
    (0, 100, 255),
    (255, 0, 255),
]


def _resolve_model_path() -> str:
    return os.environ.get("SHADOW_MODEL_PATH", MODEL_PATH)


def _resolve_camera_index() -> int:
    raw = os.environ.get("SHADOW_CAMERA_INDEX", "1").strip()
    try:
        return int(raw)
    except ValueError:
        print(f"warning: invalid SHADOW_CAMERA_INDEX={raw!r}; defaulting to 1")
        return 1


def load_model(path: str) -> nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"model file not found: {path}. "
            "Set SHADOW_MODEL_PATH in your environment or place the model under ./models/."
        )
    model = models.convnext_large(weights=None)
    in_features = model.classifier[2].in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES),
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def center_crop(frame: np.ndarray, zoom: float) -> np.ndarray:
    h, w = frame.shape[:2]
    crop_h, crop_w = int(h / zoom), int(w / zoom)
    y = (h - crop_h) // 2
    x = (w - crop_w) // 2
    return frame[y : y + crop_h, x : x + crop_w]


def preprocess(frame: np.ndarray) -> torch.Tensor:
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


BLOCKED_CLASSES = {"elephant"}

def classify(model: nn.Module, tensor: torch.Tensor) -> tuple[str, float]:
    with torch.no_grad():
        logits = model(tensor)
    scores = logits[:, :NUM_CLASSES].clone()
    for i, name in enumerate(CLASS_NAMES):
        if name in BLOCKED_CLASSES:
            scores[:, i] = -float("inf")
    probs = torch.softmax(scores, dim=1)
    confidence, idx = probs.max(dim=1)
    return CLASS_NAMES[idx.item()], confidence.item()


def adaptive_threshold(gray: np.ndarray) -> int:
    """pick a threshold so roughly SILHOUETTE_DARKEST_PCT% of pixels are 'shadow'"""
    return int(np.percentile(gray, SILHOUETTE_DARKEST_PCT))


def find_silhouettes(frame: np.ndarray) -> tuple[list[tuple[int, int, int, int]], int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = adaptive_threshold(gray)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            break
        bboxes.append(cv2.boundingRect(c))
        if len(bboxes) >= MAX_PUPPETS:
            break
    return bboxes, thresh


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, w, h = bbox
    return x + w // 2, y + h // 2


def bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ca, cb = bbox_center(a), bbox_center(b)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


# -- motion detection --------------------------------------------------------

def _detect_jump(vels: list[tuple[int, int]]) -> bool:
    if len(vels) < 8:
        return False
    up_idxs = [i for i, (_, dy) in enumerate(vels) if dy < -JUMP_THRESH]
    down_idxs = [i for i, (_, dy) in enumerate(vels) if dy > JUMP_THRESH]
    if len(up_idxs) < JUMP_MIN_PHASE_FRAMES or len(down_idxs) < JUMP_MIN_PHASE_FRAMES:
        return False
    first_up = up_idxs[0]
    first_down_after_up = next((i for i in down_idxs if i > first_up), None)
    if first_down_after_up is None:
        return False

    up_before_down = sum(1 for i in up_idxs if i < first_down_after_up)
    down_after_up = sum(1 for i in down_idxs if i > first_up)
    if up_before_down < 2 or down_after_up < JUMP_MIN_PHASE_FRAMES:
        return False

    ys = [dy for _, dy in vels]
    swing = max(ys) - min(ys)
    if swing < JUMP_MIN_SWING:
        return False
    return True


def compute_motion(velocity_history: deque[tuple[int, int]], label: str | None) -> str:
    if len(velocity_history) < 6:
        return "still"

    vels = list(velocity_history)

    if _detect_jump(vels):
        return "jumping"

    # horizontal only: require enough meaningful movement before reporting walking
    active_dx = [dx for dx, _ in vels if abs(dx) >= WALK_THRESH]
    if len(active_dx) < max(3, len(vels) // 2):
        return "still"

    # check directional consistency
    h_signs = [1 if dx > 0 else (-1 if dx < 0 else 0) for dx in active_dx]
    c = Counter(s for s in h_signs if s != 0)
    if not c:
        return "still"
    best, count = c.most_common(1)[0]
    if count / len(active_dx) < MOTION_DIR_THRESH:
        return "still"

    avg_dx = sum(active_dx) / len(active_dx)
    direction = "right" if best == 1 else "left"
    pace = "fast" if abs(avg_dx) >= WALK_FAST_THRESH else "slow"
    return f"walking {direction} {pace}"


# -- per-puppet tracking state -----------------------------------------------

class PuppetTracker:
    def __init__(self, puppet_id: int, bbox: tuple[int, int, int, int]):
        self.id = puppet_id
        self.bbox = bbox
        self.history: deque[str | None] = deque(maxlen=WINDOW_SIZE)
        self.confirmed: str | None = None
        self.trail: deque[tuple[int, int]] = deque(maxlen=TRAIL_LENGTH)
        self.velocities: deque[tuple[int, int]] = deque(maxlen=MOTION_WINDOW)
        self.prev_center: tuple[int, int] | None = None
        self._prev_right_edge: int = bbox[0] + bbox[2]
        self.motion = "still"
        self._reported_motion: str | None = None
        self._still_counter = 0
        self.frames_missing = 0
        self.color = PUPPET_COLORS[puppet_id % len(PUPPET_COLORS)]

    def update_bbox(self, bbox: tuple[int, int, int, int]):
        self.bbox = bbox
        self.frames_missing = 0
        bx, by, bw, bh = bbox
        center = bbox_center(bbox)
        right_edge = bx + bw
        self.trail.append(center)
        if self.prev_center is not None:
            dx = right_edge - self._prev_right_edge
            dy = center[1] - self.prev_center[1]
            self.velocities.append((dx, dy))
        self.prev_center = center
        self._prev_right_edge = right_edge
        raw = compute_motion(self.velocities, self.confirmed)

        if raw == "still":
            self._still_counter += 1
            if self._still_counter >= STILL_DELAY_FRAMES:
                self.motion = "still"
            # else keep previous non-still motion displayed
        else:
            self._still_counter = 0
            self.motion = raw

        if self.motion != self._reported_motion and self.motion != "still":
            self._reported_motion = self.motion
            if self.confirmed:
                print(f"   motion: [{self.id}] {self.confirmed} {self.motion}")
        elif self.motion == "still" and self._reported_motion != "still" and self._still_counter >= STILL_DELAY_FRAMES:
            self._reported_motion = "still"
            if self.confirmed:
                print(f"   motion: [{self.id}] {self.confirmed} still")

    def update_classification(self, label: str | None, conf: float):
        self.history.append(label if conf >= MIN_CONFIDENCE else None)
        votes = [v for v in self.history if v is not None]
        if votes:
            winner, count = Counter(votes).most_common(1)[0]
            stable = count / WINDOW_SIZE >= WIN_RATIO
        else:
            winner, stable = None, False

        if stable and winner != self.confirmed:
            self.confirmed = winner
            print(f">> detected: [{self.id}] {self.confirmed}")
        elif not stable and self.confirmed is not None:
            self.confirmed = None

    def mark_missing(self):
        self.frames_missing += 1

    @property
    def is_visible(self) -> bool:
        return self.frames_missing == 0

    @property
    def is_stale(self) -> bool:
        return self.frames_missing > TRACK_STALE_FRAMES

    def draw(self, display: np.ndarray):
        bx, by, bw, bh = self.bbox
        dh, dw = display.shape[:2]
        pad_x, pad_y = 80, 80
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(dw, bx + bw + pad_x)
        y2 = min(dh, by + bh + pad_y)
        cv2.rectangle(display, (x1, y1), (x2, y2), self.color, 2)
        cv2.putText(
            display,
            f"[{self.id}]",
            (x1 + 4, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.color,
            2,
        )

        pts = list(self.trail)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            c = tuple(int(ch * alpha) for ch in self.color)
            cv2.line(display, pts[i - 1], pts[i], c, max(1, int(3 * alpha)))


def _tracker_status_text(t: PuppetTracker) -> str:
    name = t.confirmed if t.confirmed else "..."
    if t.motion and t.motion != "still":
        return f"[{t.id}] {name} {t.motion}"
    return f"[{t.id}] {name}"


def _draw_label_badge(
    display: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    *,
    x: int,
    y: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    dh, dw = display.shape[:2]
    x = max(8, min(x, max(8, dw - tw - 12)))
    y = max(th + 10, min(y, max(th + 10, dh - baseline - 10)))
    pad = 6
    top_left = (x - pad, y - th - pad)
    bottom_right = (x + tw + pad, y + baseline + pad)
    cv2.rectangle(display, top_left, bottom_right, (0, 0, 0), -1)
    cv2.rectangle(display, top_left, bottom_right, color, 2)
    cv2.putText(display, text, (x, y), font, scale, color, thickness)


def draw_split_tracker_labels(display: np.ndarray, trackers: list[PuppetTracker], *, top_y: int = 35) -> None:
    """Render tracker labels in fixed top/bottom slots so two puppets stay readable."""
    if not trackers:
        return

    visible = [t for t in trackers if t.is_visible]
    ordered = sorted(visible, key=lambda t: t.id)[:2]
    if not ordered:
        return
    _draw_label_badge(
        display,
        _tracker_status_text(ordered[0]),
        ordered[0].color,
        x=10,
        y=top_y,
    )
    if len(ordered) >= 2:
        dh = display.shape[0]
        _draw_label_badge(
            display,
            _tracker_status_text(ordered[1]),
            ordered[1].color,
            x=10,
            y=dh - 20,
        )


# -- matching bboxes to existing trackers (nearest-neighbor) ------------------

def match_trackers(
    trackers: list[PuppetTracker],
    bboxes: list[tuple[int, int, int, int]],
    next_id: int,
) -> tuple[list[PuppetTracker], int]:
    used_trackers: set[int] = set()
    used_bboxes: set[int] = set()
    pairs: list[tuple[float, int, int]] = []

    for ti, t in enumerate(trackers):
        for bi, b in enumerate(bboxes):
            d = bbox_distance(t.bbox, b)
            if d < TRACK_MAX_DIST:
                pairs.append((d, ti, bi))

    pairs.sort()
    for d, ti, bi in pairs:
        if ti in used_trackers or bi in used_bboxes:
            continue
        trackers[ti].update_bbox(bboxes[bi])
        used_trackers.add(ti)
        used_bboxes.add(bi)

    for ti, t in enumerate(trackers):
        if ti not in used_trackers:
            t.mark_missing()

    for bi, b in enumerate(bboxes):
        if bi not in used_bboxes and len(trackers) < MAX_PUPPETS:
            trackers.append(PuppetTracker(next_id, b))
            next_id += 1

    trackers = [t for t in trackers if not t.is_stale]
    return trackers, next_id


# -- inference worker (round-robin across puppets) ----------------------------

def inference_worker(model, device, lock, shared):
    import time
    while shared["running"]:
        crops = shared.get("crops")
        if not crops:
            time.sleep(0.005)
            continue

        results = []
        for crop in crops:
            tensor = preprocess(crop).to(device)
            label, conf = classify(model, tensor)
            results.append((label, conf))

        with lock:
            shared["results"] = results
            shared["results_ready"] = True


# -- persistent predict state (for external callers) --------------------------

_predict_model = None
_predict_device = None
_predict_cap = None
_bg_started = False
_bg_running = False
_bg_debug = False

_frame_lock = Lock()
_result_lock = Lock()
_latest_frame = None
_latest_bboxes = []
_latest_label = ""
_latest_conf = 0.0
_latest_motion = "still"
_bg_trackers: list[PuppetTracker] = []
_bg_next_id = 0


def _open_cap() -> "cv2.VideoCapture | None":
    cam_idx = _resolve_camera_index()
    cap = cv2.VideoCapture(cam_idx)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    cap.release()
    print(f"error: webcam (index {cam_idx}) not found")
    return None


def _camera_loop():
    """Continuously capture frames, track motion, and update the debug window."""
    global _latest_frame, _latest_bboxes, _latest_motion, _bg_trackers, _bg_next_id

    while _bg_running:
        if _predict_cap is None or not _predict_cap.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = _predict_cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        bboxes, _ = find_silhouettes(frame)

        with _frame_lock:
            _latest_frame = frame
            _latest_bboxes = list(bboxes)

        # update puppet trackers for motion detection
        _bg_trackers, _bg_next_id = match_trackers(_bg_trackers, bboxes, _bg_next_id)
        visible_bg_trackers = [t for t in _bg_trackers if t.is_visible]
        motion = "still"
        if visible_bg_trackers:
            t = visible_bg_trackers[0]
            if t.motion and t.motion != "still":
                motion = t.motion
        with _result_lock:
            _latest_motion = motion

        if _bg_debug:
            with _result_lock:
                label, conf = _latest_label, _latest_conf
                cur_motion = _latest_motion

            display = frame.copy()
            for t in visible_bg_trackers:
                t.draw(display)
            draw_split_tracker_labels(display, visible_bg_trackers, top_y=72)
            result = label if conf >= MIN_CONFIDENCE else ""
            status = f"{result}  {conf:.0%}" if result else f"?  {conf:.0%}"
            if cur_motion != "still":
                status += f"  {cur_motion}"
            cv2.putText(display, status, (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.imshow("shadow debug", display)
            cv2.waitKey(1)


def _inference_loop():
    """Continuously classify the latest captured frame."""
    global _latest_label, _latest_conf

    while _bg_running:
        with _frame_lock:
            frame = _latest_frame
            bboxes = list(_latest_bboxes)

        if frame is None:
            time.sleep(0.05)
            continue

        if bboxes:
            bx, by, bw, bh = bboxes[0]
            crop = frame[max(0, by - 20): by + bh + 20,
                         max(0, bx - 20): bx + bw + 20]
        else:
            crop = frame

        tensor = preprocess(crop).to(_predict_device)
        label, conf = classify(_predict_model, tensor)

        with _result_lock:
            _latest_label = label
            _latest_conf = conf


def predict(debug: bool = False) -> tuple[str, str]:
    """Return (label, motion) for the current shadow puppet (non-blocking).

    label: animal name or "" if below confidence threshold.
    motion: e.g. "walking right slow", "jumping", "still".
    """
    global _predict_model, _predict_device, _predict_cap
    global _bg_started, _bg_running, _bg_debug

    if _predict_model is None:
        _predict_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _predict_model = load_model(_resolve_model_path()).to(_predict_device)

    if _predict_cap is None or not _predict_cap.isOpened():
        _predict_cap = _open_cap()
    if _predict_cap is None:
        return "", "still"

    _bg_debug = debug

    if not _bg_started:
        _bg_running = True
        _bg_started = True
        Thread(target=_camera_loop, daemon=True).start()
        Thread(target=_inference_loop, daemon=True).start()
        time.sleep(0.5)

    with _result_lock:
        label, conf = _latest_label, _latest_conf
        motion = _latest_motion

    return (label if conf >= MIN_CONFIDENCE else "", motion)


def cleanup() -> None:
    global _predict_cap, _bg_running, _bg_started
    _bg_running = False
    _bg_started = False
    if _predict_cap is not None:
        _predict_cap.release()
        _predict_cap = None
    cv2.destroyAllWindows()


# -- main loop ---------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(_resolve_model_path()).to(device)
    print(f"model loaded on {device}")

    cam_idx = _resolve_camera_index()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"error: cannot open camera index {cam_idx}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("press 'q' to quit")

    lock = Lock()
    shared = {
        "running": True,
        "crops": [],
        "results": [],
        "results_ready": False,
    }
    worker = Thread(target=inference_worker, args=(model, device, lock, shared), daemon=True)
    worker.start()

    trackers: list[PuppetTracker] = []
    next_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped = center_crop(frame, ZOOM)
        display = cv2.resize(cropped, (frame.shape[1], frame.shape[0]))

        bboxes, cur_thresh = find_silhouettes(display)
        trackers, next_id = match_trackers(trackers, bboxes, next_id)
        visible_trackers = [t for t in trackers if t.is_visible]

        crops = []
        if visible_trackers:
            for t in visible_trackers:
                bx, by, bw, bh = t.bbox
                # generous padding so the model sees the full puppet in context
                pad_x = max(20, bw // 4)
                pad_y = max(20, bh // 4)
                y1 = max(0, by - pad_y)
                x1 = max(0, bx - pad_x)
                y2 = min(display.shape[0], by + bh + pad_y)
                x2 = min(display.shape[1], bx + bw + pad_x)
                crops.append(display[y1:y2, x1:x2])
        else:
            crops.append(display)
        shared["crops"] = crops

        with lock:
            if shared["results_ready"]:
                results = shared["results"]
                shared["results_ready"] = False
                if visible_trackers:
                    for i, t in enumerate(visible_trackers):
                        if i < len(results):
                            t.update_classification(*results[i])
                elif results:
                    label, conf = results[0]
                    if conf >= MIN_CONFIDENCE:
                        cv2.putText(display, f"{label} {conf:.0%}", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        for t in visible_trackers:
            t.draw(display)
        draw_split_tracker_labels(display, visible_trackers, top_y=35)
        dh, dw = display.shape[:2]

        # debug inset (small)
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        _, debug_mask = cv2.threshold(gray, cur_thresh, 255, cv2.THRESH_BINARY_INV)
        small = cv2.resize(debug_mask, (dw // 6, dh // 6))
        small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        cv2.putText(small_bgr, f"t={cur_thresh}", (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
        sh, sw = small_bgr.shape[:2]
        display[dh - sh : dh, dw - sw : dw] = small_bgr

        cv2.imshow("shadow puppet", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    shared["running"] = False
    worker.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
