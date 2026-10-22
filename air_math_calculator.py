import cv2
import numpy as np
import mediapipe as mp
import math
import joblib

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
WIDTH        = 1280
HEIGHT       = 720
STROKE_THICK = 12
PINCH_DIST   = 40

# Zone layout – horizontal slices of the frame
# LEFT digit  : 0 %  → 33 %
# MIDDLE op   : 35 % → 65 %
# RIGHT digit : 67 % → 100 %
ZONES = {
    "left":   (int(WIDTH * 0.00), int(WIDTH * 0.33)),
    "middle": (int(WIDTH * 0.35), int(WIDTH * 0.65)),
    "right":  (int(WIDTH * 0.67), int(WIDTH * 1.00)),
}

# Box height: boxes are centred vertically and occupy this fraction of HEIGHT
BOX_HEIGHT_FRAC = 0.55
BOX_TOP    = int(HEIGHT * (1 - BOX_HEIGHT_FRAC) / 2)     # e.g. y=162
BOX_BOTTOM = int(HEIGHT * (1 + BOX_HEIGHT_FRAC) / 2)     # e.g. y=558

ZONE_LABELS = {
    "left":   "DIGIT  (digit_model)",
    "middle": "OPERATOR  (symbol_model)",
    "right":  "DIGIT  (digit_model)",
}

ZONE_COLORS = {
    "left":   (50,  200, 255),   # amber
    "middle": (50,  255, 120),   # green
    "right":  (50,  200, 255),   # amber
}

# ──────────────────────────────────────────
# LOAD MODELS
# ──────────────────────────────────────────
digit_model  = joblib.load("digit_model.pkl")
symbol_model = joblib.load("symbol_model.pkl")

# ──────────────────────────────────────────
# CANVAS + CAMERA
# ──────────────────────────────────────────
canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# ──────────────────────────────────────────
# MEDIAPIPE
# ──────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# ──────────────────────────────────────────
# STATE
# ──────────────────────────────────────────
prev_x, prev_y  = None, None
is_drawing       = False
active_zone      = None
expression_text  = ""
result_text      = ""


# ──────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────
def get_zone(x):
    """Return zone name for x-coordinate, or None if in a gap."""
    for name, (x0, x1) in ZONES.items():
        if x0 <= x <= x1:
            return name
    return None


def crop_zone(full_canvas, zone):
    x0, x1 = ZONES[zone]
    return full_canvas[BOX_TOP:BOX_BOTTOM, x0:x1]


def preprocess(img):
    """
    Normalise a cropped symbol to 28×28 preserving aspect ratio.
    Returns None if the region is too small or empty.
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.erode(thresh,  kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    if w < 10 or h < 10:
        return None

    symbol = thresh[y: y + h, x: x + w]

    # Resize keeping aspect ratio inside 20×20
    if h >= w:
        new_h = 20
        new_w = max(1, int(w * 20 / h))
    else:
        new_w = 20
        new_h = max(1, int(h * 20 / w))

    symbol = cv2.resize(symbol, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas28[y_off: y_off + new_h, x_off: x_off + new_w] = symbol
    return canvas28


def predict_zone(zone):
    """
    Segment strokes inside a zone, preprocess and predict.
    Uses digit_model for left/right, symbol_model for middle.
    Returns predicted label string or None.
    """
    zone_img = crop_zone(canvas, zone)

    blur = cv2.GaussianBlur(zone_img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    # Dilate to merge nearby strokes before contour detection
    kernel  = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # One symbol per zone – use the largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    if w * h < 300:
        return None

    symbol_img = thresh[y: y + h, x: x + w]   # from original thresh, not dilated
    processed  = preprocess(symbol_img)
    if processed is None:
        return None

    img_flat = (processed / 255.0).reshape(1, -1)

    if zone == "middle":
        pred = symbol_model.predict(img_flat)[0]
    else:
        pred = digit_model.predict(img_flat)[0]

    print(f"[INFO] Zone '{zone}' → '{pred}'")
    return str(pred)


def safe_eval(expr):
    allowed = set("0123456789+-*/. ")
    if not expr or not all(c in allowed for c in expr):
        return "Invalid"
    try:
        result = eval(expr)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(round(result, 6))
    except ZeroDivisionError:
        return "Div/0 Error"
    except Exception:
        return "Error"


# ──────────────────────────────────────────
# DRAW ZONE OVERLAY
# ──────────────────────────────────────────
def draw_zones(frame, active):
    for name, (x0, x1) in ZONES.items():
        color     = ZONE_COLORS[name]
        thickness = 3 if name == active else 1
        alpha     = 0.13 if name == active else 0.05

        # Tinted fill (only inside the box)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, BOX_TOP), (x1, BOX_BOTTOM), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Border
        cv2.rectangle(frame, (x0, BOX_TOP), (x1, BOX_BOTTOM), color, thickness)

        # Label just above the box
        label = ZONE_LABELS[name]
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        lx = x0 + ((x1 - x0) - tw) // 2
        cv2.putText(frame, label, (lx, BOX_TOP - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        # Faint big background letter centred inside the box
        big = {"left": "A", "middle": "OP", "right": "B"}[name]
        (bw, bh), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 4)
        bx = x0 + ((x1 - x0) - bw) // 2
        by = BOX_TOP + (BOX_BOTTOM - BOX_TOP) // 2 + bh // 2
        faint = tuple(int(c * 0.18) for c in color)
        cv2.putText(frame, big, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, faint, 4, cv2.LINE_AA)


# ──────────────────────────────────────────
# HUD (bottom bar)
# ──────────────────────────────────────────
def draw_hud(frame, expr, result, drawing):
    panel_y = HEIGHT - 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (WIDTH, HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    state_color = (0, 255, 100) if drawing else (120, 120, 200)
    state_txt   = "DRAWING" if drawing else "HOVER"
    cv2.putText(frame, state_txt, (20, panel_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Expr: {expr}", (200, panel_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Result: {result}", (750, panel_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "[P] Predict   [C] Clear   [Q] Quit",
                (20, panel_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)


# ──────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────
print("3-Zone Air Math Calculator")
print("  LEFT  zone → digit_model.pkl")
print("  MIDDLE zone → symbol_model.pkl  (operator)")
print("  RIGHT  zone → digit_model.pkl")
print("P = predict  |  C = clear  |  Q = quit")

while True:
    success, frame = cap.read()
    if not success:
        print("[ERROR] Camera read failed")
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark

        ix = int(lm[8].x * WIDTH)
        iy = int(lm[8].y * HEIGHT)
        tx = int(lm[4].x * WIDTH)
        ty = int(lm[4].y * HEIGHT)

        dist        = math.hypot(ix - tx, iy - ty)
        active_zone = get_zone(ix)

        inside_box = BOX_TOP <= iy <= BOX_BOTTOM
        if dist < PINCH_DIST and active_zone is not None and inside_box:
            is_drawing = True
            if prev_x is None:
                prev_x, prev_y = ix, iy
            cv2.line(canvas, (prev_x, prev_y), (ix, iy), 255, STROKE_THICK)
            prev_x, prev_y = ix, iy
        else:
            is_drawing = False
            prev_x, prev_y = None, None

        mp_draw.draw_landmarks(
            frame,
            res.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
        )

        dot_color = (ZONE_COLORS[active_zone]
                     if (is_drawing and active_zone) else (120, 120, 200))
        cv2.circle(frame, (ix, iy), 10, dot_color, -1)

    else:
        is_drawing   = False
        active_zone  = None
        prev_x, prev_y = None, None

    # Merge strokes onto camera frame
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    frame      = cv2.addWeighted(frame, 1.0, canvas_bgr, 1.0, 0)

    draw_zones(frame, active_zone)
    draw_hud(frame, expression_text, result_text, is_drawing)

    cv2.imshow("Air Math Calculator - 3 Zone", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        left_pred  = predict_zone("left")
        op_pred    = predict_zone("middle")
        right_pred = predict_zone("right")

        print(f"[RESULT] left={left_pred}  op={op_pred}  right={right_pred}")

        if left_pred and op_pred and right_pred:
            expression_text = f"{left_pred} {op_pred} {right_pred}"
            result_text     = safe_eval(f"{left_pred}{op_pred}{right_pred}")
        else:
            missing = []
            if not left_pred:  missing.append("left digit")
            if not op_pred:    missing.append("operator")
            if not right_pred: missing.append("right digit")
            expression_text = "Missing: " + ", ".join(missing)
            result_text     = ""

    elif key == ord('c'):
        canvas[:] = 0
        expression_text = ""
        result_text     = ""
        print("[INFO] Canvas cleared")

    elif key == ord('q'):
        print("[INFO] Quitting")
        break

cap.release()
cv2.destroyAllWindows()