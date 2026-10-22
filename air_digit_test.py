import cv2
import numpy as np
import mediapipe as mp
import math
import joblib

# Load trained model
model = joblib.load("digit_model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Canvas
width, height = 1280, 720
canvas = np.zeros((height, width), dtype=np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

prev_x, prev_y = 0, 0


def preprocess_digit(img):

    blur = cv2.GaussianBlur(img,(5,5),0)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    digit = thresh[y:y+h, x:x+w]

    if h > w:
        new_h = 20
        new_w = int(w * (20/h))
    else:
        new_w = 20
        new_h = int(h * (20/w))

    digit = cv2.resize(digit, (new_w, new_h))

    canvas28 = np.zeros((28,28), dtype=np.uint8)

    x_offset = (28-new_w)//2
    y_offset = (28-new_h)//2

    canvas28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

    return canvas28


print("Pinch to draw")
print("P = predict digit")
print("C = clear canvas")
print("Q = quit")

while True:

    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        lm = results.multi_hand_landmarks[0].landmark

        ix = int(lm[8].x * width)
        iy = int(lm[8].y * height)

        tx = int(lm[4].x * width)
        ty = int(lm[4].y * height)

        dist = math.hypot(ix - tx, iy - ty)

        if dist < 40:

            if prev_x == 0:
                prev_x, prev_y = ix, iy

            cv2.line(canvas, (prev_x, prev_y), (ix, iy), 255, 15)

            prev_x, prev_y = ix, iy

        else:
            prev_x, prev_y = 0, 0

    overlay = cv2.addWeighted(frame, 1, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), 1, 0)

    cv2.imshow("Air Digit Recognition", overlay)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):

        processed = preprocess_digit(canvas)

        if processed is None:
            print("No digit detected")
            continue

        img = processed / 255.0
        img = img.reshape(1, -1)

        pred = model.predict(img)

        print("Predicted Digit:", pred[0])

    elif key == ord('c'):
        canvas[:] = 0

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()