import cv2
import numpy as np
import mediapipe as mp
import os
import math

# Create dataset folder
save_dir = "air_digits"
os.makedirs(save_dir, exist_ok=True)

digit_label = input("Enter digit label (0-9): ")

cap = cv2.VideoCapture(0)

# Get camera size dynamically
success, img = cap.read()
h, w, _ = img.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

prev_x, prev_y = 0, 0
count = 0

print("Pinch to draw.")
print("Press S to save digit.")
print("Press C to clear.")
print("Press Q to quit.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        index_tip = landmarks[8]
        thumb_tip = landmarks[4]

        ix, iy = int(index_tip.x*w), int(index_tip.y*h)
        tx, ty = int(thumb_tip.x*w), int(thumb_tip.y*h)

        dist = math.hypot(ix - tx, iy - ty)

        # Pinch to draw
        if dist < 40:
            if prev_x == 0:
                prev_x, prev_y = ix, iy

            cv2.line(canvas, (prev_x, prev_y), (ix, iy), (255,255,255), 10)
            prev_x, prev_y = ix, iy
        else:
            prev_x, prev_y = 0, 0

    # Combine camera + canvas
    display = cv2.addWeighted(img, 1, canvas, 1, 0)
    cv2.imshow("Collect Digits", display)

    # IMPORTANT: Only ONE waitKey call
    key = cv2.waitKey(1) & 0xFF

    if key != 255:
        print("Key pressed:", key)

    if key == ord('s'):
        print("Saving...")
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        filename = f"{digit_label}_{count}.png"
        cv2.imwrite(os.path.join(save_dir, filename), thresh)
        print("Saved:", filename)
        count += 1
        canvas[:] = 0

    elif key == ord('c'):
        print("Cleared")
        canvas[:] = 0

    elif key == ord('q'):
        print("Exit")
        break

cap.release()
cv2.destroyAllWindows()