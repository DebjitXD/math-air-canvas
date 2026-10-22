import cv2
import numpy as np
import mediapipe as mp
import math
import os

label = input("Enter label (0-9 / plus / minus / mul / div): ")

save_dir = "air_symbols"
os.makedirs(save_dir, exist_ok=True)

count = 0

width = 1280
height = 720

canvas = np.zeros((height,width),dtype=np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.7)

prev_x,prev_y = 0,0

print("Pinch to draw")
print("Press S to save")
print("Press C to clear")
print("Press Q to quit")

while True:

    success,frame = cap.read()

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        lm = results.multi_hand_landmarks[0].landmark

        ix = int(lm[8].x*width)
        iy = int(lm[8].y*height)

        tx = int(lm[4].x*width)
        ty = int(lm[4].y*height)

        dist = math.hypot(ix-tx,iy-ty)

        if dist < 40:

            if prev_x == 0:
                prev_x,prev_y = ix,iy

            cv2.line(canvas,(prev_x,prev_y),(ix,iy),255,22)

            prev_x,prev_y = ix,iy

        else:
            prev_x,prev_y = 0,0

    overlay = cv2.addWeighted(frame,1,cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR),1,0)

    cv2.imshow("Collect Symbols",overlay)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):

        gray = canvas.copy()

        blur = cv2.GaussianBlur(gray,(5,5),0)

        _,thresh = cv2.threshold(blur,30,255,cv2.THRESH_BINARY)

        coords = cv2.findNonZero(thresh)

        if coords is None:
            print("Nothing drawn")
            continue

        x,y,w,h = cv2.boundingRect(coords)

        symbol = thresh[y:y+h,x:x+w]

        filename = f"{label}_{count}.png"

        cv2.imwrite(os.path.join(save_dir,filename),symbol)

        print("Saved:",filename)

        count += 1

        canvas[:] = 0

    elif key == ord('c'):

        canvas[:] = 0

    elif key == ord('q'):

        break


cap.release()
cv2.destroyAllWindows()