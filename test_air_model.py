import cv2
import numpy as np
import joblib

# Load trained model
model = joblib.load("digit_model.pkl")

# Open camera
cap = cv2.VideoCapture(0)

print("Press S to capture a digit frame")
print("Press Q to quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # Capture frame
    if key == ord('s'):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Normalize
        img = gray / 255.0

        # Flatten to match training shape
        img = img.reshape(1, -1)

        prediction = model.predict(img)

        print("Predicted Digit:", prediction[0])

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()