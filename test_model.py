import cv2
import numpy as np
import joblib

# Load trained model
model = joblib.load("digit_model.pkl")

canvas = np.zeros((480,640), dtype=np.uint8)
drawing = False


def preprocess_digit(img):

    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return None

    x,y,w,h = cv2.boundingRect(coords)

    digit = thresh[y:y+h, x:x+w]

    if h > w:
        new_h = 20
        new_w = int(w * (20/h))
    else:
        new_w = 20
        new_h = int(h * (20/w))

    digit = cv2.resize(digit, (new_w,new_h))

    canvas28 = np.zeros((28,28), dtype=np.uint8)

    x_offset = (28-new_w)//2
    y_offset = (28-new_h)//2

    canvas28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

    return canvas28


def draw(event,x,y,flags,param):

    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas,(x,y),10,255,-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


cv2.namedWindow("Draw Digit")
cv2.setMouseCallback("Draw Digit",draw)

print("Draw digit with mouse")
print("Press P to predict")
print("Press C to clear")
print("Press Q to quit")

while True:

    cv2.imshow("Draw Digit",canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):

        processed = preprocess_digit(canvas)

        if processed is None:
            print("No digit detected")
            continue

        img = processed / 255.0
        img = img.reshape(1,-1)

        prediction = model.predict(img)

        print("Predicted Digit:",prediction[0])


    elif key == ord('c'):
        canvas[:] = 0


    elif key == ord('q'):
        break


cv2.destroyAllWindows()