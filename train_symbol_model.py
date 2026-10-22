import os
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

DATASET_PATH = "air_symbols"

X = []
y = []

label_map = {
    "plus": "+",
    "minus": "-",
    "mul": "*",
    "div": "/"
}

print("Loading dataset...")

def preprocess(img):

    blur = cv2.GaussianBlur(img,(5,5),0)

    _,thresh = cv2.threshold(blur,30,255,cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return None

    x,y,w,h = cv2.boundingRect(coords)

    symbol = thresh[y:y+h,x:x+w]

    if h > w:
        new_h = 20
        new_w = int(w*(20/h))
    else:
        new_w = 20
        new_h = int(h*(20/w))

    symbol = cv2.resize(symbol,(new_w,new_h))

    canvas = np.zeros((28,28),dtype=np.uint8)

    x_offset = (28-new_w)//2
    y_offset = (28-new_h)//2

    canvas[y_offset:y_offset+new_h,x_offset:x_offset+new_w] = symbol

    return canvas


for file in os.listdir(DATASET_PATH):

    if file.endswith(".png"):

        name = file.split("_")[0]

        if name in label_map:
            label = label_map[name]
        else:
            label = name

        path = os.path.join(DATASET_PATH,file)

        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

        img = preprocess(img)

        if img is None:
            continue

        img = img/255.0
        img = img.reshape(-1)

        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset shape:",X.shape)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

print("Training model...")

model = MLPClassifier(
    hidden_layer_sizes=(256,128),
    activation="relu",
    solver="adam",
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=500,
    verbose=True
)

model.fit(X_train,y_train)

print("Evaluating model...")

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

print("\nAccuracy:",accuracy)

print("\nClassification Report:")
print(classification_report(y_test,pred))

joblib.dump(model,"symbol_model.pkl")

print("\nModel saved as symbol_model.pkl")