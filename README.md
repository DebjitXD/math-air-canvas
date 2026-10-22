# ✋ Math Air Canvas

Draw digits and math symbols **in the air** with your finger. A webcam tracks your hand,
a trained machine learning model recognises what you drew, and the app calculates the answer.

![Demo](demo.gif)
<!-- Add a screenshot or GIF named demo.gif to the repo, or delete this line -->

## What it does
1. Tracks your hand using **MediaPipe**.
2. Draws your fingertip movement on a virtual canvas using **OpenCV**.
3. Recognises each digit (0-9) and symbol (e.g. +, -, ×, ÷) using models trained with **scikit-learn**.
4. Calculates the expression and shows the result on screen.

## Project structure

| File / Folder | Purpose |
|---|---|
| `air_math_calculator.py` | **Main app.** Run this to use the calculator |
| `collect_digits.py` | Records training images for digits (0-9) |
| `collect_symbols.py` | Records training images for math symbols |
| `train_air_model.py` | Trains the digit model, saves `digit_model.pkl` |
| `train_symbol_model.py` | Trains the symbol model, saves `symbol_model.pkl` |
| `test_air_model.py`, `test_model.py`, `air_digit_test.py` | Test scripts to check model accuracy |
| `digit_model.pkl`, `symbol_model.pkl` | Trained models (ready to use) |
| `air_digits/`, `air_symbols/` | Training images collected by us |
| `demo_model_creation/` | Sample images showing how the dataset looks |

## Requirements
- **Python 3.10** (recommended)
- A **webcam**
- Good lighting and a plain background help hand tracking

## Installation

**1. Download the project**
```bash
git clone https://github.com/DebjitXD/math-air-canvas.git
cd math-air-canvas
```
(Or click the green **Code** button → **Download ZIP**, then extract it.)

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate
```
On Mac/Linux, use `source venv/bin/activate` instead.

**3. Install the libraries**
```bash
pip install -r requirements.txt
```

## How to use

### Option 1: Just run the calculator (models are included)
```bash
python air_math_calculator.py
```
1. Allow camera access if asked.
2. Raise your hand in front of the webcam.
3. **[ ] Describe your gesture here**, e.g. "Point with your index finger to draw, close your fist to stop drawing."
4. Draw a digit, then a symbol, then another digit (e.g. `5`, `+`, `3`).
5. **[ ] Describe how to get the result**, e.g. "Show an open palm to calculate."
6. **[ ] Describe how to clear and quit**, e.g. "Press `c` to clear, `q` to quit."

### Option 2: Build your own dataset and train the models
1. Collect digit images:
```bash
   python collect_digits.py
```
2. Collect symbol images:
```bash
   python collect_symbols.py
```
3. Train the models:
```bash
   python train_air_model.py
   python train_symbol_model.py
```
   This creates new `digit_model.pkl` and `symbol_model.pkl` files.
4. Test the models:
```bash
   python test_air_model.py
   python test_model.py
```

## Results
- Digit model accuracy: **[ ]%**
- Symbol model accuracy: **[ ]%**
- Dataset size: about 1,080 digit images and 230 symbol images

## Troubleshooting

| Problem | Fix |
|---|---|
| Camera does not open | Close other apps using the webcam (Zoom, Meet), or try a different camera index in the code |
| Hand not detected | Improve the lighting and keep your whole hand inside the frame |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again inside your virtual environment |
| Model file error | Make sure you use the same scikit-learn version listed in `requirements.txt` |

## Tech stack
Python · OpenCV · MediaPipe · NumPy · scikit-learn

## Team
- **Debjit Biswas** - [@DebjitXD](https://github.com/DebjitXD)
- **Sristi Seal** - [@sealsristi](https://github.com/sealsristi)

## License
Released under the [MIT License](LICENSE).
