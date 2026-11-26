# Heart Disease Predictor

A Flask web application that uses a Random Forest classifier to estimate whether a patient may have heart disease.

## Setup

```powershell
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

Open http://localhost:5000 in a browser.

## Files

- `heart.csv` — training dataset
- `train_model.py` — trains and saves the model
- `app.py` — Flask web application
- `templates/index.html` — prediction form and results
- `static/style.css` — responsive interface styles

## Important

This project is for educational purposes only and is not a medical diagnosis tool.
