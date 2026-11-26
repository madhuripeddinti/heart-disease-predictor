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

## Deploy on Render

1. Open https://dashboard.render.com and sign in.
2. Select **New → Blueprint**.
3. Connect the `madhuripeddinti/heart-disease-predictor` GitHub repository.
4. Render will detect `render.yaml` and create the web service.
5. Wait for the build and deployment to finish, then open the generated Render URL.

Alternatively, select **New → Web Service**, connect the same repository, and use:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Health check path: `/`
