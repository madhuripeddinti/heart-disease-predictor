from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_model.pkl"

FEATURES = [
    "age", "sex", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "ca", "smoking", "diabetes",
]

FIELD_LIMITS = {
    "age": (29, 76), "sex": (0, 1), "trestbps": (90, 200),
    "chol": (100, 600), "fbs": (0, 1), "restecg": (0, 2),
    "thalach": (60, 210), "exang": (0, 1), "ca": (0, 3),
    "smoking": (0, 1), "diabetes": (0, 1),
}

app = Flask(__name__)


def load_model():
    try:
        with MODEL_PATH.open("rb") as file:
            loaded_model = pickle.load(file)

        fitted_features = list(getattr(loaded_model, "feature_names_in_", []))
        if fitted_features and fitted_features != FEATURES:
            return None, (
                "The saved model was trained with old fields. "
                "Please stop Flask and run train_model.py again."
            )

        return loaded_model, None
    except FileNotFoundError:
        return None, "The model is not trained yet. Run train_model.py first."


model, model_error = load_model()


def validate_form(form):
    values = {}
    errors = []

    for field in FEATURES:
        raw_value = form.get(field, "").strip()
        if not raw_value:
            errors.append(f"Please enter {field.replace('_', ' ').title()}.")
            continue

        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"{field.replace('_', ' ').title()} must be a number.")
            continue

        minimum, maximum = FIELD_LIMITS[field]
        if value < minimum or value > maximum:
            errors.append(
                f"{field.replace('_', ' ').title()} must be between {minimum:g} and {maximum:g}."
            )
            continue

        values[field] = int(value) if field not in {"age", "trestbps", "chol", "thalach"} else value

    return values, errors


@app.route("/")
def home():
    return render_template("index.html", form_values={})


@app.route("/predict", methods=["POST"])
def predict():
    global model, model_error
    model, model_error = load_model()

    values, errors = validate_form(request.form)

    if errors:
        return render_template("index.html", form_values=request.form, errors=errors)

    input_data = pd.DataFrame([[values[field] for field in FEATURES]], columns=FEATURES)

    model, model_error = load_model()

    if model is None:
        return render_template(
            "index.html",
            form_values=request.form,
            errors=[model_error],
        )

    prediction = int(model.predict(input_data)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0][prediction]) * 100
    else:
        probability = None

    if prediction == 1:
        result = "Heart Disease Detected"
        result_class = "danger"
    else:
        result = "No Heart Disease Detected"
        result_class = "success"

    return render_template(
        "index.html",
        form_values=request.form,
        prediction_text=result,
        result_class=result_class,
        probability=probability,
    )


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
