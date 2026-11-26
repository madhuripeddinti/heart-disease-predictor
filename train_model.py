from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart.csv"
MODEL_PATH = BASE_DIR / "heart_model.pkl"
TARGET_COLUMN = "heart_disease"
REMOVED_FEATURES = ["oldpeak", "slope", "bmi", "thal", "cp"]


def main():
    data = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X = data.drop(columns=[TARGET_COLUMN] + REMOVED_FEATURES)
    y = data[TARGET_COLUMN]

    if X.isnull().sum().sum() > 0:
        raise ValueError("Training data contains missing values")

    if y.nunique() < 2:
        raise ValueError("Target column must contain at least two classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Heart Disease Class Distribution:")
    print(y.value_counts().sort_index())
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"\nModel Training Completed! Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)

    print(f"\nModel saved successfully at {MODEL_PATH}")


if __name__ == "__main__":
    main()
