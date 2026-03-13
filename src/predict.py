import pickle
import os


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "diabetes_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict(features: list) -> int:
    """
    Takes a list of 8 feature values and returns prediction:
      0 = tested_negative (no diabetes)
      1 = tested_positive (diabetes)
    """
    model = load_model()
    prediction = model.predict([features])
    return int(prediction[0])