import pandas as pd
from sklearn.datasets import fetch_openml


def load_data():
    """
    Load the Pima Indians Diabetes dataset from OpenML.
    Returns features (X) and labels (y).
    """
    dataset = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
    X = dataset.data
    y = (dataset.target == "tested_positive").astype(int)
    return X, y


FEATURE_NAMES = [
    "preg",           # Number of pregnancies
    "plas",           # Plasma glucose concentration
    "pres",           # Diastolic blood pressure (mm Hg)
    "skin",           # Triceps skin fold thickness (mm)
    "insu",           # 2-Hour serum insulin (mu U/ml)
    "mass",           # Body mass index (weight/height^2)
    "pedi",           # Diabetes pedigree function
    "age",            # Age (years)
]