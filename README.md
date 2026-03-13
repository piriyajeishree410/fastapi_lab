# Diabetes Prediction API

A FastAPI-based REST API that predicts diabetes diagnosis using a Decision Tree Classifier trained on the **Pima Indians Diabetes dataset**.

> Adapted from [FastAPI Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs) — modified to use a healthcare dataset and a diabetes prediction use case.

---

## Project Structure

```
fastapi_diabetes/
├── model/
│   └── diabetes_model.pkl
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   ├── predict.py
│   └── main.py
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

---

## Running the API

```bash
cd src

# 1. Train the model
python train.py

# 2. Start the API server
uvicorn main:app --reload
```

Visit **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

---

## API Endpoints

| Method | Endpoint   | Description              |
|--------|------------|--------------------------|
| GET    | `/`        | Health check message     |
| GET    | `/health`  | API status               |
| POST   | `/predict` | Run diabetes prediction  |

### Example Request Body (`/predict`)

```json
{
  "preg": 6,
  "plas": 148.0,
  "pres": 72.0,
  "skin": 35.0,
  "insu": 0.0,
  "mass": 33.6,
  "pedi": 0.627,
  "age": 50.0
}
```

### Example Response

```json
{
  "prediction": 1,
  "label": "Diabetes detected"
}
```

---

# Sample Output

![API Prediction Output](../images/output_image.png)

The screenshot above shows a successful `POST /predict` request via Swagger UI, returning `"prediction": 1` and `"label": "Diabetes detected"` with HTTP status **200 OK**.

---

## Dataset

The [Pima Indians Diabetes Dataset](https://www.openml.org/d/37) contains diagnostic measurements for 768 female patients. The target variable is whether the patient tested positive for diabetes.

**Features:**
- `preg` — Number of pregnancies
- `plas` — Plasma glucose concentration
- `pres` — Diastolic blood pressure
- `skin` — Triceps skin fold thickness
- `insu` — 2-hour serum insulin
- `mass` — Body mass index
- `pedi` — Diabetes pedigree function
- `age` — Age in years