from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predict import predict

app = FastAPI(
    title="Diabetes Prediction API",
    description="Predicts diabetes diagnosis using a Decision Tree Classifier trained on the Pima Indians Diabetes dataset.",
    version="1.0.0",
)


class DiabetesInput(BaseModel):
    preg: float = Field(..., description="Number of pregnancies", example=6)
    plas: float = Field(..., description="Plasma glucose concentration", example=148.0)
    pres: float = Field(..., description="Diastolic blood pressure (mm Hg)", example=72.0)
    skin: float = Field(..., description="Triceps skin fold thickness (mm)", example=35.0)
    insu: float = Field(..., description="2-Hour serum insulin (mu U/ml)", example=0.0)
    mass: float = Field(..., description="Body mass index", example=33.6)
    pedi: float = Field(..., description="Diabetes pedigree function", example=0.627)
    age:  float = Field(..., description="Age in years", example=50.0)


class DiabetesResponse(BaseModel):
    prediction: int = Field(..., description="0 = No diabetes, 1 = Diabetes")
    label: str = Field(..., description="Human-readable result")


@app.get("/")
async def root():
    return {"message": "Diabetes Prediction API is running. Visit /docs for usage."}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=DiabetesResponse)
async def predict_diabetes(data: DiabetesInput):
    try:
        features = [
            data.preg, data.plas, data.pres, data.skin,
            data.insu, data.mass, data.pedi, data.age
        ]
        result = predict(features)
        label = "Diabetes detected" if result == 1 else "No diabetes detected"
        return DiabetesResponse(prediction=result, label=label)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model file not found. Please run train.py first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))