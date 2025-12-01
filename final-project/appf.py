from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostClassifier

# --- Загружаем модель ---
model = CatBoostClassifier()
model.load_model("model(catbost)/credit_default_model.cbm")  # твоя модель

# --- FastAPI ---
app = FastAPI(title="Credit Default Prediction API")

# --- Проверка работы API ---
@app.get("/")
def root():
    return {"message": "API работает! Перейди на /docs для Swagger UI"}

# --- Входные данные ---
class CreditData(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

# --- Эндпоинт предсказания ---
@app.post("/predict")
def predict(data: CreditData):
    input_df = pd.DataFrame([data.dict()])
    cat_features = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file"
    ]

    pred = model.predict(input_df)
    proba = model.predict_proba(input_df)
    probability = float(proba[0][1])

    # Градация риска
    if probability <= 0.10:
        risk = "Низкий"
    elif probability <= 0.30:
        risk = "Средний"
    else:
        risk = "Высокий"

    return {
        "prediction": int(pred[0]),
        "probability_default": probability,
        "risk_level": risk
    }