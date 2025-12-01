from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostClassifier

app = FastAPI(title="Credit Default Prediction API")

# Подключаем папку шаблонов и статику
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Загружаем модель
model = CatBoostClassifier()
model.load_model("model(catbost)/credit_default_model.cbm")

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

# HTML фронтенд
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Предсказание
@app.post("/predict")
def predict(data: CreditData):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)
    proba = model.predict_proba(df)
    probability = float(proba[0][1])

    if probability <= 0.10:
        risk = "Низкий"
    elif probability <= 0.30:
        risk = "Средний"
    else:
        risk = "Высокий"

    return {"prediction": int(pred[0]), "probability_default": probability, "risk_level": risk}