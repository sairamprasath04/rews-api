from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="REWS Hybrid Triage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_artifacts"

model = joblib.load(MODEL_DIR / "rews_risk_model.pkl")
features = joblib.load(MODEL_DIR / "rews_feature_order.pkl")


class REWSInput(BaseModel):
    age: int
    lives_alone: int

    o2_sat: float
    heart_rate: float
    systolic_bp: float

    o2_sat_missing: int
    heart_rate_missing: int
    systolic_bp_missing: int

    speech_distress: int
    heart_racing: int
    thunderclap_headache: int

    chest_pain: int
    shortness_of_breath: int
    heavy_bleeding: int
    confusion_drowsy: int

    fever: int
    tiredness: int
    bowel_change: int = 0
    sugar_imbalance: int = 0
    joint_stiff_swelling: int = 0
    skin_rash: int = 0
    low_mood: int = 0
    panic_attacks: int = 0
    sleep_disturb: int = 0
    smoking_status: int = 0

    duration_days: int = 0
    smoking: int = 0
    alcohol: int = 0


def map_to_ats(risk_tier: str) -> dict:
    if risk_tier == "HIGH":
        return {
            "ats_category": "ATS 1-2",
            "urgency": "Emergency / immediate review"
        }
    elif risk_tier == "MEDIUM":
        return {
            "ats_category": "ATS 3",
            "urgency": "Urgent / priority follow-up"
        }
    else:
        return {
            "ats_category": "ATS 4-5",
            "urgency": "Semi-urgent / routine care"
        }


def rule_override(data: REWSInput):
    reasons = []

    if data.heavy_bleeding == 1:
        reasons.append("Heavy bleeding red flag")
        return "HIGH", reasons

    if data.chest_pain == 1 and data.shortness_of_breath == 1:
        reasons.append("Chest pain with shortness of breath")
        return "HIGH", reasons

    if data.confusion_drowsy == 1 and data.fever == 1:
        reasons.append("Confusion/drowsiness with fever")
        return "HIGH", reasons

    if data.thunderclap_headache == 1:
        reasons.append("Thunderclap headache red flag")
        return "HIGH", reasons

    if data.speech_distress == 1 and data.o2_sat_missing == 0 and data.o2_sat < 93:
        reasons.append("Speech distress with low oxygen saturation")
        return "HIGH", reasons

    return None, []


def generate_reasons(data: REWSInput, pred: str):
    reasons = []

    if data.speech_distress == 1:
        reasons.append("Speech distress increased urgency")
    if data.chest_pain == 1:
        reasons.append("Chest pain contributed to higher risk")
    if data.o2_sat_missing == 0 and data.o2_sat < 93:
        reasons.append("Low oxygen saturation increased urgency")
    if data.fever == 1 and data.duration_days >= 3:
        reasons.append("Persistent fever increased concern")
    if data.heavy_bleeding == 1:
        reasons.append("Heavy bleeding requires urgent review")
    if data.age >= 75:
        reasons.append("Older age increased risk sensitivity")
    if data.tiredness == 1 and data.lives_alone == 1:
        reasons.append("Fatigue with limited home support increased priority")
    if data.sugar_imbalance == 1:
        reasons.append("Possible glucose instability contributed to risk")
    if data.confusion_drowsy == 1:
        reasons.append("Confusion/drowsiness raised clinical concern")

    if not reasons:
        reasons.append(f"{pred} risk predicted from combined symptom pattern")

    return reasons[:3]


@app.get("/")
def root():
    return {"message": "REWS Hybrid Triage API running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
def predict(data: REWSInput):
    try:
        override_pred, override_reasons = rule_override(data)
        if override_pred is not None:
            ats_info = map_to_ats(override_pred)
            return {
                "risk_tier": override_pred,
                "source": "rule_based_override",
                "ats_category": ats_info["ats_category"],
                "urgency": ats_info["urgency"],
                "probabilities": {
                    "HIGH": 1.0,
                    "MEDIUM": 0.0,
                    "LOW": 0.0
                },
                "reasons": override_reasons
            }

        data_dict = data.dict()

        data_dict["smoking_status"] = data_dict.get("smoking_status", data_dict.get("smoking", 0))

        missing_features = [feature for feature in features if feature not in data_dict]
        print("EXPECTED FEATURES:", features)
        print("MISSING FEATURES:", missing_features)

        values = [data_dict.get(feature, 0) for feature in features]
        print("FEATURE VECTOR:", values)

        x = np.array([values])
        pred = model.predict(x)[0]
        probs = model.predict_proba(x)[0]

        class_probs = {
            cls: round(float(prob), 3)
            for cls, prob in zip(model.classes_, probs)
        }

        ats_info = map_to_ats(pred)
        reasons = generate_reasons(data, pred)

        return {
            "risk_tier": pred,
            "source": "ml_model",
            "ats_category": ats_info["ats_category"],
            "urgency": ats_info["urgency"],
            "probabilities": class_probs,
            "reasons": reasons
        }

    except Exception as e:
        print("PREDICT ERROR:", str(e))
        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction failed",
                "detail": str(e)
            }
        )