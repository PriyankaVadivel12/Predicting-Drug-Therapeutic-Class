import joblib
import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# ── NLTK Downloads (run once) ──
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Global model store ──
model_store: dict = {}

SAVE_DIR = "saved_model"

# ── Text Preprocessing (mirrors training pipeline) ──
STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update([
    "tablet", "mg", "used", "treatment", "medicine", "doctor",
    "take", "may", "also", "use", "help", "one", "drug",
    "capsule", "injection", "dose", "patient", "ml",
])
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    if pd.isna(text) or text == "":
        return ""
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model artifacts...")
    model_store["model"] = joblib.load(f"{SAVE_DIR}/random_forest_model.joblib")
    model_store["tfidf"] = joblib.load(f"{SAVE_DIR}/tfidf_vectorizer.joblib")
    model_store["le_target"] = joblib.load(f"{SAVE_DIR}/label_encoder_target.joblib")
    model_store["metadata"] = joblib.load(f"{SAVE_DIR}/metadata.joblib")
    print(f"Model loaded — {model_store['metadata']['n_classes']} classes, "
          f"{model_store['metadata']['total_features']} features")
    yield
    model_store.clear()
    print("Model artifacts unloaded.")


# ── FastAPI App ──
app = FastAPI(
    title="Therapeutic Class Predictor",
    description="Predict the therapeutic class of a drug from its text description.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ──
class PredictionRequest(BaseModel):
    drug_uses: str = Field(..., min_length=1, description="What conditions the drug treats")
    drug_mechanism: str = Field(..., min_length=1, description="How the drug works")
    drug_contains: str = Field(..., min_length=1, description="Active ingredients")
    drug_benefits: str = Field(..., min_length=1, description="Therapeutic benefits")


class TopPrediction(BaseModel):
    therapeutic_class: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_3: list[TopPrediction]


# ── Endpoints ──
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "model" in model_store,
        "classes": model_store.get("metadata", {}).get("n_classes", 0),
    }


@app.get("/classes")
def get_classes():
    """Return all therapeutic classes the model can predict."""
    meta = model_store.get("metadata")
    if not meta:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"classes": meta["classes"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = model_store["model"]
    tfidf = model_store["tfidf"]
    le_target = model_store["le_target"]
    metadata = model_store["metadata"]

    # Preprocess
    combined = f"{req.drug_uses} {req.drug_mechanism} {req.drug_contains} {req.drug_benefits}"
    processed = preprocess_text(combined)

    # Feature engineering
    text_features = tfidf.transform([processed])
    n_cat = metadata["total_features"] - metadata["tfidf_features"]
    cat_dummy = csr_matrix(np.zeros((1, n_cat)))
    features = hstack([text_features, cat_dummy])

    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    predicted_class = le_target.inverse_transform([prediction])[0]
    top3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3 = [
        TopPrediction(
            therapeutic_class=le_target.inverse_transform([i])[0],
            confidence=round(float(probabilities[i]) * 100, 2),
        )
        for i in top3_idx
    ]

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=top_3[0].confidence,
        top_3=top_3,
    )