"""
Streamlit frontend for Therapeutic Class Prediction.
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd

API_URL = "https://med-text-classifier.onrender.com"

st.set_page_config(
    page_title="Drug Therapeutic Classifier",
    page_icon="💊",
    layout="wide",
)


@st.cache_data(ttl=30)
def check_health():
    """Cached health check — only hits the API once every 30 seconds."""
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except requests.exceptions.ConnectionError:
        return None


@st.cache_data(ttl=60)
def fetch_classes():
    """Cached class list — only hits the API once every 60 seconds."""
    try:
        return requests.get(f"{API_URL}/classes", timeout=3).json()["classes"]
    except Exception:
        return None

# ── Session state defaults ──
if "result" not in st.session_state:
    st.session_state.result = None

# ── Header ──
st.title("💊 Drug Therapeutic Class Predictor")
st.markdown(
    "Enter drug information below and the ML model will predict its **therapeutic class** "
    "with confidence scores."
)

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ System Status")
    health = check_health()
    if health is None:
        st.error("API Offline — start the FastAPI backend first")
        st.code("uvicorn app:app --reload", language="bash")
    elif health["model_loaded"]:
        st.success(f"API Online — {health['classes']} classes loaded")
    else:
        st.warning("API running but model not loaded")

    st.divider()
    st.header("ℹ️ About")
    st.markdown(
        "This app uses a **Random Forest** classifier trained on **192K+ drug records** "
        "with TF-IDF text features to predict one of **22 therapeutic classes**."
    )

    with st.expander("View All Classes"):
        classes = fetch_classes()
        if classes:
            for i, cls in enumerate(classes, 1):
                st.text(f"{i:2d}. {cls}")
        else:
            st.info("Start the API to view classes.")

# ── Input Form ──
st.subheader("📝 Drug Information")

col1, col2 = st.columns(2)

with col1:
    drug_uses = st.text_area(
        "Drug Uses",
        placeholder="e.g., Treatment of bacterial infections including pneumonia, bronchitis, and UTIs",
        height=120,
        help="What conditions or diseases does this drug treat?",
        key="drug_uses",
    )
    drug_mechanism = st.text_area(
        "Mechanism of Action",
        placeholder="e.g., It works by killing bacteria. It inhibits cell wall synthesis.",
        height=120,
        help="How does the drug work in the body?",
        key="drug_mechanism",
    )

with col2:
    drug_contains = st.text_area(
        "Active Ingredients",
        placeholder="e.g., Amoxicillin (500mg)",
        height=120,
        help="What active ingredients does the drug contain?",
        key="drug_contains",
    )
    drug_benefits = st.text_area(
        "Benefits",
        placeholder="e.g., Effectively treats bacterial infections and prevents spread of infection",
        height=120,
        help="What are the therapeutic benefits?",
        key="drug_benefits",
    )

# ── Predict Button ──
st.divider()

if st.button("🔍 Predict Therapeutic Class", type="primary", use_container_width=True):
    if not all([drug_uses.strip(), drug_mechanism.strip(), drug_contains.strip(), drug_benefits.strip()]):
        st.warning("Please fill in all four fields before predicting.")
    else:
        with st.spinner("Analyzing drug information..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "drug_uses": drug_uses,
                        "drug_mechanism": drug_mechanism,
                        "drug_contains": drug_contains,
                        "drug_benefits": drug_benefits,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                st.session_state.result = response.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Make sure the FastAPI backend is running.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ── Display Results (from session state, survives reruns) ──
if st.session_state.result:
    result = st.session_state.result

    st.subheader("🎯 Prediction Result")

    res_col1, res_col2 = st.columns([2, 1])
    with res_col1:
        st.metric(label="Predicted Therapeutic Class", value=result["predicted_class"])
    with res_col2:
        st.metric(label="Confidence", value=f"{result['confidence']:.1f}%")

    # Top 3 predictions
    st.subheader("📊 Top 3 Predictions")
    top3 = result["top_3"]

    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        for i, pred in enumerate(top3):
            label = pred["therapeutic_class"]
            conf = pred["confidence"]
            icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            st.markdown(f"{icon} **{label}**")
            st.progress(conf / 100, text=f"{conf:.1f}%")

    with chart_col2:
        chart_data = pd.DataFrame({
            "Class": [p["therapeutic_class"] for p in top3],
            "Confidence (%)": [p["confidence"] for p in top3],
        }).set_index("Class")
        st.bar_chart(chart_data, horizontal=True)

# ── Quick Demo Section ──
st.divider()
with st.expander("🧪 Try a Quick Demo"):
    st.markdown("Click a button to auto-fill with sample drug data, then hit **Predict**.")

    DEMOS = {
        "💊 Antibiotic (Amoxicillin)": {
            "drug_uses": "Treatment of bacterial infections including pneumonia, bronchitis, urinary tract infections",
            "drug_mechanism": "It works by killing bacteria. It is a penicillin-type antibiotic",
            "drug_contains": "Amoxicillin (500mg)",
            "drug_benefits": "Effectively treats bacterial infections and prevents spread of infection",
        },
        "❤️ Heart Med (Amlodipine)": {
            "drug_uses": "Treatment of high blood pressure, heart failure, prevention of heart attack",
            "drug_mechanism": "Relaxes blood vessels and reduces workload on the heart",
            "drug_contains": "Amlodipine (5mg)",
            "drug_benefits": "Lowers blood pressure, reduces risk of stroke and heart attack",
        },
        "🩸 Diabetes (Metformin)": {
            "drug_uses": "Treatment of type 2 diabetes mellitus to control blood sugar levels",
            "drug_mechanism": "Increases insulin sensitivity, reduces glucose production in liver",
            "drug_contains": "Metformin Hydrochloride (500mg)",
            "drug_benefits": "Controls blood sugar levels and prevents diabetes complications",
        },
    }

    def fill_demo(data):
        """Callback — runs before widgets render, so it can set their keys."""
        for key in ["drug_uses", "drug_mechanism", "drug_contains", "drug_benefits"]:
            st.session_state[key] = data[key]
        st.session_state.result = None

    demo_cols = st.columns(len(DEMOS))
    for (name, data), col in zip(DEMOS.items(), demo_cols):
        with col:
            st.button(name, use_container_width=True, on_click=fill_demo, args=(data,))