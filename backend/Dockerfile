FROM python:3.11-slim

WORKDIR /app

# Install only backend dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    joblib \
    scikit-learn \
    numpy \
    pandas \
    nltk \
    scipy \
    pydantic

# Download NLTK data at build time
RUN python -c "\
import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('omw-1.4', quiet=True)"

# Copy app code and model artifacts
COPY app.py .
COPY saved_model/ ./saved_model/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]