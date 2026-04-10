# Predicting Drug Therapeutic Classes Using Machine Learning

Multi class text classification model that predicts the therapeutic class of a drug using textual and categorical features from the Medicines Information Dataset (192K+ rows, 22 classes).

## About

Every drug belongs to a therapeutic class based on what it treats or how it works. Classifying them manually is slow and inconsistent at scale. This project uses machine learning to automate that process by learning from drug descriptions, side effects, ingredients, and other textual information.

## Dataset

**Source:** [Kaggle — Medical Information Dataset](https://www.kaggle.com/datasets/imtkaggleteam/medical-information-dataset)

- 192,000+ rows, 15 columns
- 22 therapeutic classes (after cleaning up duplicate labels)
- Key text features: `Contains`, `ProductUses`, `HowWorks`, `ProductBenefits`, `SideEffect`
- Target variable: `Therapeutic_Class`

## What We Did

1. **Data Cleaning** — Merged duplicate class labels (44 → 22), removed duplicates, handled missing values, stripped HTML artifacts
2. **EDA** — Visualized class distribution, text lengths, and word clouds per class
3. **NLP Preprocessing** — Lowercased, tokenized, removed stopwords, lemmatized, and combined multiple text columns into one feature
4. **Feature Engineering** — Applied TF-IDF vectorization on text and encoded categorical columns
5. **Model Training** — Trained Logistic Regression, Naive Bayes, SVM, Random Forest, and XGBoost using stratified 80/20 split
6. **Evaluation** — Compared models using weighted F1 score, precision, recall, and confusion matrices

## Best Result

**Random Forest — 99.79% accuracy, 0.9979 weighted F1 score**

## Tech Stack

Python, scikit-learn, XGBoost, NLTK, Matplotlib, Seaborn, WordCloud, Jupyter Notebook, Streamlit

## Getting Started

```bash
git clone https://github.com/PriyankaVadivel12/Predicting-Drug-Therapeutic-Class.git
cd Predicting-Drug-Therapeutic-Class
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/medical-information-dataset) and place it in the `data/` directory, then run the notebook.

