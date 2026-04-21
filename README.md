# ED Wait Time Predictor

 **Live App:** https://ed-wait-time-predictor.streamlit.app/

---

## Overview
This project predicts emergency department (ED) wait times using a machine learning model (**XGBoost**) and provides explainable insights through **SHAP**.

It demonstrates how predictive modelling and explainable AI can be applied to support decision-making in healthcare environments.

---

## Project Structure
- `app.py` – Streamlit dashboard
- `/notebooks/` – Data analysis and model development
- `xgb_model.pkl` – Trained model
- `model_columns.pkl` – Feature structure

---

## Features
-  Wait time prediction
-  Explainable AI (SHAP)
-  Interactive Streamlit dashboard
-  Scenario-based inputs

---

## Tech Stack
- Python
- XGBoost
- SHAP
- Streamlit
- Pandas / NumPy

---

## Dataset

The dataset used in this project is included in this repository.

It is a synthetic dataset representing emergency department conditions and was originally sourced from Kaggle:
https://www.kaggle.com/datasets/rivalytics/er-wait-time

---

  ## Project Context
  Developed as part of a final-year Computer Science dissertation focused on machine learning, explainable AI, and real-world system prototyping.

---

##  Run Locally
- pip install -r requirements.txt
- streamlit run app.py

---

## Running the Project (Google Colab)

To run the notebook in Google Colab:

1. Upload the notebook file from the `/notebook` directory.
2. Upload the dataset file into the same Colab working directory.
3. Ensure the dataset file name matches the one used in the notebook.
4. Run all cells in order.

Note: Both the notebook and dataset must be in the same directory when running in Colab.

---
## Author
Rerosuoghene Eppiah
