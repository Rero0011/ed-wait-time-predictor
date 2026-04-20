# ED Wait Time Predictor

 **Live App:** https://ed-wait-time-predictor.streamlit.app/

---

## Overview
This project predicts emergency department (ED) wait times using a machine learning model (**XGBoost**) and provides explainable insights through **SHAP**.

It demonstrates how predictive modelling and explainable AI can be applied to support decision-making in healthcare environments.

---

## Project Structure
- `app.py` – Streamlit dashboard
- `ed_wait_time_analysis.ipynb` – Data analysis and model development
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
  Synthetic dataset simulating emergency department conditions (Urgency, time, staffing, and facility factors)

---

  ## Project Context
  Developed as part of a final-year Computer Science dissertation focused on machine learning, explainable AI, and real-world system prototyping.

---

##  Run Locally
pip install -r requirements.txt
streamlit run app.py

---

## Author
Rerosuoghene Eppiah
