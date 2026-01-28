# Diabetes Prediction & Analysis App (XAI) 🚀

A full-stack medical AI application that predicts diabetes risk using XGBoost and provides deep insights using Explainable AI (SHAP).

![App Preview](https://diabetes-risk-app.onrender.com/static/low.jpg) <!-- Example placeholder -->

## 🌟 Key Features

*   **Premium Glassmorphism UI**: Modern, sleek interface with interactive elements.
*   **Explainable AI (XAI)**:
    *   **Waterfall Plots**: Instant visualization of the top 3 risk drivers.
    *   **Complete Risk Breakdown**: Full Bar Chart analysis of all 14 clinical features.
*   **AI Health Coach**: Dynamic counterfactual advice to help users lower their risk.
*   **Metabolic Score**: A custom composite indicator for rapid health assessment.
*   **User History**: Secure historical tracking of assessments.
*   **Mobile Responsive**: Optimized for all devices.

## 🛠️ Tech Stack

*   **Frontend**: HTML5, Vanilla CSS (Glassmorphism), Chart.js
*   **Backend**: Python, Flask
*   **Intelligence**: XGBoost, SHAP (Explainable AI)
*   **Database**: SQLite

## 🚀 Deployment

The app is optimized for **Render.com** and includes:
- `Procfile` for Gunicorn deployment.
- `requirements.txt` with pinned versions for environment stability.
- Robust cross-platform model loading (Windows to Linux fix).

## 📖 How it Works

The application takes 14 health indicators (BMI, BP, activity level, etc.) and processes them through an XGBoost model. 

Individual risk is explained using **SHAP (SHapley Additive exPlanations)**, which assigns a mathematical "impact" score to each feature. This tells the user exactly *why* their score is high or low.

## 📝 Running Locally

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   python app.py
   ```
3. Open `http://localhost:5000` in your browser.

---
*Created with ❤️ for health awareness.*
