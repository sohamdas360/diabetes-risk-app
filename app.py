import flask
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import xgboost
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime

# Set Matplotlib backend to Agg to allow saving images to buffer
matplotlib.use('Agg')

app = flask.Flask(__name__)
app.secret_key = 'super_secret_key_change_this_for_production'

# --- DATABASE SETUP ---
DB_NAME = "users.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    
    # Records Table (History)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            risk_score REAL NOT NULL,
            verdict TEXT NOT NULL,
            top_factors TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized.")

init_db()

# --- LOGIN MANAGER SETUP ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    if user_data:
        return User(id=user_data[0], username=user_data[1], password_hash=user_data[2])
    return None

# Load New HbA1c-Augmented Model Bundle
model = None
model_columns = []
threshold = 0.44

try:
    with open('best_model_augmented.pkl', 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    threshold = bundle['threshold']
    model_columns = bundle['features']
    print(f"HbA1c Model loaded. Threshold={threshold}, Features={len(model_columns)}")
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize SHAP Explainer
explainer = None
if model:
    try:
        explainer = shap.TreeExplainer(model)
        print("SHAP explainer initialized.")
    except Exception as e:
        print(f"Error initializing SHAP: {e}")

# --- ROUTES ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    if flask.request.method == 'POST':
        username = flask.request.form['username']
        password = flask.request.form['password']
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # GLOBAL USER LIMIT CHECK (Max 3)
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count >= 3:
            flask.flash("Registration limits reached (Max 3 users).")
            conn.close()
            return flask.redirect(flask.url_for('login'))
        
        # Check if user exists
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            flask.flash("Username already exists!")
            conn.close()
            return flask.redirect(flask.url_for('register'))
        
        # Create new user
        hashed_pw = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        conn.close()
        
        flask.flash("Account created! Please login.")
        return flask.redirect(flask.url_for('login'))
        
    return flask.render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask.request.method == 'POST':
        username = flask.request.form['username']
        password = flask.request.form['password']
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user_data = cursor.fetchone()
        conn.close()
        
        if user_data and check_password_hash(user_data[2], password):
            user = User(id=user_data[0], username=user_data[1], password_hash=user_data[2])
            login_user(user)
            return flask.redirect(flask.url_for('dashboard'))
        else:
            flask.flash("Invalid username or password")
            return flask.redirect(flask.url_for('login'))
            
    return flask.render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return flask.redirect(flask.url_for('login'))

@app.route('/', methods=['GET'])
def index():
    if current_user.is_authenticated:
        return flask.redirect(flask.url_for('dashboard'))
    return flask.render_template('index.html')

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    # Fetch History
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT date, verdict, risk_score FROM records WHERE user_id = ? ORDER BY id DESC", (current_user.id,))
    history = cursor.fetchall()
    conn.close()
    
    return flask.render_template('dashboard.html', user=current_user, history=history)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if flask.request.method == 'GET':
        return flask.redirect(flask.url_for('dashboard'))

    if not model:
        return "Model not loaded correctly. Please check server logs.", 500

    try:
        # ── 1. RAW USER INPUTS ──────────────────────────────────────────
        input_data = {}

        # Binary features (0/1)
        binary_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                           'HeartDiseaseorAttack', 'PhysActivity',
                           'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                           'DiffWalk', 'Fruits', 'Veggies']
        for feature in binary_features:
            input_data[feature] = int(flask.request.form.get(feature) or 0)

        # Numeric / Categorical
        input_data['BMI']       = float(flask.request.form.get('BMI') or 22)
        input_data['Age']       = int(flask.request.form.get('Age') or 1)
        input_data['Sex']       = int(flask.request.form.get('Sex') or 0)
        input_data['Education'] = int(flask.request.form.get('Education') or 4)
        input_data['Income']    = int(flask.request.form.get('Income') or 5)
        input_data['GenHlth']   = int(flask.request.form.get('GenHlth') or 3)
        input_data['HbA1c']     = float(flask.request.form.get('HbA1c') or 5.5)

        # MentHlth: 1-5 scale → representative days
        ment_scale = int(flask.request.form.get('MentHlthScale') or 5)
        scale_to_days = {5: 0, 4: 2, 3: 7, 2: 15, 1: 30}
        input_data['MentHlth'] = scale_to_days.get(ment_scale, 0)

        # PhysHlth: 1-5 scale → representative days
        phys_scale = int(flask.request.form.get('PhysHlthScale') or 5)
        input_data['PhysHlth'] = scale_to_days.get(phys_scale, 0)

        # Validation
        if input_data['BMI'] < 10 or input_data['BMI'] > 100:
            return "Invalid BMI: Please enter a value between 10 and 100.", 400
        if not (4.0 <= input_data['HbA1c'] <= 14.0):
            return "Invalid HbA1c: Please enter a value between 4.0 and 14.0.", 400

        # ── 2. FEATURE ENGINEERING (matches train_augmented.py exactly) ──
        hba1c = input_data['HbA1c']
        input_data['HbA1c_diabetic']    = 1 if hba1c >= 6.5 else 0
        input_data['HbA1c_prediabetic'] = 1 if 5.7 <= hba1c < 6.5 else 0
        input_data['HbA1c_sq']          = hba1c ** 2
        input_data['HbA1c_BMI']         = hba1c * input_data['BMI']
        input_data['HbA1c_Age']         = hba1c * input_data['Age']
        input_data['HbA1c_HighBP']      = hba1c * input_data['HighBP']
        input_data['RiskScore']         = (input_data['HighBP'] + input_data['HighChol'] +
                                           input_data['HeartDiseaseorAttack'] + input_data['Stroke'] +
                                           input_data['DiffWalk'] + (1 if input_data['GenHlth'] >= 4 else 0))
        input_data['BMI_obese']         = 1 if input_data['BMI'] >= 30 else 0
        input_data['BMI_sq']            = input_data['BMI'] ** 2
        input_data['Age_BMI']           = input_data['Age'] * input_data['BMI']
        input_data['Age_GenHlth']       = input_data['Age'] * input_data['GenHlth']
        input_data['Combo_BP_Chol']     = input_data['HighBP'] * input_data['HighChol']
        input_data['HeartStroke']       = input_data['HeartDiseaseorAttack'] * input_data['Stroke']

        # Determine HbA1c status label for UI
        if hba1c >= 6.5:
            hba1c_status = ('Diabetic Range', '#ef4444')
        elif hba1c >= 5.7:
            hba1c_status = ('Pre-diabetic Range', '#f59e0b')
        else:
            hba1c_status = ('Normal Range', '#10b981')

        # Legacy metabolic score for UI display
        metabolic_score = input_data['HighBP'] + input_data['HighChol'] + input_data['BMI_obese']

        # ── 3. BUILD DATAFRAME IN MODEL COLUMN ORDER ────────────────────
        df = pd.DataFrame([input_data])
        df = df.reindex(columns=model_columns, fill_value=0)
        
        # Enforce float + string column names (cross-platform XGBoost fix)
        df = df.astype(float)
        df.columns = df.columns.astype(str)

        # ── 4. PREDICTION ───────────────────────────────────────────────
        input_array = df.values
        booster = model.get_booster()
        dtest = xgboost.DMatrix(input_array, feature_names=list(df.columns))
        probability = booster.predict(dtest)[0]

        prediction = 1 if probability >= threshold else 0
        
        verdict = "High Risk of Diabetes" if prediction == 1 else "Low Risk"
        high_risk = True if prediction == 1 else False
        prob_percent = round(probability * 100, 2)

        # Generate SHAP Plot & Extract Top Factors
        plot_url = None
        interaction_plot_url = None
        top_factors = []
        advice_list = []
        
        if explainer:
            shap_values = explainer(df)
            
            # --- EXTRACT TOP 3 CONTRIBUTORS ---
            values = shap_values[0].values
            feature_names = shap_values[0].feature_names
            feature_contributions = list(zip(feature_names, values))
            # Sort by absolute contribution strength (magnitude), but showing positive direction for risk
            positive_contributors = [(name, val) for name, val in feature_contributions if val > 0]
            positive_contributors.sort(key=lambda x: x[1], reverse=True)
            
            # --- TOP FACTORS (Simplified) ---
            # We still use SHAP to identify what matters most, but we won't show the confusing charts.
            top_3 = positive_contributors[:3]
            raw_top_names = [name for name, val in top_3]
            top_factor_values = [float(val) for name, val in top_3]
            
            # Map technical names to readable labels
            readable_map = {
                'HbA1c': 'HbA1c Level', 'HbA1c_sq': 'HbA1c (Squared)', 'HbA1c_diabetic': 'Diabetic HbA1c',
                'HbA1c_BMI': 'HbA1c × BMI Interaction', 'HbA1c_Age': 'HbA1c × Age Interaction',
                'HbA1c_HighBP': 'HbA1c × Blood Pressure', 'RiskScore': 'Clinical Risk Score',
                'BMI': 'Body Mass Index', 'BMI_sq': 'BMI (Squared)', 'Age': 'Age Group',
                'HighBP': 'High Blood Pressure', 'HighChol': 'High Cholesterol',
                'Combo_BP_Chol': 'BP + Cholesterol Combo', 'Age_BMI': 'Age × BMI Interaction',
                'HeartDiseaseorAttack': 'Heart Disease', 'Stroke': 'Stroke History',
                'DiffWalk': 'Difficulty Walking', 'GenHlth': 'General Health'
            }
            top_factor_names = [readable_map.get(f, f) for f in raw_top_names]

            # HEALTH INDICES FOR TRANSPARENCY CARDS
            health_indices = {
                'hba1c': {
                    'name': 'HbA1c Status',
                    'score': round(hba1c, 1),
                    'status_label': hba1c_status[0],
                    'status_color': hba1c_status[1],
                    'desc': 'Glycated haemoglobin — the gold-standard marker for diabetes diagnosis.'
                },
                'clinical': {
                    'name': 'Clinical Risk Score',
                    'score': round(input_data['RiskScore'], 0),
                    'status_label': 'High' if input_data['RiskScore'] >= 3 else 'Moderate' if input_data['RiskScore'] >= 1 else 'Low',
                    'status_color': '#ef4444' if input_data['RiskScore'] >= 3 else '#f59e0b' if input_data['RiskScore'] >= 1 else '#10b981',
                    'desc': 'Sum of serious comorbidities: BP, Cholesterol, Heart Disease, Stroke, Walking Difficulty.'
                },
                'metabolic': {
                    'name': 'Metabolic State',
                    'score': metabolic_score,
                    'status_label': 'Obese' if input_data['BMI_obese'] else 'Normal Weight',
                    'status_color': '#ef4444' if input_data['BMI_obese'] else '#10b981',
                    'desc': 'Captures weight, blood pressure, and cholesterol together as metabolic syndrome markers.'
                },
                'lifestyle': {
                    'name': 'Lifestyle Risk',
                    'score': int(not input_data['PhysActivity']) + int(input_data['Smoker']) + int(input_data['HvyAlcoholConsump']),
                    'status_label': 'Needs Improvement' if (not input_data['PhysActivity'] or input_data['Smoker']) else 'Healthy',
                    'status_color': '#f59e0b' if (not input_data['PhysActivity'] or input_data['Smoker']) else '#10b981',
                    'desc': 'Reflects smoking, heavy alcohol, and physical inactivity as modifiable risk factors.'
                }
            }

            # Complex charts removed as they were confusing to users
            plot_url = None
            interaction_plot_url = None

            # --- PROACTIVE HEALTH ADVICE ---
            # Now provides advice to EVERYONE with modifiable risk factors
            # (regardless of if they are currently High Risk or Low Risk)
            
            # Actionable features: (Feature, Change, readable message)
            actions = [
                ('HighBP', 0, "Lower/Manage High Blood Pressure"),
                ('BMI', -2.0, "Lower BMI by 2 points"),
                ('BMI', -5.0, "Lower BMI by 5 points"),
                ('PhysActivity', 1, "Start Regular Physical Activity"),
                ('HvyAlcoholConsump', 0, "Stop Heavy Alcohol Consumption"),
                ('Smoker', 0, "Quit Smoking")
            ]
                
            base_prob = probability
            
            for feature, change, message in actions:
                temp_data = input_data.copy()
                current_val = temp_data.get(feature, 0)
                
                # Logic to check if change is applicable
                if feature == 'HighBP' and current_val == 0: continue
                if feature == 'PhysActivity' and current_val == 1: continue
                if feature == 'Smoker' and current_val == 0: continue
                if feature == 'HvyAlcoholConsump' and current_val == 0: continue
                
                # Apply change
                if feature in ['BMI']:
                    temp_data[feature] = current_val + change
                else:
                    temp_data[feature] = change
                
                # Re-calculate Metabolic Score for BMI change
                if feature == 'BMI':
                     bmi_s = 1 if temp_data['BMI'] > 30 else 0
                     temp_data['Metabolic_Score'] = temp_data['HighBP'] + temp_data['HighChol'] + bmi_s
                
                # Re-compute HbA1c interaction features for counterfactual
                t_h = temp_data.get('HbA1c', hba1c)
                temp_data['HbA1c_sq'] = t_h ** 2
                temp_data['HbA1c_BMI'] = t_h * temp_data.get('BMI', input_data['BMI'])
                temp_data['HbA1c_Age'] = t_h * temp_data.get('Age', input_data['Age'])
                temp_data['HbA1c_HighBP'] = t_h * temp_data.get('HighBP', input_data['HighBP'])
                temp_data['BMI_obese'] = 1 if temp_data.get('BMI', input_data['BMI']) >= 30 else 0
                temp_data['BMI_sq'] = temp_data.get('BMI', input_data['BMI']) ** 2
                # Predict
                t_df = pd.DataFrame([temp_data])
                t_df = t_df.reindex(columns=model_columns, fill_value=0)
                t_df = t_df.astype(float)
                t_arr = t_df.values
                t_dtest = xgboost.DMatrix(t_arr, feature_names=list(t_df.columns))
                t_prob = booster.predict(t_dtest)[0]
                
                # Check for improvement
                # If they have the risk factor, we show the advice if it drops risk > 0.5%
                if (base_prob - t_prob) > 0.005: 
                    improvement = (base_prob - t_prob) * 100
                    new_risk = t_prob * 100
                    if t_prob < threshold and high_risk:
                         advice_list.append(f"{message} (Risk drops to {new_risk:.2f}% - LOW RISK)")
                    else:
                         advice_list.append(f"{message} (Reduces risk by {improvement:.2f}%)")

        # Save to History
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Format top factors for display in history (string)
        factors_str = ", ".join(top_factor_names) if 'top_factor_names' in locals() else ""
        
        # Check current count
        cursor.execute("SELECT COUNT(*) FROM records WHERE user_id = ?", (current_user.id,))
        count = cursor.fetchone()[0]
        
        if count >= 5:
            # Delete oldest
            cursor.execute("SELECT id FROM records WHERE user_id = ? ORDER BY id ASC LIMIT 1", (current_user.id,))
            oldest_id = cursor.fetchone()[0]
            cursor.execute("DELETE FROM records WHERE id = ?", (oldest_id,))
            
        cursor.execute("INSERT INTO records (user_id, date, risk_score, verdict, top_factors) VALUES (?, ?, ?, ?, ?)",
                       (current_user.id, datetime.now().strftime("%Y-%m-%d %H:%M"), float(prob_percent), verdict, factors_str))
        conn.commit()
        conn.close()

        return flask.render_template('result.html', 
                                     prediction=prediction, 
                                     probability=probability, 
                                     prob_percent=prob_percent,
                                     verdict=verdict,
                                     high_risk=high_risk,
                                     plot_url=plot_url,
                                     interaction_plot_url=interaction_plot_url,
                                     top_factors=top_factor_names,
                                     top_factor_values=top_factor_values,
                                     advice_list=advice_list,
                                     metabolic_score=metabolic_score,
                                     health_indices=health_indices,
                                     user=current_user)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return f"An error occurred during prediction: {e}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
