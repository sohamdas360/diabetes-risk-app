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

# --- SESSION SECURITY / COOKIE HARDENING ---
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = True if os.environ.get('RENDER') else False


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

# ═══ TWO-STAGE MODEL LOADING ═══
stage1_model = None
stage1_columns = []
stage1_threshold = 0.415
stage2_model = None
stage2_columns = []
stage2_threshold = 0.825
explainer1 = None
explainer2 = None

try:
    with open('two_stage_models.pkl', 'rb') as f:
        bundle = pickle.load(f)
    # Stage 1: Lifestyle screening (22 features)
    stage1_model = bundle['stage1']['model']
    stage1_threshold = bundle['stage1']['threshold']
    stage1_columns = bundle['stage1']['features']
    # Stage 2: Clinical confirmation (23 features with HbA1c)
    stage2_model = bundle['stage2']['model']
    stage2_threshold = bundle['stage2']['threshold']
    stage2_columns = bundle['stage2']['features']
    print(f"Two-Stage Models loaded.")
    print(f"  Stage 1: {len(stage1_columns)} features, threshold={stage1_threshold:.3f}")
    print(f"  Stage 2: {len(stage2_columns)} features, threshold={stage2_threshold:.3f}")
except Exception as e:
    print(f"Error loading models: {e}")

# Initialize SHAP Explainers
try:
    if stage1_model:
        explainer1 = shap.TreeExplainer(stage1_model)
    if stage2_model:
        explainer2 = shap.TreeExplainer(stage2_model)
    print("SHAP explainers initialized.")
except Exception as e:
    print(f"Error initializing SHAP: {e}")

# --- ROUTES ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    if flask.request.method == 'POST':
        username = flask.request.form['username']
        password = flask.request.form['password']
        
        # Server-side validation
        if len(username.strip()) < 3:
            flask.flash("Username must be at least 3 characters long.")
            return flask.redirect(flask.url_for('register'))
            
        if len(password) < 8:
            flask.flash("Password must be at least 8 characters long.")
            return flask.redirect(flask.url_for('register'))
            
        has_letter = any(c.isalpha() for c in password)
        has_digit = any(c.isdigit() for c in password)
        if not (has_letter and has_digit):
            flask.flash("Password must contain both letters and numbers.")
            return flask.redirect(flask.url_for('register'))

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # GLOBAL USER LIMIT CHECK (Max 10)
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count >= 10:
            flask.flash("Registration limits reached (Max 10 users).")
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

    if not stage1_model:
        return "Model not loaded correctly. Please check server logs.", 500

    try:
        # ── 1. RAW USER INPUTS ──────────────────────────────────────────
        input_data = {}

        # Binary features collected from form
        form_binary = ['HighBP', 'HighChol', 'Smoker', 'Stroke',
                       'HeartDiseaseorAttack', 'PhysActivity',
                       'HvyAlcoholConsump', 'AnyHealthcare',
                       'DiffWalk', 'Fruits', 'Veggies']
        for feature in form_binary:
            input_data[feature] = int(flask.request.form.get(feature) or 0)

        # Defaulted features (removed from form — low correlation with diabetes)
        input_data['CholCheck'] = 1       # r=0.065, default: Yes
        input_data['Sex'] = 0             # r=0.031, default: Female
        input_data['NoDocbcCost'] = 0     # r=0.031, default: No

        # Numeric / Categorical
        input_data['BMI']       = float(flask.request.form.get('BMI') or 22)
        input_data['Age']       = int(flask.request.form.get('Age') or 1)
        input_data['Education'] = int(flask.request.form.get('Education') or 4)
        input_data['Income']    = int(flask.request.form.get('Income') or 5)
        input_data['GenHlth']   = int(flask.request.form.get('GenHlth') or 3)

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

        # ── 2. DETECT STAGE (Is HbA1c provided?) ────────────────────────
        hba1c_raw = flask.request.form.get('HbA1c', '').strip()
        has_hba1c = hba1c_raw != '' and hba1c_raw != '0'

        if has_hba1c:
            hba1c_val = float(hba1c_raw)
            if not (4.0 <= hba1c_val <= 14.0):
                return "Invalid HbA1c: Please enter a value between 4.0 and 14.0.", 400
            input_data['HbA1c'] = hba1c_val
            stage = 2
            active_model = stage2_model
            active_columns = stage2_columns
            active_threshold = stage2_threshold
            active_explainer = explainer2
            # HbA1c status for UI
            if hba1c_val >= 6.5:
                hba1c_status = ('Diabetic Range', '#ef4444')
            elif hba1c_val >= 5.7:
                hba1c_status = ('Pre-diabetic Range', '#f59e0b')
            else:
                hba1c_status = ('Normal Range', '#10b981')
        else:
            stage = 1
            active_model = stage1_model
            active_columns = stage1_columns
            active_threshold = stage1_threshold
            active_explainer = explainer1
            hba1c_val = None
            hba1c_status = None

        # ── 3. BUILD DATAFRAME ──────────────────────────────────────────
        df = pd.DataFrame([input_data])
        df = df.reindex(columns=active_columns, fill_value=0)
        df = df.astype(float)
        df.columns = df.columns.astype(str)

        # ── 4. PREDICTION ───────────────────────────────────────────────
        input_array = df.values
        booster = active_model.get_booster()
        dtest = xgboost.DMatrix(input_array, feature_names=list(df.columns))
        probability = booster.predict(dtest)[0]

        prediction = 1 if probability >= active_threshold else 0
        verdict = "High Risk of Diabetes" if prediction == 1 else "Low Risk"
        high_risk = prediction == 1
        prob_percent = round(probability * 100, 2)

        # Also run Stage 1 if we're on Stage 2 (for comparison display)
        stage1_prob_percent = None
        if stage == 2 and stage1_model:
            df1 = pd.DataFrame([input_data])
            df1 = df1.reindex(columns=stage1_columns, fill_value=0).astype(float)
            df1.columns = df1.columns.astype(str)
            b1 = stage1_model.get_booster()
            dt1 = xgboost.DMatrix(df1.values, feature_names=list(df1.columns))
            s1_prob = b1.predict(dt1)[0]
            stage1_prob_percent = round(s1_prob * 100, 2)

        # ── 5. SHAP & HEALTH INDICES ────────────────────────────────────
        top_factors = []
        top_factor_names = []
        advice_list = []
        health_indices = {}
        metabolic_score = input_data['HighBP'] + input_data['HighChol'] + (1 if input_data['BMI'] >= 30 else 0)

        if active_explainer:
            shap_values = active_explainer(df)
            values = shap_values[0].values
            feature_names = shap_values[0].feature_names
            feature_contributions = list(zip(feature_names, values))
            positive_contributors = [(n, v) for n, v in feature_contributions if v > 0]
            positive_contributors.sort(key=lambda x: x[1], reverse=True)

            top_3 = positive_contributors[:3]
            raw_top_names = [n for n, v in top_3]

            readable_map = {
                'HbA1c': 'HbA1c Level', 'BMI': 'Body Mass Index', 'Age': 'Age Group',
                'HighBP': 'High Blood Pressure', 'HighChol': 'High Cholesterol',
                'HeartDiseaseorAttack': 'Heart Disease', 'Stroke': 'Stroke History',
                'DiffWalk': 'Difficulty Walking', 'GenHlth': 'General Health',
                'Smoker': 'Smoking', 'PhysActivity': 'Physical Inactivity',
                'Income': 'Income Level', 'MentHlth': 'Mental Health',
                'PhysHlth': 'Physical Health', 'Sex': 'Sex',
                'Education': 'Education Level', 'CholCheck': 'Cholesterol Check'
            }
            top_factor_names = [readable_map.get(f, f) for f in raw_top_names]

            # HEALTH INDICES (Correlation-backed feature combinations for user understandability)
            if stage == 2 and hba1c_val is not None:
                health_indices['hba1c'] = {
                    'name': '🧬 HbA1c Status',
                    'score': round(hba1c_val, 1),
                    'status_label': hba1c_status[0],
                    'status_color': hba1c_status[1],
                    'desc': 'Glycated haemoglobin — the gold-standard marker for diabetes diagnosis.'
                }

            # 1. Cardiovascular Risk (HighBP + HighChol + HeartDisease + Stroke) — r=0.26, 0.20, 0.18, 0.11
            cardio_score = input_data['HighBP'] + input_data['HighChol'] + input_data['HeartDiseaseorAttack'] + input_data['Stroke']
            health_indices['cardiovascular'] = {
                'name': '❤️ Cardiovascular Risk',
                'score': f"{cardio_score}/4",
                'status_label': 'Critical' if cardio_score >= 3 else 'High' if cardio_score >= 2 else 'Moderate' if cardio_score >= 1 else 'Healthy',
                'status_color': '#dc2626' if cardio_score >= 3 else '#ef4444' if cardio_score >= 2 else '#f59e0b' if cardio_score >= 1 else '#10b981',
                'desc': 'Your heart health profile — combining Blood Pressure, Cholesterol, Heart Disease, and Stroke history.'
            }

            # 2. Health Decline Score (GenHlth + DiffWalk + PhysHlth) — highly inter-correlated (r=0.52, 0.48, 0.46)
            genhlth_norm = (input_data['GenHlth'] - 1) / 4  # 0-1 scale (1=excellent→0, 5=poor→1)
            physhlth_norm = min(input_data['PhysHlth'] / 30, 1)  # 0-1 scale
            decline_score = round(genhlth_norm + physhlth_norm + input_data['DiffWalk'], 1)
            health_indices['decline'] = {
                'name': '📉 Health Decline',
                'score': f"{decline_score}/3.0",
                'status_label': 'Severe' if decline_score >= 2 else 'Declining' if decline_score >= 1 else 'Stable',
                'status_color': '#ef4444' if decline_score >= 2 else '#f59e0b' if decline_score >= 1 else '#10b981',
                'desc': 'How your body is coping — combines General Health rating, recent sick days, and mobility issues.'
            }

            # 3. Body Composition (BMI) — r=0.217
            bmi_val = input_data['BMI']
            health_indices['body'] = {
                'name': '⚖️ Body Composition',
                'score': round(bmi_val, 1),
                'status_label': 'Obese' if bmi_val >= 30 else 'Overweight' if bmi_val >= 25 else 'Normal',
                'status_color': '#ef4444' if bmi_val >= 30 else '#f59e0b' if bmi_val >= 25 else '#10b981',
                'desc': 'Your weight relative to height — a key indicator of metabolic health and diabetes risk.'
            }

            # 4. Lifestyle Risk (Smoker + HvyAlcohol + !PhysActivity) — modifiable factors
            lifestyle_score = int(not input_data['PhysActivity']) + int(input_data['Smoker']) + int(input_data['HvyAlcoholConsump'])
            health_indices['lifestyle'] = {
                'name': '🏃 Lifestyle Risk',
                'score': f"{lifestyle_score}/3",
                'status_label': 'Poor' if lifestyle_score >= 2 else 'Needs Work' if lifestyle_score >= 1 else 'Healthy',
                'status_color': '#ef4444' if lifestyle_score >= 2 else '#f59e0b' if lifestyle_score >= 1 else '#10b981',
                'desc': 'Smoking, heavy alcohol, and physical inactivity — the factors YOU can change.'
            }

            # 5. Socioeconomic Profile (Education + Income) — inter-correlated (r=0.45)
            ses_score = input_data['Education'] + input_data['Income']  # ranges from 2 to 14
            health_indices['socioeconomic'] = {
                'name': '💼 Socioeconomic Profile',
                'score': f"{ses_score}/14",
                'status_label': 'Protective' if ses_score >= 10 else 'Average' if ses_score >= 6 else 'Vulnerable',
                'status_color': '#10b981' if ses_score >= 10 else '#f59e0b' if ses_score >= 6 else '#ef4444',
                'desc': 'Higher education and income are linked to better health access and lower diabetes risk.'
            }

            # ── PROACTIVE HEALTH ADVICE ──
            actions = [
                ('HighBP', 0, "Lower/Manage High Blood Pressure"),
                ('BMI', -2.0, "Lower BMI by 2 points"),
                ('BMI', -5.0, "Lower BMI by 5 points"),
                ('PhysActivity', 1, "Start Regular Physical Activity"),
                ('HvyAlcoholConsump', 0, "Stop Heavy Alcohol Consumption"),
                ('Smoker', 0, "Quit Smoking")
            ]

            for feature, change, message in actions:
                temp_data = input_data.copy()
                current_val = temp_data.get(feature, 0)
                if feature == 'HighBP' and current_val == 0: continue
                if feature == 'PhysActivity' and current_val == 1: continue
                if feature == 'Smoker' and current_val == 0: continue
                if feature == 'HvyAlcoholConsump' and current_val == 0: continue

                if feature == 'BMI':
                    temp_data[feature] = current_val + change
                else:
                    temp_data[feature] = change

                t_df = pd.DataFrame([temp_data])
                t_df = t_df.reindex(columns=active_columns, fill_value=0).astype(float)
                t_df.columns = t_df.columns.astype(str)
                t_dtest = xgboost.DMatrix(t_df.values, feature_names=list(t_df.columns))
                t_prob = booster.predict(t_dtest)[0]

                if (probability - t_prob) > 0.005:
                    improvement = (probability - t_prob) * 100
                    new_risk = t_prob * 100
                    if t_prob < active_threshold and high_risk:
                        advice_list.append(f"{message} (Risk drops to {new_risk:.2f}% — LOW RISK!)")
                    else:
                        advice_list.append(f"{message} (Reduces risk by {improvement:.2f}%)")

        # ── 6. SAVE TO HISTORY ──────────────────────────────────────────
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        factors_str = ", ".join(top_factor_names) if top_factor_names else ""
        stage_label = "[Lab]" if stage == 2 else "[Screen]"
        verdict_with_stage = f"{stage_label} {verdict}"

        cursor.execute("SELECT COUNT(*) FROM records WHERE user_id = ?", (current_user.id,))
        count = cursor.fetchone()[0]
        if count >= 5:
            cursor.execute("SELECT id FROM records WHERE user_id = ? ORDER BY id ASC LIMIT 1", (current_user.id,))
            oldest_id = cursor.fetchone()[0]
            cursor.execute("DELETE FROM records WHERE id = ?", (oldest_id,))

        cursor.execute("INSERT INTO records (user_id, date, risk_score, verdict, top_factors) VALUES (?, ?, ?, ?, ?)",
                       (current_user.id, datetime.now().strftime("%Y-%m-%d %H:%M"), float(prob_percent), verdict_with_stage, factors_str))
        conn.commit()
        conn.close()

        return flask.render_template('result.html',
                                     prediction=prediction,
                                     probability=probability,
                                     prob_percent=prob_percent,
                                     verdict=verdict,
                                     high_risk=high_risk,
                                     stage=stage,
                                     hba1c_val=hba1c_val,
                                     hba1c_status=hba1c_status,
                                     stage1_prob_percent=stage1_prob_percent,
                                     top_factors=top_factor_names,
                                     advice_list=advice_list,
                                     metabolic_score=metabolic_score,
                                     health_indices=health_indices,
                                     user=current_user)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return f"An error occurred during prediction: {e}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
