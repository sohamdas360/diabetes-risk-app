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

# Load Model and Columns
try:
    with open('final_diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    print("Model and columns loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    model_columns = []

# Initialize SHAP Explainer (TreeExplainer for XGBoost)
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
        # Extract features from form
        input_data = {}
        
        # Binary features (0/1)
        binary_features = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 
                          'HeartDiseaseorAttack', 'PhysActivity', 
                          'HvyAlcoholConsump', 'DiffWalk', 'Fruits', 'Veggies']
        
        for feature in binary_features:
            input_data[feature] = int(flask.request.form.get(feature) or 0)

        # Numeric/Category features
        input_data['BMI'] = float(flask.request.form.get('BMI') or 0)
        input_data['Age'] = int(flask.request.form.get('Age') or 1)

        # Validation
        if input_data['BMI'] < 10 or input_data['BMI'] > 100:
             return "Invalid BMI: Please enter a value between 10 and 100.", 400

        # --- NEW FEATURE MERGING (indices) ---
        # 1. Metabolic_Index: (HighBP + HighChol) * BMI
        input_data['Metabolic_Index'] = (input_data['HighBP'] + input_data['HighChol']) * input_data['BMI']

        # 2. Physical_Fragility: Age * (DiffWalk + Stroke + HeartDiseaseorAttack + 1)
        input_data['Physical_Fragility'] = input_data['Age'] * (input_data['DiffWalk'] + input_data['Stroke'] + input_data['HeartDiseaseorAttack'] + 1)

        # 3. Lifestyle_Risk: (Smoker + HvyAlcoholConsump) - (PhysActivity + Fruits + Veggies)
        input_data['Lifestyle_Risk'] = (input_data['Smoker'] + input_data['HvyAlcoholConsump']) - (input_data['PhysActivity'] + input_data['Fruits'] + input_data['Veggies'])

        # Legacy Metabolic Score for UI display
        metabolic_score = input_data['HighBP'] + input_data['HighChol'] + (1 if input_data['BMI'] > 30 else 0)

        # Create DataFrame in the correct order
        df = pd.DataFrame([input_data])
        df = df.reindex(columns=model_columns, fill_value=0)
        
        # DATA FIX: Enforce float types and string columns to satisfy strict XGBoost checks
        df = df.astype(float)
        df.columns = df.columns.astype(str)

        # Make Prediction
        # Updated Threshold for 92%+ Recall
        threshold = 0.27
        
        # ULTIMATE FIX for 'feature names' error on Render/Linux
        # The issue: XGBoost model pickled on Windows has strict feature validation
        # Solution: Convert to numpy array and use booster's predict with feature validation disabled
        
        # Convert DataFrame to numpy array (bypasses pandas metadata issues)
        input_array = df.values
        
        # Get the underlying XGBoost Booster object
        booster = model.get_booster()
        
        # Create DMatrix from numpy array with explicit feature names
        # Note: feature_names must match exactly what model was trained with
        dtest = xgboost.DMatrix(input_array, feature_names=list(df.columns))
        
        # Predict using the booster (returns probability for binary classification)
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
            
            top_3 = positive_contributors[:3]
            raw_top_names = [name for name, val in top_3]
            top_factor_values = [float(val) for name, val in top_3]
            
            # Map technical names
            readable_map = {
                'HighBP': 'High Blood Pressure', 'HighChol': 'High Cholesterol', 'Metabolic_Score': 'Metabolic Score',
                'BMI': 'Body Mass Index', 'Age': 'Age Group',
                'DiffWalk': 'Difficulty Walking', 'HvyAlcoholConsump': 'Heavy Alcohol Consumption',
                'Smoker': 'Smoker Status', 'Stroke': 'Stroke History', 'HeartDiseaseorAttack': 'Heart Disease',
                'Metabolic_Index': 'Metabolic Index', 'Physical_Fragility': 'Physical Fragility',
                'Lifestyle_Risk': 'Lifestyle Risk'
            }
            top_factor_names = [readable_map.get(f, f) for f in raw_top_names]

            # Generate SHAP waterfall plot
            shap.plots.waterfall(shap_values[0], show=False, max_display=10)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.clf()

            # --- COMPLETE SHAP FEATURE BREAKDOWN ---
            # Lightweight alternative to interaction plot
            # Shows ALL features' contributions, color-coded by impact
            
            # Get all SHAP values and feature names
            all_shap_values = shap_values[0].values
            all_feature_names = model_columns
            
            # Create readable names mapping
            readable_map = {
                'HighBP': 'High Blood Pressure', 'HighChol': 'High Cholesterol', 
                'Metabolic_Score': 'Metabolic Score', 'BMI': 'Body Mass Index', 
                'Age': 'Age Group', 'DiffWalk': 'Difficulty Walking', 
                'HvyAlcoholConsump': 'Heavy Alcohol Use', 'Smoker': 'Smoking Status', 
                'Stroke': 'Stroke History', 'HeartDiseaseorAttack': 'Heart Disease',
                'PhysActivity': 'Physical Activity', 
                'Metabolic_Index': 'Metabolic Index (Health Indices)', 'Physical_Fragility': 'Physical Fragility Index',
                'Lifestyle_Risk': 'Lifestyle Risk Balance'
            }
            readable_names = [readable_map.get(f, f) for f in all_feature_names]
            
            # Sort by absolute value (most impactful first)
            sorted_indices = sorted(range(len(all_shap_values)), 
                                   key=lambda i: abs(all_shap_values[i]), 
                                   reverse=True)
            
            sorted_values = [all_shap_values[i] for i in sorted_indices]
            sorted_names = [readable_names[i] for i in sorted_indices]
            
            # Color code: red for increasing risk, blue for decreasing
            colors = ['#ff4444' if val > 0 else '#4444ff' for val in sorted_values]
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, 8))
            plt.barh(sorted_names, sorted_values, color=colors, edgecolor='white', linewidth=0.5)
            plt.xlabel('Impact on Diabetes Risk', fontsize=12, fontweight='bold')
            plt.title('Complete Risk Factor Analysis', fontsize=14, fontweight='bold', pad=20)
            plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
            
            # Add value labels
            for i, (name, val) in enumerate(zip(sorted_names, sorted_values)):
                label_x = val + (0.01 if val > 0 else -0.01)
                ha = 'left' if val > 0 else 'right'
                plt.text(label_x, i, f'{val:.3f}', va='center', ha=ha, fontsize=9)
            
            plt.tight_layout()
            buf_complete = io.BytesIO()
            plt.savefig(buf_complete, format='png', bbox_inches='tight', dpi=100)
            buf_complete.seek(0)
            interaction_plot_url = base64.b64encode(buf_complete.getvalue()).decode('utf-8')
            plt.clf()

            # --- COUNTERFACTUAL ADVICE ---
            if high_risk:
                # Actionable features: (Feature, Change, readable message)
                actions = [
                    ('BMI', -2.0, "Lower BMI by 2 points"),
                    ('BMI', -5.0, "Lower BMI by 5 points"),
                    ('PhysActivity', 1, "Start Regular Physical Activity"),
                    ('HvyAlcoholConsump', 0, "Stop Heavy Alcohol Consumption"),
                    ('Smoker', 0, "Quit Smoking"), 
                    ('GenHlth', -1, "Improve General Health by 1 level")
                ]
                
                base_prob = probability
                
                for feature, change, message in actions:
                    temp_data = input_data.copy()
                    current_val = temp_data.get(feature, 0)
                    
                    # Logic to check if change is applicable
                    if feature == 'PhysActivity' and current_val == 1: continue
                    if feature == 'Smoker' and current_val == 0: continue
                    if feature == 'HvyAlcoholConsump' and current_val == 0: continue
                    if feature == 'GenHlth' and current_val <= 1: continue
                    
                    # Apply change
                    if feature in ['BMI', 'GenHlth']:
                        temp_data[feature] = current_val + change
                    else:
                        temp_data[feature] = change
                    
                    # Re-calculate Metabolic Score for BMI change
                    if feature == 'BMI':
                         bmi_s = 1 if temp_data['BMI'] > 30 else 0
                         temp_data['Metabolic_Score'] = temp_data['HighBP'] + temp_data['HighChol'] + bmi_s
                    
                    # Predict
                    t_df = pd.DataFrame([temp_data])
                    t_df = t_df.reindex(columns=model_columns, fill_value=0)
                    t_prob = model.predict_proba(t_df)[:, 1][0]
                    
                    # Check for improvement
                    if (base_prob - t_prob) > 0.01: # Report >1% drop
                        improvement = round((base_prob - t_prob)*100, 1)
                        new_risk = round(t_prob*100, 1)
                        if t_prob < threshold:
                             advice_list.append(f"{message} (Risk drops to {new_risk}%)")
                        elif improvement > 2.0:
                             advice_list.append(f"{message} (Reduces risk by {improvement}%)")

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
                                     user=current_user)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return f"An error occurred during prediction: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
