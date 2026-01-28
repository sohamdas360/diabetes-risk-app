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
            return flask.redirect(flask.url_for('index'))
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
@login_required
def index():
    # Fetch History
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT date, verdict, risk_score FROM records WHERE user_id = ? ORDER BY id DESC", (current_user.id,))
    history = cursor.fetchall()
    conn.close()
    
    return flask.render_template('index.html', user=current_user, history=history)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not model:
        return "Model not loaded correctly. Please check server logs.", 500

    try:
        # Extract features from form
        input_data = {}
        
        # Binary features (0/1)
        binary_features = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 
                          'HeartDiseaseorAttack', 'PhysActivity', 
                          'HvyAlcoholConsump', 'DiffWalk']
        
        for feature in binary_features:
            input_data[feature] = int(flask.request.form.get(feature, 0))

        # Numeric/Category features
        input_data['BMI'] = float(flask.request.form.get('BMI', 0))
        input_data['GenHlth'] = int(flask.request.form.get('GenHlth', 3))
        input_data['MentHlth'] = int(flask.request.form.get('MentHlth', 0))
        input_data['PhysHlth'] = int(flask.request.form.get('PhysHlth', 0))
        input_data['Age'] = int(flask.request.form.get('Age', 1))

        # Validation
        if input_data['BMI'] < 10 or input_data['BMI'] > 100:
             return "Invalid BMI: Please enter a value between 10 and 100.", 400

        # Feature Engineering: Metabolic_Score
        # Metabolic_Score = HighBP + HighChol + (1 if BMI > 30 else 0)
        bmi_score = 1 if input_data['BMI'] > 30 else 0
        metabolic_score = input_data['HighBP'] + input_data['HighChol'] + bmi_score
        input_data['Metabolic_Score'] = metabolic_score

        # Create DataFrame in the correct order
        df = pd.DataFrame([input_data])
        
        # Ensure columns are in the exact order the model expects
        df = df.reindex(columns=model_columns, fill_value=0)

        # Make Prediction
        # Threshold 0.30 for Recall priority
        threshold = 0.30
        probability = model.predict_proba(df)[:, 1][0]
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
            positive_contributors = [(name, val) for name, val in feature_contributions if val > 0]
            positive_contributors.sort(key=lambda x: x[1], reverse=True)
            top_factors = [name for name, val in positive_contributors[:3]]
            
            # Map technical names
            readable_map = {
                'HighBP': 'High Blood Pressure', 'HighChol': 'High Cholesterol', 'Metabolic_Score': 'Metabolic Score',
                'BMI': 'Body Mass Index', 'GenHlth': 'General Health Status', 'Age': 'Age Group',
                'DiffWalk': 'Difficulty Walking', 'PhysHlth': 'Physical Health', 'HvyAlcoholConsump': 'Heavy Alcohol Consumption',
                'HeartDiseaseorAttack': 'History of Heart Disease', 'MentHlth': 'Mental Health', 'PhysActivity': 'Lack of Physical Activity'
            }
            top_factors = [readable_map.get(f, f) for f in top_factors]

            # SHAP Plot
            plt.figure()
            shap.plots.waterfall(shap_values[0], show=False, max_display=10)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
            plt.close()

            # --- SHAP INTERACTIONS & HEATMAP ---
            try:
                # Compute interaction values
                shap_interaction_values = explainer.shap_interaction_values(df)
                interaction_matrix = shap_interaction_values[0] 
                
                # Mask diagonal
                np.fill_diagonal(interaction_matrix, 0)
                
                plt.figure(figsize=(10, 8))
                import seaborn as sns
                
                # Create heatmap
                ax = sns.heatmap(interaction_matrix, xticklabels=model_columns, yticklabels=model_columns, 
                            cmap="coolwarm", center=0, annot=False)
                plt.title("Feature Interaction Risks (Off-Diagonal)")
                plt.tight_layout()
                
                buf_int = io.BytesIO()
                plt.savefig(buf_int, format='png', bbox_inches='tight')
                buf_int.seek(0)
                interaction_plot_url = base64.b64encode(buf_int.getvalue()).decode('utf8')
                plt.close()
            except Exception as e:
                print(f"Interaction Plot Risk: {e}")

        # --- SAVE TO HISTORY (Max 5 Records) ---
        from datetime import datetime
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        factors_str = ", ".join(top_factors) if top_factors else "None"
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Insert new record
        # Fix: Cast numpy float to python float to avoid storing as BLOB
        cursor.execute("INSERT INTO records (user_id, date, risk_score, verdict, top_factors) VALUES (?, ?, ?, ?, ?)", 
                       (current_user.id, now_str, float(prob_percent), verdict, factors_str))
        
        # Enforce FIFO Limit (Keep only last 5)
        cursor.execute("SELECT id FROM records WHERE user_id = ? ORDER BY id DESC", (current_user.id,))
        rows = cursor.fetchall()
        if len(rows) > 5:
            # Get IDs to delete (all except top 5)
            ids_to_keep = [r[0] for r in rows[:5]]
            placeholders = ','.join(['?'] * len(ids_to_keep))
            cursor.execute(f"DELETE FROM records WHERE user_id = ? AND id NOT IN ({placeholders})", (current_user.id, *ids_to_keep))
            
        conn.commit()
        conn.close()

        # --- COUNTERFACTUAL EXPLANATIONS ---
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
                
                # Skip invalid actions
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
                if t_prob < threshold:
                     advice_list.append(f"{message} (Risk drops to {round(t_prob*100, 1)}%)")
                elif (base_prob - t_prob) > 0.05: # Report >5% drop
                     advice_list.append(f"{message} (Reduces risk by {round((base_prob - t_prob)*100, 1)}%)")

        return flask.render_template('result.html', 
                                     verdict=verdict, 
                                     probability=prob_percent, 
                                     high_risk=high_risk,
                                     plot_url=plot_url,
                                     interaction_plot_url=interaction_plot_url,
                                     top_factors=top_factors,
                                     advice_list=advice_list,
                                     user=current_user)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return f"An error occurred during prediction: {e}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
