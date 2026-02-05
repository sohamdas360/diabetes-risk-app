import requests

data = {
    'BMI': '28.5',
    'GenHlth': '3',
    'MentHlth': '5',
    'PhysHlth': '2',
    'Age': '8',
    'HighBP': '1',
    'HighChol': '0',
    'Smoker': '1',
    'Stroke': '0',
    'HeartDiseaseorAttack': '0',
    'PhysActivity': '1',
    'HvyAlcoholConsump': '0',
    'DiffWalk': '0',
    'Sex': '1',
    'Fruits': '1',
    'Veggies': '1',
    'Education': '6',
    'Income': '8'
}

# We can't easily test the flask app without running it, 
# but we can test the data processing logic by importing from app.py
# Actually, I'll just check if the app starts.

print("Starting Flask app in background...")
import subprocess
import time

proc = subprocess.Popen(['python', 'app.py'], cwd='c:/Users/ACER/Desktop/ML 2/Prediction')
time.sleep(5)
if proc.poll() is None:
    print("App started successfully.")
    proc.terminate()
else:
    print("App failed to start.")
    exit(1)
