import pickle

try:
    with open('model_columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    print("Columns in pickle:", columns)
except Exception as e:
    print(f"Error reading pickle: {e}")
