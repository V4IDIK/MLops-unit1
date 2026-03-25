import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import time
import os

# 1. Setup Logging Configuration
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/model_monitor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("Running Titanic ML Pipeline... Check the logs folder for output!")
    logging.info("--- Starting New ML Pipeline Run ---")

    # 2. Load Dataset safely with error logging
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        df = pd.read_csv(url)
        logging.info(f"Successfully loaded dataset with {len(df)} rows.")
    except Exception as e:
        logging.error(f"CRITICAL ERROR: Failed to load dataset: {e}")
        return

    # Quick preprocessing (Selecting a few features and dropping missing data)
    df = df[['Pclass', 'Age', 'Fare', 'Survived']].dropna()
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Model and track latency (how long it takes)
    start_time = time.time()
    
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logging.info(f"Model trained successfully. Latency: {training_time:.4f} seconds.")

    # 4. Evaluate and trigger an alert if accuracy is bad
    acc = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"Model Evaluation Accuracy: {acc:.4f}")
    
    # --- ALERTING LOGIC ---
    if acc < 0.70:
        logging.warning("ALERT: Model accuracy dropped below 70% threshold! Immediate retraining required.")
    else:
        logging.info("Model performance is healthy.")

if __name__ == "__main__":
    main()