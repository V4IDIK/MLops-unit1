import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# CHANGED: Added more metrics to the import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

def main():
    # 1. Load dataset & print basic statistics
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    print("\n--- Basic Dataset Statistics ---")
    print(df.describe())

    # 2. Split data into training and testing sets
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train a simple ML model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    # 4. Evaluate the model (CHANGED: Added Precision, Recall, F1, and Classification Report)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # 5. Save the trained model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/logistic_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully to {model_path}")

if __name__ == "__main__":
    main()