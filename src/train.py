import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def main():
    # 1. Load dataset & print basic statistics
    print("Loading dataset...")
    data = load_iris(as_frame=True)
    df = data.frame
    
    print("\n--- Basic Dataset Statistics ---")
    print(df.describe())

    # 2. Split data into training and testing sets
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train a simple ML model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # 4. Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    # 5. Save the trained model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/logistic_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully to {model_path}")

if __name__ == "__main__":
    main()