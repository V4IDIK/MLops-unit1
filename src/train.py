import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

def main():
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- MLflow Setup ---
    # Set an experiment name so runs are grouped together
    mlflow.set_experiment("Breast_Cancer_Classification")

    # Define hyperparameters
    n_estimators = 100
    max_depth = 5

    # Start an MLflow run
    with mlflow.start_run():
        print(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
        
        # Train the model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")

        # --- MLflow Logging ---
        # 1. Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # 2. Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # 3. Log the model artifact
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print("\nSuccessfully logged parameters, metrics, and model to MLflow!")

if __name__ == "__main__":
    main()