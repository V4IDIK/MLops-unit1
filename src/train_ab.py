from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Training Model A and Model B for A/B testing...")
X, y = load_iris(return_X_y=True)

# Train Model A (Logistic Regression)
model_a = LogisticRegression(max_iter=200)
model_a.fit(X, y)

# Train Model B (Random Forest)
model_b = RandomForestClassifier(n_estimators=10)
model_b.fit(X, y)

# Save both models
os.makedirs('models', exist_ok=True)
joblib.dump(model_a, 'models/model_a.pkl')
joblib.dump(model_b, 'models/model_b.pkl')

print("Successfully saved models/model_a.pkl and models/model_b.pkl")