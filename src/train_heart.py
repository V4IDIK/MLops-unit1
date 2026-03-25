import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# CHANGED: Swapped the broken URL for a permanent, working raw CSV link
url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
df = pd.read_csv(url).dropna()

X = df.drop(columns=['target'])
y = df['target']

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/heart_model.pkl')

print("Model trained and saved to models/heart_model.pkl")