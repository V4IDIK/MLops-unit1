import pandas as pd
from sklearn.datasets import load_diabetes

print("Downloading Diabetes dataset...")
data = load_diabetes(as_frame=True)
df = data.frame
df.to_csv("data/diabetes.csv", index=False)
print("Dataset saved to data/diabetes.csv")