FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pandas scikit-learn joblib

COPY src/ src/

CMD ["python3", "src/train_heart.py"]
