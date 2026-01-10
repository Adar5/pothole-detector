import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

# Assume CSV files are in the directory
# For example, potholes.csv, speed_breakers.csv, braking.csv, turns.csv

def load_data():
    data = []
    labels = []
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            df = pd.read_csv(file)
            # Assume columns: timestamp, accel_x, accel_y, accel_z, latitude, longitude
            # Compute features: mean, std, max of magnitude
            df['magnitude'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
            features = df.groupby(pd.cut(df['timestamp'], bins=10)).agg({  # Segment into 10 bins
                'magnitude': ['mean', 'std', 'max']
            }).values
            label = file.split('.')[0]  # e.g., 'potholes'
            data.extend(features)
            labels.extend([label] * len(features))
    return np.array(data), np.array(labels)

# Load data
X, y = load_data()

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
import joblib
joblib.dump(model, 'vibration_classifier.pkl')
joblib.dump(le, 'label_encoder.pkl')