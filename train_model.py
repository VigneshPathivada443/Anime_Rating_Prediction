import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load dataset
data = pd.read_csv("anime.csv", escapechar="\\")

# Clean data
data['episodes'] = pd.to_numeric(data['episodes'], errors='coerce').fillna(0)
data['rating'] = data['rating'].fillna(data['rating'].mean())

X = data[['genre', 'type', 'episodes', 'members']]
y = data['rating']

# Encode categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(X[['genre', 'type']])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(['genre', 'type'])
)

X_final = pd.concat(
    [encoded_df, X[['episodes', 'members']].reset_index(drop=True)],
    axis=1
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_final, y)

# Save model & encoder
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("✅ Model and encoder saved successfully")
