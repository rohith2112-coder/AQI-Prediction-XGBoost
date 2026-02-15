import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. SETUP & DATA LOADING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "city_day.csv")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: {file_path} not found. Please ensure the CSV is in the script folder.")
    exit()

# User Input for City Selection
target_city = input("Enter city name: ").title()

# --- 2. DATA PREPROCESSING ---
# Convert Date to datetime objects and set as index for time-series analysis
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True) 

# Handle missing values: 
df.ffill(inplace=True)
df.fillna(0, inplace=True) 

# Filter data for the specific city
city_data = df[df['City'] == target_city].copy()

if city_data.empty:
    print(f"No data found for city: {target_city}")
    exit()

# --- 3. FEATURE ENGINEERING (Lag Features) ---

city_data['AQI_Yesterday'] = city_data['AQI'].shift(1)
city_data['AQI_LastWeek'] = city_data['AQI'].shift(7)
city_data['PM2.5_Yesterday'] = city_data['PM2.5'].shift(1)
city_data['NO2_Yesterday'] = city_data['NO2'].shift(1)
city_data['CO_Yesterday'] = city_data['CO'].shift(1)

# Drop rows where lag features are NaN (the first 7 days)
city_data = city_data.dropna()

if city_data.empty:
    print(f"Not enough data to generate lag features for {target_city}. Try a city with more historical transparency.")
    exit()

# Define Input Features (X) and Target Variable (y)
features = ['AQI_Yesterday', 'AQI_LastWeek', 'PM2.5_Yesterday', 'NO2_Yesterday', 'CO_Yesterday']
X = city_data[features]
y = city_data['AQI']

# --- 4. TRAIN-TEST SPLIT ---
# For Time-Series, we MUST split chronologically, not randomly.
train_size = int(len(city_data) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# --- 5. MODEL TRAINING (XGBoost) ---
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    objective='reg:squarederror'
)
model.fit(X_train, y_train)

# --- 6. PREDICTIONS & EVALUATION ---
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Accuracy Metric (MAPE-based)
def calculate_accuracy(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return 100 - mape

accuracy_grade = calculate_accuracy(y_test, predictions)

print(f"\n--- MODEL PERFORMANCE FOR {target_city} ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} AQI points")
print(f"R-squared Score: {r2:.4f}")
print(f"Overall Accuracy Grade: {accuracy_grade:.2f}%")

# --- 7. FORECASTING TOMORROW ---
latest_data = city_data.iloc[-1]
input_for_tomorrow = pd.DataFrame({
    'AQI_Yesterday': [latest_data['AQI']],       
    'AQI_LastWeek': [city_data.iloc[-6]['AQI']], 
    'PM2.5_Yesterday': [latest_data['PM2.5']],            
    'NO2_Yesterday': [latest_data['NO2']],
    'CO_Yesterday': [latest_data['CO']]
}, columns=features)

future_pred = model.predict(input_for_tomorrow)[0]

print("\n--- TOMORROW'S FORECAST ---")
print(f"Reference Date: {latest_data.name.date()}")
print(f"Predicted AQI: {future_pred:.2f}")

# Health Impact Logic
if future_pred <= 50:
    impact = "Good: Minimal health impact."
elif future_pred <= 100:
    impact = "Satisfactory: Minor breathing discomfort for sensitive people."
elif future_pred <= 200:
    impact = "Moderate: Breathing discomfort for people with lung/heart disease."
elif future_pred <= 300:
    impact = "Poor: Breathing discomfort for most people on prolonged exposure."
elif future_pred <= 400:
    impact = "Very Poor: Respiratory illness on prolonged exposure."
else:
    impact = "Hazardous: Severe respiratory illness risks."

print(f"Health Category: {impact}")
print("="*35)

# --- 8. VISUALIZATION ---
# Plot 1: Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual AQI', color='blue', alpha=0.6)
plt.plot(y_test.index, predictions, label='XGBoost Prediction', color='red', linestyle='--')
plt.title(f'AQI Prediction: Actual vs XGBoost ({target_city})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot 2: Residuals (Error Analysis)
plt.figure(figsize=(10, 5))
residuals = y_test - predictions
plt.scatter(predictions, residuals, alpha=0.5, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual Plot: Prediction Errors')
plt.xlabel('Predicted AQI')
plt.ylabel('Error (Actual - Predicted)')
plt.show()

#Plot 3: Feature Importance
#Plot 3: Feature Importance
fig, ax = plt.subplots(figsize=(10, 5))
xgb.plot_importance(model, importance_type='weight', ax=ax)
plt.title('Which factors mattered most to the model?')
plt.show()