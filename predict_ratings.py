# predict_ratings.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("RESTAURANT RATING PREDICTION MODEL")
print("=" * 50)

# Step 1: Load the data
print("\n1. Loading data...")
df = pd.read_csv('Dataset .csv')  # Make sure filename matches exactly
print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# Step 2: Initial data exploration
print("\n2. Initial data info:")
print(df.info())
print("\n   First 5 rows:")
print(df.head())

# Step 3: Clean the data
print("\n3. Cleaning data...")

# Remove restaurants with 0 rating (not rated)
initial_count = len(df)
df = df[df['Aggregate rating'] > 0]
print(f"   Removed {initial_count - len(df)} restaurants with 0 rating")
print(f"   New dataset shape: {df.shape}")

# Check for missing values
print("\n   Missing values per column:")
print(df.isnull().sum())

# Step 4: Feature Engineering
print("\n4. Feature Engineering...")

# Extract main cuisine (first cuisine listed)
df['Main_Cuisine'] = df['Cuisines'].str.split(',').str[0].str.strip()
print("   Created 'Main_Cuisine' feature")

# Convert Yes/No columns to binary (1/0)
binary_columns = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
for col in binary_columns:
    df[col + '_Binary'] = df[col].map({'Yes': 1, 'No': 0})
print("   Converted Yes/No columns to binary")

# Step 5: Select features for the model
print("\n5. Selecting features...")

# Define which columns to use as features
feature_columns = [
    'Average Cost for two',
    'Price range',
    'Votes',
    'Has Table booking_Binary',
    'Has Online delivery_Binary',
    'Country Code'
]

# Add main cuisine (will be one-hot encoded)
cuisine_dummies = pd.get_dummies(df['Main_Cuisine'], prefix='Cuisine', drop_first=True)

# Select numerical features
X_numerical = df[feature_columns].copy()

# Combine all features
X = pd.concat([X_numerical, cuisine_dummies], axis=1)
y = df['Aggregate rating']

print(f"   Final feature matrix shape: {X.shape}")
print(f"   Number of features: {X.shape[1]}")

# Step 6: Split the data
print("\n6. Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Training set size: {len(X_train)}")
print(f"   Testing set size: {len(X_test)}")

# Step 7: Train and evaluate models
print("\n7. Training and evaluating models...")
print("-" * 40)

# Dictionary to store results
results = {}

# Model 1: Linear Regression
print("\n   Model 1: Linear Regression")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

results['Linear Regression'] = {'MAE': mae_lr, 'RMSE': rmse_lr, 'R²': r2_lr}
print(f"      MAE: {mae_lr:.4f}")
print(f"      RMSE: {rmse_lr:.4f}")
print(f"      R²: {r2_lr:.4f}")

# Model 2: Decision Tree
print("\n   Model 2: Decision Tree")
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

results['Decision Tree'] = {'MAE': mae_dt, 'RMSE': rmse_dt, 'R²': r2_dt}
print(f"      MAE: {mae_dt:.4f}")
print(f"      RMSE: {rmse_dt:.4f}")
print(f"      R²: {r2_dt:.4f}")

# Model 3: Random Forest
print("\n   Model 3: Random Forest")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

results['Random Forest'] = {'MAE': mae_rf, 'RMSE': rmse_rf, 'R²': r2_rf}
print(f"      MAE: {mae_rf:.4f}")
print(f"      RMSE: {rmse_rf:.4f}")
print(f"      R²: {r2_rf:.4f}")

# Step 8: Compare results
print("\n" + "=" * 50)
print("8. MODEL COMPARISON SUMMARY")
print("=" * 50)
print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
print("-" * 50)

for model_name, metrics in results.items():
    print(f"{model_name:<20} {metrics['MAE']:<10.4f} {metrics['RMSE']:<10.4f} {metrics['R²']:<10.4f}")

# Identify best model
best_model = max(results, key=lambda x: results[x]['R²'])
print("\n" + "=" * 50)
print(f"🏆 BEST MODEL: {best_model} with R² = {results[best_model]['R²']:.4f}")
print("=" * 50)

# Step 9: Feature importance (for Random Forest)
if best_model == 'Random Forest':
    print("\n9. Top 10 Most Important Features (Random Forest):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    for idx, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

# Step 10: Save the best model (optional)
import joblib
if best_model == 'Random Forest':
    joblib.dump(rf_model, 'best_restaurant_rating_model.pkl')
    print("\n   ✅ Best model saved as 'best_restaurant_rating_model.pkl'")

print("\n" + "=" * 50)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 50)