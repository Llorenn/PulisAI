"""
XGBoost Model Training Script
Aggregates the entire dataset rather than splitting by date before aggregation.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("="*60)
print("XGBOOST MODEL TRAINING - PULISAI")
print("="*60)

# --- 1. Load Data ---
dataset_path = 'data/focus_df.csv'

try:
    focus_df = pd.read_csv(dataset_path, parse_dates=['Date'])
    print(f"\nLoaded focus_df with shape: {focus_df.shape}")
    print(f"   Date range: {focus_df['Date'].min()} to {focus_df['Date'].max()}")
except FileNotFoundError as e:
    print(f"Error: {e}")

    exit()

# --- 2. Use ENTIRE dataset for training (matching notebook approach) ---
print("\n" + "="*60)
print("DATA PREPARATION")
print("="*60)

train_df = focus_df
print(f"\nUsing entire dataset for training: {train_df.shape}")

# --- 3. Aggregation Function ---
def aggregate(df):
    """
    Aggregate crime data by Barangay, Month, Weekday, and Time_of_Day
    """
    print("\nAggregating data...")

    agg_df = df.groupby(['Barangay', 'Month', 'Weekday', 'Time_of_Day']).agg(
        Crime_Count=('Offense ID', 'count'),

        # Temporal features
        Avg_Hour=('Hour', 'mean'),
        Mode_Hour=('Hour', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]),
        Weekend_Crimes=('Is_Weekend', 'sum'),
        Weekday_Crimes=('Is_Weekend', lambda x: (~x.astype(bool)).sum()),

        # Spatial/demographic
        Population=('Population_2024', 'first'),
        Pop_Density=('Pop_Density_2024', 'first'),
        Area_sqkm=('Area_sqkm', 'first'),

        # Police presence
        Avg_Distance_Police=('Distance_from_Police', 'mean'),
        Avg_Num_Stations_1km=('Num_Police_Stations_1km', 'mean'),

        # Crime characteristics
        Avg_Victims=('Num_Victims', 'median'),
        Avg_Suspects=('Num_Suspects', 'median'),

        # Focus crime distribution
        Murder_Count=('Focus_Crime', lambda x: (x == 'Murder').sum()),
        Theft_Count=('Focus_Crime', lambda x: (x == 'Theft').sum()),
        Robbery_Count=('Focus_Crime', lambda x: (x == 'Robbery').sum()),
        Physical_Injuries_Count=('Focus_Crime', lambda x: (x == 'Physical Injuries').sum()),
        Rape_Count=('Focus_Crime', lambda x: (x == 'Rape').sum()),
        Homicide_Count=('Focus_Crime', lambda x: (x == 'Homicide').sum()),
        Carnapping_MC_Count=('Focus_Crime', lambda x: (x == 'Carnapping MC').sum()),
        Carnapping_MV_Count=('Focus_Crime', lambda x: (x == 'Carnapping MV').sum())
    ).reset_index()

    # Derived features
    agg_df['Crime_Rate_per_1000'] = (agg_df['Crime_Count'] / agg_df['Population']) * 1000
    agg_df['Crime_Density_sqkm'] = agg_df['Crime_Count'] / agg_df['Area_sqkm']
    agg_df['Weekend_Ratio'] = agg_df['Weekend_Crimes'] / (agg_df['Crime_Count'] + 1e-6)

    # Lag features
    agg_df = agg_df.sort_values(['Barangay', 'Month', 'Weekday', 'Time_of_Day'])


    print(f"   Aggregated dataset shape: {agg_df.shape}")

    return agg_df

train_df_agg = aggregate(train_df)

# --- 4. Define Alarm Level ---
print("\n" + "="*60)
print("ALARM LEVEL CLASSIFICATION")
print("="*60)

q25_baseline = train_df_agg['Crime_Count'].quantile(0.25)
q75_baseline = train_df_agg['Crime_Count'].quantile(0.75)
mean_crime_count = train_df_agg['Crime_Count'].mean()

print(f"\nCrime Count Statistics:")
print(f"   Mean:           {mean_crime_count:.3f}")
print(f"   25th Percentile: {q25_baseline:.2f} (Low/Medium boundary)")
print(f"   75th Percentile: {q75_baseline:.2f} (Medium/High boundary)")

def classify_alarm(count, q25, q75):
    """Classify crime count into alarm levels"""
    if count <= q25:
        return 'Low'
    elif count <= q75:
        return 'Medium'
    else:
        return 'High'

# Apply classification
train_df_agg['Alarm_Level'] = train_df_agg['Crime_Count'].apply(
    lambda x: classify_alarm(x, q25_baseline, q75_baseline)
)

print("\nAlarm Level Distribution:")
alarm_counts = train_df_agg['Alarm_Level'].value_counts().sort_index()
for level in ['Low', 'Medium', 'High']:
    count = alarm_counts.get(level, 0)
    percentage = (count / len(train_df_agg)) * 100
    print(f"   {level:8s}: {count:4d} records ({percentage:5.2f}%)")

# --- 5. Prepare Training Data ---
print("\n" + "="*60)
print("FEATURE SELECTION")
print("="*60)

selected_features = [
    'Population',
    'Area_sqkm',
    'Avg_Num_Stations_1km',
    'Weekend_Ratio',
    'Avg_Hour',
    'Avg_Victims',
    'Avg_Suspects'
]

print(f"\nSelected features ({len(selected_features)}):")
for i, feat in enumerate(selected_features, 1):
    print(f"   {i}. {feat}")

X = train_df_agg[selected_features]
y_raw = train_df_agg['Alarm_Level']
y = y_raw.map({'Low': 0, 'Medium': 1, 'High': 2})

# Train/test split
rand_seed = 42
test_size = 0.25

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=rand_seed, stratify=y
)

print(f"\nDataset split:")
print(f"   X_train: {X_train.shape}")
print(f"   X_test:  {X_test.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   y_test:  {y_test.shape}")

# --- 6. Train XGBoost Model ---
print("\n" + "="*60)
print("XGBOOST MODEL TRAINING")
print("="*60)

param_grid = {
# Controls the complexity of the trees
'max_depth': [5],

# Controls the step size. Smaller values require more trees.
 'learning_rate': [0.05],

# Number of boosting rounds.
'n_estimators': [400],

# Regularization parameters to prevent overfitting
'gamma': [0], # Minimum loss reduction to make a split
'subsample': [0.9], # Fraction of training data to use per tree
'colsample_bytree': [0.7], # Fraction of features to use per tree

# L1 and L2 regularization
'reg_alpha': [0.01], # L1 regularization
'reg_lambda': [1.5] # L2 regularization
}

print("\nStarting RandomizedSearchCV...")
print(f"   Cross-validation folds: 5")
print(f"   Random state: {rand_seed}")

xgb_model = XGBClassifier(random_state=rand_seed)

xgb_grid_search = RandomizedSearchCV(
    xgb_model, param_grid, cv=5, scoring="accuracy",
    n_jobs=-1, random_state=rand_seed, verbose=1
)

xgb_grid_search.fit(X_train, y_train)

best_xgb = xgb_grid_search.best_estimator_

# --- 7. Evaluate Model ---
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

print("\nBest Hyperparameters:")
for param, value in xgb_grid_search.best_params_.items():
    print(f"   {param}: {value}")

y_pred = best_xgb.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = best_xgb.score(X_train, y_train)
cv_score = xgb_grid_search.best_score_
gap_score = train_accuracy - test_accuracy

print(f"\nModel Performance:")
print(f"   Training Accuracy:      {train_accuracy:.4f}")
print(f"   Cross-Validation Score: {cv_score:.4f}")
print(f"   Test Accuracy (25%):    {test_accuracy:.4f}")
print(f"   Gap Score:              {gap_score:.4f}", end="")

if gap_score < 0.05:
    gap_label = 'Good Fit'
elif gap_score >= 0.05 and gap_score < 0.15:
    gap_label = 'Overfit (Moderate)'
else:
    gap_label = 'Underfit'

print(f" ({gap_label})")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("   (Rows=Actual, Columns=Predicted: [Low, Medium, High])")

# --- 8. Save Model and Configuration ---
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

joblib.dump(best_xgb, 'pulisai_xgb_model.joblib')
joblib.dump(selected_features, 'selected_features.joblib')
joblib.dump({'q25': q25_baseline, 'q75': q75_baseline}, 'alarm_thresholds.joblib')

print("\nModel saved successfully!")
print("   pulisai_xgb_model.joblib")
print("   selected_features.joblib")
print("   alarm_thresholds.joblib")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
