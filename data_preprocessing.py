"""
HR Employee Attrition Prediction - Data Preprocessing
Task 2: Convert categorical variables and split dataset into train/test sets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 80)
print("HR EMPLOYEE ATTRITION PREDICTION - DATA PREPROCESSING")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("1. LOADING DATASET")
print("-" * 80)

df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(f"Original Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print()

# ============================================================================
# 2. DATA CLEANING - REMOVE IRRELEVANT FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("2. DATA CLEANING")
print("-" * 80)

# Identify constant or irrelevant features
print("\nChecking for constant-value features:")
constant_features = []
for col in df.columns:
    if df[col].nunique() == 1:
        constant_features.append(col)
        print(f"  - {col}: {df[col].unique()}")

# Remove irrelevant features
features_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
print(f"\nRemoving irrelevant features: {features_to_drop}")
df_cleaned = df.drop(columns=features_to_drop, errors='ignore')
print(f"Dataset Shape After Cleaning: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
print()

# ============================================================================
# 3. CATEGORICAL VARIABLE ENCODING
# ============================================================================
print("\n" + "=" * 80)
print("3. CATEGORICAL VARIABLE ENCODING")
print("-" * 80)

# Create a copy for encoding
df_encoded = df_cleaned.copy()

# 3.1 Binary Variables - Label Encoding
print("\n3.1 Binary Variables (Label Encoding):")
print("-" * 40)

# Attrition (Target Variable)
print("\nAttrition (Target):")
print(f"  Before: {df_encoded['Attrition'].unique()}")
df_encoded['Attrition'] = df_encoded['Attrition'].map({'No': 0, 'Yes': 1})
print(f"  After:  {df_encoded['Attrition'].unique()} (No=0, Yes=1)")

# Gender
print("\nGender:")
print(f"  Before: {df_encoded['Gender'].unique()}")
df_encoded['Gender'] = df_encoded['Gender'].map({'Female': 0, 'Male': 1})
print(f"  After:  {df_encoded['Gender'].unique()} (Female=0, Male=1)")

# OverTime
print("\nOverTime:")
print(f"  Before: {df_encoded['OverTime'].unique()}")
df_encoded['OverTime'] = df_encoded['OverTime'].map({'No': 0, 'Yes': 1})
print(f"  After:  {df_encoded['OverTime'].unique()} (No=0, Yes=1)")

# 3.2 Ordinal Variables - Label Encoding
print("\n\n3.2 Ordinal Variables (Label Encoding):")
print("-" * 40)

# BusinessTravel
print("\nBusinessTravel:")
print(f"  Before: {df_encoded['BusinessTravel'].unique()}")
travel_mapping = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
df_encoded['BusinessTravel'] = df_encoded['BusinessTravel'].map(travel_mapping)
print(f"  After:  {df_encoded['BusinessTravel'].unique()}")
print(f"  Mapping: {travel_mapping}")

# 3.3 Nominal Variables - One-Hot Encoding
print("\n\n3.3 Nominal Variables (One-Hot Encoding):")
print("-" * 40)

nominal_features = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']

print(f"\nFeatures to encode: {nominal_features}")
print("\nBefore encoding:")
for feature in nominal_features:
    print(f"  {feature}: {df_encoded[feature].nunique()} categories - {list(df_encoded[feature].unique())}")

# Apply one-hot encoding with drop_first=True to avoid multicollinearity
df_encoded = pd.get_dummies(df_encoded, columns=nominal_features, drop_first=True, dtype=int)

print(f"\nAfter one-hot encoding:")
print(f"  Total features: {df_encoded.shape[1]}")
print(f"  New columns created: {df_encoded.shape[1] - df_cleaned.shape[1] + len(nominal_features)}")
print()

# Display column names after encoding
print("All columns after encoding:")
print(df_encoded.columns.tolist())
print()

# ============================================================================
# 4. PREPARE FOR TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("4. PREPARING DATA FOR MODELING")
print("-" * 80)

# Separate features (X) and target variable (y)
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

print(f"\nFeature Matrix (X): {X.shape}")
print(f"Target Vector (y): {y.shape}")
print(f"\nTarget Distribution:")
print(f"  No Attrition (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
print(f"  Yes Attrition (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")
print()

# ============================================================================
# 5. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("5. TRAIN-TEST SPLIT (80% / 20%)")
print("-" * 80)

# Split with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\nTraining Set:")
print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  Class distribution:")
print(f"    No Attrition (0): {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)")
print(f"    Yes Attrition (1): {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.2f}%)")

print(f"\nTesting Set:")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_test shape: {y_test.shape}")
print(f"  Class distribution:")
print(f"    No Attrition (0): {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.2f}%)")
print(f"    Yes Attrition (1): {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.2f}%)")
print()

# ============================================================================
# 6. SAVE PREPROCESSED DATA
# ============================================================================
print("\n" + "=" * 80)
print("6. SAVING PREPROCESSED DATA")
print("-" * 80)

# Save to CSV files
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False, header=True)
y_test.to_csv('y_test.csv', index=False, header=True)

print("\n✓ Saved: X_train.csv")
print("✓ Saved: X_test.csv")
print("✓ Saved: y_train.csv")
print("✓ Saved: y_test.csv")
print()

# ============================================================================
# 7. PREPROCESSING SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("7. PREPROCESSING SUMMARY")
print("-" * 80)

print(f"""
Data Transformation Summary:
----------------------------
Original Features:        {df.shape[1]}
After Cleaning:          {df_cleaned.shape[1]}
After Encoding:          {df_encoded.shape[1]}
Final Feature Count:     {X.shape[1]}

Encoding Applied:
-----------------
Label Encoding:          3 binary variables + 1 ordinal variable
  - Attrition (target): Yes=1, No=0
  - Gender: Male=1, Female=0
  - OverTime: Yes=1, No=0
  - BusinessTravel: Non-Travel=0, Travel_Rarely=1, Travel_Frequently=2

One-Hot Encoding:        4 nominal variables
  - Department ({df_cleaned['Department'].nunique()} categories)
  - EducationField ({df_cleaned['EducationField'].nunique()} categories)
  - JobRole ({df_cleaned['JobRole'].nunique()} categories)
  - MaritalStatus ({df_cleaned['MaritalStatus'].nunique()} categories)

Train-Test Split:
-----------------
Training Set:            {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)
Testing Set:             {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)
Stratification:          ✓ Class distribution maintained

Class Balance (Training):
  No Attrition:          {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)
  Yes Attrition:         {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.2f}%)

Class Balance (Testing):
  No Attrition:          {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.2f}%)
  Yes Attrition:         {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.2f}%)
""")

# Sample of encoded data
print("\nSample of Preprocessed Training Data (first 5 rows):")
print("-" * 80)
sample_df = pd.concat([X_train.head(), y_train.head()], axis=1)
print(sample_df)
print()

print("=" * 80)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nNext Steps:")
print("- Build Decision Tree Classification Model")
print("- Train the model on X_train and y_train")
print("- Evaluate performance on X_test and y_test")
print("=" * 80)

