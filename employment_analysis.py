"""
HR Employee Attrition Prediction - Data Loading and Exploratory Data Analysis
Task 1: Load the dataset and perform comprehensive EDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("HR EMPLOYEE ATTRITION PREDICTION - DATA LOADING AND EDA")
print("=" * 80)
print()

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("1. LOADING DATASET")
print("-" * 80)

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

print(f"Dataset loaded successfully!")
print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print()

# Display first 10 rows
print("First 10 rows of the dataset:")
print(df.head(10))
print()

# Display column names
print("Column Names:")
print(df.columns.tolist())
print()

# ============================================================================
# 2. DATA QUALITY CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("2. DATA QUALITY CHECKS")
print("-" * 80)

# Check for null values
print("\nNull Values Check:")
null_counts = df.isnull().sum()
null_percentages = (df.isnull().sum() / len(df)) * 100
null_summary = pd.DataFrame({
    'Column': null_counts.index,
    'Null Count': null_counts.values,
    'Percentage': null_percentages.values
})
null_summary = null_summary[null_summary['Null Count'] > 0]

if len(null_summary) > 0:
    print(null_summary.to_string(index=False))
else:
    print("✓ No missing values found in the dataset!")
print()

# Display data types
print("\nData Types:")
dtype_summary = pd.DataFrame({
    'Column': df.dtypes.index,
    'Data Type': df.dtypes.values
})
print(dtype_summary.to_string(index=False))
print()

# Identify numerical vs categorical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical Features ({len(numerical_features)}):")
print(numerical_features)
print()

print(f"Categorical Features ({len(categorical_features)}):")
print(categorical_features)
print()

# ============================================================================
# 3. TARGET VARIABLE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. TARGET VARIABLE ANALYSIS (Attrition)")
print("-" * 80)

# Distribution of target variable
attrition_counts = df['Attrition'].value_counts()
attrition_percentages = df['Attrition'].value_counts(normalize=True) * 100

print("\nAttrition Distribution:")
print(f"No:  {attrition_counts['No']} ({attrition_percentages['No']:.2f}%)")
print(f"Yes: {attrition_counts['Yes']} ({attrition_percentages['Yes']:.2f}%)")
print()

# Visualize target variable distribution
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
sns.countplot(data=df, x='Attrition', palette='Set2', ax=ax[0])
ax[0].set_title('Attrition Distribution (Count)', fontsize=14, fontweight='bold')
ax[0].set_xlabel('Attrition', fontsize=12)
ax[0].set_ylabel('Count', fontsize=12)
for container in ax[0].containers:
    ax[0].bar_label(container)

# Percentage pie chart
colors = sns.color_palette('Set2', 2)
ax[1].pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%',
          startangle=90, colors=colors)
ax[1].set_title('Attrition Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('attrition_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: attrition_distribution.png")
plt.close()

# ============================================================================
# 4. NUMERICAL FEATURES ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. NUMERICAL FEATURES ANALYSIS")
print("-" * 80)

# Summary statistics
print("\nSummary Statistics for Numerical Features:")
print(df[numerical_features].describe())
print()

# Select key numerical features for visualization
key_numerical = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 
                 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Create histograms for key numerical features
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(key_numerical):
    if idx < len(axes):
        df[col].hist(bins=30, ax=axes[idx], color='skyblue', edgecolor='black')
        axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(key_numerical), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('numerical_features_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: numerical_features_distribution.png")
plt.close()

# ============================================================================
# 5. CATEGORICAL FEATURES ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. CATEGORICAL FEATURES ANALYSIS")
print("-" * 80)

# Value counts for key categorical features
key_categorical = ['Department', 'JobRole', 'EducationField', 'Gender', 
                   'MaritalStatus', 'BusinessTravel', 'OverTime']

print("\nValue Counts for Key Categorical Features:\n")
for col in key_categorical:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts())
        print()

# Visualize key categorical features
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.ravel()

for idx, col in enumerate(key_categorical):
    if idx < len(axes) and col in df.columns:
        value_counts = df[col].value_counts()
        axes[idx].barh(value_counts.index, value_counts.values, color='coral', edgecolor='black')
        axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Count', fontsize=10)
        axes[idx].set_ylabel(col, fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis='x')

# Hide unused subplots
for idx in range(len(key_categorical), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('categorical_features_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: categorical_features_distribution.png")
plt.close()

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. CORRELATION ANALYSIS")
print("-" * 80)

# Calculate correlation matrix for numerical features
correlation_matrix = df[numerical_features].corr()

# Visualize correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: correlation_matrix.png")
plt.close()

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("7. EXPLORATORY DATA ANALYSIS SUMMARY")
print("-" * 80)

print(f"""
Key Findings:
- Dataset contains {df.shape[0]} employee records with {df.shape[1]} features
- No missing values detected
- Target variable (Attrition): {attrition_percentages['No']:.2f}% No, {attrition_percentages['Yes']:.2f}% Yes
- {len(numerical_features)} numerical features and {len(categorical_features)} categorical features
- Dataset includes demographics, job details, satisfaction metrics, and work history

Next Steps:
- Data preprocessing (encoding categorical variables)
- Feature selection and engineering
- Build Decision Tree Classification Model
- Model evaluation and optimization
""")

print("=" * 80)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 80)

