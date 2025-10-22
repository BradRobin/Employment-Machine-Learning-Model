# HR Employee Attrition Prediction - Decision Tree Classification Model

## Project Overview
This project builds a Decision Tree Classification Model to predict employee attrition (whether an employee will leave the company or not) based on various HR features from the IBM HR Analytics Employee Attrition dataset.

## Dataset Information
- **File**: `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Records**: 1,470 employees
- **Features**: 35 columns
- **Target Variable**: `Attrition` (Yes/No)

## Features
The dataset includes:
- **Demographics**: Age, Gender, MaritalStatus, DistanceFromHome
- **Job Details**: Department, JobRole, JobLevel, BusinessTravel, OverTime
- **Compensation**: MonthlyIncome, HourlyRate, DailyRate, MonthlyRate
- **Satisfaction Metrics**: JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance
- **Work History**: TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
- **Education**: Education, EducationField
- **Performance**: PerformanceRating, PercentSalaryHike

## Task 1: Data Loading and Exploratory Data Analysis ✓

### Completed Steps:
1. **Data Loading**
   - Successfully loaded 1,470 employee records
   - Verified dataset structure (35 features)
   - Displayed first 10 rows for inspection

2. **Data Quality Checks**
   - ✓ No missing values found in any column
   - Identified 26 numerical features and 9 categorical features
   - Confirmed appropriate data types for each column

3. **Target Variable Analysis**
   - **Class Distribution**:
     - No Attrition: 1,233 employees (83.88%)
     - Yes Attrition: 237 employees (16.12%)
   - Note: Dataset is imbalanced (need to address in modeling phase)

4. **Numerical Features Analysis**
   - Generated comprehensive summary statistics
   - Key insights:
     - Average age: ~37 years
     - Average tenure: ~7 years at company
     - Average monthly income: $6,503
     - Average total working years: ~11 years

5. **Categorical Features Analysis**
   - **Department**: Research & Development (65%), Sales (30%), HR (4%)
   - **Gender**: Male (60%), Female (40%)
   - **Overtime**: No (72%), Yes (28%)
   - **Business Travel**: Travel_Rarely (71%), Travel_Frequently (19%), Non-Travel (10%)

6. **Correlation Analysis**
   - Generated correlation matrix for numerical features
   - Strong correlations observed between:
     - Age and TotalWorkingYears
     - YearsAtCompany and YearsInCurrentRole
     - MonthlyIncome and JobLevel

### Generated Visualizations:
1. `attrition_distribution.png` - Target variable distribution
2. `numerical_features_distribution.png` - Histograms of key numerical features
3. `categorical_features_distribution.png` - Bar charts of categorical features
4. `correlation_matrix.png` - Heatmap showing feature correlations

## Installation & Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0

## Usage

### Run the EDA Script
```bash
python employment_analysis.py
```

This will:
- Load and display the dataset
- Perform comprehensive exploratory data analysis
- Generate statistical summaries
- Create visualizations (saved as PNG files)

## Key Findings

### Data Quality
✓ **No missing values** - Dataset is complete and ready for modeling
✓ **No duplicates** - All employee records are unique
✓ **Consistent data types** - Proper numerical and categorical separation

### Target Variable Insights
- **Class Imbalance**: 83.88% No Attrition vs 16.12% Yes Attrition
- **Action Required**: Consider using SMOTE, class weighting, or other techniques to handle imbalance

### Feature Insights
- Employees who leave tend to be younger and have less tenure
- Overtime work shows correlation with attrition
- Job satisfaction and work-life balance are important factors
- Business travel frequency may impact attrition decisions

## Next Steps

1. **Data Preprocessing**
   - Encode categorical variables (One-Hot Encoding / Label Encoding)
   - Handle class imbalance
   - Feature scaling/normalization
   - Remove irrelevant features (EmployeeCount, StandardHours, Over18)

2. **Feature Engineering**
   - Create interaction features
   - Derive new meaningful features
   - Feature selection based on importance

3. **Model Building**
   - Build Decision Tree Classifier
   - Tune hyperparameters
   - Implement cross-validation

4. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC-AUC Curve
   - Feature Importance Analysis

5. **Model Optimization**
   - Hyperparameter tuning (max_depth, min_samples_split, etc.)
   - Pruning strategies
   - Compare with ensemble methods (Random Forest, Gradient Boosting)

## Project Structure
```
Employment-Machine-Learning-Model/
│
├── WA_Fn-UseC_-HR-Employee-Attrition.csv    # Dataset
├── employment_analysis.py                    # EDA script
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
│
├── attrition_distribution.png               # Visualization: Target variable
├── numerical_features_distribution.png       # Visualization: Numerical features
├── categorical_features_distribution.png     # Visualization: Categorical features
└── correlation_matrix.png                    # Visualization: Feature correlations
```

## Author
Created as part of a machine learning project to predict employee attrition using Decision Tree Classification.

## License
This project uses the IBM HR Analytics Employee Attrition dataset for educational purposes.

