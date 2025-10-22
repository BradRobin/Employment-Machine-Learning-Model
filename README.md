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

## Task 2: Data Preprocessing ✓

### Completed Steps:
1. **Data Cleaning**
   - Removed 4 irrelevant/constant features:
     - `EmployeeCount` (constant value: 1)
     - `EmployeeNumber` (identifier, not predictive)
     - `Over18` (constant value: 'Y')
     - `StandardHours` (constant value: 80)
   - Reduced from 35 to 31 features

2. **Categorical Variable Encoding**
   - **Label Encoding** (3 binary + 1 ordinal variable):
     - `Attrition`: No=0, Yes=1 (target variable)
     - `Gender`: Female=0, Male=1
     - `OverTime`: No=0, Yes=1
     - `BusinessTravel`: Non-Travel=0, Travel_Rarely=1, Travel_Frequently=2
   
   - **One-Hot Encoding** (4 nominal variables):
     - `Department` (3 categories → 2 dummy variables)
     - `EducationField` (6 categories → 5 dummy variables)
     - `JobRole` (9 categories → 8 dummy variables)
     - `MaritalStatus` (3 categories → 2 dummy variables)
   
   - Applied `drop_first=True` to avoid multicollinearity
   - Final feature count: **43 features**

3. **Train-Test Split (80/20)**
   - **Training Set**: 1,176 samples (80%)
     - No Attrition: 986 (83.84%)
     - Yes Attrition: 190 (16.16%)
   
   - **Testing Set**: 294 samples (20%)
     - No Attrition: 247 (84.01%)
     - Yes Attrition: 47 (15.99%)
   
   - Used `stratify=y` to maintain class distribution
   - Random state set to 42 for reproducibility

4. **Saved Preprocessed Data**
   - `X_train.csv` - Training features (1,176 × 43)
   - `y_train.csv` - Training labels (1,176 × 1)
   - `X_test.csv` - Testing features (294 × 43)
   - `y_test.csv` - Testing labels (294 × 1)

### Key Achievements:
✓ All categorical variables successfully encoded
✓ Proper train-test split with stratification
✓ Class balance maintained in both sets
✓ Data ready for model training

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

### Run the Data Preprocessing Script
```bash
python data_preprocessing.py
```

This will:
- Clean the dataset (remove irrelevant features)
- Encode categorical variables (label & one-hot encoding)
- Split data into training (80%) and testing (20%) sets
- Save preprocessed datasets: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

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

1. ~~**Data Preprocessing**~~ ✓ COMPLETED
   - ~~Encode categorical variables (One-Hot Encoding / Label Encoding)~~ ✓
   - ~~Handle class imbalance~~ (Maintained via stratification)
   - ~~Remove irrelevant features (EmployeeCount, StandardHours, Over18)~~ ✓

2. **Build Decision Tree Model**
   - Initialize DecisionTreeClassifier
   - Train on X_train and y_train
   - Make predictions on X_test

3. **Model Evaluation**
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
├── WA_Fn-UseC_-HR-Employee-Attrition.csv    # Original dataset
│
├── employment_analysis.py                    # Task 1: EDA script
├── data_preprocessing.py                     # Task 2: Preprocessing script
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
│
├── X_train.csv                               # Training features (1,176 × 43)
├── X_test.csv                                # Testing features (294 × 43)
├── y_train.csv                               # Training labels
├── y_test.csv                                # Testing labels
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

