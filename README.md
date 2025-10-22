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

## Task 3: Model Building and Training ✓

### Completed Steps:
1. **Decision Tree Classifier Initialization**
   - Algorithm: CART (Classification and Regression Trees)
   - Criterion: Gini Impurity
   - Class Weight: Balanced (to handle class imbalance)
   - Random State: 42 (reproducibility)

2. **Model Training**
   - Trained on 1,176 samples with 43 features
   - Model Structure:
     - Tree Depth: 15
     - Number of Leaves: 156
     - All 43 features used

3. **Model Performance**
   
   **Training Set:**
   - Accuracy: 100.00% (Perfect fit - indicates overfitting)
   - Precision: 1.0000
   - Recall: 1.0000
   - F1-Score: 1.0000
   
   **Testing Set:**
   - Accuracy: 76.53%
   - Precision: 0.2609 (for attrition class)
   - Recall: 0.2553 (for attrition class)
   - F1-Score: 0.2581
   - ROC-AUC: 0.5588

4. **Confusion Matrix (Test Set)**
   - True Negatives: 213 (correctly predicted no attrition)
   - False Positives: 34 (incorrectly predicted attrition)
   - False Negatives: 35 (missed actual attrition)
   - True Positives: 12 (correctly predicted attrition)

5. **Feature Importance Analysis**
   
   **Top 10 Most Important Features:**
   1. TotalWorkingYears (12.04%)
   2. Age (9.64%)
   3. OverTime (7.17%)
   4. DailyRate (7.17%)
   5. MonthlyIncome (7.00%)
   6. NumCompaniesWorked (6.40%)
   7. StockOptionLevel (6.31%)
   8. YearsSinceLastPromotion (4.40%)
   9. DistanceFromHome (3.82%)
   10. PercentSalaryHike (3.82%)

6. **Generated Outputs**
   - `decision_tree_model.pkl` - Trained model
   - `y_train_predictions.csv` - Training predictions
   - `y_test_predictions.csv` - Testing predictions
   - `feature_importance.csv` - Complete feature rankings
   - `confusion_matrix.png` - Confusion matrix visualization
   - `roc_curve.png` - ROC curve (AUC = 0.5588)
   - `feature_importance.png` - Top 20 features chart
   - `decision_tree_visualization.png` - Tree structure
   - `model_evaluation_report.txt` - Comprehensive analysis

### Key Findings:
⚠️ **Significant Overfitting Detected**
- Training accuracy (100%) >> Testing accuracy (76.53%)
- Difference of 23.47% indicates the model memorized training data

⚠️ **Poor Minority Class Performance**
- Recall for attrition: 25.53% (missed 74% of employees who left)
- Precision for attrition: 26.09% (many false alarms)
- Low ROC-AUC (0.5588) suggests limited discrimination ability

✓ **Feature Insights**
- Employee tenure and age are strongest predictors
- Overtime work is a significant factor
- Compensation-related features are important

### Recommendations for Improvement:
1. **Address Overfitting:**
   - Prune the tree (limit max_depth to 5-10)
   - Set min_samples_split and min_samples_leaf
   - Implement cross-validation

2. **Improve Minority Class Detection:**
   - Try SMOTE for synthetic sampling
   - Adjust decision threshold
   - Consider ensemble methods (Random Forest, XGBoost)

3. **Hyperparameter Tuning:**
   - Use GridSearchCV or RandomizedSearchCV
   - Optimize max_depth, min_samples_split, min_samples_leaf
   - Experiment with different criteria (gini vs entropy)

## Task 4: Enhanced Model Visualization ✓

### Completed Steps:
1. **Multiple plot_tree() Visualizations**
   - Created visualizations at different detail levels
   - Used matplotlib for high-quality PNG outputs
   - All visualizations use color coding and proper labels

2. **Visualization Outputs Created:**
   
   **Quick Overview:**
   - `tree_top3_levels.png` - Top 3 levels (main decision points)
     - Shows the most important splits
     - Best for presentations and quick understanding
     - Most readable and clearest visualization
   
   **Moderate Detail:**
   - `tree_top5_levels.png` - Top 5 levels
     - Balance between detail and readability
     - Good for technical discussions
   
   **Complete Structure:**
   - `tree_full_visualization.png` - All 15 levels (LARGE FILE)
     - Shows complete tree structure
     - All 311 nodes and 156 leaves
     - Use for detailed analysis and zooming
   
   **Special Views:**
   - `tree_with_proportions.png` - Top 4 levels with class proportions
     - Shows percentage distributions instead of counts
     - Useful for understanding class balance at each node

3. **Graphviz Outputs**
   - `tree_graphviz.dot` - DOT source file created
   - Can be rendered to PDF/SVG if Graphviz system package is installed
   - Optional enhancement for publication-quality diagrams

4. **Decision Rules & Analysis**
   
   **Text-Based Rules:**
   - `tree_rules.txt` - Complete decision rules (if-then format)
     - Top 5 levels exported for readability
     - Shows exact thresholds and conditions
     - Easy to copy and reference
   
   **Decision Path Examples:**
   - `tree_decision_paths.txt` - Analyzed example predictions
     - True Positive: Correctly caught attrition
     - False Negative: Missed attrition case  
     - True Negative: Correctly predicted retention
     - False Positive: False alarm case
     - Shows exact path from root to leaf for each

5. **Tree Structure Analysis**
   - `tree_structure_analysis.txt` - Comprehensive statistics
     - 311 total nodes, 156 leaf nodes
     - Depth: 15 levels
     - Feature usage counts for all splits
   
   - `tree_structure_analysis.png` - Visual statistics
     - Samples per node distribution
     - Gini impurity distribution
     - Top 10 features used in splits
     - Summary statistics panel

6. **Comprehensive Report**
   - `tree_visualization_report.txt` - Full documentation
     - How to use each visualization
     - Interpretation guidelines
     - Key insights and recommendations
     - Business context and applications

### Key Insights from Visualizations:

**Primary Decision Point (Root Node):**
- **Feature**: TotalWorkingYears
- **Threshold**: ≤ 2.50 years
- This single split divides the dataset into two major groups
- Employees with ≤2.5 years experience have higher attrition risk

**Most Used Features in Tree:**
1. Age (14 splits throughout tree)
2. MonthlyIncome (12 splits)
3. DailyRate (10 splits)
4. MonthlyRate (9 splits)
5. DistanceFromHome (8 splits)

**Tree Complexity Indicators:**
- 311 nodes total (very complex!)
- Average 36.9 samples per node
- Median 7.0 samples per node (many small leaf nodes)
- This confirms overfitting - many branches with few samples

**Decision Path Insights:**
- True Positive case: 6-node path using TotalWorkingYears, OverTime, MonthlyIncome
- False Negative case: 4-node path - tree predicted retention but employee left
- Paths can be very deep (up to 15 levels) showing model complexity

### Visualization Usage Guide:

**For Stakeholders/Presentations:**
- Use `tree_top3_levels.png`
- Shows only the most critical 3 decision points
- Easy to explain: "First we check TotalWorkingYears..."

**For Technical Teams:**
- Use `tree_top5_levels.png` or `tree_structure_analysis.png`
- Shows enough detail to understand logic
- Good balance of information

**For Deep Analysis:**
- Use `tree_full_visualization.png` 
- Can zoom in to inspect specific branches
- Useful for debugging model decisions

**For Documentation:**
- Reference `tree_rules.txt` for exact rules
- Include `tree_decision_paths.txt` for examples
- Cite `tree_visualization_report.txt` for methodology

### Files Generated (11 total):
✓ 5 PNG visualizations (plot_tree)  
✓ 1 DOT file (Graphviz source)  
✓ 3 Text analysis files (rules, paths, structure)  
✓ 1 PNG structure analysis  
✓ 1 Comprehensive report

### Optional Enhancement:
Install Graphviz system package to render DOT file to PDF/SVG:
1. Download from: https://graphviz.org/download/
2. Add to system PATH
3. Run script again to generate vector formats

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

### Run the Model Training Script
```bash
python model_training.py
```

This will:
- Load preprocessed training and testing data
- Train Decision Tree Classification Model
- Generate predictions and evaluation metrics
- Create visualizations (confusion matrix, ROC curve, feature importance)
- Save trained model and comprehensive evaluation report

### Run the Tree Visualization Script
```bash
python tree_visualization.py
```

This will:
- Load the trained Decision Tree model
- Create multiple tree visualizations at different detail levels
- Extract and document decision rules
- Analyze example decision paths
- Generate tree structure statistics
- Save comprehensive visualization report

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

2. ~~**Build Decision Tree Model**~~ ✓ COMPLETED
   - ~~Initialize DecisionTreeClassifier~~ ✓
   - ~~Train on X_train and y_train~~ ✓
   - ~~Make predictions on X_test~~ ✓

3. ~~**Model Evaluation**~~ ✓ COMPLETED
   - ~~Accuracy, Precision, Recall, F1-Score~~ ✓
   - ~~Confusion Matrix~~ ✓
   - ~~ROC-AUC Curve~~ ✓
   - ~~Feature Importance Analysis~~ ✓

4. ~~**Model Visualization**~~ ✓ COMPLETED
   - ~~Visualize decision tree using plot_tree()~~ ✓
   - ~~Create multiple visualization levels~~ ✓
   - ~~Extract decision rules~~ ✓
   - ~~Analyze decision paths~~ ✓
   - ~~Generate structure analysis~~ ✓

5. **Model Optimization** (Recommended Next Task)
   - Address overfitting through hyperparameter tuning
   - Tune: max_depth (5-10), min_samples_split (10-50), min_samples_leaf (5-20)
   - Implement cross-validation for robust evaluation
   - Try SMOTE for better minority class handling
   - Compare with ensemble methods (Random Forest, XGBoost)
   - Optimize decision threshold for better recall

## Project Structure
```
Employment-Machine-Learning-Model/
│
├── WA_Fn-UseC_-HR-Employee-Attrition.csv    # Original dataset
│
├── employment_analysis.py                    # Task 1: EDA script
├── data_preprocessing.py                     # Task 2: Preprocessing script
├── model_training.py                         # Task 3: Model training script
├── tree_visualization.py                     # Task 4: Tree visualization script
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
├── PREPROCESSING_SUMMARY.txt                 # Data preprocessing summary
├── MODEL_TRAINING_SUMMARY.txt                # Model training summary
│
├── X_train.csv                               # Training features (1,176 × 43)
├── X_test.csv                                # Testing features (294 × 43)
├── y_train.csv                               # Training labels
├── y_test.csv                                # Testing labels
│
├── decision_tree_model.pkl                   # Trained Decision Tree model
├── y_train_predictions.csv                   # Training set predictions
├── y_test_predictions.csv                    # Testing set predictions
├── feature_importance.csv                    # Feature importance rankings
├── model_evaluation_report.txt               # Comprehensive model evaluation
│
├── attrition_distribution.png               # EDA: Target variable distribution
├── numerical_features_distribution.png       # EDA: Numerical features
├── categorical_features_distribution.png     # EDA: Categorical features
├── correlation_matrix.png                    # EDA: Feature correlations
│
├── confusion_matrix.png                      # Model: Confusion matrices
├── roc_curve.png                             # Model: ROC curve (AUC)
├── feature_importance.png                    # Model: Top 20 features
├── decision_tree_visualization.png           # Model: Tree structure (depth 3)
│
├── tree_top3_levels.png                      # Tree Viz: Top 3 levels (best for presentations)
├── tree_top5_levels.png                      # Tree Viz: Top 5 levels (moderate detail)
├── tree_full_visualization.png               # Tree Viz: Complete tree (all 15 levels)
├── tree_with_proportions.png                 # Tree Viz: With class proportions
├── tree_structure_analysis.png               # Tree Viz: Structure statistics
│
├── tree_graphviz.dot                         # Tree: Graphviz DOT source
├── tree_rules.txt                            # Tree: Decision rules (if-then format)
├── tree_decision_paths.txt                   # Tree: Example decision paths
├── tree_structure_analysis.txt               # Tree: Detailed statistics
└── tree_visualization_report.txt             # Tree: Comprehensive visualization guide
```

## Author
Created as part of a machine learning project to predict employee attrition using Decision Tree Classification.

## License
This project uses the IBM HR Analytics Employee Attrition dataset for educational purposes.

