"""
HR Employee Attrition Prediction - Model Training
Task 3: Build and Train Decision Tree Classification Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix,
                             roc_curve, roc_auc_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("HR EMPLOYEE ATTRITION PREDICTION - DECISION TREE MODEL TRAINING")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("1. LOADING PREPROCESSED DATA")
print("-" * 80)

# Load training and testing sets
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Convert to arrays for sklearn
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print(f"✓ Training Set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"✓ Testing Set:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"\nTraining Class Distribution:")
print(f"  No Attrition (0): {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)")
print(f"  Yes Attrition (1): {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.2f}%)")
print()

# ============================================================================
# 2. INITIALIZE DECISION TREE CLASSIFIER
# ============================================================================
print("\n" + "=" * 80)
print("2. INITIALIZING DECISION TREE CLASSIFIER")
print("-" * 80)

# Create Decision Tree model with class_weight='balanced' to handle imbalance
dt_model = DecisionTreeClassifier(
    random_state=42,
    criterion='gini',
    class_weight='balanced'  # Handle class imbalance
)

print("\nModel Parameters:")
print(f"  Algorithm: Decision Tree Classifier")
print(f"  Criterion: {dt_model.criterion}")
print(f"  Random State: {dt_model.random_state}")
print(f"  Class Weight: {dt_model.class_weight}")
print()

# ============================================================================
# 3. TRAIN THE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("3. TRAINING THE MODEL")
print("-" * 80)

print("\nTraining Decision Tree on training data...")
dt_model.fit(X_train, y_train)
print("✓ Model training completed successfully!")

print(f"\nTrained Model Information:")
print(f"  Tree Depth: {dt_model.get_depth()}")
print(f"  Number of Leaves: {dt_model.get_n_leaves()}")
print(f"  Number of Features Used: {dt_model.n_features_in_}")
print()

# ============================================================================
# 4. MAKE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("4. GENERATING PREDICTIONS")
print("-" * 80)

# Predictions on training set
y_train_pred = dt_model.predict(X_train)
y_train_proba = dt_model.predict_proba(X_train)[:, 1]

# Predictions on testing set
y_test_pred = dt_model.predict(X_test)
y_test_proba = dt_model.predict_proba(X_test)[:, 1]

print(f"✓ Training predictions generated: {len(y_train_pred)} samples")
print(f"✓ Testing predictions generated: {len(y_test_pred)} samples")
print()

# ============================================================================
# 5. MODEL EVALUATION - TRAINING SET
# ============================================================================
print("\n" + "=" * 80)
print("5. MODEL EVALUATION - TRAINING SET")
print("-" * 80)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

print(f"\nTraining Set Metrics:")
print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")

print(f"\nTraining Set Classification Report:")
print(classification_report(y_train, y_train_pred, target_names=['No Attrition', 'Yes Attrition']))

# ============================================================================
# 6. MODEL EVALUATION - TESTING SET
# ============================================================================
print("\n" + "=" * 80)
print("6. MODEL EVALUATION - TESTING SET")
print("-" * 80)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\nTesting Set Metrics:")
print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  ROC-AUC:   {test_auc:.4f}")

print(f"\nTesting Set Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Attrition', 'Yes Attrition']))

# ============================================================================
# 7. CONFUSION MATRIX VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("7. CONFUSION MATRIX ANALYSIS")
print("-" * 80)

# Confusion matrices
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print("\nTraining Set Confusion Matrix:")
print(cm_train)
print(f"  True Negatives:  {cm_train[0, 0]}")
print(f"  False Positives: {cm_train[0, 1]}")
print(f"  False Negatives: {cm_train[1, 0]}")
print(f"  True Positives:  {cm_train[1, 1]}")

print("\nTesting Set Confusion Matrix:")
print(cm_test)
print(f"  True Negatives:  {cm_test[0, 0]}")
print(f"  False Positives: {cm_test[0, 1]}")
print(f"  False Negatives: {cm_test[1, 0]}")
print(f"  True Positives:  {cm_test[1, 1]}")

# Visualize confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training confusion matrix
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Attrition', 'Yes Attrition'],
            yticklabels=['No Attrition', 'Yes Attrition'])
axes[0].set_title('Training Set Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_xlabel('Predicted', fontsize=12)

# Testing confusion matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['No Attrition', 'Yes Attrition'],
            yticklabels=['No Attrition', 'Yes Attrition'])
axes[1].set_title('Testing Set Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_xlabel('Predicted', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: confusion_matrix.png")
plt.close()

# ============================================================================
# 8. ROC CURVE
# ============================================================================
print("\n" + "=" * 80)
print("8. ROC CURVE ANALYSIS")
print("-" * 80)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curve.png")
plt.close()

# ============================================================================
# 9. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("9. FEATURE IMPORTANCE ANALYSIS")
print("-" * 80)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# Save feature importance to CSV
feature_importance.to_csv('feature_importance.csv', index=False)
print("\n✓ Saved: feature_importance.csv")

# Visualize top 20 features
plt.figure(figsize=(12, 8))
top_20 = feature_importance.head(20)
plt.barh(range(len(top_20)), top_20['Importance'], color='steelblue', edgecolor='black')
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Top 20 Most Important Features - Decision Tree', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
plt.close()

# ============================================================================
# 10. DECISION TREE VISUALIZATION (LIMITED DEPTH)
# ============================================================================
print("\n" + "=" * 80)
print("10. DECISION TREE VISUALIZATION")
print("-" * 80)

# Only visualize if tree is not too deep
if dt_model.get_depth() <= 10:
    plt.figure(figsize=(25, 15))
    plot_tree(dt_model, 
              feature_names=X_train.columns,
              class_names=['No Attrition', 'Yes Attrition'],
              filled=True,
              rounded=True,
              fontsize=8,
              max_depth=3)  # Limit visualization depth for readability
    plt.title('Decision Tree Visualization (Max Depth 3 for Display)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: decision_tree_visualization.png (limited to depth 3 for readability)")
    plt.close()
else:
    print(f"⚠ Tree depth ({dt_model.get_depth()}) too large for full visualization")
    print("  Creating simplified visualization (depth limited to 3)...")
    plt.figure(figsize=(25, 15))
    plot_tree(dt_model, 
              feature_names=X_train.columns,
              class_names=['No Attrition', 'Yes Attrition'],
              filled=True,
              rounded=True,
              fontsize=8,
              max_depth=3)
    plt.title('Decision Tree Visualization (Simplified - Depth 3)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: decision_tree_visualization.png")
    plt.close()

# ============================================================================
# 11. SAVE MODEL AND PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("11. SAVING MODEL AND PREDICTIONS")
print("-" * 80)

# Save the trained model
joblib.dump(dt_model, 'decision_tree_model.pkl')
print("\n✓ Saved: decision_tree_model.pkl")

# Save predictions
pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred, 'Probability': y_train_proba})\
  .to_csv('y_train_predictions.csv', index=False)
print("✓ Saved: y_train_predictions.csv")

pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred, 'Probability': y_test_proba})\
  .to_csv('y_test_predictions.csv', index=False)
print("✓ Saved: y_test_predictions.csv")

# ============================================================================
# 12. GENERATE COMPREHENSIVE EVALUATION REPORT
# ============================================================================
print("\n" + "=" * 80)
print("12. GENERATING EVALUATION REPORT")
print("-" * 80)

report_content = f"""
================================================================================
HR EMPLOYEE ATTRITION PREDICTION - MODEL EVALUATION REPORT
================================================================================

Date: Generated automatically
Model: Decision Tree Classifier
Algorithm: CART (Classification and Regression Trees)
Criterion: Gini Impurity
Class Weight: Balanced (to handle class imbalance)

================================================================================
DATASET INFORMATION
================================================================================

Training Set: {X_train.shape[0]} samples, {X_train.shape[1]} features
Testing Set:  {X_test.shape[0]} samples, {X_test.shape[1]} features

Class Distribution (Training):
  No Attrition (0): {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)
  Yes Attrition (1): {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.2f}%)

Class Distribution (Testing):
  No Attrition (0): {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.2f}%)
  Yes Attrition (1): {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.2f}%)

================================================================================
MODEL STRUCTURE
================================================================================

Tree Depth: {dt_model.get_depth()}
Number of Leaves: {dt_model.get_n_leaves()}
Number of Features: {dt_model.n_features_in_}

================================================================================
TRAINING SET PERFORMANCE
================================================================================

Accuracy:  {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)
Precision: {train_precision:.4f}
Recall:    {train_recall:.4f}
F1-Score:  {train_f1:.4f}

Confusion Matrix:
  True Negatives:  {cm_train[0, 0]}
  False Positives: {cm_train[0, 1]}
  False Negatives: {cm_train[1, 0]}
  True Positives:  {cm_train[1, 1]}

================================================================================
TESTING SET PERFORMANCE
================================================================================

Accuracy:  {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)
Precision: {test_precision:.4f}
Recall:    {test_recall:.4f}
F1-Score:  {test_f1:.4f}
ROC-AUC:   {test_auc:.4f}

Confusion Matrix:
  True Negatives:  {cm_test[0, 0]}
  False Positives: {cm_test[0, 1]}
  False Negatives: {cm_test[1, 0]}
  True Positives:  {cm_test[1, 1]}

================================================================================
PERFORMANCE COMPARISON (Training vs Testing)
================================================================================

Metric          Training    Testing     Difference
--------------------------------------------------------
Accuracy        {train_accuracy:.4f}      {test_accuracy:.4f}      {abs(train_accuracy - test_accuracy):.4f}
Precision       {train_precision:.4f}      {test_precision:.4f}      {abs(train_precision - test_precision):.4f}
Recall          {train_recall:.4f}      {test_recall:.4f}      {abs(train_recall - test_recall):.4f}
F1-Score        {train_f1:.4f}      {test_f1:.4f}      {abs(train_f1 - test_f1):.4f}

Overfitting Assessment:
{('✓ Model generalizes well (minimal overfitting)' if abs(train_accuracy - test_accuracy) < 0.05 
  else '⚠ Model shows signs of overfitting' if train_accuracy - test_accuracy > 0.10 
  else '⚠ Model may be slightly overfitting')}

================================================================================
TOP 10 MOST IMPORTANT FEATURES
================================================================================

{feature_importance.head(10).to_string(index=False)}

================================================================================
MODEL INTERPRETATION
================================================================================

Key Insights:
1. Model Accuracy: {test_accuracy * 100:.2f}% on unseen test data
2. Recall for Attrition: {test_recall:.4f} (ability to catch employees who will leave)
3. Precision for Attrition: {test_precision:.4f} (accuracy of positive predictions)
4. ROC-AUC Score: {test_auc:.4f} (overall discrimination ability)

Business Context:
- True Positives ({cm_test[1, 1]}): Correctly identified employees at risk of leaving
- False Negatives ({cm_test[1, 0]}): Missed employees who actually left
- False Positives ({cm_test[0, 1]}): Incorrectly flagged stable employees
- True Negatives ({cm_test[0, 0]}): Correctly identified stable employees

The model prioritizes {('recall (catching all potential attritors)' if test_recall > test_precision 
  else 'precision (accurate attrition predictions)')} based on class weighting.

================================================================================
RECOMMENDATIONS
================================================================================

1. Model Performance:
   - {'The model shows good generalization to unseen data.' if abs(train_accuracy - test_accuracy) < 0.05 else 'Consider pruning or regularization to reduce overfitting.'}
   
2. Class Imbalance:
   - Used class_weight='balanced' to handle 84/16 split
   - Consider SMOTE or other sampling techniques for improvement
   
3. Feature Engineering:
   - Focus on top features: {', '.join(feature_importance.head(3)['Feature'].values)}
   - Consider creating interaction features from important variables
   
4. Model Optimization:
   - Tune hyperparameters: max_depth, min_samples_split, min_samples_leaf
   - Implement cross-validation for robust evaluation
   - Compare with ensemble methods (Random Forest, XGBoost)

5. Business Application:
   - Use model for early identification of at-risk employees
   - Implement retention strategies for high-probability attrition cases
   - Regular model retraining with new data

================================================================================
OUTPUT FILES GENERATED
================================================================================

Model:
✓ decision_tree_model.pkl          - Trained Decision Tree model

Predictions:
✓ y_train_predictions.csv          - Training set predictions
✓ y_test_predictions.csv           - Testing set predictions

Feature Analysis:
✓ feature_importance.csv           - Complete feature importance rankings

Visualizations:
✓ confusion_matrix.png             - Confusion matrices (train & test)
✓ roc_curve.png                    - ROC curve with AUC score
✓ feature_importance.png           - Top 20 features chart
✓ decision_tree_visualization.png  - Tree structure visualization

Reports:
✓ model_evaluation_report.txt      - This comprehensive report

================================================================================
END OF REPORT
================================================================================
"""

# Save report to file
with open('model_evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("✓ Saved: model_evaluation_report.txt")
print()

# ============================================================================
# 13. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("13. MODEL TRAINING SUMMARY")
print("-" * 80)

print(f"""
Decision Tree Classification Model - Training Complete!

Model Performance on Test Set:
  ✓ Accuracy:  {test_accuracy * 100:.2f}%
  ✓ Precision: {test_precision:.4f}
  ✓ Recall:    {test_recall:.4f}
  ✓ F1-Score:  {test_f1:.4f}
  ✓ ROC-AUC:   {test_auc:.4f}

Model Characteristics:
  • Tree Depth: {dt_model.get_depth()}
  • Number of Leaves: {dt_model.get_n_leaves()}
  • Top Feature: {feature_importance.iloc[0]['Feature']}

Overfitting Check:
  Training Accuracy: {train_accuracy * 100:.2f}%
  Testing Accuracy:  {test_accuracy * 100:.2f}%
  Difference: {abs(train_accuracy - test_accuracy) * 100:.2f}%
  Status: {('✓ Good generalization' if abs(train_accuracy - test_accuracy) < 0.05 else '⚠ May need tuning')}

All outputs saved successfully!
""")

print("=" * 80)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nNext Steps:")
print("- Review model_evaluation_report.txt for detailed analysis")
print("- Examine feature_importance.csv for key predictors")
print("- Consider hyperparameter tuning for optimization")
print("- Explore ensemble methods for potential improvement")
print("=" * 80)

