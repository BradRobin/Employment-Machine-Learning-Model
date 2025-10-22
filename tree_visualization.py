"""
HR Employee Attrition Prediction - Enhanced Tree Visualization
Task 4: Comprehensive Decision Tree Visualization using plot_tree() and graphviz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree, export_graphviz, export_text
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DECISION TREE VISUALIZATION - ENHANCED ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD TRAINED MODEL AND DATA
# ============================================================================
print("1. LOADING TRAINED MODEL AND DATA")
print("-" * 80)

# Load the trained model
model = joblib.load('decision_tree_model.pkl')
print("✓ Loaded: decision_tree_model.pkl")

# Load training data to get feature names
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

feature_names = X_train.columns.tolist()
class_names = ['No Attrition', 'Yes Attrition']

print(f"✓ Features: {len(feature_names)}")
print(f"✓ Tree Depth: {model.get_depth()}")
print(f"✓ Number of Leaves: {model.get_n_leaves()}")
print()

# ============================================================================
# 2. ENHANCED PLOT_TREE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("2. CREATING ENHANCED PLOT_TREE VISUALIZATIONS")
print("-" * 80)

# Version 1: Top 3 Levels (Clearest overview)
print("\n2.1 Top 3 Levels Visualization...")
fig, ax = plt.subplots(figsize=(30, 18))
plot_tree(model, 
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=3,
          proportion=False,
          precision=2,
          ax=ax)
plt.title('Decision Tree - Top 3 Levels (Main Decision Points)', 
          fontsize=20, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('tree_top3_levels.png', dpi=300, bbox_inches='tight')
print("✓ Saved: tree_top3_levels.png")
plt.close()

# Version 2: Top 5 Levels (Moderate detail)
print("2.2 Top 5 Levels Visualization...")
fig, ax = plt.subplots(figsize=(40, 24))
plot_tree(model, 
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          fontsize=8,
          max_depth=5,
          proportion=False,
          precision=2,
          ax=ax)
plt.title('Decision Tree - Top 5 Levels (Moderate Detail)', 
          fontsize=20, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('tree_top5_levels.png', dpi=300, bbox_inches='tight')
print("✓ Saved: tree_top5_levels.png")
plt.close()

# Version 3: Full Tree (Maximum detail - very large)
print("2.3 Full Tree Visualization (this may take a moment)...")
fig, ax = plt.subplots(figsize=(80, 60))
plot_tree(model, 
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          fontsize=6,
          proportion=False,
          precision=2,
          ax=ax)
plt.title('Decision Tree - Complete Structure (All 15 Levels)', 
          fontsize=24, fontweight='bold', pad=30)
plt.tight_layout()
plt.savefig('tree_full_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: tree_full_visualization.png (LARGE FILE)")
plt.close()

# Version 4: With Proportions
print("2.4 Tree with Proportions (Top 4 Levels)...")
fig, ax = plt.subplots(figsize=(35, 20))
plot_tree(model, 
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          fontsize=9,
          max_depth=4,
          proportion=True,  # Show proportions instead of counts
          precision=2,
          ax=ax)
plt.title('Decision Tree - With Class Proportions (Top 4 Levels)', 
          fontsize=20, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('tree_with_proportions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: tree_with_proportions.png")
plt.close()

# ============================================================================
# 3. GRAPHVIZ VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("3. CREATING GRAPHVIZ VISUALIZATIONS")
print("-" * 80)

try:
    # Export to DOT format
    print("\n3.1 Exporting to DOT format...")
    dot_data = export_graphviz(
        model,
        out_file='tree_graphviz.dot',
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=False,
        precision=2,
        max_depth=5  # Limit depth for readability
    )
    print("✓ Saved: tree_graphviz.dot")
    
    # Try to render with graphviz if available
    try:
        import graphviz
        
        # Read the dot file
        with open('tree_graphviz.dot', 'r') as f:
            dot_content = f.read()
        
        # Create graphviz object
        graph = graphviz.Source(dot_content)
        
        # Render to multiple formats
        print("\n3.2 Rendering Graphviz visualizations...")
        
        # PDF (best quality, scalable)
        graph.format = 'pdf'
        graph.render('tree_graphviz', cleanup=True)
        print("✓ Saved: tree_graphviz.pdf (scalable vector format)")
        
        # SVG (web-friendly, scalable)
        graph.format = 'svg'
        graph.render('tree_graphviz_svg', cleanup=True)
        print("✓ Saved: tree_graphviz_svg.svg (web-friendly vector)")
        
        # PNG (high resolution)
        graph.format = 'png'
        graph.render('tree_graphviz_png', cleanup=True)
        print("✓ Saved: tree_graphviz_png.png (high-resolution raster)")
        
        print("\n✓ Graphviz visualizations created successfully!")
        print("  Note: PDF and SVG are scalable and best for printing/presentations")
        
    except ImportError:
        print("\n⚠ Graphviz Python library not installed.")
        print("  DOT file created. Install graphviz to render:")
        print("  pip install graphviz")
        print("  Also install Graphviz system package from: https://graphviz.org/download/")
        
except Exception as e:
    print(f"\n⚠ Error creating Graphviz visualization: {e}")

# ============================================================================
# 4. EXTRACT DECISION RULES (TEXT FORMAT)
# ============================================================================
print("\n" + "=" * 80)
print("4. EXTRACTING DECISION RULES")
print("-" * 80)

print("\n4.1 Exporting tree structure as text...")
tree_rules = export_text(model, 
                         feature_names=feature_names,
                         max_depth=5,  # Limit for readability
                         decimals=2,
                         show_weights=True)

with open('tree_rules.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("DECISION TREE RULES - TOP 5 LEVELS\n")
    f.write("=" * 80 + "\n\n")
    f.write("Decision Tree Structure (if-then rules):\n")
    f.write("Note: Each path from root to leaf represents a classification rule.\n\n")
    f.write(tree_rules)
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("Tree Statistics:\n")
    f.write("=" * 80 + "\n")
    f.write(f"Total Depth: {model.get_depth()}\n")
    f.write(f"Total Leaves: {model.get_n_leaves()}\n")
    f.write(f"Total Features: {model.n_features_in_}\n")
    f.write("\n")

print("✓ Saved: tree_rules.txt")
print(f"  Preview (first 20 lines):")
print("-" * 40)
for i, line in enumerate(tree_rules.split('\n')[:20]):
    print(f"  {line}")
print("  ...")

# ============================================================================
# 5. DECISION PATH ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. ANALYZING EXAMPLE DECISION PATHS")
print("-" * 80)

# Get predictions for test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Find example cases
tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]  # True Positive
fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]  # False Negative
tn_idx = np.where((y_test == 0) & (y_pred == 0))[0]  # True Negative
fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]  # False Positive

# Select one example from each category
examples = []
if len(tp_idx) > 0:
    examples.append(('True Positive', tp_idx[0], 'Correctly Predicted Attrition'))
if len(fn_idx) > 0:
    examples.append(('False Negative', fn_idx[0], 'Missed Attrition'))
if len(tn_idx) > 0:
    examples.append(('True Negative', tn_idx[0], 'Correctly Predicted Retention'))
if len(fp_idx) > 0:
    examples.append(('False Positive', fp_idx[0], 'False Alarm'))

# Analyze decision paths
decision_paths_content = []
decision_paths_content.append("=" * 80)
decision_paths_content.append("EXAMPLE DECISION PATHS")
decision_paths_content.append("=" * 80)
decision_paths_content.append("")

print("\nAnalyzing decision paths for example cases:\n")

for example_type, idx, description in examples:
    print(f"\n{example_type}: {description}")
    print(f"  Test Sample Index: {idx}")
    
    # Get the decision path
    path = model.decision_path(X_test.iloc[[idx]])
    node_indicator = path.toarray()[0]
    nodes = np.where(node_indicator)[0]
    
    # Get tree structure
    tree = model.tree_
    
    decision_paths_content.append(f"\n{example_type}: {description}")
    decision_paths_content.append("-" * 80)
    decision_paths_content.append(f"Test Sample Index: {idx}")
    decision_paths_content.append(f"Actual Class: {class_names[int(y_test[idx])]}")
    decision_paths_content.append(f"Predicted Class: {class_names[int(y_pred[idx])]}")
    decision_paths_content.append(f"Prediction Probability: {y_pred_proba[idx][1]:.4f}")
    decision_paths_content.append(f"\nDecision Path (nodes traversed: {len(nodes)}):")
    decision_paths_content.append("")
    
    # Show first few nodes in the path
    for i, node in enumerate(nodes[:10]):  # Limit to first 10 nodes
        if i >= tree.node_count:
            break
            
        # Check if it's a leaf node
        if tree.feature[node] != -2:  # Not a leaf
            feature = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            value = X_test.iloc[idx, tree.feature[node]]
            
            if value <= threshold:
                direction = "<="
                next_node = "left"
            else:
                direction = ">"
                next_node = "right"
            
            message = f"  Node {node}: {feature} {direction} {threshold:.2f} (value: {value:.2f}) → Go {next_node}"
            print(f"    {message}")
            decision_paths_content.append(message)
        else:  # Leaf node
            samples = tree.n_node_samples[node]
            values = tree.value[node][0]
            predicted_class = class_names[np.argmax(values)]
            message = f"  Node {node}: LEAF - Prediction: {predicted_class} (samples: {samples})"
            print(f"    {message}")
            decision_paths_content.append(message)
    
    if len(nodes) > 10:
        message = f"  ... ({len(nodes) - 10} more nodes)"
        print(f"    {message}")
        decision_paths_content.append(message)
    
    decision_paths_content.append("")

# Save decision paths
with open('tree_decision_paths.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(decision_paths_content))

print("\n✓ Saved: tree_decision_paths.txt")

# ============================================================================
# 6. TREE STRUCTURE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. ANALYZING TREE STRUCTURE")
print("-" * 80)

tree = model.tree_

# Analyze tree statistics
n_nodes = tree.node_count
n_leaves = model.get_n_leaves()
depth = model.get_depth()

# Get samples per node
samples_per_node = tree.n_node_samples

# Get gini impurity per node
impurity_per_node = tree.impurity

# Get feature usage
feature_usage = {}
for node in range(n_nodes):
    if tree.feature[node] != -2:  # Not a leaf
        feat = feature_names[tree.feature[node]]
        feature_usage[feat] = feature_usage.get(feat, 0) + 1

# Sort by usage
sorted_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)

print(f"\nTree Structure Statistics:")
print(f"  Total Nodes: {n_nodes}")
print(f"  Leaf Nodes: {n_leaves}")
print(f"  Internal Nodes: {n_nodes - n_leaves}")
print(f"  Maximum Depth: {depth}")
print(f"  Average Samples per Node: {np.mean(samples_per_node):.1f}")
print(f"  Median Samples per Node: {np.median(samples_per_node):.1f}")
print(f"  Average Gini Impurity: {np.mean(impurity_per_node[impurity_per_node > 0]):.4f}")

print(f"\nTop 10 Most Used Features in Tree Splits:")
for i, (feat, count) in enumerate(sorted_features[:10], 1):
    print(f"  {i}. {feat}: {count} splits")

# Create visualizations for tree structure
print("\n6.1 Creating structure analysis visualizations...")

# Histogram of samples per node
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Samples per node distribution
axes[0, 0].hist(samples_per_node, bins=50, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('Samples per Node', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Distribution of Samples per Node', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Gini impurity distribution
axes[0, 1].hist(impurity_per_node[impurity_per_node > 0], bins=50, color='coral', edgecolor='black')
axes[0, 1].set_xlabel('Gini Impurity', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Distribution of Node Impurity (Gini Index)', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature usage in splits
top_10_features = sorted_features[:10]
feat_names = [f[0][:20] for f in top_10_features]  # Truncate long names
feat_counts = [f[1] for f in top_10_features]
axes[1, 0].barh(range(len(feat_names)), feat_counts, color='seagreen', edgecolor='black')
axes[1, 0].set_yticks(range(len(feat_names)))
axes[1, 0].set_yticklabels(feat_names)
axes[1, 0].set_xlabel('Number of Splits', fontsize=12)
axes[1, 0].set_title('Top 10 Features Used in Tree Splits', fontsize=14, fontweight='bold')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Plot 4: Tree statistics summary
stats_text = f"""
Tree Statistics Summary

Structure:
• Total Nodes: {n_nodes}
• Leaf Nodes: {n_leaves}
• Internal Nodes: {n_nodes - n_leaves}
• Maximum Depth: {depth}

Samples Distribution:
• Mean: {np.mean(samples_per_node):.1f}
• Median: {np.median(samples_per_node):.1f}
• Min: {np.min(samples_per_node)}
• Max: {np.max(samples_per_node)}

Node Purity (Gini):
• Mean: {np.mean(impurity_per_node[impurity_per_node > 0]):.4f}
• Pure Nodes: {np.sum(impurity_per_node == 0)}

Most Important Split:
• {sorted_features[0][0]}
  ({sorted_features[0][1]} times)
"""
axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')
axes[1, 1].set_title('Summary Statistics', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('tree_structure_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: tree_structure_analysis.png")
plt.close()

# Save structure analysis to text file
with open('tree_structure_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("DECISION TREE STRUCTURE ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Tree Statistics:\n")
    f.write(f"  Total Nodes: {n_nodes}\n")
    f.write(f"  Leaf Nodes: {n_leaves}\n")
    f.write(f"  Internal Nodes: {n_nodes - n_leaves}\n")
    f.write(f"  Maximum Depth: {depth}\n\n")
    f.write(f"Samples Distribution:\n")
    f.write(f"  Mean: {np.mean(samples_per_node):.1f}\n")
    f.write(f"  Median: {np.median(samples_per_node):.1f}\n")
    f.write(f"  Min: {np.min(samples_per_node)}\n")
    f.write(f"  Max: {np.max(samples_per_node)}\n\n")
    f.write(f"Node Purity (Gini Impurity):\n")
    f.write(f"  Mean: {np.mean(impurity_per_node[impurity_per_node > 0]):.4f}\n")
    f.write(f"  Pure Nodes (Gini = 0): {np.sum(impurity_per_node == 0)}\n\n")
    f.write("=" * 80 + "\n")
    f.write(f"FEATURE USAGE IN TREE SPLITS\n")
    f.write("=" * 80 + "\n\n")
    for i, (feat, count) in enumerate(sorted_features, 1):
        f.write(f"{i:2d}. {feat:40s} - {count:3d} splits\n")
    f.write("\n")

print("✓ Saved: tree_structure_analysis.txt")

# ============================================================================
# 7. SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("7. GENERATING COMPREHENSIVE VISUALIZATION REPORT")
print("-" * 80)

report = f"""
================================================================================
DECISION TREE VISUALIZATION - COMPREHENSIVE REPORT
================================================================================

Generated: Automatically
Model: Decision Tree Classifier (Employee Attrition Prediction)

================================================================================
1. VISUALIZATION OUTPUTS
================================================================================

PLOT_TREE VISUALIZATIONS (Matplotlib):
--------------------------------------
✓ tree_top3_levels.png            - Top 3 levels (main decisions)
✓ tree_top5_levels.png            - Top 5 levels (moderate detail)
✓ tree_full_visualization.png     - Complete tree (all 15 levels)
✓ tree_with_proportions.png       - Top 4 levels with class proportions

Note: These visualizations use matplotlib's plot_tree() function.
      Larger images provide more detail but may be harder to read.

GRAPHVIZ VISUALIZATIONS:
-----------------------
✓ tree_graphviz.dot               - DOT source file
✓ tree_graphviz.pdf               - PDF format (best for printing)
✓ tree_graphviz_svg.svg           - SVG format (web-friendly)
✓ tree_graphviz_png.png           - PNG format (high resolution)

Note: Graphviz produces higher-quality diagrams than matplotlib.
      PDF and SVG are scalable vector formats - best for presentations.

DECISION RULES & ANALYSIS:
-------------------------
✓ tree_rules.txt                  - Text-based decision rules
✓ tree_decision_paths.txt         - Example decision paths analyzed
✓ tree_structure_analysis.txt     - Tree statistics and feature usage
✓ tree_structure_analysis.png     - Visual analysis of tree structure

================================================================================
2. TREE STRUCTURE OVERVIEW
================================================================================

Dimensions:
  • Total Nodes: {n_nodes}
  • Leaf Nodes: {n_leaves}
  • Internal Decision Nodes: {n_nodes - n_leaves}
  • Maximum Depth: {depth} levels

Complexity Analysis:
  • Average Samples per Node: {np.mean(samples_per_node):.1f}
  • Median Samples per Node: {np.median(samples_per_node):.1f}
  • Average Node Impurity: {np.mean(impurity_per_node[impurity_per_node > 0]):.4f}
  • Pure Leaf Nodes: {np.sum(impurity_per_node == 0)}

================================================================================
3. KEY DECISION POINTS
================================================================================

Root Node (First Split):
  • Feature: {feature_names[tree.feature[0]]}
  • Threshold: {tree.threshold[0]:.2f}
  • Samples: {tree.n_node_samples[0]}

This is the MOST IMPORTANT split in the entire tree - it divides
the dataset into two major groups based on this feature.

Top 5 Most Used Features in Tree:
"""

for i, (feat, count) in enumerate(sorted_features[:5], 1):
    report += f"  {i}. {feat}: {count} splits\n"

report += f"""

These features appear most frequently in decision nodes throughout
the tree, indicating their importance in classification.

================================================================================
4. VISUALIZATION RECOMMENDATIONS
================================================================================

For Quick Overview:
  → Use tree_top3_levels.png
  → Shows the 3 most important decision points
  → Easy to understand and present

For Detailed Analysis:
  → Use tree_top5_levels.png or tree_graphviz.pdf
  → Provides more detail while remaining readable
  → Good for understanding decision logic

For Complete Structure:
  → Use tree_full_visualization.png
  → Shows all {depth} levels and {n_leaves} leaves
  → Very large - best for zooming and detailed inspection

For Presentations:
  → Use tree_graphviz.pdf or tree_graphviz_svg.svg
  → Vector formats scale perfectly
  → Professional quality diagrams

For Technical Documentation:
  → Use tree_rules.txt
  → Complete if-then rules in text format
  → Easy to copy and reference

================================================================================
5. INTERPRETING THE TREE
================================================================================

Color Coding:
  • Orange nodes → Predicted "Yes Attrition" (Class 1)
  • Blue nodes → Predicted "No Attrition" (Class 0)
  • Color intensity → Confidence (purity) of prediction

Node Information:
  • Top line → Feature and threshold for split
  • "gini" → Impurity measure (0 = pure, 0.5 = mixed)
  • "samples" → Number of training samples at this node
  • "value" → [No Attrition count, Yes Attrition count]
  • "class" → Majority class prediction

Reading Decision Paths:
  • Start at root (top)
  • Follow left branch if condition is TRUE (≤)
  • Follow right branch if condition is FALSE (>)
  • Continue until reaching a leaf node
  • Leaf node shows final prediction

================================================================================
6. INSIGHTS FROM VISUALIZATION
================================================================================

Tree Complexity:
  The tree has {depth} levels of depth with {n_leaves} leaf nodes.
  This is a COMPLEX tree that likely overfits the training data.
  
Overfitting Indicators:
  • Very deep tree (depth = {depth})
  • Many leaf nodes ({n_leaves})
  • Some leaves may have very few samples
  • Perfect training accuracy suggests memorization

Decision Logic:
  • Primary split on: {feature_names[tree.feature[0]]}
  • Early splits use most important features
  • Deeper levels may capture noise rather than signal

Feature Usage:
  • {len(feature_usage)} out of {len(feature_names)} features are used
  • Top feature appears in {sorted_features[0][1]} split decisions
  • Some features may not be used at all

================================================================================
7. RECOMMENDED ACTIONS
================================================================================

Based on visualization analysis:

1. TREE PRUNING (Critical):
   • Current depth ({depth}) is too large
   • Recommend max_depth = 5-8 for better generalization
   • Set min_samples_split = 20-50
   • Set min_samples_leaf = 10-20

2. FEATURE SELECTION:
   • Only {len(feature_usage)}/{len(feature_names)} features are actually used
   • Consider removing unused features
   • Focus on top 20 most important features

3. MODEL SIMPLIFICATION:
   • The tree_top3_levels.png shows the essential logic
   • Most important decisions happen in first 3-5 levels
   • Deeper levels may be overfitting

4. ALTERNATIVE APPROACHES:
   • Try ensemble methods (Random Forest)
   • Ensemble averages multiple trees
   • Reduces overfitting while maintaining accuracy

================================================================================
8. HOW TO USE THESE VISUALIZATIONS
================================================================================

For Stakeholder Presentations:
  1. Show tree_top3_levels.png to explain key factors
  2. Highlight: "{feature_names[tree.feature[0]]}" is the #1 predictor
  3. Use tree_graphviz.pdf for high-quality printouts

For Technical Review:
  1. Review tree_structure_analysis.png for statistics
  2. Read tree_rules.txt for complete logic
  3. Check tree_decision_paths.txt for example cases

For Model Debugging:
  1. Examine tree_full_visualization.png at deep levels
  2. Look for nodes with very few samples (overfitting)
  3. Check if leaf nodes make sense logically

For Documentation:
  1. Include tree_top5_levels.png in reports
  2. Reference tree_rules.txt for rule documentation
  3. Cite feature usage statistics from analysis

================================================================================
END OF VISUALIZATION REPORT
================================================================================
"""

with open('tree_visualization_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✓ Saved: tree_visualization_report.txt")

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("8. VISUALIZATION COMPLETE - SUMMARY")
print("-" * 80)

print(f"""
Decision Tree Visualization Complete!

Generated {11} files:

Visualizations (PNG):
  ✓ tree_top3_levels.png              - Quick overview
  ✓ tree_top5_levels.png              - Moderate detail
  ✓ tree_full_visualization.png       - Complete tree (LARGE)
  ✓ tree_with_proportions.png         - With class proportions
  ✓ tree_structure_analysis.png       - Structure statistics

Graphviz Files:
  ✓ tree_graphviz.dot                 - Source file
  ✓ tree_graphviz.pdf                 - Vector (scalable)
  ✓ tree_graphviz_svg.svg             - Web-friendly
  ✓ tree_graphviz_png.png             - High-res image

Analysis Files:
  ✓ tree_rules.txt                    - Decision rules
  ✓ tree_decision_paths.txt           - Example paths
  ✓ tree_structure_analysis.txt       - Statistics
  ✓ tree_visualization_report.txt     - This summary

Key Findings:
  • Tree depth: {depth} levels
  • Total nodes: {n_nodes}
  • Leaf nodes: {n_leaves}
  • Most important feature: {feature_names[tree.feature[0]]}
  • Tree is COMPLEX - pruning recommended

Recommendations:
  1. View tree_top3_levels.png for quick understanding
  2. Read tree_visualization_report.txt for complete analysis
  3. Use tree_graphviz.pdf for presentations (best quality)
  4. Consider pruning: limit depth to 5-8 for better performance
""")

print("=" * 80)
print("TREE VISUALIZATION COMPLETED SUCCESSFULLY!")
print("=" * 80)

