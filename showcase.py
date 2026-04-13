"""
NBA Draft Prediction Showcase
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, 
                             confusion_matrix, ConfusionMatrixDisplay, classification_report )
from xgboost import XGBClassifier
import xgboost as xgb

# Load all the datasets
test_df = pd.read_csv('Data/test.csv')
X_test  = test_df.drop(columns = ['Drafted'])
y_test  = test_df['Drafted']
 
model = XGBClassifier()
model.load_model('xgb_final.json')
 
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Compute metrics
metrics = {
    'Accuracy':  accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall':    recall_score(y_test, y_pred),
    'F1 Score':  f1_score(y_test, y_pred),
    'ROC-AUC':   roc_auc_score(y_test, y_pred_prob),
}
print("Model Metrics:")
for k, v in metrics.items():
    print(f"  {k:12s}: {v:.4f}")

# Build figure
fig = plt.figure(figsize = (16, 13))
fig.suptitle("NBA Draft Prediction Model — Results", fontsize = 18, fontweight = 'bold', y = 0.98)
gs = gridspec.GridSpec(2, 2, figure = fig, hspace = 0.38, wspace = 0.32)

# Plot 1 : ROC Curve
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc_score   = roc_auc_score(y_test, y_pred_prob)
 
ax1.plot(fpr, tpr, color = 'red', linewidth = 2.5, label = f'XGBoost (AUC = {auc_score:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', linewidth = 1.2, label = 'Random Classifier (AUC = 0.5)')
ax1.fill_between(fpr, tpr, alpha = 0.12, color = 'red')
ax1.set_xlabel('False Positive Rate', fontsize = 11)
ax1.set_ylabel('True Positive Rate', fontsize = 11)
ax1.set_title('ROC-AUC Curve', fontsize = 13, fontweight = 'bold')
ax1.legend(loc = 'lower right', fontsize = 10)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.02])
ax1.grid(True, alpha = 0.3)

# Plot 2 : Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm,
                               display_labels = ['Not Drafted', 'Drafted'])
disp.plot(ax = ax2, colorbar = False, cmap = 'Blues')
ax2.set_title('Confusion Matrix', fontsize = 13, fontweight = 'bold')
ax2.set_xlabel('Predicted Label', fontsize = 11)
ax2.set_ylabel('True Label', fontsize = 11)
 
# Add counts annotation helper
tn, fp, fn, tp = cm.ravel()
ax2.text(0.5, -0.18,
         f'True Negatives: {tn}  |  False Positives: {fp}\n'
         f'False Negatives: {fn}  |  True Positives: {tp}',
         transform = ax2.transAxes, ha = 'center', fontsize = 9, color = '#444')

# Plot 3 : Feature Importance
ax3 = fig.add_subplot(gs[1, 0])
importances = model.get_booster().get_score(importance_type = 'weight')
importances = dict(sorted(importances.items(), key = lambda x: x[1], reverse = True))
 
name_map = {
    'PPG': 'Points Per Game', 'RPG': 'Rebounds Per Game',
    'APG': 'Assists Per Game', 'SPG': 'Steals Per Game',
    'BPG': 'Blocks Per Game', 'TOVPG': 'Turnovers Per Game',
    'MPG': 'Minutes Per Game', 'GSPG': 'Start Rate',
    '3PAr': '3-Point Attempt Rate', 'FTr': 'Free Throw Rate',
    'AST_TOV': 'Assist/Turnover Ratio', 'TS%': 'True Shooting %',
    'impact_per_min': 'Impact Per Minute',
    'POS_enc': 'Position', 'Class_enc': 'Class Year',
}
 
top_keys   = list(importances.keys())[:12]
top_vals   = [importances[k] for k in top_keys]
top_labels = [name_map.get(k, k) for k in top_keys]
 
colors = ['darkred' if v == max(top_vals) else 'darkblue' for v in top_vals]
bars = ax3.barh(top_labels[::-1], top_vals[::-1], color=colors[::-1], edgecolor = 'white')
ax3.set_xlabel('Importance (Weight)', fontsize = 11)
ax3.set_title('Top Feature Importances', fontsize = 13, fontweight = 'bold')
ax3.grid(True, axis = 'x', alpha = 0.3)
for bar, val in zip(bars, top_vals[::-1]):
    ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.0f}', va = 'center', fontsize = 9)
plt.tight_layout(rect = [0, 0, 1, 0.95])

# Plot 4 : Metric bar chart
ax4 = fig.add_subplot(gs[1, 1])
metric_names  = list(metrics.keys())
metric_values = list(metrics.values())
bar_colors = ['darkred' if n == 'ROC-AUC' else 'darkblue' for n in metric_names]
 
bars2 = ax4.bar(metric_names, metric_values, color=bar_colors,
                edgecolor = 'white', width = 0.55)
ax4.set_ylim([0, 1.1])
ax4.set_ylabel('Score', fontsize = 11)
ax4.set_title('Model Performance Metrics', fontsize = 13, fontweight = 'bold')
ax4.axhline(y = 0.5, color = 'gray', linestyle = '--', linewidth = 1, alpha = 0.6,
            label = 'Random baseline (0.5)')
ax4.legend(fontsize = 9)
ax4.grid(True, axis = 'y', alpha = 0.3)
for bar, val in zip(bars2, metric_values):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             f'{val:.3f}', ha = 'center', fontsize = 10, fontweight = 'bold')
 
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('showcase_plots.png', dpi = 150, bbox_inches = 'tight', facecolor = 'white')
plt.show()
print("\nPlots saved → showcase_plots.png")

