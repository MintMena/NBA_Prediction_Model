import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Load preprocessed data
train_df = pd.read_csv('Data/train.csv')
test_df  = pd.read_csv('Data/test.csv')

X_train = train_df.drop(columns = ['Drafted'])
y_train = train_df['Drafted']
X_test  = test_df.drop(columns = ['Drafted'])
y_test  = test_df['Drafted']

# Calculate scale_pos_weight for handling class imbalance in XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


# Define the model with initial parameters
param_grid = {
    'max_depth':        [3], # it is often beneficial to keep trees relatively shallow to prevent overfitting
    'min_child_weight': [5, 7],
    'gamma':            [0.3, 0.5], # minimum loss reduction required to make a further partition on a leaf node of the tree
    'reg_lambda':       [2.0, 5.0],  # L2 regularization term on weights (analogous to Ridge regression)
    'reg_alpha':        [0, 0.5, 1.0], # L1 regularization term on weights (analogous to Lasso regression)
    'n_estimators':     [300, 500, 700], # number of trees to fit
    'learning_rate':    [0.01, 0.02, 0.03], # step size shrinkage used in update to prevents overfitting
    'subsample':        [0.7, 0.8], # fraction of samples to be used for fitting the individual base learners
    'colsample_bytree': [0.7, 0.8],
}

search = GridSearchCV(
    estimator = XGBClassifier(
        scale_pos_weight = scale_pos_weight, # handle class imbalance
        eval_metric = 'logloss', # evaluation metric for early stopping
        random_state = 67,
        n_jobs = 1
    ),
    param_grid = param_grid,
    cv = 5, # 5-fold cross-validation
    scoring = 'roc_auc', # use ROC-AUC as the evaluation metric for hyperparameter
    n_jobs = -1, # use all available CPU cores
    verbose = 1 # print progress during gripythd search
)

model = search.fit(X_train, y_train)
print(f"Best parameters: {search.best_params_}")
print(f"Best ROC-AUC: {search.best_score_:.4f}")

model_best = search.best_estimator_
y_pred      = model_best.predict(X_test)
y_pred_prob = model_best.predict_proba(X_test)[:, 1]

print(f"\nAccuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# best is 0.8496 (405 fits)
# with n_estimators = 400, max_depth = 3, learning_rate = 0.01

# now adding reg_alpha and reg_lambda (3645 fits) gets us to 0.8582 
# with n_estimators = 400, max_depth = 3, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'subsample': 0.7
# Feture importance analysis with the best model

# After tuning we got 8.605 
# with n_estimators = 300, max_depth = 3, learning_rate = 0.01, reg_alpha = 0.5, reg_lambda = 2.0, subsample = 0.7, colsample_bytree = 0.7

# final adjusting 0.8761 by feature engineering

import matplotlib.pyplot as plt

importances = model_best.get_booster().get_score(importance_type = 'weight')
importances = dict(sorted(importances.items(), key = lambda x : x[1], reverse = True))

plt.figure(figsize = (10, 6))
plt.barh(list(importances.keys())[:15], list(importances.values())[:15], color = 'skyblue')
plt.xlabel('Feature Importance (Weight)')
plt.title('Top 15 Feature Importances from XGBoost')
plt.gca().invert_yaxis() # highest importance at the top
plt.tight_layout()
plt.show()