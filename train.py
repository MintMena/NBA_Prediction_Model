import pandas as pd
from sklearn.model_selection import train_test_split

# Train the model on the preprocessed data

# Load preprocessed data
train_df = pd.read_csv('Data/train.csv')
test_df  = pd.read_csv('Data/test.csv')

X_train = train_df.drop(columns = ['Drafted'])
y_train = train_df['Drafted']
X_test  = test_df.drop(columns = ['Drafted'])
y_test  = test_df['Drafted']

# XGBoost : a powerful gradient boosting library that often performs well on tabular data

from xgboost import XGBClassifier

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() # ratio of negative to positive samples
model = XGBClassifier(
    n_estimators = 300, # number of trees
    max_depth = 4, # depth of each tree
    learning_rate = 0.05, # step size for boosting
    subsample = 0.8, # fraction of samples to use for each tree
    colsample_bytree = 0.8, # fraction of features to use for
    scale_pos_weight = scale_pos_weight, # handle class imbalance
    eval_metric = 'logloss', # evaluation metric for early stopping
    random_state = 67,
)

model.fit(X_train, y_train)

# Evaluate the model on the test set

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall : {recall_score(y_test, y_pred):.4f}")

print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred_prob):.4f}")   # ← competition metric

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
