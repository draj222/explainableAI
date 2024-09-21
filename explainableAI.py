import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


import shap
import lime.lime_tabular


# Load dataset from Kaggle
df = pd.read_csv('creditcard.csv')


print(df.head())
print(df['Class'].value_counts())  # Check  imbalance (0: Non-fraud, 1: Fraud)

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

#Split to train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

#base models 
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
dt = DecisionTreeClassifier(random_state=42)

#ensemble model - VotingClassifier
ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('gb', gb),
    ('dt', dt)
], voting='soft')  # soft voting for probability based predictions

#train ensemble
ensemble.fit(X_train_scaled, y_train)

#predict on test  data
y_pred = ensemble.predict(X_test_scaled)

# evaluation
from sklearn.metrics import classification_report, accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

#train on resampled
ensemble.fit(X_train_res, y_train_res)
y_pred_res = ensemble.predict(X_test_scaled)

print("Resampled Classification Report:\n", classification_report(y_test, y_pred_res))
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)  # No direct class_weight in GB
dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
from sklearn.metrics import roc_auc_score, precision_recall_curve

#probability estimates on positive class
y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc}")

precision, recall, _ = precision_recall_curve(y_test, y_prob)
#precision-recall curve
from sklearn.model_selection import GridSearchCV

param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 20, None],
    'gb__learning_rate': [0.01, 0.1],
    'dt__max_depth': [5, 10, 15]
}

grid_search = GridSearchCV(ensemble, param_grid, cv=3, scoring='recall')
grid_search.fit(X_train_scaled, y_train)

print("Best parameters found: ", grid_search.best_params_)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

#feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]})")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-fraud', 'Fraud'], yticklabels=['Non-fraud', 'Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import precision_recall_curve

#probabilitis for class 1
y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]

# precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)


plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

# ROC curve AUC score
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)


plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--') 
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend(loc='lower right')
plt.show()

import numpy as np

#RandomForestClassifer importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]


plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

#class distribution
plt.figure(figsize=(6, 4))
sns.countplot(df['Class'], palette="Blues")
plt.title("Class Distribution (Imbalanced Data)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks([0, 1], ['Non-fraud', 'Fraud'])
plt.show()

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(ensemble, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)

#mean and std for train and test scores
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

#learning curve
plt.figure(figsize=(6, 4))
plt.plot(train_sizes, train_scores_mean, label="Training accuracy")
plt.plot(train_sizes, test_scores_mean, label="Cross-validation accuracy")
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

import shap

#SHAP summary plot
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test_scaled)


shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)