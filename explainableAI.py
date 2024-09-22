# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_curve, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

# Load the Credit Record dataset
credit_record = pd.read_csv('/mnt/data/credit_record.csv')

# Check the columns in credit_record
print("Credit Record Columns:", credit_record.columns)
print("First few rows of Credit Record:")
print(credit_record.head())

# The STATUS column contains categorical values. Let's define our target 'Class' based on it.
# Convert STATUS to binary 'Class' based on the risk: Assume '0', '1', '2' are low risk (Class 0) and '3', '4', '5' are high risk (Class 1).
# You can adjust this logic based on your business rules.

# First, let's map the STATUS values to numerical risk levels
# '0', '1', '2': low risk -> Class 0
# '3', '4', '5': high risk -> Class 1
# 'C', 'X' might mean no risk or no credit activity, so we'll exclude them from training for now.
risk_mapping = {
    '0': 0, '1': 0, '2': 0,  # Low risk
    '3': 1, '4': 1, '5': 1,  # High risk
    'C': np.nan,  # Completed/no activity
    'X': np.nan   # No credit history
}

# Apply the mapping
credit_record['Class'] = credit_record['STATUS'].map(risk_mapping)

# Remove rows where 'Class' is NaN (where STATUS is 'C' or 'X')
credit_record_cleaned = credit_record.dropna(subset=['Class'])

# Check for imbalance in the target variable 'Class'
print("Class Distribution in Credit Record after mapping:")
print(credit_record_cleaned['Class'].value_counts())

# Separate features and target
X_credit = credit_record_cleaned.drop(['Class', 'STATUS'], axis=1)
y_credit = credit_record_cleaned['Class']

# Preprocess categorical features (e.g., if 'MONTHS_BALANCE' is categorical, convert it)
# There may not be categorical columns after dropping 'STATUS', but verify this based on your dataset.
X_credit = pd.get_dummies(X_credit, drop_first=True)

# Split the data into training and testing sets
X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.3, random_state=42, stratify=y_credit)

# Scale the data
scaler_credit = StandardScaler()
X_credit_train_scaled = scaler_credit.fit_transform(X_credit_train)
X_credit_test_scaled = scaler_credit.transform(X_credit_test)

# Create classifiers
rf_credit = RandomForestClassifier(n_estimators=100, random_state=42)
gb_credit = GradientBoostingClassifier(n_estimators=100, random_state=42)
dt_credit = DecisionTreeClassifier(random_state=42)

# Ensemble model
ensemble_credit = VotingClassifier(estimators=[('rf', rf_credit), ('gb', gb_credit), ('dt', dt_credit)], voting='soft')
ensemble_credit.fit(X_credit_train_scaled, y_credit_train)

# Predictions and evaluation
y_credit_pred = ensemble_credit.predict(X_credit_test_scaled)
print("Classification Report for Credit Record:")
print(classification_report(y_credit_test, y_credit_pred))

# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_credit_train_res, y_credit_train_res = smote.fit_resample(X_credit_train_scaled, y_credit_train)

# Train on resampled data
ensemble_credit.fit(X_credit_train_res, y_credit_train_res)
y_credit_pred_res = ensemble_credit.predict(X_credit_test_scaled)

# Resampled evaluation
print("Resampled Classification Report for Credit Record:")
print(classification_report(y_credit_test, y_credit_pred_res))

# ROC-AUC Score
y_credit_prob = ensemble_credit.predict_proba(X_credit_test_scaled)[:, 1]
roc_auc_credit = roc_auc_score(y_credit_test, y_credit_prob)
print(f"ROC-AUC Score for Credit Record: {roc_auc_credit}")

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_credit_test, y_credit_prob)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_credit_test, y_credit_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_credit:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend(loc='lower right')
plt.show(), 
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_curve, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

# Load the Application Record dataset
application_record = pd.read_csv('/mnt/data/application_record.csv')

# Check the columns in application_record
print("Application Record Columns:", application_record.columns)
print("First few rows of Application Record:")
print(application_record.head())

# Define the target 'Class'. Let's assume we are categorizing applications based on 'AMT_INCOME_TOTAL'
# For example, applicants with income less than a threshold are classified as high risk (Class 1)
# and others are low risk (Class 0). You can adjust this logic based on your dataset.

# Let's assume we want to classify based on income. This threshold is just an example.
income_threshold = 50000

# Create the 'Class' column
application_record['Class'] = np.where(application_record['AMT_INCOME_TOTAL'] < income_threshold, 1, 0)

# Check for imbalance in the target variable
print("Class Distribution in Application Record:")
print(application_record['Class'].value_counts())

# Separate features and target
X_application = application_record.drop('Class', axis=1)
y_application = application_record['Class']

# Preprocess categorical features (if any)
# Convert categorical features to dummy variables
X_application = pd.get_dummies(X_application, drop_first=True)

# Split the data into training and testing sets
X_application_train, X_application_test, y_application_train, y_application_test = train_test_split(
    X_application, y_application, test_size=0.3, random_state=42, stratify=y_application
)

# Scale the data
scaler_application = StandardScaler()
X_application_train_scaled = scaler_application.fit_transform(X_application_train)
X_application_test_scaled = scaler_application.transform(X_application_test)

# Create classifiers
rf_application = RandomForestClassifier(n_estimators=100, random_state=42)
gb_application = GradientBoostingClassifier(n_estimators=100, random_state=42)
dt_application = DecisionTreeClassifier(random_state=42)

# Ensemble model
ensemble_application = VotingClassifier(estimators=[('rf', rf_application), ('gb', gb_application), ('dt', dt_application)], voting='soft')
ensemble_application.fit(X_application_train_scaled, y_application_train)

# Predictions and evaluation
y_application_pred = ensemble_application.predict(X_application_test_scaled)
print("Classification Report for Application Record:")
print(classification_report(y_application_test, y_application_pred))

# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_application_train_res, y_application_train_res = smote.fit_resample(X_application_train_scaled, y_application_train)

# Train on resampled data
ensemble_application.fit(X_application_train_res, y_application_train_res)
y_application_pred_res = ensemble_application.predict(X_application_test_scaled)

# Resampled evaluation
print("Resampled Classification Report for Application Record:")
print(classification_report(y_application_test, y_application_pred_res))

# ROC-AUC Score
y_application_prob = ensemble_application.predict_proba(X_application_test_scaled)[:, 1]
roc_auc_application = roc_auc_score(y_application_test, y_application_prob)
print(f"ROC-AUC Score for Application Record: {roc_auc_application}")

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_application_test, y_application_prob)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_application_test, y_application_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_application:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend(loc='lower right')
plt.show()
