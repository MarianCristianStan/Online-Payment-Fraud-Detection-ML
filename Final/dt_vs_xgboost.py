
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('onlinefraud.csv')
data = data.drop(columns=['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

# Encode(convert to numeric)
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

X = data.drop(columns=['isFraud'])
y = data['isFraud']


dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=5000, min_samples_leaf=5000)
dt_model.fit(X, y)
y_pred_dt = dt_model.predict(X)

# Confusion Matrix for DT
conf_matrix_dt = confusion_matrix(y, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='YlGn',
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
plt.title('Confusion Matrix for Decision Tree Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Train and predict with XGBoost model
xgb_model = XGBClassifier(max_depth=3, min_child_weight=5000,subsample=0.8, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X, y)
y_pred_xgb = xgb_model.predict(X)

# Confusion Matrix for XGBoost
conf_matrix_xgb = confusion_matrix(y, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='YlGn',
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
plt.title('Confusion Matrix for XGBoost Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Accuracy and Classification Report for Decision Tree
dt_accuracy = accuracy_score(y, y_pred_dt)
print(f'Accuracy for Decision Tree: {dt_accuracy:.2f}')
print("Classification Report for Decision Tree:")
print(classification_report(y, y_pred_dt, target_names=['Legitimate', 'Fraudulent']))

# Accuracy and Classification Report for XGBoost
xgb_accuracy = accuracy_score(y, y_pred_xgb)
print(f'Accuracy for XGBoost: {xgb_accuracy:.2f}')
print("Classification Report for XGBoost:")
print(classification_report(y, y_pred_xgb, target_names=['Legitimate', 'Fraudulent']))