import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier,plot_tree
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('onlinefraud.csv')
data = data.drop(columns=['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

# Encode(convert to numeric)
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

X = data.drop(columns=['isFraud'])
y = data['isFraud']

# Train the XGBoost model and predict
xgb_model = XGBClassifier(max_depth=3, min_child_weight=1000, subsample=0.8 ,use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X, y)
y_pred = xgb_model.predict(X)

accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGn',
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
plt.title('Confusion Matrix for XGBoost Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Visualizing feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x=xgb_model.feature_importances_, y=X.columns)
plt.title('Feature Importances for XGBoost Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

plt.figure(figsize=(30, 10))
plot_tree(xgb_model)
plt.show()