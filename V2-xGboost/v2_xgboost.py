import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('onlinefraud.csv')
data = data.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])

le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

X = data.drop(columns=['isFraud'])
y = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

# Apply SMOTETomek for balancing
smote_tomek = SMOTETomek(random_state=0)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
X_test_resampled, y_test_resampled = smote_tomek.fit_resample(X_test, y_test)

scale_pos_weight = len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1])

# Train the XGBoost model with hyperparameters
xgb_model = XGBClassifier(
    max_depth=4,  #default 6
    learning_rate=0.1, #default 0.3
    n_estimators=50, #default 100
    min_child_weight=10, #default 1
    subsample=0.8, #default 1 fraction of samples used for building each tree.
    colsample_bytree=0.8, #default 1 fraction of features used for building each tree
    scale_pos_weight=scale_pos_weight, #default 1
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=0
)
xgb_model.fit(X_train_resampled, y_train_resampled)

y_pred = xgb_model.predict(X_test_resampled)

accuracy = accuracy_score(y_test_resampled, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test_resampled, y_pred)
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

from xgboost import to_graphviz
import graphviz

dot = to_graphviz(xgb_model)
dot.format = 'png'
dot.render('xgb_tree')