import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import plot_tree
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

smote_tomek = SMOTETomek(random_state=0)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
X_test_resampled, y_test_resampled = smote_tomek.fit_resample(X_test, y_test)

# BRF model with hyperparameters
brf_model = BalancedRandomForestClassifier(
    n_estimators=50,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=0
)
brf_model.fit(X_train_resampled, y_train_resampled)

y_pred = brf_model.predict(X_test_resampled)

accuracy = accuracy_score(y_test_resampled, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test_resampled, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGn',
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
plt.title('Confusion Matrix for Balanced Random Forest Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Visualizing feature importance
feature_importances = brf_model.feature_importances_
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=X.columns)
plt.title('Feature Importances for Balanced Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


plt.figure(figsize=(20, 10))
plot_tree(brf_model.estimators_[0], feature_names=X.columns, filled=True, fontsize=8)
plt.title('Visualization of a Single Decision Tree from Balanced Random Forest')
plt.show()
