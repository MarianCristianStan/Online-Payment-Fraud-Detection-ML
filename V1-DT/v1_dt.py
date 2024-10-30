
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import  confusion_matrix
from sklearn.preprocessing import LabelEncoder

# load the dataset and drop useless column
data = pd.read_csv('onlinefraud.csv')
data = data.drop(columns=['step','nameOrig', 'nameDest', 'isFlaggedFraud'])

# encode type of transaction (convert to numeric)
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

X = data.drop(columns=['isFraud'])
y = data['isFraud']

# train the DT model and predict
dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=50, min_samples_leaf=25)
dt_model.fit(X, y)
y_pred = dt_model.predict(X)

# confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGn',
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Visualize the Decision Tree
plt.figure(figsize=(25, 10))
plot_tree(dt_model,
          feature_names=X.columns,
          class_names=['Legitimate', 'Fraudulent'],
          filled=True,
          rounded=True,
          fontsize=8,
          impurity=False)
plt.title('Decision Tree for Fraud Detection')
plt.show()
