import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

data = pd.read_csv('onlinefraud.csv')

#Calculate fraud percentage
fraud_counts = data['isFraud'].value_counts(normalize=True) * 100
fraud_labels = ['Legit (0)', 'Fraud (1)']

plt.figure(figsize=(10, 6))
sns.barplot(x=fraud_labels, y=fraud_counts.values, color='green')
plt.title('Percentage of Fraudulent vs Legitimate Transactions')
plt.ylabel('Percentage')
plt.xlabel('isFraud')

# Annotate percentage on each bar
for i, v in enumerate(fraud_counts.values):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center')
plt.show()

#transaction type
transactions_type = data["type"].value_counts()
transactions = transactions_type.index
quantity = transactions_type.values
figure = px.pie(data, values=quantity, names=transactions, hole = 0.3, title="Distribution of Transaction Type")
figure.show()

top_orig_fraud = data[data['isFraud'] == 1]['nameOrig'].value_counts().head(10)
top_dest_fraud = data[data['isFraud'] == 1]['nameDest'].value_counts().head(10)

# Fraudulent Senders
plt.figure(figsize=(10, 6))
sns.barplot(x=top_orig_fraud.index, y=top_orig_fraud.values, color='cyan')
plt.title('Top 10 Fraudulent Senders')
plt.xlabel('Sender ID')
plt.ylabel('Number of Fraudulent Transactions',)
plt.xticks(rotation=45)
plt.show()

# Fraudulent Receivers
plt.figure(figsize=(10, 6))
sns.barplot(x=top_dest_fraud.index, y=top_dest_fraud.values, color='red')
plt.title('Top 10 Fraudulent Receivers')
plt.xlabel('Receiver ID')
plt.ylabel('Number of Fraudulent Transactions')
plt.xticks(rotation=45)
plt.show()
