<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>Online Payment Fraud Detection Using Machine Learning</title>
</head>
<body>

    <h1>Online Payment Fraud Detection Using Machine Learning</h1>

    <h2>Project Description</h2>
    <p>
        This project aims to detect fraudulent transactions on an online payment platform using machine learning algorithms. 
        Using a historical dataset, which includes both real and simulated transactions, our model will learn to classify 
        transactions into two categories: fraudulent and non-fraudulent.
    </p>

    <h3>Objectives:</h3>
    <ul>
        <li><strong>Build a classification model</strong> capable of detecting fraud with high accuracy.</li>
        <li><strong>Explore the dataset</strong> to understand the factors contributing to financial fraud.</li>
        <li><strong>Evaluate model performance</strong> using metrics specific to binary classification problems 
            (precision, recall, AUC-ROC, etc.).
        </li>
    </ul>

    <h2>Data</h2>
    
    <h3>Dataset Link:</h3>
    <p>
        <a href="https://www.kaggle.com/datasets/ntnu-testimon/paysim1" target="_blank">Online Payment Fraud Detection Dataset - Kaggle</a>
    </p>

    <h3>Dataset Description:</h3>
    <p>
        This dataset contains simulated financial transactions from an online payment platform. Each transaction is labeled 
        as either fraudulent or non-fraudulent. The dataset contains <strong>636,262 examples</strong> and <strong>10 features</strong>.
    </p>

    <h3>Dataset Features:</h3>
    <table border="1" cellpadding="5" cellspacing="0">
        <thead>
            <tr>
                <th>Column</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>step</strong></td>
                <td>Time unit (1 step = 1 hour).</td>
            </tr>
            <tr>
                <td><strong>type</strong></td>
                <td>Type of transaction (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).</td>
            </tr>
            <tr>
                <td><strong>amount</strong></td>
                <td>The amount involved in the transaction.</td>
            </tr>
            <tr>
                <td><strong>nameOrig</strong></td>
                <td>Unique ID of the customer initiating the transaction.</td>
            </tr>
            <tr>
                <td><strong>oldbalanceOrg</strong></td>
                <td>Initial balance of the sender before the transaction.</td>
            </tr>
            <tr>
                <td><strong>newbalanceOrig</strong></td>
                <td>Sender’s balance after the transaction.</td>
            </tr>
            <tr>
                <td><strong>nameDest</strong></td>
                <td>Unique ID of the recipient of the transaction.</td>
            </tr>
            <tr>
                <td><strong>oldbalanceDest</strong></td>
                <td>Initial balance of the recipient before the transaction.</td>
            </tr>
            <tr>
                <td><strong>newbalanceDest</strong></td>
                <td>Recipient’s balance after the transaction.</td>
            </tr>
            <tr>
                <td><strong>isFraud</strong></td>
                <td>Target variable indicating whether the transaction is fraudulent (1 = fraud, 0 = non-fraud).</td>
            </tr>
        </tbody>
    </table>

    <h2>Project Task</h2>
    <p>
        The goal of the project is to train and evaluate a <strong>binary classification</strong> model to detect fraudulent transactions. 
        We will analyze the available data to understand the patterns in fraudulent transactions and build an efficient model to predict 
        fraud based on various transaction features.
    </p>

    <h2>Algorithms Used</h2>
    <p>For this project, we will use the following machine learning algorithms:</p>
    <ul>
        <li><strong>Random Forest</strong> - An ensemble algorithm that builds multiple decision trees and aggregates the results for robust classification.</li>
        <li><strong>Gradient Boosting (GBM)</strong> - An advanced boosting method for building high-performance classification models.</li>
        <li><strong>Logistic Regression</strong> - A simple binary classification model that is fast and interpretable.</li>
        <li><strong>Support Vector Machines (SVM)</strong> - Used for both linear and non-linear separation of the data.</li>
        <li><strong>Neural Networks</strong> - Useful for capturing complex, non-linear patterns in financial transactions.</li>
    </ul>

    <h2>Project Steps</h2>
    <ol>
        <li><strong>Data Exploration</strong> (Exploratory Data Analysis - EDA)
            <ul>
                <li>Data cleaning and statistical analysis to understand the structure of the dataset and relationships between features.</li>
                <li>Detect and handle missing or anomalous values.</li>
            </ul>
        </li>
        <li><strong>Data Preprocessing</strong>
            <ul>
                <li>Normalize the data and encode categorical variables (e.g., transaction type).</li>
                <li>Handle class imbalance (fraud cases are rare) using resampling techniques like <strong>SMOTE</strong> (Synthetic Minority Over-sampling Technique) or class weighting.</li>
            </ul>
        </li>
        <li><strong>Model Training</strong>
            <ul>
                <li>Build and train multiple binary classification models.</li>
                <li>Use cross-validation to validate model performance.</li>
            </ul>
        </li>
        <li><strong>Performance Evaluation</strong>
            <ul>
                <li>Use appropriate metrics for imbalanced data: <strong>precision</strong>, <strong>recall</strong>, <strong>f1-score</strong>, <strong>AUC-ROC</strong>.</li>
                <li>Analyze the performance of each algorithm to determine the most effective model for fraud detection.</li>
            </ul>
        </li>
        <li><strong>Model Tuning</strong> - Hyperparameter tuning to improve the performance of the final model.</li>
        <li><strong>Conclusions</strong> - Interpret the results and provide recommendations for implementing the model in production.</li>
    </ol>

    <h2>System Requirements</h2>
    <p>To run this project, make sure you have the following Python packages installed:</p>
    <pre><code>pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm</code></pre>

    <h2>How to Run the Project</h2>
    <ol>
        <li>Clone this repository:
            <pre><code>git clone https://github.com/username/online-fraud-detection.git</code></pre>
        </li>
        <li>Navigate to the project directory:
            <pre><code>cd online-fraud-detection</code></pre>
        </li>
        <li>Run the script to train the model:
            <pre><code>python train_model.py</code></pre>
        </li>
    </ol>

    <h2>Expected Results</h2>
    <p>
        At the end of this project, we expect to develop a model capable of detecting fraud in online financial transactions with 
        high accuracy, along with a high <strong>recall</strong> for the minority class (fraud), which is essential for such problems.
    </p>


</body>
</html>
