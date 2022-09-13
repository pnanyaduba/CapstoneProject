<!-- markdownlint-disable -->
<h1 align="center">
   Capstone Project
    <br>
</h1>


<h2 align="center">
   **Credit Card Fruad Detection**
    <br>
</h2>

---

<p align="center">
    <strong>🏆&nbsp; In partial fulfillment for the award of the Professional Certificate in Machine Learning and Artificial Intelligence</strong>
</p>

<p align="center">
    <a href="https://github.com/pnanyaduba/kraftwerk/tree/main/practical_application_II_starter" title="Best-of-badge"><img src="https://ci.appveyor.com/api/projects/status/32r7s2skrgm9ubva?svg=true&passingText=master%20-%20OK"></a>
    <a href="#Contents" title="Project Count"><img src="https://img.shields.io/badge/projects-2nd-blue.svg?color=5ac4bf"></a>
    <a href="#Contribution" title="Contributions are welcome"><img src="https://img.shields.io/badge/contributions-welcome-green.svg"></a>
    <a href="#" title="Best-of Updates"><img src="https://img.shields.io/github/release-date/ml-tooling/best-of-ml-python?color=green&label=updated"></a>
    <a href="https://twitter.com/peteberc" title="Follow on Twitter"><img src="https://img.shields.io/twitter/follow/mltooling.svg?style=social&label=Follow"></a>
</p>

In this capstone project, the datasets explored contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. The dataset is from the following site: [Data Source Site](https://data.world/raghu543/credit-card-fraud-data)

#### Business Objective
The business objective is to help a financial institution to lower the risk of loss of funds for its customers. Secondly, it will also enable the bank to improve on its public image and reputation.

<h4>
   <b>Selected Model: Support Vector Machine was selected</b>
</h4>
<div style="font-size:13px;font-family:'Arial'; font-weight:400">The proprietary information has been provided by the source: <span style="font-size:14px;font-family:'Times New Roman'; font-style:italic;">Machine Learning Group of Université Libre de Bruxelles</span></div>

---

<p align="center">
     🧙‍♂️&nbsp; Click here to access the Project Notebook <a href="https://github.com/pnanyaduba/CapstoneProject/blob/main/CapstoneNotebookCopy2.ipynb">Capstone Jupyter Notebook</a> <br>
    📫&nbsp; Subscribe to my Twitter handle <a href="http://twitter.com/hashtharg">@hashtharg</a> for updates and my tweets.
</p>

---


<h2>
   Executive Summary
    <br>
</h2>


All over the world millions of credit card transactions are constantly processed by thousands of financial institutions. 
These transactions often are processed in real time and often across geographical regions and borders. More often than not, 
speed and timeliness is the key factors that holders of credit cards view as motivation for using online transactions. However, 
these qualities have their downsides. One key downside is the risk that customers will loose their funds if their credit card 
details are stolen.

Thousands of fraudulent transactions from stolen credit cards occur on a daily basis. Most customers whose credit cards have 
entered the wrong hands only find out when they receive invoices or receipts of what they didn't pay for.

Often this constitute a risk for the issuing financial institutions who sometimes might face serious consequences as a result of processing transactions from stolen or lost credit card. This high risk factor can cost financial organizations dearly. 

Most financial organizations as a result have turned to technology to help detect fraudulent transactions and hence lower their risk. In addition to lowering their risk, detecting and foiling fraudulent credit card transactions will improve the image and integrity of the institutions.

**Methodology.** The methodology used involves 5 key machine learning techniques which include imbalanced class handling, categorical features, clustering and classification, sequential modelling and performance measures.

The dataset utilized contains a 2-day transactions made by European credit card users in September 2013. About 492 fraudulent transactions were identified from 284,807 transactions. The dataset has highly imbalanced classes. These dataset have been processed using Principal Component Analysis. The target feature is the 'Class' field which can either be a 'zero' or a 'one' representing legitimate or fraudulent transactions respectively. 

The dataset used has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.

In executing the project, the general information about the dataset is obtained. Existence of null values is checked. The distribution of the dataset is plotted to get a clear picture of the data. Next, the distribution of the classes are separated for analysis. The statistical characteristics of both legitimate and fraudulent transactions are obtained and compared. Below is the definition of the dataset:

- The dataset contains a total of 284,807 transactions
- There are two classes of data. Fraud class which contains 492 transactions and Legitimate class which contains of 284,315 transactions.
- The dataset is highly unbalanced and Fraud class represents 0.172% of all transactions.
- The dataset has been pre-prepared and has the numerical features resulting from PCA transformation.
- Features include Time, V1, V2, V3 to V28 and 'Amount'.
- Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.
- The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning.
- Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.


The next step involves handling the Imbalanced classes in which under-sampling was utilized instead of over-sampling in order to minimize the computational cost. An equal amount of both legitimate and fraudulent transactions are then obtained. The under-sampled data is not split into Train and Test sets.

Models were developed and tuned using the under-sampled train dataset. The models used include LogisticRegression, KNNClassifier, RandomForestClassifier, and the DecisionTreeClassifier. RandomForestClassifier model was finally selected based on its performance. The model's accuracy is obtained and the threshold is determined.


**The Findings.** The findings from the study are as follows:

1. RandomForestClassifer produced the best F1 Score and Threshold to predict fraud detection. 
2. The mean of fraudulent transactions is about $122.29.
3. The maximum amount of all fraudulent transactions is about $2,125.87
4. The accuracy of the RandomForestClassifier is 0.9538
6. The threshold that will maximize the model's ability to predict fraudulent transactions is 0.60
7. Most Fraudulent transactions are under $1,000
8. Most transactions greater that $1,000 can be considered as outliers


**Recommendation.** It is recommended that focus should be placed on transactions that are less that $1,000 as most fraudulent transactions occur at amounts less or within that range.

**Conclusion.** In conclusion, Using  a RandomForestClassifer provides an effective machine learning mechanism to detect fraudulent credit card transactions.

---

<h2>
   Rationale
    <br>
</h2>

The project is targeted at Banks or financial institutions whose objective is to combat and foil fraudulent credit card transactions in real time.



<h2>
   Research Question
    <br>
</h2>

How can financial institutions and Banks effectively combat fraudulent credit card transactions?

<h2>
   Data Sources
    <br>
</h2>


#### **Sourcing**
The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. The dataset used can be attributed to Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

<h2>
   Data Preparation
    <br>
</h2>

Below are steps followed for preparing the data for analysis:

- The dataset has been pre-prepared before it was obtained and contains numerical features resulting from PCA transformation.
- The dataset contains fields which have been pre-prepared and contains a 2-day credit cards transactions by european cardholders in Sept. 2013.
- The dataset contains a total of 284,807 transactions
- There are two classes of data. Fraud class which contains 492 transactions and Legitimate class which contains of 284,315 transactions.
- The dataset is highly unbalanced and Fraud class represents 0.172% of all transactions.
- Features include Time, V1, V2, V3 to V28 and 'Amount'.
- Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.
- The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning.
- Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
- Lastly, the imbalanced dataset was processed using under-sampling in order to obtain a balanced dataset
- The balanced data obtained as follows: Fraudulent Transactions: 492, Legitimate Transactions: 492
- Mean of Fraudulent Transactions of Balanced Data: 122.21
- Mean of Legitimate Transactions: 99.80
- The Balanced Dataset is split into Train and Test data

<h2>
   Final Dataset
    <br>
</h2>

The final dataset contains:
- A <b>Fraudulent Transactions Class</b> of <b>492</b> and a <b>Legitimate Transactions Class</b> of <b>284,315</b>
- The Features include: 
   - Time
   - Amount 
   - V1 TO V28, 
   - Class


<h2>
   Methodology
    <br>
</h2>

The following methods were utilized in the project.

1. Information about the dataset utilized was obtained
2. Null values were checked. The dataset do not have null values.
3. A plot of the distribution of the dataset was generated and the level of imbalanced classes were determined 
4. A plot of the features of the dataset was obtained.
5. A time series of the dataset was plotted 
6. A plot of the Amount of the dataset was generated which showed a normalized distribution
7. A count of the classes of the dataset was obtained: Fraudulent Transactions: 492, Legitimate Transactions: 284315
8. Both Fraudulent and Legitimate Transactions were separated
9. Statistics of Fraudulent Transactions were generated
10. Statistics of Legitimate Transactions were generated
11. Histogram and Scatterplot of both Fraudulent and Legitimate Transactions were generated 
12. Imbalanced Data was handled using Under-sampling
13. Balanced Dataset is obtained.
14. Balanced data obtained as follows: Fraudulent Transactions: 492, Legitimate Transactions: 492
15. Mean of Fraudulent Transactions: 122.21
16. Mean of Legitimate Transactions: 99.80
17. The Balanced Dataset is split into Train and Test data

#### **Methodology for Models**
18. Use One-Versus-Rest, One-Versus-One and Multinomial Classification to determine best accuracy values.
19. Four models were developed using the following classifiers each: LogisticRegression, KNNClassifier, RandomForestClassifer and DecisionTreeClassifier
21. Hyperparameter tuning was performed on each Model using the GridSearchCV and their best scores were obtained
22. Their best scores were presented alongside each other in a tabular form and the best performing score was selected
23. The RandomForestClassifer was selected because it produced the best performance score.
24. Lastly, the RandomForestClassifier was fine-tuned 
25. The ClassificationReport report was obtained
26. The ConfusionMatrix was plotted
27. The threshold that will maximize the models ability to predict fraudulent transactions was producted
28. Finally, the Precision was obtained.


<h2>
   Results
    <br>
</h2>
The Analysis of the four models created and hyperparameter-tuned which include, LogisticticRegression, DecisionTree Classifier, KNNClassifier and RandomForestClassifier was performed according to the BEST SCORE criteria. 

The outcome of the analysis, based on best-score performance figures for hyperparameter tuning shows that the RandomForestClassifer is the best mechanism to predict if a transaction is fraudulent. Out of the four models created and tuned, the RandomForestClassifer was selected. 

<h2>
   Outline of Project
    <br>
</h2>

https://github.com/pnanyaduba/CapstoneProject/blob/main/CapstoneNotebookCopy2.ipynb

<h2>
   Conclusion
    <br>
</h2>

Using  a RandomForestClassifer provides an effective machine learning mechanism to detect fraudulent credit card transactions. Improvements to the project will involve an indepth hyperparameter tuning and possibly explore using neural networks to improve on the results of the project.

<h2>
   Contact and Further Information
    <br>
</h2>


Nwachukwu Peter Anyaduba <br />
+234 708.688.3202<br />
Twitter: @pete2ai<br />
Email: pete2ai.co@gmail.com<br />






## Report
The Analysis of the Logistic Regress, Decision Tree Classifier, KNeighborsClassifier and the Support Vector Machines was performed according to the following criteria
1. Imbalance Class Handling
2. Model Training Speed
3. Interpretable Results

Other criteria observed include
1. Accuracy
2. Precision
3. Recall
4. Specificity
5. Mean Squared Error

- **Logistic Regression Classifier**
    - SMOTE was used to handle imbalanced classes
    - Speed of Training is moderately high at 0.58s
    - Train Score performs slightly better than the Test Score
    - Accuracy and Specificity are not too high at 59. and 33.
    - Precision is low at 56.%
    - Recall is high at 85.%
    - Train and Test MSEs are relatively equal
- **Decision Tree Classifier**
    - SMOTE was used to handle imbalanced classes
    - Speed of Training is moderately high at 0.4s
    - Accuracy and Specificity are very high at 75.9% and 69.2% respectively
    - Train Score performs slightly better than the Test Score
    - Precision is high at 72.7%
    - Recall is low at 82.6%
    - However, the Decision Tree Classifier appears to overfit as Train MSE is lower than Test MSE
- **KNearest Neighbors Classifier**
    - SMOTE was used to handle imbalanced classes
    - Speed of Training is high at 0.45s
    - Accuracy and Specificity are very high at 72% and 66% respectively
    - Train Score is slightly higher than the Test Score
    - Precision is high at 69.8%
    - Recall is low at 78.4%
    - However, the KNNeighbors Classifier appears to slightly overfit since Test MSE is higher than Train MSE
    
- **Support Vector Machine**
    - SMOTE was used to handle imbalanced classes
    - Speed of Training is least at 12s
    - Accuracy and Specificity are very high at 62.8% and 48% respectively
    - Test Score is lower than Train Score
    - Precision is 58.%
    - Recall is high at 77.%
    - Train and Test MSE are relatively equal

### Selecting Best Model: Support Vector Machine
- The Support Vector Machine was selected as best model because the Train and Test MSEs are relatively equal.

## Contents on Jupyter Notebook Steps

- [Jupyter Notebook Link](https://github.com/pnanyaduba/PortugueseBank/blob/main/PortugueseBank.ipynb)
- Information about the data
- Dropping Unwanted Columns
- Encode the whole categorical columns
- Get the Correlation Matrix
- Plot the Scatter Matrix
- Define the Modelling Data
- Perform Principal Component Analysis of the Scaled Data
- Create four models - Logistic Regression, DecisionTree, KNN and SVC
- Evaluate the four models
- Summarize the results of the Evaluations
- Select the Best Model out of the four based on the summary
- Create a Report of the Analysis

<br>

## Machine Learning Libraries Used on Jupyter Notebook

- statsmodels.tsa.filters.filtertools as ft
- sklearn.metrics import mean_squared_error
- statsmodels.tsa.filters.filtertools import convolution_filter
- sklearn.feature_selection import SequentialFeatureSelector
- statsmodels.tsa.seasonal import _extrapolate_trend
- pandas.testing as tm
- statsmodels.tsa.arima_process as arima_process
- statsmodels.graphics.tsaplots as tsaplots
- numpy as np
- pandas as pd
- matplotlib.pyplot as plt
- statsmodels.api as sm
- statsmodels.tsa.seasonal import seasonal_decompose
- sklearn.preprocessing import OneHotEncoder
- sklearn.pipeline import Pipeline
- sklearn.preprocessing import StandardScaler
- sklearn.impute import SimpleImputer
- sklearn.compose import ColumnTransformer
- sklearn.decomposition import PCA
- sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
- sklearn.pipeline import make_pipeline
- sklearn.model_selection import train_test_split, GridSearchCV
- sklearn.preprocessing import PolynomialFeatures, OrdinalEncoder
- sklearn.linear_model import Ridge
- scipy import stats
- scipy.linalg import svd
- warnings
- warnings.filterwarnings('ignore')
