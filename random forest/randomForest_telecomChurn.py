import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

rcParams.update({'figure.autolayout': True})

data = pd.read_csv("./datasets/telecom churn.csv")

data.head()


label = data["Churn"].value_counts(sort=True).index.tolist()

size = data["Churn"].value_counts(sort=True)
'''
                    ./results/plt-churn_percentage.png
'''
plt.pie(size, autopct="%1.1f%%", shadow=True, startangle=55, labels=label, explode=(0.1, 0.1))
plt.title("Churn%")
plt.show()

# dropping irrelevant columns
data.drop(['customerID'], axis=1, inplace=True)
data.isnull().any() # all False indicating no null values

data['TotalCharges'] = pd.to_numeric(data['TotalCharges']) # error
'''
upon extensive search i found that the error is  because of a whitespace
in the TotalCharges column. Since it's not technoically a NaN i got an error
to combat this we use the following line to replace it with nans
you'll see that there are 11 NaNs in TotalCharges now'''
data = data.replace('^\s*$', np.nan, regex=True)
data.isna().sum()

data.dropna(axis=0, inplace=True)

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])

df = data.copy()

# convert predictor variable to binary vategorical
df['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df['Churn'].replace(to_replace='No',  value=0, inplace=True)

# convert all categorical variables into dummy variables
df_dummies = pd.get_dummies(df)
df_dummies.columns

# logistic model
y = df_dummies['Churn'].values
x = df_dummies.drop(columns=['Churn'])

features = x.columns.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x)
x = pd.DataFrame(scaler.transform(x))
x.columns = features

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)


rfmodel = RandomForestClassifier(n_estimators=1000,
                                 oob_score=True, n_jobs=-1,
                                 random_state=42,
                                 max_features="auto",
                                 max_leaf_nodes=30)

rfmodel.fit(xtrain, ytrain)
y_pred = rfmodel.predict(xtest)
print(metrics.accuracy_score(ytest, y_pred)) #79.43% accuracy

# important features
importances = rfmodel.feature_importances_
weights = pd.Series(importances, index=x.columns.values)
'''
                    ./results/rf-feature_importance_weights
'''
weights.sort_values()[-10:].plot(kind="barh")

'''
From random forest algorithm, monthly contract, tenure and total charges 
are the most important predictor variables to predict churn.
'''
