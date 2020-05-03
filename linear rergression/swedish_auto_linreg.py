'''
dataset description
X: number of claims
Y: total payment for claims
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_score, train_test_split, KFold
#from lcp import plot_learning_curve
from sklearn.model_selection import learning_curve
from scipy.spatial import ConvexHull

data = pd.read_csv("./datasets/swedish auto claims.csv")
data.info()

# basic stats
data.describe()

# visualisation

'''
                    ./results/plot1-Distributions.png
'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.set_title('Distribution: Number of Claims')
sns.distplot(data.X, bins=50, ax=ax1)

ax2.set_title('Distribution: Total Payment for Claims')
sns.distplot(data.Y, bins=50, ax=ax2)

'''
It can be seen that the distributions are almost similar in shape indicating 
a strong linear relationship between the dependent and independent variable

Although I see outliers, which I will confirm using Boxplots
'''

'''
                    ./results/plot2-Boxplot_x.png
'''
fig, (ax1) = plt.subplots(figsize=(10, 5))
ax1.set_ylim(-50, 150)
ax1.set_title('Boxplot: Number of Claims')
sns.boxplot(y='X', data=data, ax=ax1,)
sns.stripplot(y='X', color='green', data=data, jitter=True, ax=ax1, alpha=0.5)

'''
We can see clearly that there are some outliers
'''

# handling outliers
'''
for this i would like to use convex hull algorithm
the following procedure is followed:
    1. scatter plot of X vs Y showing the outliers
    2. scatter plot of convex hull marking the outliers as vertices
    3. removing and reshaping the data back to its original form
    4. scatter plot of clean data without outliers
'''

'''
                    ./results/plot3-scatter plot with outliers.png
'''

plt.scatter("X", "Y", data=data)
plt.xlabel("Claims")
plt.ylabel("Payment")

# Convex Hull
points = data[["X", "Y"]].values
hull = ConvexHull(points)
print(points[hull.vertices])

'''
                    ./results/plot4-convexHull_outlierVertices.png
'''

plt.plot(data["X"], data["Y"], 'ok')
plt.plot(points[hull.vertices, 0], points[hull.vertices,1], 'r--', lw = 2)
plt.plot(points[hull.vertices, 0], points[hull.vertices,1], 'ro', lw = 2)
plt.xlabel("Claims")
plt.ylabel("Payments")

clean_data = pd.DataFrame((np.delete(points[:,0], hull.vertices), np.delete(points[:,1], hull.vertices), 'ok'))
clean_data = clean_data.transpose()
clean_data = clean_data.loc[:, 0:1]
clean_data.columns = ["X", "Y"]

'''
                    ./results/plot5-clean data.png
'''
plt.scatter("X", "Y", data=clean_data)
plt.xlabel("Claims")
plt.ylabel("Payments")

# traininf the model
x = pd.DataFrame(clean_data.X)
y = clean_data.Y

regressionModel = linear_model.LinearRegression().fit(x, y)
y_pred = regressionModel.predict(x)
mse = metrics.mean_squared_error(y_pred, y)
print("Root Mean Squared Error: ", np.sqrt(mse))

# cross validation
regressionModel_cv = linear_model.LinearRegression()
scores = cross_val_score(regressionModel_cv, x, y, cv=10, scoring="neg_mean_squared_error")
scores = scores * -1
print("Root Mean Squared Error: ", np.mean(np.sqrt(scores)))

###

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

'''
                    ./results/plot6-train_test_samples.png
'''
fig, ax = plt.subplots()
ax.set_xlabel('Claims')
ax.set_ylabel('Payments')
ax.set_title('Scatter plot showing train and test sample split')
ax.scatter(xtrain,ytrain,marker='*',label='Train')
ax.scatter(xtest,ytest,c='red',label='Test')
ax.legend()

regressionModel_final = linear_model.LinearRegression().fit(xtrain, ytrain)
y_pred_final = regressionModel_final.predict(xtest)
rmse = np.sqrt(metrics.mean_squared_error(ytest, y_pred_final))
print("Root Mean Squared Error final", rmse)

# plotting the regression line
'''
                    ./results/plot7-regression line.png
'''
fig, ax = plt.subplots()
ax.set_xlabel('Claims')
ax.set_ylabel('Payments')
ax.set_title('Scatter plot showing train and test sample split')
ax.scatter(xtrain,ytrain,marker='*',label='Train')
ax.scatter(xtest,ytest,c='red',label='Test')
ax.legend()
ax.plot(x, y_pred, color="black")
