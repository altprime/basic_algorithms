import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap


data = pd.read_csv('./datasets/social_network_ads.csv')
x = data.iloc[:, [2,3]].values
y = data.iloc[:, 4].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)

# feature scaling
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# model fitting and predicting
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(xtrain, ytrain)

y_pred = classifier.predict(xtest)

# confusion matrix

cm = confusion_matrix(ytest, y_pred)

# visualising training results
'''
                    ./results/ads-training.png
'''
xset, yset = xtrain, ytrain
X1, X2 = np.meshgrid(np.arange(start = xset[:, 0].min() - 1, stop=xset[:, 0].max() + 1, step=0.01),
                     np.arange(start = xset[:, 1].min() - 1, stop=xset[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('pink', 'cyan')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Train Set')
plt.xlabel('Age')
plt.ylabel('Est Sal')
plt.legend()
plt.show()

# visualising training results
'''
                    ./results/ads-testing.png
'''
xset, yset = xtest, ytest
X1, X2 = np.meshgrid(np.arange(start = xset[:, 0].min() - 1, stop = xset[:, 0].max() + 1, step = 0.01),
                     np.arange(start = xset[:, 1].min() - 1, stop = xset[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'cyan')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Test Set')
plt.xlabel('Age')
plt.ylabel('Est Sal')
plt.legend()
plt.show()
