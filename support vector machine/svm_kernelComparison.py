import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import datasets
iris = datasets.load_iris()

colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
data = pd.read_csv("./datasets/iris_data - Copy.csv", names=colnames)

# preprocessing
x = data.drop("Class", axis=1)
y = data["Class"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# training
## linear
lin = SVC(kernel="linear")
lin.fit(xtrain, ytrain)
lin_pred = lin.predict(xtest)

'''
                    ./results/svm-lin-classification_report.png
'''
print(classification_report(ytest, lin_pred))

## polynomial kernel
poly = SVC(kernel="poly", degree=8)
poly.fit(xtrain, ytrain)
poly_pred = poly.predict(xtest)

'''
                    ./results/svm-poly-classification_report.png
'''
print(classification_report(ytest, poly_pred))

## gaussian kernel
gauss = SVC(kernel="rbf")
gauss.fit(xtrain, ytrain)
gauss_pred = gauss.predict(xtest)

'''
                    ./results/svm-gaussian-classification_report.png
'''
print(classification_report(ytest, gauss_pred))

## sigmoid kernel
sig = SVC(kernel="sigmoid")
sig.fit(xtrain, ytrain)
sig_pred = sig.predict(xtest)

'''
                    ./results/svm-sigmoid-classification_report.png
'''
print(classification_report(ytest, sig_pred))

'''
Sigmoid performs the worst. This is because sigmoid function returns 
either 0 or 1 and is thus more suitable for binary classification.

Between the other two, Gaussian achieved 100% accuracy, prediction rate.
This makes Gaussian looks like the best but there isn't a hard and fast
rule. All depends on dataset.
'''

# visualising
X = iris.data[:, :2]
Y = iris.target

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

'''
we only take the first two features. We could avoid the slicing by
using a two dimensional dataset
'''

# make a mesh
h = 0.2 # step size in mesh

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

svc = SVC(kernel='linear', C=1).fit(X, Y)
poly_svc = SVC(kernel='poly', gamma=0.7, C=1).fit(X, Y)
rbf_svc = SVC(kernel='rbf', degree=3, C=1).fit(X, Y)
sig_svc = SVC(kernel='sigmoid', degree=3, C=1).fit(X, Y)


# title for the plots
titles = ['Linear',
          'Polynomial',
          'Gaussian',
          'Sigmoid']
'''
                    ./results/svm-plot-all.png
'''
for i, clf in enumerate((svc, poly_svc, rbf_svc, sig_svc)):
    # plot decision boundary.
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()