from PizzaManager import *
#import csv
from sklearn import datasets
#import sklearn as sk
#from sklearn import *

X_train,y_train = getNumericTraining()
print(y_train)

iris = datasets.load_iris()
iX_train, iX_test, iy_train, iy_test = sk.cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(iX_train, iy_train)
print(clf.score(iX_test, iy_test))

iX_train, iX_test, iy_train, iy_test = sk.cross_validation.train_test_split(X_train, y_train, test_size=0.4, random_state=0)
#clf = svm.SVC(kernel='linear', C=1).fit(iX_train, iy_train)

getNumericTest()
