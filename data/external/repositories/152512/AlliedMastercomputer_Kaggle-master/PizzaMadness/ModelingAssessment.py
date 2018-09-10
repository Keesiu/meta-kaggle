from PizzaManager import *

X_train = None
y_train = None
X_test  = None
scaling = None
cheat   = None

#word = WordVecObj()

def initWords():
    print "Loading word2vec model. Could take a while."
    loadModel()
    print "Finished loading. Now loading EVecs."
    df = getTrainingData()
    loadEVecs(df,numVecs=10)
    print "Finished loading EVecs."

def addFeatures(df):
    addTextLen(df)
    addPostHour(df)
    addWordVecFeatures(df)

def initModeling():
# Set up training data
    df = getTrainingData()
    addFeatures(df)
    X_train,y_train = getNumericTraining(df)
    scaling = sk.preprocessing.StandardScaler()
    X_train = scaling.fit_transform(X_train)
# Set up test data
    testdf = getTestData()
    addFeatures(testdf)
    X_test = getNumericTest(testdf)
    X_test = scaling.transform(X_test)
    cheat = getTestCheat()

def modelSVM(xdata = X_train, ydata = y_train, cval = 0.01):
    pizzaSVM = svm.SVC(kernel='linear', C=cval)
    scores = cross_validation.cross_val_score(pizzaSVM, xdata, ydata, cv=10,scoring='f1')
    pizzaSVM.fit(xdata,ydata)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return (pizzaSVM2,scores)

def modelLog(xdata = X_train, ydata = y_train, cval = 0.01):
    pizzaLog = sk.linear_model.LogisticRegression(penalty='l2',C=cval)
    scores = cross_validation.cross_val_score(pizzaLog, xdata, ydata, cv=10,scoring='f1')
    pizzaLog.fit(xdata,ydata)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return (pizzaLog,score)

def modelRandomTree(xdata = X_train, ydata = y_train, ntrees = 10):
    pizzaTree = sk.ensemble.RandomForestClassifier(n_estimators = ntrees)
    scores = cross_validation.cross_val_score(pizzaTree, xdata, ydata, cv=10,scoring='f1')
    pizzaTree.fit(xdata,ydata)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return (pizzaTree,scores)

def getPrediction(mlmodel,xpredict=X_train,ypredict=y_train,metrics=True):
    predicted = mlmodel.predict(xpredict)
    if metrics:
        print(sk.metrics.confusion_matrix(ypredict, predicted))
        print(sk.metrics.classification_report(ypredict, predicted))
    return predicted



