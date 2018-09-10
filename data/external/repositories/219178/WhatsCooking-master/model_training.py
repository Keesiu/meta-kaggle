#
# Model training using sk-learn.
#

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from numpy import genfromtxt, savetxt

def trainRF(train, target, test, test_stripped, n_estimators=1000, n_jobs=8):
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
    rf.fit(train, target)
    prediction = [[test[index][0], x] for index, x in enumerate(rf.predict(test_stripped))]
    return prediction

def trainLR(train, target, test, test_stripped):
    lr = LogisticRegression()
    lr.fit(train, target)
    prediction = [[test[index][0], x] for index, x in enumerate(lr.predict(test_stripped))]
    return prediction

def trainGBM(train, target, test, test_stripped):
    gbm = GradientBoostingClassifier()
    gbm.fit(train, target)
    prediction = [[test[index][0], x] for index, x in enumerate(gbm.predict(test_stripped))]
    return prediction

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt('trainProcessedDF.csv', delimiter='::')[1:]
    target = [x[1] for x in dataset]
    train = [x[2:] for x in dataset]
    test = genfromtxt('testProcessedDF.csv', delimiter='::')[1:]
    test_stripped = [x[2:] for x in test]

    prediction = trainRF(train, target, test, test_stripped, 2000, 8)

    savetxt('submission.csv', prediction, delimiter=',', fmt='%d,%f',
            header='id,cuisine', comments = '')

if __name__=="__main__":
    main()
