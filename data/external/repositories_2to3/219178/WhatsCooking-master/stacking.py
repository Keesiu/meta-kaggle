from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

import EnsembleClassifier

from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt('trainProcessedDF.csv', delimiter='::')[1:]
    target = [x[1] for x in dataset]
    train = [x[2:] for x in dataset]
    test = genfromtxt('testProcessedDF.csv', delimiter='::')[1:]
    test_stripped = [x[2:] for x in test]

    rf = RandomForestClassifier(n_estimators=2000, n_jobs=8)
    lr = LogisticRegression()
    sgd = SGDClassifier()

    ensemble = EnsembleClassifier(clfs=[rf, lr, sgd], voting='hard')
    ensemble.fit(target, train)

    prediction = [[test[index][0], x] for index, x in enumerate(ensemble.predict(test_stripped))]

    savetxt('submission.csv', prediction, delimiter=',', fmt='%d,%f',
            header='id,cuisine', comments = '')

if __name__=="__main__":
    main()
