import pandas as pd
import numpy as np
import ml_metrics as metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
np.random.seed(131)
path = '../Data/'
print("read training data")
train = pd.read_csv(path+"train_2nd.csv")
label = train['target']
trainID = train['id']
del train['id'] 
del train['target']
tsne = pd.read_csv(path+"train_2nd_tsne.csv")
train = train.join(tsne)

clf = ExtraTreesClassifier(n_jobs=-1, n_estimators=300, verbose=3, random_state=131)
iso_clf = CalibratedClassifierCV(clf, method='isotonic', cv=10)
iso_clf.fit(train.values, label)

print("read test data")
test  = pd.read_csv(path+"test_2nd.csv")
ID = test['id']
del test['id']
tsne = pd.read_csv(path+"test_2nd_tsne.csv")
test = test.join(tsne)

clf_probs = iso_clf.predict_proba(test.values)

sample = pd.read_csv(path+'sampleSubmission.csv')
print("writing submission data")
submission = pd.DataFrame(clf_probs, index=ID, columns=sample.columns[1:])
submission.to_csv(path+"extraTree_2nd.csv",index_label='id')

# retrain

sample = pd.read_csv(path+'sampleSubmission.csv')
submission = pd.DataFrame(index=trainID, columns=sample.columns[1:])
nfold=5
skf = StratifiedKFold(label, nfold, random_state=131)
score = np.zeros(nfold)
i=0
for tr, te in skf:
	X_train, X_test, y_train, y_test = train.values[tr], train.values[te], label[tr], label[te]
	clf = ExtraTreesClassifier(n_jobs=-1, n_estimators=300, verbose=3, random_state=131)
	iso_clf = CalibratedClassifierCV(clf, method='isotonic', cv=10)
	iso_clf.fit(X_train, y_train)
	pred = iso_clf.predict_proba(X_test)
	tmp = pd.DataFrame(pred, columns=sample.columns[1:])
	submission.iloc[te] = pred
	score[i]= log_loss(y_test,pred,eps=1e-15, normalize=True)
	print((score[i]))
	i+=1

print(("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score))))

# nfold 5: 0.421715 + 0.006959383 

print((log_loss(label,submission.values,eps=1e-15, normalize=True)))
submission.to_csv(path+"extraTree_2nd_retrain.csv",index_label='id')

