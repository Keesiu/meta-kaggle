import pandas as pd
import numpy as np
import ml_metrics as metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

path = '../Data/'
print("read training data")
train = pd.read_csv(path+'train_tfidf.csv')
label = train['target']
trainID = train['id']
del train['id'] 
del train['target']
tsne = pd.read_csv(path+'tfidf_train_tsne.csv')
train = train.join(tsne)

clf = ExtraTreesClassifier(n_jobs=-1, n_estimators=300, verbose=3, random_state=131)
iso_clf = CalibratedClassifierCV(clf, method='isotonic', cv=10)
iso_clf.fit(train.values, label)

print("read test data")
test  = pd.read_csv(path+'test_tfidf.csv')
ID = test['id']
del test['id']
tsne = pd.read_csv(path+'tfidf_test_tsne.csv')
test = test.join(tsne)

clf_probs = iso_clf.predict_proba(test.values)

sample = pd.read_csv(path+'sampleSubmission.csv')
print("writing submission data")
submission = pd.DataFrame(clf_probs, index=ID, columns=sample.columns[1:])
submission.to_csv(path+'extraTree_tfidf.csv',index_label='id')

# retrain

sample = pd.read_csv(path+'sampleSubmission.csv')
submission = pd.DataFrame(index=trainID, columns=sample.columns[1:])
nfold=5
score = np.zeros(nfold)
skf = StratifiedKFold(label, nfold, random_state=131)
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
	print(score[i])
	i+=1

print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))

# cv 10, 0.4640333 + 0.008130822 
# nfold 5: 0.4662828, stddev 0.008184
# nfold 4: 0.4688583, stddev 0.004398
# nfold 3: 0.4706140, stddev 0.004154

print(log_loss(label,submission.values,eps=1e-15, normalize=True))
submission.to_csv(path+'extraTree_tfidf_retrain.csv',index_label='id')

