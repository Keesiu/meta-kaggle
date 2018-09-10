'''
This script gets log loss on the validation set from full_val_set.pkl, 
(generated by the full_validation_set.py script) for some simple, 
no-learning models like the HistCTR, all 0's, or mean-value benchmark.

author: David Thaler
date: July 2015
'''
import avito2_io
from datetime import datetime
from eval import logloss

maxlines_val = None

start = datetime.now()
val_ids = avito2_io.get_artifact('full_val_set.pkl')
print 'validation set ids read'
train_etl = {'ad'      : lambda l : l['AdID'],
             'pos'     : lambda l : l['Position'],
             'ctr'     : lambda l : l['HistCTR']}
search_etl = {'cat'    : lambda l : l['CategoryID']}
# validation run
input = avito2_io.rolling_join(True, 
                               train_etl, 
                               search_etl, 
                               do_validation=True, 
                               val_ids=val_ids)
loss = 0.0
for (k, (x, y)) in enumerate(input):
  #loss += logloss(float(x['ctr']), y)
  loss += logloss(0.006, y)
  if k == maxlines_val:
    break
  if (k + 1) % 250000 == 0:
    print 'processed %d lines on validation pass' % (k + 1)
    
print 'validation set log loss: %.5f' % (loss/(k + 1))
print 'elapsed time: %s' % (datetime.now() - start)









