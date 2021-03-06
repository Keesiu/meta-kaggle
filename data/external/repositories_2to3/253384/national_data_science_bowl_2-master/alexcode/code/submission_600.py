

import csv
import numpy as np
import logging
logging.basicConfig(format = '[%(asctime)s]  %(message)s', level = logging.INFO)

from model import get_model
from utils import real_to_cdf, preprocess, correct_cdf


def load_validation_data():
    """
    Load validation data from .npy files.
    """
    X = np.load('../input/X_validate.npy')
    ids = np.load('../input/ids_validate.npy')

    X = X.astype(np.float32)
    X /= 255

    return X, ids


def accumulate_study_results(ids, prob):
    """
    Accumulate results per study (because one study has many SAX slices),
    so the averaged CDF for all slices is returned.
    """
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    for i in range(size):
        study_id = ids[i]
        idx = int(study_id)
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]), dtype=np.float32)
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in list(cnt_result.keys()):
        sum_result[i][:] /= cnt_result[i]
    return sum_result


def submission():
    """
    Generate submission file for the trained models.
    """
    logging.info('Loading and compiling models...')
    model_systole = get_model()
    model_diastole = get_model()

    logging.info('Loading models weights...')
    model_systole.load_weights('../models/weights/weights_systole_best.hdf5')
    model_diastole.load_weights('../models/weights/weights_diastole_best.hdf5')

    logging.info('Loading validation data...')
    X, ids = load_validation_data()

    logging.info('Pre-processing images...')
    X = preprocess(X)

    batch_size = 32
    logging.info('Predicting on validation data...')
    pred_systole = model_systole.predict(X, batch_size=batch_size, verbose=1)
    pred_diastole = model_diastole.predict(X, batch_size=batch_size, verbose=1)

    # real predictions to CDF
    cdf_pred_systole = correct_cdf(pred_systole)
    cdf_pred_diastole = correct_cdf(pred_diastole)

    logging.info('Accumulating results...')
    sub_systole = accumulate_study_results(ids, cdf_pred_systole)
    sub_diastole = accumulate_study_results(ids, cdf_pred_diastole)

    # write to submission file
    logging.info('Writing submission to file...')
    fi = csv.reader(open('../input/sample_submission_validate.csv'))
    f = open('../submissions/submission_13.csv', 'w')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(next(fi))
    for line in fi:
        idx = line[0]
        key, target = idx.split('_')
        key = int(key)
        out = [idx]
        if key in sub_systole:
            if target == 'Diastole':
                out.extend(list(sub_diastole[key][0]))
            else:
                out.extend(list(sub_systole[key][0]))
        else:
            logging.info('Miss {0}'.format(idx))
        fo.writerow(out)
    f.close()

    logging.info('Done.')

submission()
