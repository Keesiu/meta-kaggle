

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import logging
logging.basicConfig(format = '[%(asctime)s]  %(message)s', level = logging.INFO)

from model import get_model
from utils import crps, real_to_cdf, real_to_ans, preprocess, rotation_augmentation, shift_augmentation, correct_cdf

seed = 1377713

def load_train_data():
    """
    Load training data from .npy files.
    """
    X = np.load('../input/X_train.npy')
    y = np.load('../input/y_train.npy')

    X = X.astype(np.float32)
    X /= 255

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    return X, y


def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]

    return X_train, y_train, X_test, y_test


def train():
    """
    Training systole and diastole models.
    """
    logging.info('Loading and compiling models...')
    model_systole = get_model()
    model_diastole = get_model()

    logging.info('Loading training data...')
    X, y = load_train_data()

    logging.info('Pre-processing images...')
    X = preprocess(X)

    # split to training and test
    X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.15)

    nb_iter = 500
    epochs_per_iter = 1
    batch_size = 32
    calc_crps = 1  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_val_loss_systole = sys.float_info.max
    min_val_loss_diastole = sys.float_info.max

    logging.info('-'*50)
    logging.info('Training...')
    logging.info('-'*50)

    for i in range(nb_iter):
        logging.info('-'*50)
        logging.info('Iteration {0}/{1}'.format(i + 1, nb_iter))
        logging.info('-'*50)

        logging.info('Augmenting images - rotations')
        X_train_aug = rotation_augmentation(X_train, 20)
        logging.info('Augmenting images - shifts')
        X_train_aug = shift_augmentation(X_train_aug, 0.1, 0.1)

        logging.info('Fitting systole model...')
        hist_systole = model_systole.fit(X_train_aug, real_to_ans(y_train[:, 0]), shuffle=True, nb_epoch=epochs_per_iter,
                                         batch_size=batch_size, validation_data=(X_test, real_to_ans(y_test[:, 0])))

        logging.info('Fitting diastole model...')
        hist_diastole = model_diastole.fit(X_train_aug, real_to_ans(y_train[:, 1]), shuffle=True, nb_epoch=epochs_per_iter,
                                           batch_size=batch_size, validation_data=(X_test, real_to_ans(y_test[:, 1])))

        # sigmas for predicted data, actually loss function values (RMSE)
        loss_systole = hist_systole.history['loss'][-1]
        loss_diastole = hist_diastole.history['loss'][-1]
        val_loss_systole = hist_systole.history['val_loss'][-1]
        val_loss_diastole = hist_diastole.history['val_loss'][-1]

        if calc_crps > 0 and i % calc_crps == 0:
            logging.info('Evaluating CRPS...')
            pred_systole = model_systole.predict(X_train, batch_size=batch_size, verbose=1)
            pred_diastole = model_diastole.predict(X_train, batch_size=batch_size, verbose=1)
            val_pred_systole = model_systole.predict(X_test, batch_size=batch_size, verbose=1)
            val_pred_diastole = model_diastole.predict(X_test, batch_size=batch_size, verbose=1)

            # CDF for train and test data (actually a step function)
            cdf_train = real_to_cdf(np.concatenate((y_train[:, 0], y_train[:, 1])))
            cdf_test = real_to_cdf(np.concatenate((y_test[:, 0], y_test[:, 1])))

            # CDF for predicted data
            cdf_pred_systole = correct_cdf(pred_systole)
            cdf_pred_diastole = correct_cdf(pred_diastole)
            cdf_val_pred_systole = correct_cdf(val_pred_systole)
            cdf_val_pred_diastole = correct_cdf(val_pred_diastole)

            # evaluate CRPS on training data
            crps_train = crps(cdf_train, np.concatenate((cdf_pred_systole, cdf_pred_diastole)))
            logging.info('CRPS(train) = {0}'.format(crps_train))

            # evaluate CRPS on test data
            crps_test = crps(cdf_test, np.concatenate((cdf_val_pred_systole, cdf_val_pred_diastole)))
            logging.info('CRPS(test) = {0}'.format(crps_test))

        logging.info('Saving weights...')
        # save weights so they can be loaded later
        model_systole.save_weights('../models/weights/weights_systole_1.hdf5', overwrite=True)
        model_diastole.save_weights('../models/weights/weights_diastole_1.hdf5', overwrite=True)

        # for best (lowest) val losses, save weights
        if val_loss_systole < min_val_loss_systole:
            min_val_loss_systole = val_loss_systole
            model_systole.save_weights('../models/weights/weights_systole_best_1.hdf5', overwrite=True)

        if val_loss_diastole < min_val_loss_diastole:
            min_val_loss_diastole = val_loss_diastole
            model_diastole.save_weights('../models/weights/weights_diastole_best_1.hdf5', overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open('./logs/val_loss.txt_1', mode='w+') as f:
            f.write(str(min_val_loss_systole))
            f.write('\n')
            f.write(str(min_val_loss_diastole))


train()
