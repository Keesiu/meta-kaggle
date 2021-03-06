# Script to train specialist
from __future__ import division, print_function

import argparse
from time import time

import numpy as np

from plankton import utils
from plankton import get_net

parser = argparse.ArgumentParser()
parser.add_argument('net_name')
parser.add_argument('base_net_fname')
parser.add_argument('--X_train_train_npy', default='data/X_train_train')
parser.add_argument('--X_train_test_npy', default='data/X_train_test')
parser.add_argument('--y_train_train_npy', default='data/y_train_train')
parser.add_argument('--y_train_test_npy', default='data/y_train_test')
parser.add_argument('--hw', default=48, type=int)
parser.add_argument('--out_dir', default='models/')
args = parser.parse_args()

if __name__ == '__main__':
    print('Loading training images')
    X_train, X_test = np.load('%s_%i.npy' % (args.X_train_train_npy, args.hw)), np.load('%s_%i.npy' % (args.X_train_test_npy, args.hw))
    y_train, y_test = np.load('%s_%i.npy' % (args.y_train_train_npy, args.hw)), np.load('%s_%i.npy' % (args.y_train_test_npy, args.hw))
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    print('Loading base net from %s' % args.base_net_fname)
    base_net = utils.load_from_pickle(args.base_net_fname)

    print('Loading model definition from %s' % args.net_name)
    net = get_net(args.net_name)

    net.load_weights_from(base_net)

    t0 = time()
    print('Started training at %s' % t0)
    net.fit(X_train, y_train)
    print('Finished training. Took %i seconds' % (time() - t0))

    y_test_pred = net.predict(X_test)
    y_test_pred_proba = net.predict_proba(X_test)
    lscore = utils.multiclass_log_loss(y_test, y_test_pred_proba)
    ascore = net.score(X_test, y_test)

    print('Accuracy test score is %.4f' % ascore)
    print('Multiclass log loss test score is %.4f' % lscore)

    model_fname = utils.save_to_pickle(net, '%snet-%s-%s' % (args.out_dir, args.net_name, lscore))
    print('Model saved to %s' % model_fname)
