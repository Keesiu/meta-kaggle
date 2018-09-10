# from __future__ import true_division
import numpy as np
from numpy.random import randint
# Lasagne (& friends) imports
import theano
from lasagne import regularization
from nolearn.lasagne import BatchIterator, NeuralNet
from lasagne.objectives import categorical_crossentropy, aggregate, binary_crossentropy
from lasagne.layers import (InputLayer, DropoutLayer, DenseLayer, FeaturePoolLayer, Conv1DLayer, Conv2DLayer,
                            MaxPool1DLayer, MaxPool2DLayer, GaussianNoiseLayer, ReshapeLayer, NINLayer, 
                            DimshuffleLayer, ConcatLayer, SliceLayer, flatten)
from lasagne.updates import nesterov_momentum
from theano.tensor.nnet import sigmoid, softmax
from theano import tensor as T
from lasagne.nonlinearities import leaky_rectify, very_leaky_rectify
# Local imports
from nnet import AdjustVariable, EarlyStopping, WeightDumper, LayerFactory

from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta, lfilter, butter
from index_batch_iterator import IndexNeuralNet, IndexTrainSplit

SAMPLE_RATE = 500
CHANNELS = 32
N_EVENTS = 6

SAMPLE_SIZE = 4096
FEATURES = 6

# Experiment with larger / smaller dense layers
# 1024 on subj1 gave 0.035 vs 0.037 for 512 (40 epochs), although with more epochs 512 got down there
    

class IndexBatchIterator(BatchIterator):

    def __init__(self, source, *args, **kwargs):
        super(IndexBatchIterator, self).__init__(*args, **kwargs)
        self.set_source(source)
        self.Xbuf = np.zeros([self.batch_size, CHANNELS, SAMPLE_SIZE], np.float32) 
        self.Ybuf = np.zeros([self.batch_size, N_EVENTS], np.float32) 
    
    def set_source(self, source):
        self.source = source
        if source is None:
            self.augmented = None
        else:
            self.augmented = self.augment(source)
    
    @staticmethod
    def augment(source):
        offset = SAMPLE_SIZE-1
        augmented = np.zeros([CHANNELS, len(source)+offset], dtype=np.float32)
        augmented[:,offset:] = source.data.transpose()
        augmented[:,:offset] = augmented[:,offset][:,None][::-1]
        return augmented
    
    def transform(self, X_indices, y_indices):
        X_indices, y_indices = super(IndexBatchIterator, self).transform(X_indices, y_indices)
        [count] = X_indices.shape
        # Use preallocated space
        X = self.Xbuf[:count]
        Y = self.Ybuf[:count]
        for i, ndx in enumerate(X_indices):
            if ndx == -1:
                ndx = np.random.randint(len(self.source.events))
            augmented = self.augmented[:,ndx:ndx+SAMPLE_SIZE]
            X[i] = augmented
            if y_indices is not None:
                Y[i] = self.source.events[ndx]
        Y = None if (y_indices is None) else Y
        return X, Y
    


    
def create_net(train_source, test_source, filter0_width=9, filter1_width=7, filter2_width=5,
               batch_size=32, max_epochs=100, rate=0.04, patience=20): 
        
    learning_rate = theano.shared(np.float32(rate))
    momentum = theano.shared(np.float32(0.9))

    batch_iter_train = IndexBatchIterator(train_source, batch_size=batch_size)
    batch_iter_test  = IndexBatchIterator(test_source, batch_size=batch_size)
    LF = LayerFactory()

    maxout = 2
    dense = 1024
        
    # This model seems to work better with grasp_50 (50 Hz cutoff) than 
    # grasp_inf (no cutoff).
    # Surprisingly pad="same" or nonoverlapping pools seems to help
    layers = [
        LF(InputLayer, shape=(None, CHANNELS, SAMPLE_SIZE)), 
        #
        LF(Conv1DLayer, num_filters=FEATURES, filter_size=filter0_width, nonlinearity=None, pad="same"),
        # The sample rate is about 4x the cutoff frequency and I want to reduce the
        # data down to FEATURES size. Since I don't know the optimal reduction strategy
        # try to let the net figure it out by reducing CHANNELSx4 blocks down to 
        # FEATURESx1 blocks using a linear NINLayer.
        LF(DimshuffleLayer, pattern=(0,2,1)),
        LF(ReshapeLayer, shape=([0], -1, 16*FEATURES)),
        LF(DimshuffleLayer, pattern=(0,2,1)),        
        LF(NINLayer, num_units=FEATURES, nonlinearity=None),  
        #
        LF(Conv1DLayer, num_filters=64, filter_size=filter1_width, pad="same",
            nonlinearity=very_leaky_rectify, untie_biases=False),
        LF(MaxPool1DLayer, pool_size=3, stride=2, ignore_border=False),
        #
        LF(Conv1DLayer, num_filters=128, filter_size=filter2_width, pad="same",
            nonlinearity=None, untie_biases=True, layer_name="last_conv"),
        LF(FeaturePoolLayer, pool_size=2),
        LF(MaxPool1DLayer, pool_size=12, stride=8),  
        LF(flatten, layer_name="all_time"),
        #
        LF(SliceLayer, incoming="last_conv", indices=slice(-8,None,None)),
        LF(FeaturePoolLayer, pool_size=2),
        LF(flatten, layer_name="recent"),
        #
        LF(ConcatLayer, incomings=["all_time", "recent"]),
        #   
        LF(DropoutLayer, p=0.5),  
        #
        LF(DenseLayer, nonlinearity=None, num_units=maxout*dense),
        LF(FeaturePoolLayer, pool_size=maxout),
        LF(FeaturePoolLayer, pool_size=2),
        LF(DropoutLayer, p=0.5),
        #
        LF(DenseLayer, nonlinearity=None, num_units=maxout*dense),
        LF(FeaturePoolLayer, pool_size=maxout),
        LF(FeaturePoolLayer, pool_size=2),
        LF(DropoutLayer, p=0.5),
        #
        LF(DenseLayer, layer_name="output", num_units=N_EVENTS, nonlinearity=sigmoid)
    ]


    
    def loss(x,t):
        return aggregate(binary_crossentropy(x, t))
    
    on_epoch_finished = [AdjustVariable(learning_rate, target=0, half_life=20)]
    on_training_finished = []
    on_training_started = []
    if patience:
        earlyStopper = EarlyStopping(patience=patience)
        on_epoch_finished.append(earlyStopper)
        on_training_finished.append(earlyStopper.finished)
        on_training_started.append(earlyStopper.started)
    
        
    nnet =  IndexNeuralNet(
        y_tensor_type = T.matrix,
        train_split = IndexTrainSplit(),
        layers = layers,
        batch_iterator_train = batch_iter_train,
        batch_iterator_test = batch_iter_test,
        max_epochs = max_epochs,
        verbose=1,
        update = nesterov_momentum, 
        update_learning_rate = learning_rate,
        update_momentum = 0.9,
        objective_loss_function = loss,
        regression = True,
        on_epoch_finished = on_epoch_finished,
        on_training_started = on_training_started,
        on_training_finished = on_training_finished,
        **LF.kwargs
        )


    return nnet
    
create_net.max_freq = 50
 
# Tied for best as of 8/30 with stf7b. Local 0.98179841184535777, LB 0.97138
 