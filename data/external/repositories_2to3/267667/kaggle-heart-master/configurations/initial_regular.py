from .default import *

import theano.tensor as T
from layers import MuLogSigmaErfLayer
import objectives

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPoolLayer
from lasagne.layers import InputLayer
from lasagne.layers import reshape
from lasagne.layers import DenseLayer
from postprocess import upsample_segmentation
from volume_estimation_layers import GaussianApproximationVolumeLayer

validate_every = 100
validate_train_set = False
save_every = 100
restart_from_save = False

batches_per_chunk = 1

batch_size = 2
sunny_batch_size = 4
num_chunks_train = 8400

image_size = 64

learning_rate_schedule = {
    0:   0.0003,
    250:  0.00003,
    5000: 0.000003,
    8000: 0.0000003,
}

data_sizes = {
    "sliced:data:ax": (batch_size, 30, 15, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:shape": (batch_size, 2,),
    "sunny": (sunny_batch_size, 1, image_size, image_size)
    # TBC with the metadata
}


data_sizes = {
    "sliced:data:ax": (batch_size, 30, 15, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:shape": (batch_size, 2,),
    "sunny": (sunny_batch_size, 1, image_size, image_size)
    # TBC with the metadata
}

def build_model():
    #################
    # Regular model #
    #################
    l0 = InputLayer(data_sizes["sliced:data:ax"])
    l0r = reshape(l0, (-1, 1, ) + data_sizes["sliced:data:ax"][-2:])

    # first do the segmentation steps
    l1a = ConvLayer(l0r, num_filters=32, filter_size=(3, 3),
                    pad='same',
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1),
                    )
    l1b = ConvLayer(l1a, num_filters=32, filter_size=(3, 3),
                    pad='same',
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1),
                    )
    l1c = ConvLayer(l1b, num_filters=64, filter_size=(3, 3),
                    pad='same',
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1),
                    )
    l1f = ConvLayer(l1c, num_filters=1, filter_size=(3, 3),
                    pad='same',
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1),
                    nonlinearity=lasagne.nonlinearities.sigmoid)

    l_1r = reshape(l1f, data_sizes["sliced:data:ax"])

    l_d3 = lasagne.layers.DenseLayer(l_1r,
                              num_units=2)
    l_systole = MuLogSigmaErfLayer(l_d3)

    l_d3b = lasagne.layers.DenseLayer(l_1r,
                              num_units=2)
    l_diastole = MuLogSigmaErfLayer(l_d3b)

    return {
        "inputs":{
            "sliced:data:ax": l0,
        },
        "outputs":{
            "systole": l_systole,
            "diastole": l_diastole
        }
    }


def build_objective(l_ins, l_outs):
    return objectives.KaggleObjective(l_outs)
