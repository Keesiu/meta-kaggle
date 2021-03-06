import glob
from scipy import ndimage
from scipy import misc
import numpy as np
import os
import matplotlib.pyplot as plot

error = [0.754204,0.694717,0.659362,0.620961,0.596266,0.573457,0.551741,0.541843,0.522941,0.518935,0.528072,0.500530,0.478019,0.495663,0.487354,0.470041,0.461732,0.447696,0.441936,0.427767,0.445776,0.444452,0.415387,0.409461,0.417406,0.411348,0.414923,0.415718,0.391784,0.385130,0.391022,0.377648,0.376225,0.383872,0.367883,0.382084,0.372716,0.371756,0.369273,0.359739,0.356429,0.343253,0.362851,0.348815,0.35268]
plot.plot(list(range(1,46)), error , 'b--')
plot.ylabel('Train Error')
plot.xlabel('Epoch Round')
plot.title('Round vs Train Error')
plot.savefig('error.png')
plot.show()