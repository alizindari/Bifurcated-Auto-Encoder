import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers




def xconv2D (nOf_filters,d_rate=1,name=None,inp_layer=None):
    nOf_filters = int(nOf_filters/2)
    d_rate = (d_rate,d_rate)
    t_layer1 =  Conv2D(nOf_filters,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inp_layer)
    t_layer1 =  Conv2D(nOf_filters,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(t_layer1)
    
    t_layer2 =  Conv2D(nOf_filters,7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dilation_rate=d_rate)(inp_layer)
    t_layer2 =  Conv2D(nOf_filters,5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dilation_rate=d_rate)(t_layer2)
    
    t_layer3 =  Conv2D(nOf_filters,9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dilation_rate=d_rate)(inp_layer)
    t_layer3 =  Conv2D(nOf_filters,7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dilation_rate=d_rate)(t_layer3)
    
    t_layer4 =  MaxPooling2D((7,7),strides=(1,1),padding = 'same')(inp_layer)
    
    t_cc = concatenate([inp_layer,t_layer1,t_layer2,t_layer3,t_layer4],axis=3)
    drop = Dropout(0.07)(t_cc) #0.07 -> 0
    
    res =  Conv2D(nOf_filters*3,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dilation_rate=d_rate)(drop)
    res =  Conv2D(nOf_filters*2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',dilation_rate=d_rate,name=name)(res)
    
    return res