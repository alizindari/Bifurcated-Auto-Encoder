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




def channel_attention(input_feature):

    channel = input_feature.shape[3]
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,channel,1))(avg_pool)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,channel,1))(max_pool)
    
    concat = concatenate([avg_pool,max_pool],axis=1)
    print("<<<<<concat shape : >>>>>>>>",concat.shape)
    conv = Conv2D(8, (2,1), activation='relu',kernel_initializer='he_normal')(concat)
    print("<<<<<conv1 shape : >>>>>>>>",conv.shape)
    conv = Conv2D(1, 1, activation='relu',kernel_initializer='he_normal')(conv)
    print("<<<<<conv2 shape : >>>>>>>>",conv.shape)
    conv = Reshape((1,1,channel))(conv)
    print("<<<<<final shape : >>>>>>>>",conv.shape)
    d1 = Dense(int(channel*1.4),activation='relu')(conv)
    drop = Dropout(0.1)(d1)
    d1 = Dense(int(channel*1),activation='sigmoid')(drop)
    
    return multiply([input_feature, d1])


############
def res_attention(ratio=2,inp=None):
    inp = Conv2D(int(inp.shape[3]*ratio),3,activation = 'relu',padding = 'same',kernel_initializer = 'he_normal')(inp) 

    drop = Dropout(0.1)(inp)

    wire = Conv2D(int(inp.shape[3]*1.5), 3,padding='same',activation='relu',kernel_initializer='he_normal')(drop)
    box1 = Conv2D(int(inp.shape[3]), 3,padding='same',kernel_initializer='he_normal',activation = 'relu')(wire)

    wire = Conv2D(int(inp.shape[3]*1.5), 5,padding='same',dilation_rate = 2,activation='relu',kernel_initializer='he_normal')(drop)
    box2 = Conv2D(int(inp.shape[3]), 7,padding='same',dilation_rate = 2,kernel_initializer='he_normal',activation = 'relu')(wire)

    wire = concatenate([box1,box2],axis=3)
    wire  = channel_attention(wire)
    wire = Conv2D(int(inp.shape[3]),3,padding = 'same',kernel_initializer = 'he_normal')(wire) 
    
    
    final = Add()([inp,wire])
    
    return final

############

def corrector(inp=None):
    wire  = channel_attention(inp)
    wire = spatial_attention(wire)  #<<<<< A >>>>>
    wire = Conv2D(int(inp.shape[3]),3,activation = 'relu',padding = 'same',kernel_initializer = 'he_normal')(wire) 
    final = Add()([inp,wire])
    return final

#########
def spatial_attention(inp):
    
    avg_pool = K.mean(inp,axis=3, keepdims=True)
    max_pool = K.max(inp, axis=3, keepdims=True)
    print('max pool',max_pool.shape)
    print('avg pool',avg_pool.shape)
    concat = Concatenate(axis=3)([avg_pool, max_pool])  
    print('concat',concat.shape)
    conv = Conv2D(8,3,activation='relu',padding='same',kernel_initializer='he_normal')(concat)
    conv = Conv2D(4,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv)
    final = Conv2D(1,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv)
    print('final ',final.shape)
    return mask_layer(final,inp) 