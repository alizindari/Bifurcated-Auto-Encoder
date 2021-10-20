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




def ura_pipe(encoder_lines, nOf_filters = [32,64,100,128,256],name=None):
    F1 = nOf_filters[0]
    F2 = nOf_filters[1]
    F3 = nOf_filters[2]
    F4 = nOf_filters[3]
    F5 = nOf_filters[4]
    
    wire = encoder_lines[3]
    
    wire = UpSampling2D(size = (2,2)) (wire)
    wire = res_attention(ratio=1/2,inp= wire)
    wire = concatenate([wire, encoder_lines[2]],axis= 3)
    wire = res_attention(ratio=1/2,inp= wire)
   
           
    wire = UpSampling2D(size = (2,2)) (wire)
    # wire = Dropout(0.1)(wire)
    wire = res_attention(ratio=1/2,inp= wire)
    wire = concatenate([wire, encoder_lines[1]],axis= 3)
    wire = res_attention(ratio=1/2,inp= wire)
           
    
    wire = UpSampling2D(size = (2,2))(wire)
    # wire = Dropout(0.1)(wire)
    wire = Conv2D(64,3,activation= 'relu',padding= 'same',kernel_initializer= 'he_normal')(wire)
    wire = Conv2D(64,3,activation='relu',padding= 'same',kernel_initializer= 'he_normal')(wire)
    wire = concatenate([wire, encoder_lines[0]],axis= 3)
    wire = Conv2D(32,3,activation = 'relu',padding = 'same',kernel_initializer = 'he_normal')(wire) 

    wire = xconv2D(32,d_rate=1, inp_layer= wire)
    
    return wire


##############

def ura_encoder(nOf_filters = [32,64,100,128,256],inp_layer=None):
    F1 = nOf_filters[0]
    F2 = nOf_filters[1]
    F3 = nOf_filters[2]
    F4 = nOf_filters[3]
    F5 = nOf_filters[4]

  

    wire = Conv2D(20,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inp_layer)
    E1 = xconv2D(40,2,inp_layer=wire)

    wire = MaxPool2D((2,2),name='pool1_E') (E1)
    # wire = Dropout(0.2)(wire)
    wire = res_attention(ratio=2,inp=wire)
    E2=wire

    wire = MaxPool2D((2,2),name='pool2_E') (E2)
    # wire = Dropout(0.2)(wire)
    wire = res_attention(ratio=2,inp=wire)
    E3=wire

    wire = MaxPool2D((2,2),name='pool3_E') (E3)
    E4 = res_attention(ratio=1,inp=wire)
    return [E1,E2,E3,E4]

##############
def mask_layer(mask,inp_layer):

    l=[mask]*inp_layer.shape[3]
    mask_tens = concatenate(l,axis= 3)
    res = Multiply()([inp_layer, mask_tens])

    return res




def u_encoder(nOf_filters = [32,64,128,256,512],inp_layer=None):
    F1 = nOf_filters[0]
    F2 = nOf_filters[1]
    F3 = nOf_filters[2]
    F4 = nOf_filters[3]
    F5 = nOf_filters[4]


  

    wire = Conv2D(F1,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inp_layer)
    wire = xconv2D(F1,3,inp_layer=wire)
    #conv1_E = Conv2D(64,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_E) # new
    wire = channel_attention(wire)

    E1=wire
    wire = MaxPool2D((2,2),name='pool1_E') (wire)
    wire = Dropout(0.1)(wire)


    wire = Conv2D( F2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    wire = Conv2D(F2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    wire = Conv2D(F3,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv2_E')(wire)
    wire = corrector(wire)
    
    E2=wire
    wire = MaxPool2D((2,2),name='pool2_E') (wire)
    wire = Dropout(0.1)(wire)

    wire = Conv2D( F3,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    wire = Conv2D( F3,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    wire = Conv2D( F4,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv3_E')(wire)
    wire = corrector(wire)
    
    E3=wire
    wire = MaxPool2D((2,2),name='pool3_E') (wire)
    wire = Dropout(0.1)(wire)
    
    wire = Conv2D( F4,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire) # 128 -> 256
    wire = Conv2D( F4,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    wire = Conv2D( F5,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='conv4_E')(wire)
    wire = corrector(wire)

    E4=wire

    return [E1,E2,E3,E4]