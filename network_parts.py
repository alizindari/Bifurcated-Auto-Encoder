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


#############
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

#########
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

##########
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
#########
def u_pipe(encoder_lines, nOf_filters = [32,64,100,128,256]):
    F1 = nOf_filters[0]
    F2 = nOf_filters[1]
    F3 = nOf_filters[2]
    F4 = nOf_filters[3]
    F5 = nOf_filters[4]
    
    wire  = UpSampling2D(size = (2,2)) (encoder_lines[3])
    # wire = Dropout(0.2)(wire)
    
    wire = Conv2D(F5,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    wire = Conv2D(F4,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    wire = Conv2D( F4,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    
    
    wire = concatenate([wire,encoder_lines[2]],axis= 3)
    wire = Conv2D( F4,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire) # f3 -> f4
    wire = Conv2D( F3,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire) # f2 -> f3
    #conv2_Dt = Conv2D( F2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_Dt) # new
    
    wire   = UpSampling2D(size = (2,2),) (wire)
    wire = Dropout(0.1)(wire)
    
    wire = Conv2D( F3,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal') (wire)
    wire = Conv2D( F2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal') (wire)
    wire = Conv2D( F2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal') (wire)


    wire   = concatenate([wire,encoder_lines[1]],axis= 3)
    wire = Conv2D( F3,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    wire = Conv2D( F2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
   # conv3_Dt = Conv2D( F2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_Dt) #new 
    
    wire  = UpSampling2D(size = (2,2)) (wire)
    wire = Dropout(0.1)(wire)
    
    wire = Conv2D(F3,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    wire = Conv2D(F2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    wire = Conv2D(F2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    
    
    wire   = concatenate([wire,encoder_lines[0]],axis= 3)

    
    wire = Conv2D(F2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire) # f2 -> f3
    wire = Conv2D(F2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(wire)
    #convf_Dt = Conv2D(F2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convf_Dt) #new
    wire = xconv2D(F1,d_rate=1, inp_layer= wire)
    
    return wire


##############

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