
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import cv2 
import tqdm
from tensorflow.keras.utils import plot_model
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K


def find_thresh( metric, y_true, y_pred,accu=3):
    best = (0,0)
    for i in range(1, 20):
        th = i/20
        pred = np.where(y_pred > th,1,0).astype(np.float32)
        m = np.round(metric(y_true, pred).numpy(),accu)
        
        if best[0] < m:
            best = (m, th)
        print(th,'    ',m)
    return best[1]



def dice_score(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     intersection = K.abs(y_true * y_pred)
#     intersection = K.sum(K.square(intersection),1)
#     intersection = K.sum(intersection,1)
    
#     s_true = K.sum(K.square(y_true),1)
#     s_true = K.sum(s_true,1)

#     s_pred = K.sum(K.square(y_pred),1)
#     s_pred = K.sum(s_pred,1)
    
#     return K.mean((2. * intersection + smooth) / (s_true + s_pred + smooth)    )

    
def dice_loss(y_true, y_pred, smooth=1e-6):

#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     answer = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1-dice_score(y_true,y_pred)
###################################################

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f1_score(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    f1_l = []        

    for i in range(y_true.shape[1]):
        cf= tf.math.confusion_matrix(y_true_f, y_pred_f)
    
        if (cf.shape[0] == 2):
            TP = cf[1,1]
            TN = cf[0,0]
            FP = cf[0,1]
            FN = cf[1,0]
            f1_l        .append (2*TP/(2*TP+FP+FN))
                             
        else:
            f1_l.append (1)
            
    f1_l = tf.constant(np.array(f1_l))
    return K.mean(f1_l )
   
###################################################

def metricss(y_true,y_pred):
 
   
    iou_l       = []    
    f1_l        = []        
    ppv_l       = []       
    sensivity_l = []
    spec_l      = [] 
    cfs_l       = []

    for i in tqdm.trange(y_true.shape[0]):
        cf= confusion_matrix(y_true[i].reshape(-1).astype(np.bool), y_pred[i].reshape(-1))
        cfs_l.append(cf)
        if (cf.shape[0] == 2):
            TP = cf[1,1]
            TN = cf[0,0]
            FP = cf[0,1]
            FN = cf[1,0]
            
            iou_l       .append (TP/(TP+FP+FN))
            f1_l        .append (2*TP/(2*TP+FP+FN))
            ppv_l       .append (TP/(TP+FP+1e-8))
            sensivity_l .append (TP/(TP+FN+1e-8))
            spec_l      .append (TN/(FP+TN))
                            
         
        else:
            iou_l       .append (1)
            f1_l        .append (1)
            ppv_l       .append (1)
            sensivity_l .append (1)
            spec_l      .append (1)
            

    
        
    iou       = np.mean(iou_l      )          
    f1        = np.mean(f1_l       )     
    ppv       = np.mean(ppv_l      )    
    sensivity = np.mean(sensivity_l) 
    spec      = np.mean(spec_l     )    
    
    
    print('iou       : ',iou)
    print('f1_score  : ',f1)
    print('ppv       : ',ppv)
    print('sensivity : ',sensivity)
    print('specifity : ',spec)
    return cfs_l