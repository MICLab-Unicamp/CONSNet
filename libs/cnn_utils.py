from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate
from keras.layers import UpSampling2D, Dropout
from keras.layers.noise import GaussianNoise
from keras.layers import BatchNormalization
import numpy as np


smooth = 1. #CNN dice coefficient smooth

''' Limit memory for CNN training in TF session'''
def limit_mem():
    K.tf.Session.close
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

''' Exponential decay for the learning rate'''
class ExpDecay:
    def __init__(self, initial_lr, decay):
        self.initial_lr = initial_lr
        self.decay = decay
    
    def scheduler(self, epoch):
        print ("Current lr: ", self.initial_lr * np.exp(-self.decay*epoch))
        return self.initial_lr * np.exp(-self.decay*epoch)

''' Metric used for CNN training'''
def dice_coef(y_true, y_pred):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


''' Loss function'''
def dice_coef_loss(y_true, y_pred):
    
    return -dice_coef(y_true, y_pred)

''' U-NET RECOD W/ BatchNorm (kernel size is fixed to 3,3)'''
def get_unet2_recod_bn(ch, tag = 'train', patch_size = (None,None)):
    
    gaussian_noise_std = 0.025
    dropout = 0.5
    
    if tag == 'train':
        inputs = Input((patch_size[0], patch_size[1], ch))
    elif tag == 'test':
        inputs = Input((patch_size[0], patch_size[1], ch))
    else:
        print '[Wrong tag option:', tag,']', 'Use either train ou test tags'
        return
        
    inputs_weights = GaussianNoise(gaussian_noise_std)(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs_weights)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = GaussianNoise(gaussian_noise_std)(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = GaussianNoise(gaussian_noise_std)(pool2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)
    
    pool3  = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    pool3 = BatchNormalization(axis=3)(pool3)
    pool3 = Dropout(dropout)(pool3)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(pool3), conv3],axis=3)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Dropout(dropout)(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv2],axis=3)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Dropout(dropout)(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv1],axis=3)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Dropout(dropout)(up8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

    conv9 = Conv2D(1, (1, 1), activation='sigmoid')(conv8)

    model = Model(inputs=[inputs], outputs=[conv9])

    return model