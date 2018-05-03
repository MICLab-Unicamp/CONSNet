import os
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate
from keras.layers import UpSampling2D, Dropout
from keras.layers.noise import GaussianNoise
from keras.layers import BatchNormalization

from sklearn.metrics import confusion_matrix
import SimpleITK as sitk

''' Save dict to a file '''
def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


''' Load dict from a file '''
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


''' Save list to a file'''
def save_file(fname, data):
    f = open(fname, 'w')
    for c in data:
        f.write('%s\n' % c)
    f.close()


''' Read list from a file'''
def read_file(filename):
    content = []
    f = open(filename, 'r')
    content = np.asarray([x.strip('\n') for x in f.readlines()])
    f.close()
    return content

''' create a new directory if it is not exist'''
def create_new_dir(dstDir):
    if not os.path.exists(dstDir):
        os.makedirs(dstDir)


''' Convert voxel intensity to range from 0 to up_bound. Default value is 1000'''
def convert_nifti_range(data, up_bound=1000):
    mn = np.float32(np.min(data))
    mx = np.float32(np.max(data))

    return (data - mn) / (mx - mn) * up_bound

''' CNN architecture '''
def get_unet2_recod_bn(ch, tag='train', patch_size=(None, None)):
    gaussian_noise_std = 0.025
    dropout = 0.5

    if tag == 'train':
        inputs = Input((patch_size[0], patch_size[1], ch))
    elif tag == 'test':
        inputs = Input((patch_size[0], patch_size[1], ch))
    else:
        print '[Wrong tag option:', tag, ']', 'Use either train ou test tags'
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

    pool3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    pool3 = BatchNormalization(axis=3)(pool3)
    pool3 = Dropout(dropout)(pool3)

    up6 = concatenate([UpSampling2D(size=(2, 2))(pool3), conv3], axis=3)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Dropout(dropout)(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv2], axis=3)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Dropout(dropout)(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv1], axis=3)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Dropout(dropout)(up8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

    conv9 = Conv2D(1, (1, 1), activation='sigmoid')(conv8)

    model = Model(inputs=[inputs], outputs=[conv9])

    return model


''' Some volume resolutions are not divided by 16 (factor of sucessive max-pooling layers from U-Net) and 
result in different in the reconstruction part due to rounding numbers. Therefore, a process of zero paddind 
was used to overcome this impairment.'''

''' Get offset values to be used in the padding '''
def get_near_offset(h, d):
    n = 16
    x = h % n  # 16 (4 layers of size 2x2) because of the number of max pooling layers
    y = d % n

    if x != 0:
        offset_x = n - x
    else:
        offset_x = 0

    if y != 0:
        offset_y = n - y
    else:
        offset_y = 0

    return offset_x, offset_y


''' Zero-padding function '''
def zero_pad(a, offset_x, offset_y):
    if offset_x == offset_y == 0:
        return a
    else:
        result = np.zeros((a.shape[0] + offset_x, a.shape[1] + offset_y))
        result[:a.shape[0], :a.shape[1]] = a

    return result


''' Remove the zero-padding from the volume '''
def remove_padding(pred, data, offset_x, offset_y):
    # removing zero-padding to compare with the mask
    if offset_x != 0:
        pred = pred[:, :-offset_x, :]
        data = data[:, :-offset_x, :]
    else:
        pred = pred
        data = data

    if offset_y != 0:
        pred = pred[:, :, :-offset_y]
        data = data[:, :, :-offset_y]
    else:
        pred = pred
        data = data

    return pred, data


''' Normalize volume using mean an standard deviation '''
def normalize(data, mean, std):
    # Normalizing data
    data -= mean
    data /= std

    return data


''' Zero-padding the volume '''
def pad(data):
    w, h, d = data.shape

    # Padding data
    offset_x, offset_y = get_near_offset(h, d)
    data = np.asarray([zero_pad(ii, offset_x, offset_y) for ii in data])
    w, h, d = data.shape

    return data, offset_x, offset_y, w, h, d


''' Compute Dice, sensitiviyt, and specificity '''
def get_dice_sensitivity_specificity(unet_segmentation, ref_segmentation):
    ref_segmentation = np.ravel(np.array(ref_segmentation))
    unet_segmentation = np.ravel(np.array(unet_segmentation))

    cMat = confusion_matrix(ref_segmentation, unet_segmentation)

    dice = float(2 * cMat[1][1]) / (2 * cMat[1][1] + cMat[0][1] + cMat[1][0])
    sensitivity = float(cMat[1][1]) / (cMat[1][1] + cMat[1][0])
    specificity = float(cMat[0][0]) / (cMat[0][0] + cMat[0][1])

    return dice, sensitivity, specificity


''' Compute Hausdorff distance and symmetric mean distance '''
def get_overlap_surface_measures(pred_path, mask_path):
    pred = sitk.ReadImage(pred_path)
    mask = sitk.ReadImage(mask_path)

    pred = pred > 0
    mask = mask > 0

    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.SetGlobalDefaultCoordinateTolerance(1e+3)
    hausdorff_distance_filter.SetGlobalDefaultDirectionTolerance(1e+3)
    hausdorff_distance_filter.Execute(pred, mask)

    hauss_dist = hausdorff_distance_filter.GetHausdorffDistance()
    mean_dist = hausdorff_distance_filter.GetAverageHausdorffDistance()

    return hauss_dist, mean_dist




