import numpy as np
from sklearn.metrics import confusion_matrix
import SimpleITK as sitk 

''' Some volume resolutions are not divided by 16 (factor of sucessive max-pooling layers from U-Net) and 
result in different in the reconstruction part due to rounding numbers. Therefore, a process of zero paddind 
was used to overcome this impairment.'''


''' Get offset values to be used in the padding '''
def get_near_offset(h,d):

    n = 16
    x = h % n # 16 (4 layers of size 2x2) because of the number of max pooling layers
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
    
    if offset_x == offset_y ==0:
        return a
    else:
        result = np.zeros((a.shape[0] + offset_x,a.shape[1] + offset_y))
        result[:a.shape[0],:a.shape[1]] = a
    
    return result

''' Remove the zero-padding from the volume '''
def remove_padding(pred,data,offset_x,offset_y):
    
    # removing zero-padding to compare with the mask 
    if offset_x != 0:
        pred = pred[:,:-offset_x,:]
        data = data[:,:-offset_x,:]
    else:
        pred = pred
        data = data

    if offset_y != 0:
        pred = pred[:,:,:-offset_y]
        data = data[:,:,:-offset_y]
    else:
        pred = pred
        data = data

    return pred,data

''' Normalize volume using mean an standard deviation '''
def normalize(data,mean,std):
    
    # Normalizing data
    data -= mean
    data /= std
    
    return data

''' Zero-padding the volume '''
def pad(data):
    
    w,h,d = data.shape
    
    # Padding data
    offset_x , offset_y = get_near_offset(h,d)
    data = np.asarray([zero_pad(ii, offset_x, offset_y) for ii in data])
    w,h,d = data.shape
    
    return data, offset_x , offset_y, w,h,d

''' Compute Dice, sensitiviyt, and specificity '''
def get_dice_sensitivity_specificity(unet_segmentation,ref_segmentation):

    ref_segmentation =  np.ravel(np.array(ref_segmentation))
    unet_segmentation = np.ravel(np.array(unet_segmentation))
    
    cMat = confusion_matrix(ref_segmentation, unet_segmentation)

    dice = float(2*cMat[1][1])/(2*cMat[1][1] + cMat[0][1] + cMat[1][0])
    sensitivity = float(cMat[1][1])/(cMat[1][1] + cMat[1][0])
    specificity = float(cMat[0][0])/(cMat[0][0] + cMat[0][1])

    return dice,sensitivity,specificity

''' Compute Hausdorff distance and symmetric mean distance '''
def get_overlap_surface_measures(pred_path,mask_path):

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

''' Plot 3D volume in a mosaic '''
def mosaic(f,N):
    d,h,w = f.shape
    nLines = int(np.ceil(float(d)/N))
    nCells = int(nLines*N)
        
    fullf = np.resize(f, (nCells,h,w))
    fullf[d:nCells,:,:] = 0        
        
    Y,X = np.indices((nLines*h,N*w))
    Pts = np.array([
                   (np.floor(Y/h)*N + np.floor(X/w)).ravel(),
                   np.mod(Y,h).ravel(),
                   np.mod(X,w).ravel() ]).astype(int).reshape((3,int(nLines*h),int(N*w)))
    g = fullf[Pts[0],Pts[1],Pts[2]]
    return g

