import glob
import os
import nibabel as nib
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import pickle

''' Save dict to a file '''
def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

''' Load dict from a file '''
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

''' Save list to a file'''
def save_file(fname,data):
    
    f = open(fname,'w')
    for c in data:
        f.write('%s\n' % c)
    f.close()

''' Read list from a file'''
def read_file(filename):
    f = open(filename,'r')
    content = np.asarray([x.strip('\n') for x in f.readlines()])
    f.close()
    return content

''' create a new directory if it is not exist'''
def create_new_dir(dstDir):
    
    if not os.path.exists(dstDir):
        os.makedirs(dstDir)
        
''' Remove volumes with manual segmentations from train list filenames'''
def remove_manual_segs(manual_data_names,original_data_names):
    
    original_path = [name.split('/C')[0] for name in original_data_names]
    indexes = []
    
    manual = [name.split('/')[-1] for name in manual_data_names]
    original = [name.split('/')[-1] for name in original_data_names]
       
    for name in manual:
        name = name.split('_man')[0] + '.nii.gz'

        if name in original:
            indexes.append(original.index(name))
            original.remove(name) 

    [original_path.remove(original_path[idx]) for idx in indexes]
    original = [os.path.join(original_path[i],original[i]) for i in range(len(original))]

    return original
        
''' Convert voxel intensity to range from 0 to up_bound. Default value is 1000'''
def convert_nifti_range(data, up_bound=1000):
    
    mn = np.float32(np.min(data))
    mx = np.float32(np.max(data))


    return (data - mn)/(mx-mn)*up_bound

''' Get interval where data is non-zero in the volume'''
def get_min_max_brain_interval(data, tag):
    
    miN = maX = np.asarray([])
    check = aux = i = 0
    
    if tag == 'sagittal':
        rangE = data.shape[0]
    elif tag == 'coronal':
        rangE = data.shape[1]
    elif tag == 'axial':
        rangE = data.shape[2]
    else:
        print "[Error] You need to specify a tag for the analysis, th options are: sagittal,coronal or axial!"
        return
    
    for Hc in range(rangE):
        
        if tag == 'sagittal':
            slicE =  data[Hc,:,:]
        if tag == 'coronal':
            slicE =  data[:,Hc,:]
        if tag == 'axial':
            slicE =  data[:,:,Hc]

        if slicE.sum() == 0:
            if aux !=0:
                maX = aux - 1
            i+=1
            
        elif Hc == (rangE-1) and slicE.sum() != 0: # condition when last slice contain data (added later)
            maX = Hc
            
        else:
            if aux == 0:
                miN = i
                aux = i
            aux +=1
            
        
    return miN, maX

''' Extract patches from a volume'''
def extract_patches(tr_original,tr_consensus,patch_size,max_patches, tag):

    data = nib.load(tr_original).get_data()
    data = data.astype(np.float32)
    data = convert_nifti_range(data)

    consensus_data = nib.load(tr_consensus).get_data() >= 0.5 # average threshold
    consensus_data = consensus_data.astype(np.uint8)

    interval = (get_min_max_brain_interval(consensus_data,tag))

    total = (interval[1] - interval[0])*max_patches
    r_begin = interval[0]
    r_end = interval[1]

    train_cut_patches = np.ndarray((total,patch_size[0], patch_size[1]), dtype=np.float32)
    train_consensus_patches = np.ndarray((total,patch_size[0], patch_size[1]), dtype=np.uint8)

    i = 0
    random_value = np.random.randint(100) #random number from 0 to 100

    for Hc in range(r_begin, r_end):

        if tag == 'sagittal':
            cut = data[Hc,:,:]
            consensus = consensus_data[Hc,:,:]
        elif tag == 'coronal':
            cut = data[:,Hc,:]
            consensus = consensus_data[:,Hc,:]
        elif tag == 'axial':
            cut = data[:,:,Hc]
            consensus = consensus_data[:,:,Hc]
        else:
            print ("[Error] You need to specify a tag for the analysis, th options are: sagittal,coronal or axial!")
            return

        cut_patches = extract_patches_2d(cut, patch_size, max_patches, random_state = random_value)
        consensus_patches = extract_patches_2d(consensus, patch_size, max_patches, random_state = random_value)

        train_cut_patches[i*max_patches:max_patches*(i+1)] = cut_patches
        train_consensus_patches[i*max_patches:max_patches*(i+1)] = consensus_patches

        i+=1


    return train_cut_patches,train_consensus_patches



''' Iterate along the volumes provided in the filename list to extract patches from each one'''
def iterate_volume(tr_original,tr_consensus,patch_size,max_patches,tag
                   ,imgs_train,imgs_masks_train):
    
    total_shape = [0,patch_size[0],patch_size[1]]
       
    for i in range(tr_original.size):
        train_patches,train_consensus_patches = extract_patches(tr_original[i],tr_consensus[i],
                                                             patch_size,max_patches,tag)

        np.save(imgs_train + '_{num:0{width}}'.format(num=i, width=3) +'.npy'
                , train_patches.astype(np.float32))
        np.save(imgs_masks_train + '_{num:0{width}}'.format(num=i, width=3) +'.npy'
                , train_consensus_patches.astype(np.uint8))
        total_shape[0] = total_shape[0] + train_patches.shape[0]


    return tuple(total_shape)

''' Load patches numpy files and concatenate them to feed the CNN'''
def load_and_conc_patches(imgs_train,imgs_masks_train, total_shape):
    
    imgs = np.sort(glob.glob(os.path.join(imgs_train, 'imgs_train*')))
    masks = np.sort(glob.glob(os.path.join(imgs_masks_train, 'imgs_masks_train*')))

    imgs_tr = np.ndarray(total_shape, dtype=np.float32)
    imgs_mask_tr = np.ndarray(total_shape, dtype=np.uint8)

    aux = 0
    for i in range(imgs.size):
        
        im = np.load(imgs[i])
        mask = np.load(masks[i])

        if im.shape != mask.shape:
            print "mismatch shapes"
            return
        
        imgs_tr[aux:aux + im.shape[0]] = im
        imgs_mask_tr[aux:aux + mask.shape[0]] = mask
        
        aux = aux + im.shape[0]


    return imgs_tr, imgs_mask_tr
