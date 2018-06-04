import os, time, argparse
import nibabel as nib
import numpy as np
from libs import evaluation_utils as evl
from libs import prep_utils as prep
from libs import cnn_utils as cnn
from scipy import ndimage as ndi

def clean_mask(img):
    """Clean up a brain mask by selecting largest connected region.
    """
    # get rid of islands by finding largest region
    labels, n = ndi.measurements.label(img)
    hist = np.histogram(labels.flat, bins=(n + 1), range=(-0.5, n + 0.5))[0]
    i = np.argmax(hist[1:]) + 1
    mask = (labels != i).astype(np.uint8)
    # get rid of holes by allowing only one background region
    labels, n = ndi.measurements.label(mask)
    hist = np.histogram(labels.flat, bins=(n + 1), range=(-0.5, n + 0.5))[0]
    i = np.argmax(hist[1:]) + 1
    return (labels != i).astype(np.uint8)


def output_prob_maps(opt, shapes, shapes_rev, tags,
                     name):

    # Reading data and normalizing to range (0,1000)
    volume_in = nib.load(name)
    data = volume_in.get_data()
    data = data.astype(np.float32)
    data = prep.convert_nifti_range(data)
    preds = []

    for count in range(len(tags[:-1])):

        # Suffixes for models, means, and stds
        mean_filename = os.path.join(opt.mean_filename + tags[count] +'.npy')
        std_filename = os.path.join(opt.std_filename + tags[count] +'.npy')

        # Load model, mean, and std
        mean = np.load(mean_filename)
        std = np.load(std_filename)

        model_name = opt.models + tags[count] + '.model'
        model = cnn.get_unet2_recod_bn(ch = 1, tag='test')
        model.load_weights(model_name)

        # Reshaping for each slice
        dat = np.copy(data)
        dataa = np.transpose(dat, shapes[count])

        # Normalizing and padding data
        dataa = evl.normalize(dataa, mean, std)
        dataa, offset_x, offset_y, w, h, d = evl.pad(dataa)

        # Predicting data
        dataa = dataa.reshape(w, h, d, 1)
        pred = model.predict(dataa, batch_size=8, verbose=0)
        dataa = dataa.reshape(w, h, d)
        pred = pred.reshape(w, h, d)

        # Remove padding
        pred, dataa = evl.remove_padding(pred, dataa, offset_x, offset_y)
        pred = np.transpose(pred, shapes_rev[count])
        preds.append(pred)

    return preds, volume_in.affine


def predict_cbp(opt, pred_folder, shapes, shapes_rev, tags,
                input_name):

    start_time = time.time()
    sigma = 0.5

    root = input_name.split('/')[-1].split('.nii.gz')[0]
    print("[INFO] Predicting Subject", root)

    preds, affine = output_prob_maps(opt, shapes, shapes_rev, tags, input_name)

    # Making consensus
    consensus_pred = np.mean(preds, axis=0)
    preds.append(consensus_pred)
    preds = [pred > sigma for pred in preds]

    # Saving consensus prediction
    print('[INFO] Saving Consensus-based and Single Plane predictions')
    for tag, pred in zip(tags,preds):
        volume_name = input_name.split('/')[-1].split('.nii.gz')[0] + '_' + tag + '.nii.gz'
        volume_out = nib.Nifti1Image(pred.astype(np.uint8), affine=affine)
        nib.save(volume_out, os.path.join(pred_folder, volume_name))

    print("--- %s seconds ---" % (time.time() - start_time))

def post_proc(pred_folder):

    imgs = [f for f in os.listdir(pred_folder) if f.endswith('.nii.gz')]

    start_time = time.time()
    for img in imgs:
        img2 = img[:-7] + "_pp.nii.gz"
        print(img2)
        data = nib.load(os.path.join(pred_folder, img))
        affine = data.get_affine()
        data = data.get_data()
        new_data = clean_mask(data)
        nii = nib.Nifti1Image(new_data, affine)
        nib.save(nii, os.path.join(pred_folder, img2))
    print("--- %s seconds ---" % (time.time() - start_time))


def run_inference(opt):


    pred_folder = opt.pred_path
    prep.create_new_dir(pred_folder)

    tags = ['axial', 'coronal', 'sagittal', 'consensus']
    shapes = [(2, 0, 1), (1, 0, 2), (0, 1, 2)]  # Shapes for transpose volume before prediction
    shapes_rev = [(1, 2, 0), (1, 0, 2), (0, 1, 2)]  # Shapes for transpose volume after prediction

    predict_cbp(opt, pred_folder, shapes, shapes_rev, tags,
                opt.input)

    print('[INFO] Running post-processing algorithm')
    post_proc(pred_folder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-pred_path',
                        type=str,
                        default='./preds_x',
                        help='directory where predictions will be saved')


    parser.add_argument('-model_path',
                        type=str,
                        default='./models',
                        help='directory where models are saved')

    parser.add_argument('-input',
                        type=str,
                        default='./input_data.txt',
                        help='txt file containing input data filenames')


    parser.add_argument('-models', type=str, default='./models_paper_2/3_patches_128/ss_model_',
                        help='models prefix directory')

    parser.add_argument('-mean_filename', type=str, default='./npy_paper_2/3_patches_128/mean_',
                        help='train data mean filename prefix')
    parser.add_argument('-std_filename', type=str, default='./npy_paper_2/3_patches_128/std_',
                        help='train data std filename prefix')


    opt = parser.parse_args()
    run_inference(opt)
