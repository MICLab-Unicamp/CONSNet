import os,time,argparse
import nibabel as nib
import numpy as np
import siamxt
from libs import utils as futil
from scipy import ndimage


def output_prob_maps(shapes, shapes_rev, npy, models, tags,
                     name, prefix_model):

    # Reading data and normalizing to range (0,1000)
    volume_in = nib.load(name)
    data = volume_in.get_data()
    data = data.astype(np.float32)
    data = futil.convert_nifti_range(data)
    preds = []

    for count in range(len(tags[:-1])):

        # Suffixes for models, means, and stds
        suffix =  prefix_model + '_' + tags[count] + '_staple'
        suf = '_cc347_stacked' + '_' + tags[count] + '_staple'
        mean_filename = os.path.join(npy,'mean'+ suf +'.npy')
        std_filename = os.path.join(npy,'std'+ suf +'.npy')

        # Load model, mean, and std
        mean = np.load(mean_filename)
        std = np.load(std_filename)

        model_name = os.path.join(models, 'ss-unet' + suffix)
        model = futil.get_unet2_recod_bn(ch = 1, tag='test')
        model.load_weights(model_name + '.model')

        # Reshaping for each slice
        dat = np.copy(data)
        dataa = np.transpose(dat, shapes[count])

        # Normalizing and padding data
        dataa = futil.normalize(dataa, mean, std)
        dataa, offset_x, offset_y, w, h, d = futil.pad(dataa)

        # Predicting data
        dataa = dataa.reshape(w, h, d, 1)
        pred = model.predict(dataa, batch_size=8, verbose=0)
        dataa = dataa.reshape(w, h, d)
        pred = pred.reshape(w, h, d)

        # Remove padding
        pred, dataa = futil.remove_padding(pred, dataa, offset_x, offset_y)
        pred = np.transpose(pred, shapes_rev[count])
        preds.append(pred)

    return preds, volume_in.affine


def predict_cbp(pred_folder, shapes, shapes_rev, npy, models, tags,
                ts_original_data_names, prefix_model):

    start_time = time.time()
    sigma = 0.5
    ii = 0

    for name in ts_original_data_names:

        ii += 1
        root = name.split('/')[-1].split('.nii.gz')[0]
        print "[INFO] Predicting Subject", root, '{0}/{1} volumes'.format(ii, len(ts_original_data_names))

        preds, affine = output_prob_maps(shapes, shapes_rev, npy, models, tags,
                     name, prefix_model)

        # Making consensus
        consensus_pred = np.mean(preds, axis=0)
        preds.append(consensus_pred)
        preds = [pred > sigma for pred in preds]

        # Saving consensus prediction
        print '[INFO] Saving Consensus-based and Single Plane predictions'
        for tag, pred in zip(tags,preds):
            volume_name = name.split('/')[-1].split('.nii.gz')[0] + '_' + tag + '.nii.gz'
            volume_out = nib.Nifti1Image(pred.astype(np.uint8), affine=affine)
            nib.save(volume_out, os.path.join(pred_folder, volume_name))

    print("--- %s seconds ---" % (time.time() - start_time))

def post_proc(pred_folder):

    imgs = [f for f in os.listdir(pred_folder) if f.endswith('.nii.gz')]

    Bc = np.ones((3, 3, 3), dtype=bool)

    start_time = time.time()
    for img in imgs:
        img2 = img[:-7] + "_pp.nii.gz"
        print img2
        data = nib.load(os.path.join(pred_folder, img))
        affine = data.get_affine()
        data = data.get_data()
        mxt = siamxt.MaxTreeAlpha(data, Bc)
        mxt.areaOpen(mxt.node_array[3, 1:].max() - 5)
        data2 = mxt.getImage()
        new_data = data2
        nii = nib.Nifti1Image(new_data, affine)
        nib.save(nii, os.path.join(pred_folder, img2))
    print("--- %s seconds ---" % (time.time() - start_time))


def run_inference(opt):

    npy = opt.npy_path
    models = opt.model_path
    pred_folder = opt.pred_path
    futil.create_new_dir(pred_folder)
    input_data = opt.input
    input_data = futil.read_file(input_data)

    tags = ['axial', 'coronal', 'sagittal', 'consensus']
    prefix_model = '_cc347_unet2_recod_bn_stacked'

    shapes = [(2, 0, 1), (1, 0, 2), (0, 1, 2)]  # Shapes for transpose volume before prediction
    shapes_rev = [(1, 2, 0), (1, 0, 2), (0, 1, 2)]  # Shapes for transpose volume after prediction

    predict_cbp(pred_folder, shapes, shapes_rev, npy, models, tags,
                input_data, prefix_model)

    print '[INFO] Running post-processing algorithm'
    post_proc(pred_folder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-pred_path',
                        type=str,
                        default='./preds',
                        help='directory where predictions will be saved')

    parser.add_argument('-npy_path',
                        type=str,
                        default='./mean_std',
                        help='directory where means and stds are saved')

    parser.add_argument('-model_path',
                        type=str,
                        default='./models',
                        help='directory where models are saved')

    parser.add_argument('-input',
                        type=str,
                        default='./input_data.txt',
                        help='txt file containing input data filenames')

    opt = parser.parse_args()
    run_inference(opt)
    # isometric_voxels()