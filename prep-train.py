from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, os, time
import numpy as np
from libs import prep_utils as prep
from libs import cnn_utils as cnn
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import RMSprop



def patches_creation(opt, patch_size, max_patches, tag):

    # Reading data
    tr_original_data_names = prep.read_file(opt.tr_original_filename)
    tr_consensus_data_names = prep.read_file(opt.tr_consensus_filename)
    npy = os.path.join(opt.npy, str(opt.max_patches) + '_patches_' + str(opt.patch_size_row))
    imgs_train = os.path.join(npy, tag, opt.imgs_train)
    imgs_masks_train = os.path.join(npy, tag, opt.imgs_masks_train)
    total_shape_filename = os.path.join(npy, opt.total_shape_filename + tag)
    prep.create_new_dir(npy)
    prep.create_new_dir(os.path.join(npy, tag))

    # Creating patches
    total_shape_data = prep.iterate_volume(tr_original_data_names, tr_consensus_data_names, patch_size
                     ,max_patches, tag, imgs_train, imgs_masks_train)

    np.save(total_shape_filename, total_shape_data)

def train(opt, patch_size, tag):

    # Reading data
    npy = os.path.join(opt.npy, str(opt.max_patches) + '_patches_' + str(opt.patch_size_row))
    imgs_train = os.path.join(npy, tag)
    imgs_masks_train = os.path.join(npy, tag)
    mean_filename = os.path.join(npy, opt.mean_filename + tag +'.npy')
    std_filename = os.path.join(npy, opt.std_filename + tag +'.npy')
    models = os.path.join(opt.models, str(opt.max_patches) + '_patches_' + str(patch_size[0]))
    prep.create_new_dir(models)
    model_name = os.path.join(models,'ss_model_' + tag)

    total_shape = tuple(np.load(os.path.join(npy, opt.total_shape_filename + tag + '.npy'))) # total number of patches
    imgs, imgs_mask = prep.load_and_conc_patches(imgs_train,imgs_masks_train, total_shape)

    w,h,d = imgs.shape
    imgs = imgs.reshape(w,h,d,1)
    imgs_mask = imgs_mask.reshape(w,h,d,1)


    mean = np.mean(imgs)
    std = np.std(imgs)

    np.save(mean_filename, mean)
    np.save(std_filename, std)

    imgs -= mean
    imgs /= std

    # Early stopping callback to shut down training after 10 epochs with no improvement
    earlyStopping = EarlyStopping(monitor='val_loss',
                                   patience= opt.patience,
                                   verbose=1, mode='min')

    checkpoint = ModelCheckpoint(model_name + '.model', monitor='val_loss', save_best_only=True, verbose=0)
    lr_decay = LearningRateScheduler(cnn.ExpDecay(opt.lr, opt.decay).scheduler)
    optimizer= RMSprop()

    cnn_model = cnn.get_unet2_recod_bn(imgs.shape[-1], 'train', patch_size)
    cnn_model.compile(optimizer = optimizer,loss=[cnn.dice_coef_loss], metrics=[cnn.dice_coef])

    hist = cnn_model.fit(imgs,
                 imgs_mask,
                 batch_size=opt.batch_size,
                 epochs=opt.nepochs,
                 verbose=1,
                 validation_split= opt.validation_split,
                 shuffle=True,callbacks=[checkpoint, earlyStopping, lr_decay])

    prep.save_obj(hist.history, model_name + '.history')

    print ('[INFO] CNN training done!')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('-nepochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate of the optimizer')
    parser.add_argument('-decay', type=float, default=0.995, help='learning rate of the optimizer')
    parser.add_argument('-validation_split', type=float, default=0.1, help='validation split for training')
    parser.add_argument('-patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('-patch_size_row', type=int, default='128', help='patch width')
    parser.add_argument('-patch_size_col', type=int, default='128', help='patch height')
    parser.add_argument('-max_patches', type=int, default=3, help='number of patches per slice')



    parser.add_argument('-npy', type=str, default='./npy_paper_2x',
                        help='processed data directory')
    parser.add_argument('-models', type=str, default='./models_paper_2x',
                        help='models directory')

    parser.add_argument('-imgs_train', type=str, default='imgs_train',
                        help='train data directory name')
    parser.add_argument('-imgs_masks_train', type=str, default='imgs_masks_train',
                        help='train mask data directory name')
    parser.add_argument('-total_shape_filename', type=str, default='total_shape_data_',
                        help='train data total shape filename')
    parser.add_argument('-mean_filename', type=str, default='mean_',
                        help='train data mean filename')
    parser.add_argument('-std_filename', type=str, default='std_',
                        help='train data std filename')
    parser.add_argument('-tr_original_filename', type=str, default='./txt_paper_2/original_train_staple.txt',
                        help='train data filenames')
    parser.add_argument('-tr_consensus_filename', type=str, default='./txt_paper_2/original_train_masks_staple.txt',
                        help='train mask data filenames')


    opt = parser.parse_args()

    # Parameters used for the patch creation
    patch_size = opt.patch_size_row, opt.patch_size_col

    start_time = time.time()

    tags = ['axial', 'coronal', 'sagittal']

    for tt in tags:
        print('[INFO] Image plane', tt)

        print('[INFO] Running patches creation')
        patches_creation(opt, patch_size, opt.max_patches, tt)

        print('[INFO] Running cnn training')
        cnn.limit_mem()
        train(opt, patch_size, tt)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()






