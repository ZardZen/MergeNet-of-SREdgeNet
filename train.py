import numpy as np
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import argparse
from MergeNet import merge
import os
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def mae(hr, sr):
    hr, sr = _crop_hr_in_training(hr, sr)
    return mean_absolute_error(hr, sr)

def psnr(hr, sr):
    hr, sr = _crop_hr_in_training(hr, sr)
    return tf.image.psnr(hr, sr, max_val=255)

def _crop_hr_in_training(hr, sr):
    """
    Remove margin of size scale*2 from hr in training phase.
    The margin is computed from size difference of hr and sr
    so that no explicit scale parameter is needed. This is only
    needed for WDSR models.

    """
    margin = (tf.shape(hr)[1] - tf.shape(sr)[1]) // 2
    # crop only if margin > 0
    hr_crop = tf.cond(tf.equal(margin, 0),
                      lambda: hr,
                      lambda: hr[:, margin:-margin, margin:-margin, :])
    hr = K.in_train_phase(hr_crop, hr)
    hr.uses_learning_phase = True
    return hr, sr
def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description='Keras Training')
    # ========= paths for training
    ap.add_argument("-npath", "--npy_path", default="data/", required=False,
                    help="path to npy. files to train")
    ap.add_argument("-mpath", "--model_path", default="model_save/", required=False,
                    help="path to save the output model")
    ap.add_argument("-lpath", "--log_path", default="log/", required=False,
                    help="path to save the 'log' files")
    ap.add_argument("-name","--model_name", default="MergeNet.h5", required=False,
                    help="output of model name")
    # ========= parameters for training
    ap.add_argument("-p", "--pretrain", default=0, required=False, type=int,
                    help="load pre-train model or not")

    ap.add_argument('-bs', '--batch_size', default=2, type=int,
                    help='batch size')
    ap.add_argument('-ep', '--epoch', default=5, type=int,
                    help='epoch')
    ap.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    args = vars(ap.parse_args())
    return args


def train(args):
    scale = 1
    X_train = np.load(args["npy_path"] + 'sr.npy')
    X_mask = np.load(args["npy_path"] + 'mask.npy')
    #X_val = np.load(args["npy_path"] + 'X_val.npy')
    y_train = np.load(args["npy_path"] + 'hr.npy')
    #y_val = np.load(args["npy_path"] + 'y_val.npy')
    if args["pretrain"]:
        model = load_model(args["model_path"] + args["model_name"],
                       custom_objects={'loss': mae, 'psnr': psnr})
    else:
        model = merge(scale, num_filters=128, num_res_blocks=16, res_block_scaling=None, tanh_activation=False)

    model.summary()
    lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, min_lr=1e-5)
    checkpointer = ModelCheckpoint(args["model_path"] + args["model_name"], verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=args["log_path"])
    callback_list = [lr_decay, checkpointer, tensorboard]

    #optimizer = SGD(lr=1e-5, momentum=args["momentum"], nesterov=False)
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    #parallel_model = multi_gpu_model(model, gpus=1)
    #parallel_model.compile(loss=mae,metrics=psnr,optimizer=optimizer)
    model.compile(loss=mae,metrics=[psnr],optimizer=optimizer)

#     EDSR = model.fit(X_train, y_train,validation_data=(X_val, y_val),
#                                 batch_size=args["batch_size"], epochs=args["epoch"],
#                                 callbacks=callback_list, verbose=1)

    #EDSR = parallel_model.fit(X_train, y_train,,validation_split=0.2,batch_size=args["batch_size"], epochs=args["epoch"], callbacks=callback_list, verbose=1)
    MergeNet = model.fit([X_train,X_mask], y_train,validation_split=0.2,batch_size=args["batch_size"], epochs=args["epoch"], callbacks=callback_list, verbose=1)


if __name__ == "__main__":
    args = args_parse()
    train(args)
