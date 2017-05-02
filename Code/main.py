from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models import ResNET
from utils import *
import os

import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_integer("n_epochs", 500, "Epoch to train [500]")
flags.DEFINE_integer("train_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("val_size", 256, "The size of batch images [128]")
flags.DEFINE_integer("test_size", 512, "The size of batch images [128]")
flags.DEFINE_string("optimizer","RMSProp","The name of the optimizer to be used [AdaDelta,mom,RMSProp,SGD]")
FLAGS = flags.FLAGS

def main(_):
    homedir = os.path.expanduser('~')
    path = homedir + '/Datasets/cifar10/cifar-10-batches-py'
    hdf5_path = "./data"

    if not os.path.exists(hdf5_path+"/data.hdf5"):
        print ("hdf5 doesn't exist, creating it in data folder")
        trX, trY, teX, teY, valX, valY = read_cifar10(path)
        write_hdf5(trX, trY, teX, teY, valX, valY, hdf5_path)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        ResNet_model = ResNET(sess, img_shape=get_img_shape(),optimizer=FLAGS.optimizer)
        ResNet_model.train(n_epochs=FLAGS.n_epochs,train_batch=FLAGS.train_size,val_batch=FLAGS.val_size, test_batch=FLAGS.test_size)

if __name__ == "__main__":
    tf.app.run()