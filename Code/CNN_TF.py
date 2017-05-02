import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import cPickle
import os
batch_size = 128
test_size = 256

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def onehot(value,out_size):
    output = np.zeros((out_size))
    output[value] = 1
    return output

def read_cifar10(path):
    data_batches = []
    labels_batches = []
    for i in xrange(1,6):
        dict = unpickle(path+"/data_batch_"+str(i))
        data = np.array(dict['data'])
        labels = dict['labels']
        data_batches.extend(data)
        labels_batches.extend(labels)

    labels_batches = [onehot(v,10) for v in labels_batches]
    trX = np.array(data_batches[:40000])
    trY= np.array(labels_batches[:40000])

    teX = np.array(data_batches[40000:])
    teY = np.array(labels_batches[40000:])
    return trX,trY,teX,teY


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, wh, wo, p_keep_conv, p_keep_hidden,conv_list,maxpool_list):
    layer = X
    for i in xrange(len(conv_list)):
        layer = tf.nn.relu(tf.nn.conv2d(layer, w[i], strides=conv_list[i][1], padding=conv_list[i][2]))
        layer = tf.nn.max_pool(layer, ksize=maxpool_list[i][0], strides=maxpool_list[i][1], padding=maxpool_list[i][2])
        layer = tf.nn.dropout(layer,p_keep_conv)
    layer = tf.reshape(layer, [-1, wh.get_shape().as_list()[0]])  # reshape to (?, ?)
    layer = tf.nn.dropout(layer, p_keep_conv)

    layer = tf.nn.relu(tf.matmul(layer, wh))
    layer = tf.nn.dropout(layer, p_keep_hidden)

    pyx = tf.matmul(layer, wo)
    return pyx

class CNN(object):
    def __init__(self,conv_list,maxpool_list, hid_layer, out_n = 10):

        self.X = tf.placeholder("float", [None, 32, 32, 3])
        self.Y = tf.placeholder("float", [None, 10])
        self.w = []
        for conv in conv_list:
            self.w.append(init_weights(conv[0]))
        self.wh = init_weights([128*4*4,hid_layer])
        self.wo = init_weights([hid_layer,out_n])
        self.p_keep_conv = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")

        py_x = model(self.X, self.w, self.wh,self.wo, self.p_keep_conv, self.p_keep_hidden,conv_list,maxpool_list)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=self.Y))
        self.train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(self.cost)
        self.predict_op = tf.argmax(py_x, 1)

    def train(self,trX,trY,teX,teY,n_epochs=100):
        with tf.Session() as sess:
            # you need to initialize all variables
            tf.global_variables_initializer().run()

            for i in range(n_epochs):
                training_batch = zip(range(0, len(trX), batch_size),
                                     range(batch_size, len(trX) + 1, batch_size))
                for start, end in training_batch:
                    sess.run(self.train_op, feed_dict={self.X: trX[start:end], self.Y: trY[start:end],
                                                  self.p_keep_conv: 0.8, self.p_keep_hidden: 0.5})

                test_indices = np.arange(len(teX))  # Get A Test Batch
                np.random.shuffle(test_indices)
                test_indices = test_indices[0:test_size]

                print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                                 sess.run(self.predict_op, feed_dict={self.X: teX[test_indices],
                                                                 self.Y: teY[test_indices],
                                                                 self.p_keep_conv: 1.0,
                                                                 self.p_keep_hidden: 1.0})))
                print ('cost - ', sess.run(self.cost, feed_dict={self.X: teX[test_indices],
                                                                 self.Y: teY[test_indices],
                                                                 self.p_keep_conv: 1.0,
                                                                 self.p_keep_hidden: 1.0}))
if __name__ == "__main__":
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    homedir = os.path.expanduser('~')
    path = homedir + '/Datasets/cifar10/cifar-10-batches-py'
    trX, trY, teX, teY = read_cifar10(path)
    trX = trX.reshape(-1, 32, 32, 3) # 32x32x1 input img
    teX = teX.reshape(-1, 32, 32, 3)  # 32x32x1 input img


    conv_list = []
    maxpool_list =[]
    conv_list.append(([3, 3, 3, 32] , [1,1,1,1],'SAME'))
    conv_list.append(([3, 3, 32, 64], [1, 1, 1, 1], 'SAME'))
    conv_list.append(([3, 3, 64, 128], [1, 1, 1, 1], 'SAME'))
    maxpool_list.append(([1,2,2,1],[1,2,2,1],'SAME')) # 14
    maxpool_list.append(([1, 2, 2, 1], [1, 2, 2, 1], 'SAME')) # 7
    maxpool_list.append(([1, 2, 2, 1], [1, 2, 2, 1], 'SAME')) # 4

    CNN_model = CNN(conv_list,maxpool_list,625,10)
    CNN_model.train(trX,trY,teX,teY)
