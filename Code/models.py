import tensorflow as tf
from ops import *
from utils import *
import os
import time
#from ValGrad import *
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

class ResNET(object):
    def __init__(self, sess,img_shape=[32,32,3],optimizer='RMSProp'):
        self.sess = sess
        self.is_train = tf.placeholder(tf.bool)
        self.lrn_rate = tf.placeholder(tf.float16)
        self.batch_size= tf.placeholder(tf.int32)
        self.batch_shape = [self.batch_size,128]
        self.img_shape = img_shape
        self.X = tf.placeholder("float", [None, ]+img_shape)
        #self.X = self.mod_image(X)
        self.Y = tf.placeholder("float", [None, 10])
        self.optimizer = optimizer
        self.py_x = self.build_resnet20()
        self.out_sum = tf.summary.histogram("out", self.out)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.py_x, labels=self.Y))
        #py_x = tf.nn.softmax(self.build_resnet20())
        #self.cost=-tf.reduce_mean(py_x*tf.log(self.Y))
        tf.summary.scalar('cross_entropy', self.cost)
        if self.optimizer == 'mom':
            self.train_op = tf.train.MomentumOptimizer(self.lrn_rate, 0.9).minimize(self.cost)
        elif self.optimizer == 'RMSProp':
            self.train_op = tf.train.RMSPropOptimizer(self.lrn_rate, 0.9).minimize(self.cost)
        elif self.optimizer == "AdaGrad":
            self.train_op = tf.train.AdagradOptimizer(self.lrn_rate,1e-6).minimize(self.cost)
        elif self.optimizer == 'SGD':
            self.train_op = tf.train.GradientDescentOptimizer(self.lrn_rate).minimize(self.cost)
            #self.train_op = ValgradOptimizer(self.lrn_rate).minimize(self.cost)
        else:
            self.train_op = self.own_optimizer(self.cost,self.lrn_rate)
        self.predict_op = tf.argmax(self.py_x, 1)
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict_op,tf.argmax(self.Y,1)),"float"))
        tf.summary.scalar('accuracy', self.accuracy)
        self.checkpoint_dir = 'checkpoint'
        self.saver = tf.train.Saver()
        # For early stopping
        self.best_acc = 0
        self.max_no_imp = 50
        self.no_imp_counter = 0

    def mod_image(self,X):
        X = tf.image.resize_image_with_crop_or_pad(
            X, 32 + 4, 32 + 4)
        X = tf.random_crop(X, [-1,32, 32, 3])
        #image = tf.image.random_flip_left_right(image)
        # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
        # image = tf.image.random_brightness(image, max_delta=63. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        X = tf.image.per_image_standardization(X)
        return X
    def own_optimizer(self,cost, lrn_rate):
        var = [v for v in tf.trainable_variables()]
        grads = tf.gradients(cost,var)
        hessian = self.compute_hessian(cost,var)
        print 1
    def compute_hessian(self,fn, vars):
        mat = []
        for v1 in vars:
            temp = []
            for v2 in vars:
                # computing derivative twice, first w.r.t v2 and then w.r.t v1
                temp.append(tf.gradients(tf.gradients(fn, v2)[0], v1)[0])
            temp = [self.cons(0) if t == None else t for t in
                    temp]  # tensorflow returns None when there is no gradient, so we replace None with 0
            temp = tf.stack(temp)
            mat.append(temp)
        mat = tf.stack(mat)
        return mat

    def cons(self,x):
        return tf.constant(x, dtype=tf.float32)
    def simple_block(self, input_feats, name1, name2, n, stride=1):
        """ A basic block of ResNets - used for the smaller versions, for more advanced ones, check the bottleneck version"""
        branchSa_feats = convolution(input_feats, 1, 1, 2 * n, stride, stride, name1 + '_branchSa')
        branchSa_feats = batch_norm(branchSa_feats, name2 + '_branchSa',self.is_train)

        branchSb_feats = convolution(input_feats, 3, 3, n, stride, stride, name1 + '_branchSb')
        branchSb_feats = batch_norm(branchSb_feats, name2 + '_branchSb',self.is_train)
        branchSbfeats = nonlinear(branchSb_feats, 'relu')

        branchSc_feats = convolution(branchSbfeats, 3, 3, n*2, 1, 1, name1 + '_branchSc')
        branchSc_feats = batch_norm(branchSc_feats, name2 + '_branchSc',self.is_train)

        output_feats = branchSa_feats + branchSc_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def build_resnet20(self):
        """ Build the ResNet20 net. """
        """ This net can be modified to any other size stride by changing the n, I am recreating
        the specific residual network that was used by facebook designed for CIFAR. If want to change
        this network to a larger one by using only the basic blocks"""

        imgs = self.X
        #imgs = tf.placeholder(tf.float32, [self.batch_size] + self.img_shape)
        #is_train = tf.placeholder(tf.bool)
        # TODO: Relu and batch normalization separately!!!
        conv1_feats = convolution(imgs, 3, 3, 16, 1, 1, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1',self.is_train)
        conv1_feats = nonlinear(conv1_feats, 'relu')

        res2a_feats = self.simple_block(conv1_feats,'res2a','bn2a', 16)
        res3a_feats = self.simple_block(res2a_feats, 'res3a', 'bn3a', 32, 2)
        res4a_feats = self.simple_block(res3a_feats, 'res4a', 'bn4a', 64,2)


        self.res2_sum = tf.summary.histogram("res2", res2a_feats)
        self.res3_sum = tf.summary.histogram("res3", res3a_feats)
        self.res4_sum = tf.summary.histogram("res4", res4a_feats)
        # in torch this is spatial average pool
        pool4_feats = tf.nn.avg_pool(res4a_feats, ksize=[1,8,8,1], strides=[1,1,1,1], padding = 'VALID')

        res5a_feats_flat = tf.reshape(pool4_feats,self.batch_shape)

        res6a_fc_feats = fully_connected(res5a_feats_flat, 10, "FCa", init_w='he', init_b=0, stddev=0.01, group_id=0)

        self.out = res6a_fc_feats

        return self.out

    def train(self,train_batch=128,val_batch=256, test_batch=1024,n_epochs=500):
        # Need to initialize all variables
        trX, trY, valX, valY, teX, teY = get_hdf5()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter("./logs/train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter("./logs/test", self.sess.graph)

        tf.global_variables_initializer().run()
        lrn_rate = 0.1
        count = 0
        curr_level = 2
        total_time = time.time()
        i = 0
        rate_counter = 0
        best_index = 0
        for i in range(n_epochs):
            # Rate starts to descend after starting to train on the curriculum of the whole data
            if curr_level == 10:
                rate_counter += 1
                if rate_counter == 20:
                    lrn_rate=0.01
                elif rate_counter ==50:
                    lrn_rate=0.001
                elif rate_counter == 80:
                    lrn_rate = 0.0001

            with trY.astype('float32'):
                batch_labels = trY[...]
            working_indices = []
            for curr in xrange(curr_level):
                indices = [ind for ind, x in enumerate(batch_labels) if x[curr] == 1]
                working_indices.extend(indices)
            # Adding randomality to reduce overfitting
            #train_indices = np.arange(len(trX))
            np.random.shuffle(working_indices)
            training_batch = zip(range(0, len(working_indices), train_batch),
                                 range(train_batch, len(working_indices) + 1, train_batch))
            for start, end in training_batch:
                tr_ind = list(np.sort(working_indices[start:end]))
                with trX.astype('float32'):
                    batch_data = trX[tr_ind]
                with trY.astype('float32'):
                    batch_labels=trY[tr_ind]
                #batch_data, batch_labels = get_train_batch(trX,trY,train_indices[start:end])
                # Adding random modification to reduce overfitting
                batch_data = random_crop_and_flip(batch_data,5)
                summary,_,tr_acc, tr_cost = self.sess.run([self.merged, self.train_op, self.accuracy, self.cost], feed_dict={self.X: batch_data, self.Y: batch_labels,
                                                   self.is_train:True, self.lrn_rate:lrn_rate,self.batch_size:train_batch})

                self.train_writer.add_summary(summary, count)
                count+=1
            if tr_acc > 0.8 and curr_level < 10:
                curr_level += 1
                print ("Extending curriculum to level:", curr_level)
            batch_data, batch_labels = get_test_batch(valX,valY, val_batch)
            summary, val_acc = self.sess.run([self.merged, self.accuracy], feed_dict={self.X: batch_data,
                                                                                 self.Y: batch_labels,
                                                                                 self.is_train: True,
                                                                                 self.batch_size: val_batch})
            batch_data = None
            if val_acc > self.best_acc:
                self.save(self.checkpoint_dir, count)
                self.no_imp_counter = 0
                self.best_acc = val_acc
                best_index = i
            else:
                self.no_imp_counter += 1
                if self.no_imp_counter > self.max_no_imp:
                    break

            self.test_writer.add_summary(summary,count)
            #self.train_writer.add_summary(summary, i)
            print('train ', i, tr_acc)
            #self.test_writer.add_summary(summary, i)
            print('validation ',i, val_acc)
            print ('cost - ',tr_cost)
            if (i%10==0):
                batch_data,batch_labels = get_test_batch(teX,teY,test_batch)
                summary, test_acc = self.sess.run([self.merged, self.accuracy], feed_dict={self.X: batch_data,
                                                                       self.Y: batch_labels,
                                                                       self.is_train: True,
                                                                       self.batch_size: test_batch})
                print('test ', i, test_acc)
        total_time -= time.time()
        # Print out the best run (early stopping implementation)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            batch_data, batch_labels = get_test_batch(teX,teY, test_batch)
            train_data, train_labels = get_test_batch(trX, trY, test_batch)
            val_data, val_labels = get_test_batch(valX, valY, test_batch)

            val_acc = self.sess.run([self.accuracy], feed_dict={self.X: val_data,
                                                                self.Y: val_labels,
                                                                self.is_train: True,
                                                                self.batch_size: test_batch})
            tr_acc = self.sess.run([self.accuracy], feed_dict={self.X: train_data,
                                                                self.Y: train_labels,
                                                                self.is_train: True,
                                                                self.batch_size: test_batch})
            te_acc = self.sess.run([self.accuracy], feed_dict={self.X: batch_data,
                                                               self.Y: batch_labels,
                                                               self.is_train: True,
                                                               self.batch_size: test_batch})
            print ('average time per iteration is: ',total_time/(i+1))
            print ('iteration ',best_index,'best validation accuracy - ',val_acc)
            print ('with train accuracy of - ', tr_acc)
            print ('with test accuracy of - ', te_acc)

    @property
    def model_dir(self):
        return "".format()

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0