# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


# example vgg16
class VGG16(object):
    " This class define your model"
    def __init__(self):
        self.create_model()
    def get_weight_variable(self,shape,regularizer=None,\
                            trainable=True,validate_shape=True):
        
        weights = tf.get_variable("weights",shape,dtype=tf.float32,\
                    initializer=tf.truncated_normal_initializer(0.0,0.001),\
                    #initializer=tf.contrib.layers.xavier_initializer(),
                    trainable=trainable,validate_shape=True)
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(weights))

        return weights

    def get_bias_variable(self,shape,regularizer=None,\
                            trainable=True,validate_shape=True):
        
        biases = tf.get_variable("biases",shape,dtype=tf.float32,\
                    initializer=tf.constant_initializer(0),\
                    trainable=trainable,validate_shape=True)
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(biases))

        return biases

    def create_model(self):
        reuse = False
        # conv1_1
        with tf.variable_scope('conv1_1',reuse=reuse):
            self.conv1_1_w = self.get_weight_variable([3, 3, 3, 64],trainable=False)
            self.conv1_1_b = self.get_bias_variable([64],trainable=False)
        # conv1_2
        with tf.variable_scope('conv1_2',reuse=reuse):
            self.conv1_2_w = self.get_weight_variable([3, 3, 64, 64],trainable=False)
            self.conv1_2_b = self.get_bias_variable([64],trainable=False)
        # conv2_1
        with tf.variable_scope('conv2_1',reuse=reuse):
            self.conv2_1_w = self.get_weight_variable([3, 3, 64, 128],trainable=False)
            self.conv2_1_b = self.get_bias_variable([128],trainable=False)
        # conv2_2
        with tf.variable_scope('conv2_2',reuse=reuse):
            self.conv2_2_w = self.get_weight_variable([3, 3, 128, 128],trainable=False)
            self.conv2_2_b = self.get_bias_variable([128],trainable=False)
        # conv3_1
        with tf.variable_scope('conv3_1',reuse=reuse):
            self.conv3_1_w = self.get_weight_variable([3, 3, 128, 256],trainable=False)
            self.conv3_1_b = self.get_bias_variable([256],trainable=False)
        # conv3_2
        with tf.variable_scope('conv3_2',reuse=reuse):
            self.conv3_2_w = self.get_weight_variable([3, 3, 256, 256],trainable=False)
            self.conv3_2_b = self.get_bias_variable([256],trainable=False)
        # conv3_3
        with tf.variable_scope('conv3_3',reuse=reuse):
            self.conv3_3_w = self.get_weight_variable([3, 3, 256, 256],trainable=False)
            self.conv3_3_b = self.get_bias_variable([256],trainable=False)
        # conv4_1
        with tf.variable_scope('conv4_1',reuse=reuse):
            self.conv4_1_w = self.get_weight_variable([3, 3, 256, 512],trainable=True)
            self.conv4_1_b = self.get_bias_variable([512],trainable=True)
        # conv4_2
        with tf.variable_scope('conv4_2',reuse=reuse):
            self.conv4_2_w = self.get_weight_variable([3, 3, 512, 512],trainable=True)
            self.conv4_2_b = self.get_bias_variable([512],trainable=True)
        # conv4_3
        with tf.variable_scope('conv4_3',reuse=reuse):
            self.conv4_3_w = self.get_weight_variable([3, 3, 512, 512],trainable=True)
            self.conv4_3_b = self.get_bias_variable([512],trainable=True)
        # conv5_1
        with tf.variable_scope('conv5_1',reuse=reuse):
            self.conv5_1_w = self.get_weight_variable([3, 3, 512, 512],trainable=True)
            self.conv5_1_b = self.get_bias_variable([512],trainable=True)
        # conv5_2
        with tf.variable_scope('conv5_2',reuse=reuse):
            self.conv5_2_w = self.get_weight_variable([3, 3, 512, 512],trainable=True)
            self.conv5_2_b = self.get_bias_variable([512],trainable=True)
        # conv5_3
        with tf.variable_scope('conv5_3',reuse=reuse):
            self.conv5_3_w = self.get_weight_variable([3, 3, 512, 512],trainable=True)
            self.conv5_3_b = self.get_bias_variable([512],trainable=True)
        # fc1   
        with tf.variable_scope('fc1',reuse=reuse):
            self.fc1_w = self.get_weight_variable([7*7*512, 4096],trainable=True)
            self.fc1_b = self.get_bias_variable([4096],trainable=True)
        # fc2
        with tf.variable_scope('fc2',reuse=reuse):
            self.fc2_w = self.get_weight_variable([4096, 4096],trainable=True)
            self.fc2_b = self.get_bias_variable([4096],trainable=True)
        # fc3
        with tf.variable_scope('fc3',reuse=reuse):
            self.fc3_w = self.get_weight_variable([4096, 1000],trainable=True)
            self.fc3_b = self.get_bias_variable([1000],trainable=True)
        
    def vgg16_inference(self,imgs):
        with tf.name_scope('conv1_1') as scope:
            conv = tf.nn.conv2d(imgs, self.conv1_1_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv1_1_b)
            self.conv1_1 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv1_2') as scope:
            conv = tf.nn.conv2d(self.conv1_1, self.conv1_2_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv1_2_b)
            self.conv1_2 = tf.nn.relu(out, name=scope)
        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')
        with tf.name_scope('conv2_1') as scope:
            conv = tf.nn.conv2d(self.pool1, self.conv2_1_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv2_1_b)
            self.conv2_1 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv2_2') as scope:
            conv = tf.nn.conv2d(self.conv2_1, self.conv2_2_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv2_2_b)
            self.conv2_2 = tf.nn.relu(out, name=scope)
        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')
        with tf.name_scope('conv3_1') as scope:
            conv = tf.nn.conv2d(self.pool2, self.conv3_1_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv3_1_b)
            self.conv3_1 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv3_2') as scope:
            conv = tf.nn.conv2d(self.conv3_1, self.conv3_2_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv3_2_b)
            self.conv3_2 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv3_3') as scope:
            conv = tf.nn.conv2d(self.conv3_2, self.conv3_3_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv3_3_b)
            self.conv3_3 = tf.nn.relu(out, name=scope)
        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')
        
        with tf.name_scope('conv4_1') as scope:
            conv = tf.nn.conv2d(self.pool3, self.conv4_1_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv4_1_b)
            self.conv4_1 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv4_2') as scope:
            conv = tf.nn.conv2d(self.conv4_1, self.conv4_2_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv4_2_b)
            self.conv4_2 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv4_3') as scope:
            conv = tf.nn.conv2d(self.conv4_2, self.conv4_3_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv4_3_b)
            self.conv4_3 = tf.nn.relu(out, name=scope)
        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
        with tf.name_scope('conv5_1') as scope:
            conv = tf.nn.conv2d(self.pool4, self.conv5_1_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv5_1_b)
            self.conv5_1 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv5_2') as scope:
            conv = tf.nn.conv2d(self.conv5_1, self.conv5_2_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv5_2_b)
            self.conv5_2 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv5_3') as scope:
            conv = tf.nn.conv2d(self.conv5_2, self.conv5_3_w, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, self.conv5_3_b)
            self.conv5_3 = tf.nn.relu(out, name=scope)
        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, self.fc1_w), self.fc1_b)
            self.fc1 = tf.nn.relu(fc1l)
        # fc2
        with tf.name_scope('fc2') as scope:
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, self.fc2_w), self.fc2_b)
            self.fc2 = tf.nn.relu(fc2l) 
        # fc3
        with tf.name_scope('fc3') as scope:
            self.fc3 = tf.nn.bias_add(tf.matmul(self.fc2, self.fc3_w), self.fc3_b)
        # prob
        return tf.nn.softmax(self.fc3)