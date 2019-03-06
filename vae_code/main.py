import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import input_data
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *


class LatentAttention:
    """
    使用vae方法将原始的encoder和decoder方法变成了哈斯用均值和方差求带变化结果的程序
    """
    def __init__(self):
        """
        mnist: 数据集
        n_samples: 数据的个数
        n_hidden: 隐藏层的个数
        n_z: 每次使用的图像的数量
        batchsize: 使用图像的尺寸大小
        """
        self.mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.n_hidden = 500
        self.n_z = 20
        self.batchsize = 100
        # 获取图像矩阵
        self.images = tf.placeholder(tf.float32, [None, 784])
        image_matrix = tf.reshape(self.images, [-1, 28, 28, 1])
        # 使用encoder将均值和方差计算出来
        z_mean, z_stddev = self.recognition(image_matrix)
        # 使用标准差中取出的一个随机值,加上原始的均值,获得一个随机样本,生成变量通过这个获得
        samples = tf.random_normal([self.batchsize, self.n_z], 0, 1, dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)
        # 使用这个方法将编码后添加了kl散度的值解码出
        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])
        # 生成误差,用于衡量将原始的图片和新生成图片之间的差距
        self.generation_loss = tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) +
                                             (1 - self.images) * tf.log(1e-8 + 1 - generated_flat), 1)
        # 均方误差,使用一个求均值方差的方法,判断生成图像与原始图像之间KL散度之间的关系
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) +
                                               tf.square(z_stddev) -
                                               tf.log(tf.square(z_stddev)) - 1, 1)

        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    def recognition(self, input_images):
        with tf.variable_scope('recongnition'):
            h1 = lrelu(conv2d(input_images, 1, 16, 'd_h1'))
            h2 = lrelu(conv2d(h1, 16, 32, 'd_h2'))
            h2_flat = tf.reshape(h2, [self.batchsize, 7*7*32])

            w_mean = dense(h2_flat, 7*7*32, self.n_z, 'w_mean')
            w_stddev = dense(h2_flat, 7*7*32, self.n_z, 'w_stddev')

        return w_mean, w_stddev

    def generation(self, z):
        with tf.variable_scope('generation'):
            z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 7, 7, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 14, 14, 16], 'g_h1'))
            h2 = conv_transpose(h1, [self.batchsize, 28, 28, 1], 'g_h2')
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        visualization = self.mnist.train.next_batch(self.batchsize)[0]
        reshaped_vis = visualization.reshape(self.batchsize, 28, 28)
        print(np.shape(reshaped_vis))
        ims('results/base2.jpg', merge(reshaped_vis[:64], [8, 8]))

        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(10):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss),
                                                     feed_dict={self.images: batch})

                    if idx % (self.n_samples - 3) == 0:
                        print('epoch %d: genloss %f latloss %f' % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
                        saver.save(sess, os.getcwd()+'training/train', global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize, 28, 28)
                        ims('result/' + str(epoch) + '.jpg', merge(generated_test[:64], [8, 8]))


model = LatentAttention()
model.train()

