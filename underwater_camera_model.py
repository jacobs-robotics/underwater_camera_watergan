from __future__ import division
import os
import PIL
import scipy.stats as st
import numpy as np
from PIL import Image
import time
import scipy
from scipy import misc
import scipy.misc
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import scipy.io as sio
from ops import *
from utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from fill_depth import *


class WGAN(object):
    def __init__(self, sess, input_height=640, input_width=480, input_water_height=1360, input_water_width=1024,
                 is_crop=True,
                 batch_size=64, sample_num=64, output_height=256, output_width=256,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=3, max_depth=3.0,
                 save_epoch=100,
                 water_dataset_name='default', input_fname_pattern='*.png', checkpoint_dir=None, results_dir=None,
                 sample_dir=None, use_batchsize_for_prediction=True):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.input_water_height = input_water_height
        self.input_water_width = input_water_width
        self.save_epoch = save_epoch
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.max_depth = max_depth

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        self.sw = 640
        self.sh = 480

        self.use_batchsize_for_prediction=use_batchsize_for_prediction

        # batch normalization : deals with poor initialization helps gradient flow
        # DexROV: using batch size = 1 for prediction (as opposed to batch size = 64 for training)
        self.d_bn1 = batch_norm(name='d_bn1', use_batchsize_for_prediction=self.use_batchsize_for_prediction)
        self.d_bn2 = batch_norm(name='d_bn2', use_batchsize_for_prediction=self.use_batchsize_for_prediction)

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3', use_batchsize_for_prediction=self.use_batchsize_for_prediction)

        self.g_bn0 = batch_norm(name='g_bn0', use_batchsize_for_prediction=self.use_batchsize_for_prediction)
        self.g_bn1 = batch_norm(name='g_bn1', use_batchsize_for_prediction=self.use_batchsize_for_prediction)
        self.g_bn2 = batch_norm(name='g_bn2', use_batchsize_for_prediction=self.use_batchsize_for_prediction)

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3', use_batchsize_for_prediction=self.use_batchsize_for_prediction)
            self.g_bn4 = batch_norm(name='g_bn4', use_batchsize_for_prediction=self.use_batchsize_for_prediction)

        self.water_dataset_name = water_dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir

        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        image_dims = [self.output_height, self.output_width, self.c_dim]
        sample_dims = [self.output_height, self.output_width, self.c_dim]
        self.water_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.air_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='air_images')
        self.depth_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + [self.output_height, self.output_width, 1], name='depth')
        self.water_sample_inputs = tf.placeholder(
            tf.float32, [1] + image_dims, name='sample_inputs')
        self.depth_small_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + [self.output_height, self.output_width, 1], name='depth_small')

        self.R2 = tf.placeholder(tf.float32, [self.output_height, self.output_width], name='R2')
        self.R4 = tf.placeholder(tf.float32, [self.output_height, self.output_width], name='R4')
        self.R6 = tf.placeholder(tf.float32, [self.output_height, self.output_width], name='R6')

        self.sample_air_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + [self.sh, self.sw, 3], name='sample_air_images')
        self.sample_depth_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + [self.sh, self.sw, 1], name='sample_depth')
        self.sample_fake_inputs = tf.placeholder(
            tf.float32, [self.batch_size] + [self.sh, self.sw, 3], name='sample_fake')

        sample_air_inputs = self.sample_air_inputs
        sample_depth_inputs = self.sample_depth_inputs
        depth_small_inputs = self.depth_small_inputs
        sample_fake_inputs = self.sample_fake_inputs

        water_inputs = self.water_inputs
        water_sample_inputs = self.water_sample_inputs
        air_inputs = self.air_inputs
        depth_inputs = self.depth_inputs

        R2 = self.R2
        R4 = self.R4
        R6 = self.R6

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)
        self.sample_z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.G, eta_r, eta_g, eta_b, C1, C2, C3, A = self.wc_generator(self.z, air_inputs, depth_inputs, R2, R4, R6)
        self.D, self.D_logits = self.discriminator(water_inputs)

        self.wc_sampler = self.wc_sampler(self.sample_z, sample_air_inputs, sample_depth_inputs, depth_small_inputs, R2,
                                          R4, R6)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G, max_outputs=200)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        self.c1_loss = -tf.minimum(tf.reduce_min(C1), 0) * 10000
        self.c2_loss = -tf.minimum(tf.reduce_min(-1 * (4 * C2 * C2 - 12 * C1 * C3)), 0) * 10000

        self.eta_r_loss = -tf.minimum(tf.reduce_min(eta_r), 0) * 10000
        self.eta_g_loss = -tf.minimum(tf.reduce_min(eta_g), 0) * 10000
        self.eta_b_loss = -tf.minimum(tf.reduce_min(eta_b), 0) * 10000
        self.A_loss = -tf.minimum(tf.reduce_min(A), 0) * 10000
        self.g_loss = self.c1_loss + self.c2_loss + self.g_loss + self.eta_r_loss + self.eta_g_loss + self.eta_b_loss + self.A_loss

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.D = tf.summary.scalar("D_realdata", self.D)
        self.D_ = tf.summary.scalar("D_fakedata", self.D_)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.saver = tf.train.Saver()

    def load_model(self, config):

        self.config = config

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss,
                                                                                            var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss,
                                                                                            var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.r2 = np.ones([self.output_height, self.output_width], np.float32)
        self.r4 = np.ones([self.output_height, self.output_width], np.float32)
        self.r6 = np.ones([self.output_height, self.output_width], np.float32)

        cx = self.output_width / 2
        cy = self.output_height / 2
        for i in range(0, self.output_height):
            for j in range(0, self.output_width):
                r = np.sqrt((i - cy) * (i - cy) + (j - cx) * (j - cx)) / (np.sqrt(cy * cy + cx * cx))
                self.r2[i, j] = r * r
                self.r4[i, j] = r * r * r * r
                self.r6[i, j] = r * r * r * r * r * r

        print(' [*] WaterGAN initialized, settings: ')
        print(self.sess.run('wc_generator/g_atten/g_eta_r:0'))
        print(self.sess.run('wc_generator/g_atten/g_eta_g:0'))
        print(self.sess.run('wc_generator/g_atten/g_eta_b:0'))
        print(self.sess.run('wc_generator/g_vig/g_amp:0'))
        print(self.sess.run('wc_generator/g_vig/g_c1:0'))
        print(self.sess.run('wc_generator/g_vig/g_c2:0'))
        print(self.sess.run('wc_generator/g_vig/g_c3:0'))

    # predict for single image
    def predict(self, rgb_image, depth_image_raw, smooth_depth=False):

        for epoch in xrange(self.config.epoch):

            # testing data
            #air_data = '/home/tobi/data/watergan/uw-rgbd-images/01-00000-color.png'
            #depth_data = '/home/tobi/data/watergan/uw-rgbd-depth/01-00000-depth.png'
            #air_data = '/home/tobi/data/watergan/VOCB3DO/KinectColor/img_0099.png' # same image as in the WaterGAN paper
            #depth_data = '/home/tobi/data/watergan/VOCB3DO/RegisteredDepthData/img_0099_abs_smooth.png' # same image as in the WaterGAN paper
            #depth_data_raw = '/home/tobi/data/watergan/VOCB3DO/RegisteredDepthData/img_0099_abs.png' # same image as in the WaterGAN paper

            rgb_name = 'watergan_rgb_image.png'
            depth_name_raw = 'watergan_depth_image.png'
            depth_name_smooth = 'watergan_depth_image_smooth.png'
            underwater_name = 'watergan_underwater_image.png'

            scipy.misc.imsave(rgb_name, rgb_image)
            scipy.misc.imsave(depth_name_raw, depth_image_raw)

            # fill/smooth depth, see https://gist.github.com/bwaldvogel/6892721, original: www.cs.huji.ac.il/~yweiss/Colorization/
            # and used in "A Category-Level 3-D Object Dataset: Putting the Kinect to Work", A. Janoch et al.
            # this has no effect in the current setup where NaN depth is clamped to maximum depth as permitted by hardware
            if smooth_depth:
                print(' [*] Smoothing depth image, this takes a while... ')
                depth_image_smooth = fill_depth_colorization(rgb_image, depth_image_raw)
            else:
                depth_image_smooth = depth_image_raw
            scipy.misc.imsave(depth_name_smooth, depth_image_smooth)

            for idx in xrange(self.batch_size):
                # use the same images self.batch_size times because the pipeline has to use the same batch size as in training
                sample_air_batch_files = [rgb_name] * self.batch_size
                sample_depth_batch_files = [depth_name_smooth] * self.batch_size
                if self.is_crop:
                    sample_air_batch = [self.read_img_sample(sample_air_batch_file) for sample_air_batch_file in
                                        sample_air_batch_files]
                    sample_depth_small_batch = [self.read_depth_small(sample_depth_batch_file) for
                                                sample_depth_batch_file in sample_depth_batch_files]
                    sample_depth_batch = [self.read_depth_sample(sample_depth_batch_file) for sample_depth_batch_file in
                                          sample_depth_batch_files]
                else:
                    sample_air_batch = [scipy.misc.imread(sample_air_batch_file) for sample_air_batch_file in
                                        sample_air_batch_files]
                    sample_depth_batch = [self.read_depth_sample(sample_depth_batch_file) for sample_depth_batch_file in
                                          sample_depth_batch_files]
                    sample_depth_small_batch = [self.read_depth_small(sample_depth_batch_file) for
                                                sample_depth_batch_file in sample_depth_batch_files]
                sample_air_images = np.array(sample_air_batch).astype(np.float32)
                sample_depth_small_images = np.expand_dims(sample_depth_small_batch, axis=3)
                sample_depth_images = np.expand_dims(sample_depth_batch, axis=3)
                sample_z = np.random.uniform(-1, 1, [self.config.batch_size, self.z_dim]).astype(np.float32)

                # run the prediction method
                samples = self.sess.run([self.wc_sampler],
                                        feed_dict={self.sample_z: sample_z, self.sample_air_inputs: sample_air_images,
                                                   self.sample_depth_inputs: sample_depth_images,
                                                   self.depth_small_inputs: sample_depth_small_images, self.R2: self.r2,
                                                   self.R4: self.r4, self.R6: self.r6})

                sample_ims = np.asarray(samples)
                # remove only first dimension to prevent from errors with batch size = 1
                sample_ims = np.squeeze(sample_ims, axis=0)
                sample_fake_images = sample_ims[:, 0:self.sh, 0:self.sw, 0:3]
                sample_fake_images_small = np.empty([0, self.sh, self.sw, 3])

                # discard all but the first generated image
                # (others were only created to leave the pipeline intact with the original batch size)
                for img_idx in range(1):
                    out_file = "fake_%0d_%02d_%02d.png" % (epoch, img_idx, idx)
                    out_name = os.path.join(self.config.water_dataset, self.results_dir, out_file)
                    sample_im = sample_ims[img_idx, 0:self.sh, 0:self.sw, 0:3]
                    sample_im = np.squeeze(sample_im)

                    try:
                        scipy.misc.imsave(out_name, sample_im)
                        scipy.misc.imsave(underwater_name, sample_im)
                    except OSError:
                        print(out_name)
                        print("ERROR!")
                        pass
                    out_file2 = "air_%0d_%02d_%02d.png" % (epoch, img_idx, idx)
                    out_name2 = os.path.join(self.config.water_dataset, self.results_dir, out_file2)
                    sample_im2 = sample_air_images[img_idx, 0:self.sh, 0:self.sw, 0:3]
                    sample_im2 = np.squeeze(sample_im2)
                    try:
                        scipy.misc.imsave(out_name2, sample_im2)
                    except OSError:
                        print(out_name)
                        print("ERROR!")
                        pass
                    out_file3 = "depth_%0d_%02d_%02d.png" % (epoch, img_idx, idx)
                    out_name3 = os.path.join(self.config.water_dataset, self.results_dir, out_file3)
                    try:
                        scipy.misc.imsave(out_name3, depth_image_smooth)
                    except OSError:
                        print(out_name)
                        print("ERROR!")
                        pass
                    out_file4 = "depth_raw_%0d_%02d_%02d.png" % (epoch, img_idx, idx)
                    out_name4 = os.path.join(self.config.water_dataset, self.results_dir, out_file4)
                    try:
                        scipy.misc.imsave(out_name4, depth_image_raw)
                    except OSError:
                        print(out_name)
                        print("ERROR!")
                        pass
                    sample_fake = sample_fake_images[img_idx, 0:self.sh, 0:self.sw, 0:3]
                    sample_fake = np.squeeze(sample_fake)
                    sample_fake = scipy.misc.imresize(sample_fake, [self.sh, self.sw, 3], interp='bicubic')
                    sample_fake = np.expand_dims(sample_fake, axis=0)
                    sample_fake_images_small = np.append(sample_fake_images_small, sample_fake, axis=0)

                #return sample_im
                # read image again to avoid conversion issues
                return scipy.misc.imread(underwater_name)

    def discriminator(self, image, depth=None, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def sample_discriminator(self, image, depth=None, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4)

    # the actual WaterGAN generator, i.e. training method
    def wc_generator(self, z, image, depth, r2, r4, r6, y=None):
        with tf.variable_scope("wc_generator") as scope:
            # water-based attenuation and backscatter
            with tf.variable_scope("g_atten"):
                init_r = tf.random_normal([1, 1, 1], mean=0.35, stddev=0.01, dtype=tf.float32)
                eta_r = tf.get_variable("g_eta_r", initializer=init_r)
                init_b = tf.random_normal([1, 1, 1], mean=0.0194, stddev=0.01, dtype=tf.float32)
                eta_b = tf.get_variable("g_eta_b", initializer=init_b)
                init_g = tf.random_normal([1, 1, 1], mean=0.038, stddev=0.01, dtype=tf.float32)
                eta_g = tf.get_variable("g_eta_g", initializer=init_g)
                eta = tf.stack([eta_r, eta_g, eta_b], axis=3)
                eta_d = tf.exp(tf.multiply(-1.0, tf.multiply(depth, eta)))

            h0 = tf.multiply(image, eta_d)

            # backscattering
            self.z_, self.h0z_w, self.h0z_b = linear(
                z, self.output_width * self.output_height * self.batch_size * 1, 'g_h0_lin', with_w=True)

            self.h0z = tf.reshape(
                self.z_, [-1, self.output_height, self.output_width, self.batch_size * 1])
            h0z = tf.nn.relu(self.g_bn0(self.h0z))
            h0z = tf.multiply(h0z, depth)

            with tf.variable_scope('g_h1_conv'):
                w = tf.get_variable('g_w', [5, 5, h0z.get_shape()[-1], 1],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1z = tf.nn.conv2d(h0z, w, strides=[1, 1, 1, 1], padding='SAME')
            h_g = lrelu(self.g_bn1(h1z))

            with tf.variable_scope('g_h1_convr'):
                wr = tf.get_variable('g_wr', [5, 5, h0z.get_shape()[-1], 1],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1zr = tf.nn.conv2d(h0z, wr, strides=[1, 1, 1, 1], padding='SAME')
            h_r = lrelu(self.g_bn3(h1zr))

            with tf.variable_scope('g_h1_convb'):
                wb = tf.get_variable('g_wb', [5, 5, h0z.get_shape()[-1], 1],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1zb = tf.nn.conv2d(h0z, wb, strides=[1, 1, 1, 1], padding='SAME')
            h_b = lrelu(self.g_bn4(h1zb))

            h_r = tf.squeeze(h_r, axis=3)
            h_g = tf.squeeze(h_g, axis=3)
            h_b = tf.squeeze(h_b, axis=3)

            h_final = tf.stack([h_r, h_g, h_b], axis=3)

            h2 = tf.add(h_final, h0)

            # camera model
            with tf.variable_scope("g_vig"):
                A = tf.get_variable('g_amp', [1],
                                    initializer=tf.truncated_normal_initializer(mean=0.9, stddev=0.01))
                C1 = tf.get_variable('g_c1', [1],
                                     initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.001))
                C2 = tf.get_variable('g_c2', [1],
                                     initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.001))
                C3 = tf.get_variable('g_c3', [1],
                                     initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.001))
            h11 = tf.multiply(r2, C1)
            h22 = tf.multiply(r4, C2)
            h33 = tf.multiply(r6, C3)
            h44 = tf.ones([self.output_height, self.output_width], tf.float32)
            h1 = tf.add(tf.add(h44, h11), tf.add(h22, h33))
            V = tf.expand_dims(h1, axis=2)
            h1a = tf.divide(h2, V)
            h_out = tf.multiply(h1a, A)
            return h_out, eta_r, eta_g, eta_b, C1, C2, C3, A

    # the actual WaterGAN sample creation, i.e. testing method
    def wc_sampler(self, z, image, depth, depth_small, r2, r4, r6, y=None):
        with tf.variable_scope("wc_generator", reuse=True) as scope:
            # water-based attenuation
            with tf.variable_scope("g_atten", reuse=True):
                init_r = tf.random_normal([1, 1, 1], mean=0.35, stddev=0.01, dtype=tf.float32)
                eta_r = tf.get_variable("g_eta_r", initializer=init_r)
                init_b = tf.random_normal([1, 1, 1], mean=0.0194, stddev=0.01, dtype=tf.float32)
                eta_b = tf.get_variable("g_eta_b", initializer=init_b)
                init_g = tf.random_normal([1, 1, 1], mean=0.038, stddev=0.01, dtype=tf.float32)
                eta_g = tf.get_variable("g_eta_g", initializer=init_g)
                eta = tf.stack([eta_r, eta_g, eta_b], axis=3)

                eta_d = tf.exp(tf.multiply(-1.0, tf.multiply(depth, eta)))
                h0 = tf.multiply(image, eta_d)

            self.z_, self.h0z_w, self.h0z_b = linear(
                z, self.output_width * self.output_height * self.batch_size * 1, 'g_h0_lin', with_w=True)

            self.h0z = tf.reshape(
                self.z_, [-1, self.output_height, self.output_width, self.batch_size * 1])
            h0z = tf.nn.relu(self.g_bn0(self.h0z))
            h0z = tf.multiply(h0z, depth_small)

            # backscattering
            with tf.variable_scope('g_h1_conv', reuse=True):
                w = tf.get_variable('g_w', [5, 5, h0z.get_shape()[-1], 1],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1z = tf.nn.conv2d(h0z, w, strides=[1, 1, 1, 1], padding='SAME')
            h_g = lrelu(self.g_bn1(h1z))

            with tf.variable_scope('g_h1_convr', reuse=True):
                wr = tf.get_variable('g_wr', [5, 5, h0z.get_shape()[-1], 1],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1zr = tf.nn.conv2d(h0z, wr, strides=[1, 1, 1, 1], padding='SAME')
            h_r = lrelu(self.g_bn3(h1zr))

            with tf.variable_scope('g_h1_convb', reuse=True):
                wb = tf.get_variable('g_wb', [5, 5, h0z.get_shape()[-1], 1],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1zb = tf.nn.conv2d(h0z, wb, strides=[1, 1, 1, 1], padding='SAME')
            h_b = lrelu(self.g_bn4(h1zb))

            h_r1 = tf.image.resize_images(h_r, [120, 160], method=2)
            h_g1 = tf.image.resize_images(h_g, [120, 160], method=2)
            h_b1 = tf.image.resize_images(h_b, [120, 160], method=2)
            h_rxlt = tf.image.resize_images(h_r1, [240, 320], method=2)
            h_gxlt = tf.image.resize_images(h_g1, [240, 320], method=2)
            h_bxlt = tf.image.resize_images(h_b1, [240, 320], method=2)

            h_rxl = tf.image.resize_images(h_rxlt, [480, 640], method=2)
            h_gxl = tf.image.resize_images(h_gxlt, [480, 640], method=2)
            h_bxl = tf.image.resize_images(h_bxlt, [480, 640], method=2)

            h_rxl = tf.squeeze(h_rxl, axis=3)
            h_gxl = tf.squeeze(h_gxl, axis=3)
            h_bxl = tf.squeeze(h_bxl, axis=3)
            h_final = tf.stack([h_rxl, h_gxl, h_bxl], axis=3)
            h2 = tf.add(h_final, h0)

            # camera model
            with tf.variable_scope("g_vig", reuse=True):
                A = tf.get_variable('g_amp', [1],
                                    initializer=tf.truncated_normal_initializer(mean=0.9, stddev=0.01))
                C1 = tf.get_variable('g_c1', [1],
                                     initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.001))
                C2 = tf.get_variable('g_c2', [1],
                                     initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.001))
                C3 = tf.get_variable('g_c3', [1],
                                     initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.001))

            h11 = tf.multiply(r2, C1)
            h22 = tf.multiply(r4, C2)
            h33 = tf.multiply(r6, C3)
            h44 = tf.ones([self.output_height, self.output_width], tf.float32)
            h1 = tf.add(tf.add(h44, h11), tf.add(h22, h33))
            V = tf.expand_dims(h1, axis=2)
            h1a = V
            h1a1 = tf.image.resize_images(h1a, [120, 160], method=2)
            h1_xlt = tf.image.resize_images(h1a1, [240, 320], method=2)
            h1_xl = tf.image.resize_images(h1_xlt, [480, 640], method=2)
            h_out1 = tf.divide(h2, h1_xl)
            h_out = tf.multiply(h_out1, A)
            return h_out

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.water_dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(self.model_dir, checkpoint_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        checkpoint_dir = os.path.join(self.model_dir, checkpoint_dir)
        print(" [*] Reading model from {}...".format(checkpoint_dir))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a model in {}!".format(checkpoint_dir))
            return False

    def read_depth(self, filename):
        if filename.split('.'[-1]) == '.mat':
            depth_mat = sio.loadmat(filename)
            depthtmp = depth_mat["depth"]
        else:
            depthtmp = scipy.misc.imread(filename)
        ds = depthtmp.shape
        if self.is_crop:
            depth = scipy.misc.imresize(depthtmp, (self.output_height, self.output_width), mode='F')
        depth = np.array(depth).astype(np.float32)
        depth = np.multiply(self.max_depth, np.divide(depth, depth.max()))

        return depth

    def read_img(self, filename):
        imgtmp = scipy.misc.imread(filename)
        ds = imgtmp.shape
        if self.is_crop:
            img = scipy.misc.imresize(imgtmp, (self.output_height, self.output_width, 3))
        img = np.array(img).astype(np.float32)
        return img

    def read_depth_small(self, filename):
        if filename.split('.'[-1]) == '.mat':
            depth_mat = sio.loadmat(filename)
            depthtmp = depth_mat["depth"]
        else:
            depthtmp = scipy.misc.imread(filename)
        ds = depthtmp.shape

        if self.is_crop:
            depth = scipy.misc.imresize(depthtmp, (self.output_height, self.output_width), mode='F')
        depth = np.array(depth).astype(np.float32)
        depth = np.multiply(self.max_depth, np.divide(depth, depth.max()))

        return depth

    def read_depth_sample(self, filename):
        if filename.split('.'[-1]) == '.mat':
            depth_mat = sio.loadmat(filename)
            depthtmp = depth_mat["depth"]
        else:
            depthtmp = scipy.misc.imread(filename)
        ds = depthtmp.shape
        if self.is_crop:
            depth = scipy.misc.imresize(depthtmp, (self.sh, self.sw), mode='F')
        else:
            depth = depthtmp
        depth = np.array(depth).astype(np.float32)
        depth = np.multiply(self.max_depth, np.divide(depth, depth.max()))

        return depth

    def read_img_sample(self, filename):
        imgtmp = scipy.misc.imread(filename)
        ds = imgtmp.shape
        if self.is_crop:
            img = scipy.misc.imresize(imgtmp, (self.sh, self.sw, 3))
        img = np.array(img).astype(np.float32)
        return img
