#!/usr/bin/env python

import rospy
import os
import scipy.misc
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from model_training import WGAN
from yaml import load, Loader

import tensorflow as tf

flags = None

def init():

    global flags
    rospy.init_node('train_underwater_camera_model', anonymous=True)
    # load configuration
    config_filename = rospy.get_param('~config_filename')
    config = load(file(config_filename, 'r'), Loader=Loader)

    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
    flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
    flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
    flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
    flags.DEFINE_integer("input_height", 480, "The size of image to use (will be center cropped). [108]")
    flags.DEFINE_integer("input_width", 640, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
    flags.DEFINE_integer("input_water_height", 1024, "The size of image to use (will be center cropped). [108]")
    flags.DEFINE_integer("input_water_width", 1360, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
    flags.DEFINE_integer("output_height", 48, "The size of the output images to produce [64]")
    flags.DEFINE_integer("output_width", 64, "The size of the output images to produce. If None, same value as output_height [None]")
    flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
    flags.DEFINE_float("max_depth", 1.5, "Dimension of image color. [3.0]")
    flags.DEFINE_string("water_dataset", config['water_dataset'], "The name of dataset [celebA, mnist, lsun]")
    flags.DEFINE_string("air_dataset",config['rgb_dataset'],"The name of dataset with air images")
    flags.DEFINE_string("depth_dataset",config['depth_dataset'],"The name of dataset with depth images")
    flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
    flags.DEFINE_string("checkpoint_dir", config['model_directory'], "Directory name to save the models [model]")
    flags.DEFINE_string("results_dir", "results", "Directory name to save the checkpoints [results]")
    flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
    flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
    flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
    flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
    flags.DEFINE_integer("num_samples",64, "True for visualizing, False for nothing [4000]")
    flags.DEFINE_integer("save_epoch",10, "The size of the output images to produce. If None, same value as output_height [None]")
    flags = flags.FLAGS
    
    if flags.input_width is None:
        flags.input_width = flags.input_height
    if flags.output_width is None:
        flags.output_width = flags.output_height

    if not os.path.exists(flags.checkpoint_dir):
        os.makedirs(flags.checkpoint_dir)

def main(_):
    
    global flags

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    with tf.Session(config=run_config) as sess:
        wgan = WGAN(
            sess,
            input_width=flags.input_width,
            input_height=flags.input_height,
            input_water_width=flags.input_water_width,
            input_water_height=flags.input_water_height,
            output_width=flags.output_width,
            output_height=flags.output_height,
            batch_size=flags.batch_size,
            c_dim=flags.c_dim,
            max_depth = flags.max_depth,
            save_epoch=flags.save_epoch,
            water_dataset_name=flags.water_dataset,
            air_dataset_name = flags.air_dataset,
            depth_dataset_name = flags.depth_dataset,
            input_fname_pattern=flags.input_fname_pattern,
            is_crop=flags.is_crop,
            checkpoint_dir=flags.checkpoint_dir,
            results_dir = flags.results_dir,
            sample_dir=flags.sample_dir,
            num_samples = flags.num_samples)

        if flags.is_train:
            print('TRAINING')
            wgan.train(flags)
        else:
            print('TESTING')
            if not wgan.load(flags.checkpoint_dir):
                raise Exception("[!] Train a model first, then run test mode")
            wgan.test(flags)
      
    rospy.spin()

if __name__ == '__main__':
    init()
    tf.app.run()
