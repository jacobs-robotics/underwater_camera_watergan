import os
import numpy as np
import rospy
import message_filters
import tensorflow as tf
from sensor_msgs.msg import Image, CameraInfo
import ros_numpy
import cv2
from cv_bridge import CvBridge, CvBridgeError

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from underwater_camera_model import WGAN

wgan = None
flags = None
is_initialized = False
underwater_image_pub = None
max_depth = 5.0 # maximum depth in meters to regard with underwater camera (clamp all computations to this depth if exceeded)

def init():

    global flags
    global underwater_image_pub
    global underwater_camera_info_pub

    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
    flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
    flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
    flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
    flags.DEFINE_integer("input_height", 480, "The size of image to use (will be center cropped). [108]")
    flags.DEFINE_integer("input_width", 640,
                         "The size of image to use (will be center cropped). If None, same value as input_height [None]")
    flags.DEFINE_integer("input_water_height", 1024, "The size of image to use (will be center cropped). [108]")
    flags.DEFINE_integer("input_water_width", 1360,
                         "The size of image to use (will be center cropped). If None, same value as input_height [None]")
    flags.DEFINE_integer("output_height", 48, "The size of the output images to produce [64]")
    flags.DEFINE_integer("output_width", 64,
                         "The size of the output images to produce. If None, same value as output_height [None]")
    flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
    flags.DEFINE_float("max_depth", 1.5, "Dimension of image color. [3.0]")
    flags.DEFINE_string("water_dataset", "water_images", "The name of dataset [celebA, mnist, lsun]")
    flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
    flags.DEFINE_string("checkpoint_dir", "model", "Directory name to save the models [model]")
    flags.DEFINE_string("results_dir", "results", "Directory name to save the checkpoints [results]")
    flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
    flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
    flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
    flags.DEFINE_integer("save_epoch", 10,
                         "The size of the output images to produce. If None, same value as output_height [None]")
    flags.DEFINE_boolean("use_batchsize_for_prediction", True, "Use batch size for prediction (True) or one sample only (False) [True]")
    flags = flags.FLAGS

    if flags.input_width is None:
        flags.input_width = flags.input_height
    if flags.output_width is None:
        flags.output_width = flags.output_height

    if not os.path.exists(flags.checkpoint_dir):
        os.makedirs(flags.checkpoint_dir)
    if not os.path.exists(flags.sample_dir):
        os.makedirs(flags.sample_dir)

    # initialize image subscribers and publisher
    rospy.init_node('underwater_camera_node', anonymous=True)
    image_sub = message_filters.Subscriber('/depth_camera/image_raw', Image)
    depth_sub = message_filters.Subscriber('/depth_camera/depth/image_raw', Image)
    camera_info_sub = message_filters.Subscriber('/depth_camera/camera_info', CameraInfo)
    underwater_image_pub = rospy.Publisher('/underwater_camera_watergan/image_raw', Image, queue_size=10)
    underwater_camera_info_pub = rospy.Publisher('/underwater_camera_watergan/camera_info', CameraInfo, queue_size=10)

    ts = message_filters.TimeSynchronizer([image_sub, depth_sub, camera_info_sub], 10)
    ts.registerCallback(camera_callback)

def camera_callback(rgb_image, depth_image, camera_info):
    global wgan
    global is_initialized
    global underwater_image_pub
    global underwater_camera_info_pub
    global max_depth

    if is_initialized:
        try:
            cv_rgb_image = CvBridge().imgmsg_to_cv2(rgb_image, "rgb8")
            data = ros_numpy.numpify(depth_image)
            # fill NaNs with maximum depth value
            # because simulated Kinect reports NaN where depth is higher than maximum hardware depth
            where_are_NaNs = np.isnan(data)
            data[where_are_NaNs] = max_depth
            #cv2.normalize(data, data, 0, 255, cv2.NORM_MINMAX)
            cv_depth_image = data

        except CvBridgeError as e:
            print(e)
        cv_underwater_image = wgan.predict(cv_rgb_image, cv_depth_image)
        #cv2.normalize(cv_underwater_image, cv_underwater_image, 0, 255, cv2.NORM_MINMAX)
        #cv_underwater_image = np.array(cv_underwater_image).astype(np.uint8)
        underwater_image = ros_numpy.msgify(Image, cv_underwater_image, "rgb8")
        underwater_image.header = rgb_image.header
        underwater_image_pub.publish(underwater_image)
        underwater_camera_info_pub.publish(camera_info)


def main(_):

    global wgan
    global flags
    global is_initialized

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
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
            max_depth=flags.max_depth,
            save_epoch=flags.save_epoch,
            water_dataset_name=flags.water_dataset,
            input_fname_pattern=flags.input_fname_pattern,
            is_crop=flags.is_crop,
            checkpoint_dir=flags.checkpoint_dir,
            results_dir=flags.results_dir,
            sample_dir=flags.sample_dir,
            use_batchsize_for_prediction=flags.use_batchsize_for_prediction
        )

        print('Running Underwater Camera ROS Node')
        wgan.load_model(flags)
        is_initialized = True

        rospy.spin()


if __name__ == '__main__':
    init()
    tf.app.run()
