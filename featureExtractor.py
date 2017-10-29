import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import constant
from models import resnet_v2 as resnet


class FrameFeatures():
    def __init__(self, height, width, channel=3, architecture=constant.RESNET152, trainable=False):
        self.height = height
        self.width = width
        self.channel = channel
        self.inputs = tf.placeholder(shape=[None, self.height, self.width, self.channel],
                                     name=constant.FRAME_FEATURE_INPUTS,
                                     dtype=tf.float32)
        if architecture == constant.RESNET152:
            with slim.arg_scope(resnet.resnet_arg_scope()):
                self.features, _ = resnet.resnet_v2_152(inputs=self.inputs,
                                                        #num_classes=1001,
                                                        global_pool=True,
                                                        is_training=trainable)
                self.features = tf.squeeze(self.features,[1,2])
        print('Resnet 152 model loaded.')

    def restore_diff_scope(self, sess, scope=''):
        self.sess = sess
        net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'resnet_v2_152')
        assignments = []
        for i_var in net_vars:
            ckpt_var = tf.contrib.framework.load_variable(constant.PRETRAINED_ROOT + constant.RESNET_152_CKPT,
                                                          i_var.name.replace(scope, ''))
            assignments.append(i_var.assign(ckpt_var))
        sess.run(assignments)
        print('Resnet 152 checkpoints restored.')

    def restore(self, sess, scope='resnet_v2_152'):
        self.sess = sess
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        self.saver.restore(sess, constant.PRETRAINED_ROOT + constant.RESNET_152_CKPT)
        print('Resnet 152 checkpoints restored.')

    def getFeatures(self, images):
        # Images: [batch, height, width, channel]
        images = images / 127.5 - 1.0
        predictedFeatures = self.sess.run(self.features, feed_dict={self.inputs: images})
        return predictedFeatures


# Example of usage
if __name__ == '__main__':
    with tf.Session() as sess:
        with tf.name_scope(constant.RESNET152) as scope:
            features = FrameFeatures(225, 225)
            features.restore(sess)
            img = tf.read_file('./download.jpg')
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize_images(img, [225, 225])
            img = tf.expand_dims(img, axis=0)

            r1 = np.argmax(features.getFeatures(img.eval()))
            print(r1)
