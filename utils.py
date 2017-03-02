import tensorflow as tf
import tensorflow.contrib.slim as slim
import faces
import numpy as np
import os
import cv2
import math


def lrelu(x,leak=0.2,name="lrelu"):
	"""with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2* abs(x)"""
	return tf.maximum(x,leak*x)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                    tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def conv2d(input_, output_dim, 
        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
        name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                    initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv

def deconv2d(input_, output_shape,
        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
        name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                    initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]],      initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def generator(z,im_size,layers,batch_size,initializer = tf.truncated_normal_initializer(stddev=0.002)):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = im_size, im_size
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')
        g_bn3 = batch_norm(name='g_bn3')

        # project `z` and reshape
        z_, h0_w, h0_b = linear(
            z, im_size*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        h0 = tf.reshape(
            z_, [-1, s_h16, s_w16, im_size * 8])
        h0 = tf.nn.relu(g_bn0(h0))

        h1, h1_w, h1_b = deconv2d(
            h0, [batch_size, s_h8, s_w8, im_size*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(g_bn1(h1))

        h2, h2_w, h2_b = deconv2d(
            h1, [batch_size, s_h4, s_w4, im_size*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(g_bn2(h2))

        h3, h3_w, h3_b = deconv2d(
            h2, [batch_size, s_h2, s_w2, im_size*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(g_bn3(h3))

        h4, h4_w, h4_b = deconv2d(
            h3, [batch_size, s_h, s_w, layers], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
	

def discriminator(bottom,batch_size,reuse=False,initializer = tf.truncated_normal_initializer(stddev=0.002)):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        size = 64
        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')
        d_bn3 = batch_norm(name='d_bn3')

        h0 = lrelu(conv2d(bottom, size, name='d_h0_conv'))
        h1 = lrelu(d_bn1(conv2d(h0, size*2, name='d_h1_conv')))
        h2 = lrelu(d_bn2(conv2d(h1, size*4, name='d_h2_conv')))
        h3 = lrelu(d_bn3(conv2d(h2, size*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')

        return h4	# -> this breaks things tf.nn.sigmoid(h4)

def train(real_in,fake_in,update_G,g_loss,update_D,d_loss,sess,im_size=64,batch_size=1,z_size=100):
	fake_data = np.random.uniform(-1,1,size=[batch_size,z_size])
	real_data = faces.get_batch(im_size,batch_size)
	_,dLoss = sess.run([update_D,d_loss],feed_dict={fake_in:fake_data,real_in:real_data})
	_,gLoss = sess.run([update_G,g_loss],feed_dict={fake_in:fake_data})
	_,gLoss = sess.run([update_G,g_loss],feed_dict={fake_in:fake_data})
	return gLoss,dLoss
	

def sample(z_size,batch_size,spongebob,z_in,sess,sample_directory='./sample',i=0):
	test_input = np.random.uniform(-1,1,size=[batch_size,z_size]).astype(np.float32)
	newZ = sess.run(spongebob,feed_dict={z_in: test_input})
	newZ *= 255.0			
	newZ += 128.0
	if not os.path.exists(sample_directory):
		os.makedirs(sample_directory)
	cv2.imwrite(sample_directory+'/gen'+str(i)+'.png',newZ[0])



