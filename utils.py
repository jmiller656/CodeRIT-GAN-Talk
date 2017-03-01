import tensorflow as tf
import tensorflow.contrib.slim as slim
import faces
import numpy as np
import os
import cv2

def lrelu(x,leak=0.2,name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2* abs(x)

def generator(z,im_size,layers,initializer = tf.truncated_normal_initializer(stddev=0.002)):
	zP = slim.fully_connected(
	z,4*4*256,normalizer_fn=slim.batch_norm,
	activation_fn=tf.nn.relu,scope='g_project',
	weights_initializer=initializer)

	zCon = tf.reshape(zP,[-1,4,4,256])
	
	gen1 = slim.convolution2d_transpose(
	zCon,num_outputs=64,kernel_size=[5,5],stride=  
	[2,2],padding="SAME",normalizer_fn=slim.batch_norm,
	activation_fn=tf.nn.relu,scope='g_conv1',
	weights_initializer=initializer)
	
	gen2 = slim.convolution2d_transpose(
	gen1,num_outputs=32,kernel_size=[5,5],stride=  
	[2,2],padding="SAME",normalizer_fn=slim.batch_norm,
	activation_fn=tf.nn.relu,scope='g_conv2',
	weights_initializer=initializer)

	gen3 = slim.convolution2d_transpose(
	gen2,num_outputs=16,kernel_size=[5,5],stride=  
	[2,2],padding="SAME",normalizer_fn=slim.batch_norm,
	activation_fn=tf.nn.relu,scope='g_conv3',
	weights_initializer=initializer)

	gen4 = slim.convolution2d_transpose(
	gen3,num_outputs=8,kernel_size=[5,5],stride=  
	[2,2],padding="SAME",normalizer_fn=slim.batch_norm,
	activation_fn=tf.nn.relu,scope='g_conv4',
	weights_initializer=initializer)

	g_out = slim.convolution2d_transpose(
	gen4,num_outputs=layers,kernel_size=[im_size,im_size],padding="SAME",
        activation_fn=tf.nn.tanh,
        scope='g_out', weights_initializer=initializer)
	
	return g_out

def discriminator(bottom,reuse=False,initializer = tf.truncated_normal_initializer(stddev=0.002)):
	net = bottom
	size = 16
	kernel = [4,4]
	stride = [2,2]
	padding = "SAME"
	for i in range(3):
		net = slim.convolution2d(net,size,kernel,stride=stride,
		padding=padding,biases_initializer=None,activation_fn=lrelu,
		reuse=reuse,scope='d_conv_'+str(i),weights_initializer=initializer)

	net = slim.fully_connected(slim.flatten(net),1,activation_fn=tf.nn.sigmoid,
	reuse=reuse,scope='d_out',weights_initializer=initializer)

	return net

def train(real_in,fake_in,update_G,g_loss,update_D,d_loss,sess,im_size=64,batch_size=1,z_size=100):
	fake_data = np.random.uniform(0.0,1.0,size=[batch_size,z_size])
	real_data = faces.get_batch(im_size,batch_size)
	_,dLoss = sess.run([update_D,d_loss],feed_dict={fake_in:fake_data,real_in:real_data})
	_,gLoss = sess.run([update_G,g_loss],feed_dict={fake_in:fake_data})
	_,gLoss = sess.run([update_G,g_loss],feed_dict={fake_in:fake_data})
	return gLoss,dLoss
	

def sample(z_size,spongebob,z_in,sess,sample_directory='./sample',i=0):
	test_input = np.random.uniform(0.0,1.0,size=[1,z_size]).astype(np.float32)
	newZ = sess.run(spongebob,feed_dict={z_in: test_input})
	newZ *= 255			
	newZ += 128
	if not os.path.exists(sample_directory):
		os.makedirs(sample_directory)
	cv2.imwrite(sample_directory+'/gen'+str(i)+'.png',newZ[0])



