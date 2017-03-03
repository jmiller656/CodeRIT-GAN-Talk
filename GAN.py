import tensorflow as tf
import os
import utils
tf.reset_default_graph()
batch_size = 100
iterations = 100000
sample_directory = './figs'
model_directory = './models'
im_size = 64
layers = 3
z_size = 100
z_in = tf.placeholder(shape=[batch_size,z_size],dtype=tf.float32)
real_in = tf.placeholder(shape=[batch_size,im_size,im_size,layers],dtype=tf.float32)
spongebob = utils.generator(z_in,im_size,layers,batch_size)
reggie = utils.discriminator(real_in,batch_size)
reggie2 = utils.discriminator(spongebob,batch_size,reuse=True)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(reggie,tf.ones_like(reggie)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(reggie2,tf.zeros_like(reggie2)))
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(reggie2,tf.ones_like(reggie2)))
tvars = tf.trainable_variables()
spongebob_params = []
reggie_params = []
for var in tvars:
	if 'g_' in var.name:
		spongebob_params.append(var)
	elif 'd_' in var.name:
		reggie_params.append(var)


trainerD = tf.train.AdamOptimizer(2e-4,beta1=0.5).minimize(d_loss,var_list=reggie_params)
trainerG = tf.train.AdamOptimizer(2e-4,beta1=0.5).minimize(g_loss,var_list=spongebob_params)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	saver = tf.train.Saver()
	"""try:
		saver.restore(sess, tf.train.latest_checkpoint("models"))
	except:
		print "Previous weights not found"
	"""
	for i in range(iterations):

		gLoss,dLoss = utils.train(real_in,z_in,trainerG,g_loss,trainerD,d_loss,sess,im_size=im_size,batch_size=batch_size,z_size=z_size)
		if i%1 ==0:
			print "Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss)
			utils.sample(z_size,batch_size,spongebob,z_in,sess,i=i%5000)
			# sample here
		if i % 1000 == 0 and i != 0:
			if not os.path.exists(model_directory):
				os.makedirs(model_directory)
			saver.save(sess,model_directory+'/model-'+str(i)+'.ckpt')
			print "Saved Model"







