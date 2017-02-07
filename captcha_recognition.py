import tensorflow as tf
import cv2
import sys
from captcha_separate import separate_captcha
from image_preprocess import image_preprocess
from tensorflow.examples.tutorials.mnist import input_data


#weight initialization
def weight(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

#bias initialization
def bias(shape_bias):
	initial = tf.ones(shape_bias)/10
	return tf.Variable(initial)

#conv layer
#'SAME'mode means maintain the same size with apt padding
#'VALID'mode means no paddings
def conv2d(x,W):
	return tf.nn.conv2d(x,W, strides = [1,conv_stride,conv_stride,1],padding = 'SAME')

#max_pool layer
#[batch, height, width, channels]
def max_pool(x):
	return tf.nn.max_pool(x, ksize = [1,pool_kernal_height,pool_kernal_weight,1],\
		strides = [1,pool_stride,pool_stride,1], padding = 'SAME')


if __name__=='__main__':
	train_test_flag = sys.argv[1]
	#save path
	model_filepath = "mnist_model.ckpt"

	conv_stride=1
	pool_kernal_height = pool_kernal_weight = 2
	pool_stride = 2
	
	sess = tf.InteractiveSession()

	# original input
	x = tf.placeholder(tf.float32, [None,784])
	#Grounf truth
	y_ = tf.placeholder(tf.float32, [None, 10])

	#first layer
	W_conv1 = weight([3,3,1,16])
	b_conv1 = bias([16])
	x_image = tf.reshape(x, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
	W_conv12 = weight([3,3,16,32])
	b_conv12 = bias([32])
	h_conv12 = tf.nn.relu(conv2d(h_conv1,W_conv12) + b_conv12)
	h_pool1 = max_pool(h_conv12)

	#second layer
	W_conv2 = weight([5,5,32,64])
	b_conv2 = bias([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
	h_pool2 = max_pool(h_conv2)

	#fc layer1
	#image size is 7*7 after pooling 2 times
	W_fc1 = weight([7*7*64, 1024])
	b_fc1 = bias([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	#dropout
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	#final fc2 layer
	W_fc2 = weight([1024, 10])
	b_fc2 = bias([10])
	h_fc2 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	saver = tf.train.Saver()
	if train_test_flag == 'train':
		#download mnist dataset
		mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
		#cost function
		#reduction_indices=[1] means add by row ;[0] means add by column
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(h_fc2), \
			reduction_indices=[1]))
		#train
		lr = 0.0001
		train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
		correct_pred = tf.equal(tf.argmax(h_fc2, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		sess.run(tf.initialize_all_variables())

		for i in range(5001):
			batch = mnist.train.next_batch(64)
			sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
			#show the result every 50 steps
			if i%50 == 0:
				#train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
				acc = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:1})
				test_acc = sess.run(accuracy, \
					feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1})
				print 'training step:%d'%i, 'trainig accuracy:%5f'%acc, 'test accuracy:%5f'%test_acc
		#save the model
			if i%1000 ==0:
				save = saver.save(sess, model_filepath+'_%d'%i)
	else:
		#restore from disk
		saver.restore(sess, model_filepath)
		print 'model restored'
		pred = tf.argmax(h_fc2,1)

		captcha_path = sys.argv[2]
		#saperate 4 numbers
		image0,image1,image2,image3 = \
			separate_captcha(captcha_path)
		#number 1
		image_A = image_preprocess(image0)
		prediction_A = sess.run(pred, feed_dict={x:image_A, keep_prob:1})
		#number 2
		image_B = image_preprocess(image1)
		prediction_B = sess.run(pred, feed_dict={x:image_B, keep_prob:1})
		#number 3
		image_C = image_preprocess(image2)
		prediction_C = sess.run(pred, feed_dict={x:image_C, keep_prob:1})
		#number 1
		image_D = image_preprocess(image3)
		prediction_D = sess.run(pred, feed_dict={x:image_D, keep_prob:1})
		print prediction_A,prediction_B,prediction_C,prediction_D
