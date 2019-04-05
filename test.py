
import tensorflow as tf
import numpy as np

mod = tf.load_op_library('./interleave_kernels.so')
tf.logging.set_verbosity( tf.logging.DEBUG )

config = tf.ConfigProto(log_device_placement = True)
config.graph_options.optimizer_options.opt_level = -1

in1 = tf.constant([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], shape = [1,3,3,1])
in2 = tf.constant([2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0], shape = [1,3,3,1])
in3 = tf.constant([3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0], shape = [1,3,3,1])
in4 = tf.constant([4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0], shape = [1,3,3,1])
		
with tf.Session(config = config):
	with tf.device('/gpu:0'):
		#in1 = np.reshape( [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], (1,3,3,1) )
		#in2 = np.reshape( [2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0], (1,3,3,1) )
		#in3 = np.reshape( [3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0], (1,3,3,1) )
		#in4 = np.reshape( [4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0], (1,3,3,1) )
		#tf.reshape(in1, [1,3,3,1])
		#tf.reshape(in2, [1,3,3,1])
		#tf.reshape(in3, [1,3,3,1])
		#tf.reshape(in4, [1,3,3,1])
		finish = mod.interleave(in1, in2, in3, in4).eval()
		print(finish)
		print(np.shape(finish))