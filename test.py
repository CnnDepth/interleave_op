
import tensorflow as tf
import numpy as np

mod = tf.load_op_library('./interleave_kernels.so')
tf.logging.set_verbosity( tf.logging.DEBUG )

config = tf.ConfigProto(log_device_placement = True)
config.graph_options.optimizer_options.opt_level = -1
		
with tf.device('/cpu:0'):
	in1 = tf.constant([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], shape = [1,3,3,1])
	in2 = tf.constant([2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0], shape = [1,3,3,1])
	in3 = tf.constant([3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0], shape = [1,3,3,1])
	in4 = tf.constant([4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0], shape = [1,3,3,1])

	finish_cpu = mod.interleave(in1, in2, in3, in4)

with tf.device('/gpu:0'):
	in1 = tf.constant([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], shape = [1,3,3,1])
	in2 = tf.constant([2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0], shape = [1,3,3,1])
	in3 = tf.constant([3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0], shape = [1,3,3,1])
	in4 = tf.constant([4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0], shape = [1,3,3,1])

	finish_gpu = mod.interleave(in1, in2, in3, in4)

	
sess = tf.Session()

#Test CPU version
print( sess.run(finish_cpu) )
print( finish_cpu )
print( np.shape(finish_cpu) )

#Test GPU version
print( sess.run(finish_gpu) )
print( finish_cpu )
print( np.shape(finish_gpu) )
