import tensorflow as tf
import numpy as np

mod = tf.load_op_library('./interleave_kernels.so')
tf.logging.set_verbosity( tf.logging.INFO )

config = tf.ConfigProto(log_device_placement = True)
config.graph_options.optimizer_options.opt_level = -1

in1_np = np.ones((4, 3, 3, 1))
in2_np = np.ones((4, 3, 3, 1)) * 2
in3_np = np.ones((4, 3, 3, 1)) * 3
in4_np = np.ones((4, 3, 3, 1)) * 4

in1 = tf.constant(in1_np)
in2 = tf.constant(in2_np)
in3 = tf.constant(in3_np)
in4 = tf.constant(in4_np)
finish_cpu = mod.interleave(in1, in2, in3, in4)

sess = tf.Session(config=config)
f_cpu = sess.run(finish_cpu)