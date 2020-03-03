import numpy as np
import tensorflow as tf


print("")
print("")

tf.debugging.set_log_device_placement(True)


#boxes = np.random.uniform(size=(1,500,1,4))
#scores = np.random.uniform(size=(1,500,1))

boxes = np.random.uniform(size=(2500,4))
scores = np.random.uniform(size=(2500))

bx = tf.constant(boxes, dtype=tf.float32)
sc = tf.constant(scores, dtype=tf.float32)

with tf.device("/GPU:0"):
	indx = tf.image.combined_non_max_suppression(bx,sc,max_output_size_per_class=500,max_total_size=500)
print("")
print("")
print("")
print(indx)
