
import numpy as np
import tensorflow as tf
import time
boxes = np.random.uniform(size=(500, 4))
scores = np.random.uniform(size=(500,))

bx = tf.constant(boxes, dtype=tf.float32)
sc = tf.constant(scores, dtype=tf.float32)

@tf.function
def function(b,s,num):
        indx = tf.image.non_max_suppression(b,s,num)
        return indx

t = []
for i in range(1000):
	with tf.device("GPU:0"):
		inicio = time.time()
		indx = function(bx, sc, 1000)
		fin = time.time()
		t.append(fin-inicio)
print("")
print("Tiempo promedio de ejecución en GPU",np.mean(t))
print("Tiempo medio de ejecución en GPU",np.median(t))

t=[]
for i in range(1000):
	with tf.device("CPU:0"):
		inicio = time.time()
		indx = tf.image.non_max_suppression(bx,sc,1000)
		fin = time.time()
		t.append(fin-inicio)
print("")
print("Tiempo promedio de ejecución en CPU",np.mean(t))
print("Tiempo medio de ejecución en CPU",np.median(t))

