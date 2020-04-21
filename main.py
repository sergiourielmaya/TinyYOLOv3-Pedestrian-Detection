#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on TUe Feb 25 16:11:36 2020

@author: sergio
"""


from YOLOblocks import TinyYOLOv3,ReadModelConfig
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from skimage.io import imread,imshow
from skimage.transform import resize 
import time
from tensorflow.compat.v1.image import decode_image



img_raw = tf.image.decode_image(
            open("dog.jpg", 'rb').read(), channels=3)
img = tf.expand_dims(img_raw, 0)
img = tf.image.resize(img, (416, 416))/255

#anchors =[[10/416,14/416],[23/416,27/416],[37/416,58/416],[81/416,82/416],[135/416,169/416],[344/416,319/416]]
anchors =[[10/416,14/416],[23/416,27/416],[37/416,58/416],[81/416,82/416]]

mod = TinyYOLOv3(1,anchor_boxes=anchors)

mod.build(batch_input_shape=(None,416,416,3))
mod.summary()
print(mod.load_weights("yolov3-tiny.weights"))

#img = np.float32(img[np.newaxis,:,:,:])
#print(img)

tiempo=[]
with tf.device("GPU:0"):
    for i in range(1000):
        inicio = time.time()
        output = mod(img)
        fin = time.time()
        tiempo.append(fin-inicio)
        #print(fin-inicio)

print(output)

print("Tiempo medio: ",np.median(tiempo))
print("FPS: ", 1./np.median(tiempo))
print("Tiempo promedio: ",np.mean(tiempo))
print("Tiempo mínimo: ",np.min(tiempo))
print("tiempo máximo: ",np.max(tiempo))


plt.plot(tiempo[1:])
plt.ylabel("Time (ms)")
plt.show()

'''
sample_image = np.random.random((1,416,416,3))
print(sample_image.dtype)



import time

tiempo= []
for i in range(100):
	if i%1000==0:
		print(i)
	inicio = time.time()
	salida = mod(img)
	fin = time.time()

	tiempo.append(fin-inicio)



print(np.median(tiempo))
print(1./np.median(tiempo))
print(np.mean(tiempo))
print(np.min(tiempo))

inicio = time.time()
_ = mod(sample_image)
fin= time.time()
print("TIEMPO FINAL",fin-inicio)

plt.plot(tiempo[1:])
plt.ylabel("Time (ms)")

#plt.hist(tiempo[1:],bins = 50)
plt.show()

print(salida)

print("Procedemos a guardar el Modelo")
tf.saved_model.save(mod,"saved_model")


from tensorflow.python.compiler.tensorrt import trt_convert as trt
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(precision_mode="FP32")

converter = trt.TrtGraphConverterV2(input_saved_model_dir="saved_model",conversion_params=conversion_params)
converter.convert()
converter.save("output_saved_model_dir")


b = TinyConvnet(80,None)
b.build(batch_input_shape=(None,416,416,3))
print(b.summary())
print(weights.shape)
total_param = b.load_weights("yolov3-tiny.weights")
print(total_param)
 

names = [weight.name for layer in b.layers for weight in layer.weights]
weights = b.get_weights()

for name,weights in zip(names,weights):
    #print(name,weights.shape)
    pass

#sub_model = b.get_layer("Block1").get_weights()

for layer in b.layers:
    print("")
    print("Nuevo bloque")
    for  j in layer.weights:
        print(j.name,j.shape)
        #print(j.name)
    #print(layer.shape)
    pass

for layer in b.layers:
    print(len(layer.get_weights()))

#aux = 
#print(b.get_layer("Block1").get_weights())

#prueba = ReadModelConfig("yolov3-tiny.cfg")
#print(prueba)

'''