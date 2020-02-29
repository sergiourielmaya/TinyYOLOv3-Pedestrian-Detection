#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on TUe Feb 25 16:11:36 2020

@author: sergio
"""

from YOLOblocks import TinyYOLOv3,ReadModelConfig
import numpy as np
import matplotlib.pyplot as plt



mod = TinyYOLOv3(1,anchor_boxes=[[0.2,0.5],[0.3,0.8],[0.4,0.4],[0.2,0.5]])
mod.build(batch_input_shape=(None,416,416,3))
mod.summary()
print(mod.load_weights("yolov3-tiny.weights"))

sample_image = np.random.random((1,416,416,3))

import time

tiempo= []
for i in range(1000):
	if i%100==0:
		print(i)
	inicio = time.time()
	_ = mod(sample_image)
	fin = time.time()

	tiempo.append(fin-inicio)

print(np.median(tiempo))
print(1./np.median(tiempo))
print(np.mean(tiempo))
print(np.min(tiempo))

plt.plot(tiempo[1:])
#plt.hist(tiempo[1:],bins = 50)
plt.show()

'''
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