#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on TUe Feb 25 16:11:36 2020

@author: sergio
"""

from YOLOblocks import TinyConvnet,ReadModelConfig
import numpy as np

fp = open("yolov3-tiny.weights", "rb")
header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values
header_info = header
seen = header[3] #Número de imágenes totales para el entrenamiento
weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
fp.close()



b = TinyConvnet(80,None)
b.build(batch_input_shape=(None,416,416,3))
print(b.summary())
print(weights.shape)
b.load_weights("yolov3-tiny.weights")



'''

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