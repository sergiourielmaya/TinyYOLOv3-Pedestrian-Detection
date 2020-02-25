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
bias_conv = False


#prueba = ReadModelConfig("yolov3-tiny.cfg")
#print(prueba)