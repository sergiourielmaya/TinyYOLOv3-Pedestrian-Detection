#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:18:29 2020

@author: sergio
"""

#Import Tensorflow sub classes
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


#Import basic blocks
from tensorflow.keras.layers import Conv2D,BatchNormalization,ZeroPadding2D,MaxPool2D, LeakyReLU,UpSampling2D,Concatenate

import numpy as np
import matplotlib as plt


class BasicBlock(Layer):
    def __init__(self,num_filters=3,
                 kernel_size=3,
                 max_pooling=True,
                 max_pool_stride=2,
                 activation= LeakyReLU,
                 batch_norm=True,
                 root=False,
                 **kwargs):
        super(BasicBlock,self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.max_pooling = max_pooling
        self.max_pool_stride = max_pool_stride
        self.root = root

        #pad = (kernel_size-1)//2
        self.conv = Conv2D(self.num_filters,
                           kernel_size=self.kernel_size,
                           strides=(np.int64(1),np.int64(1)),
                           padding="same",
                           use_bias = not self.batch_norm)

        if self.batch_norm:
            self.bn = BatchNormalization(axis = -1)

        if self.max_pool_stride == 1:
            #Padding Order : ((top_pad,botton_pad),(left_pad,right_pad))
            self.fixed_padding = ZeroPadding2D(padding = ((0,1),(0,1)))

        if self.max_pooling:
            self.max_pool = MaxPool2D(pool_size =(2,2), strides= (self.max_pool_stride,self.max_pool_stride))

        self.act = LeakyReLU(0.1)
    
    def call(self,X):

        x= self.conv(X)

        if self.batch_norm:
            x_root = self.bn(x)
        else:
            x_root = x

        x = self.act(x_root)

        if self.max_pool_stride ==1:
            x = self.fixed_padding(x)

        if self.max_pooling:
            x=self.max_pool(x)

        if self.root:
            return x,x_root
        else:
            return x

class PredictionLayer(Layer):

    def __init__(self,anchor_boxes,conf_thresh,grid_size):
        super(PredictionLayer,self).__init__()

        self.anchors = anchor_boxes
        self.conf_thresh = conf_thresh
        self.grid_size = grid_size
        self.cx = tf.tile(tf.reshape(tf.repeat(tf.linspace(0.,1,grid_size)[tf.newaxis,:],grid_size,axis=0),shape=(grid_size,grid_size))[:,:,tf.newaxis],tf.constant([1,1,2]))[:,:,:,tf.newaxis] 
        self.cy = tf.tile(tf.reshape(tf.repeat(tf.linspace(0.,1,grid_size)[:,tf.newaxis],grid_size,axis=0),shape=(grid_size,grid_size))[:,:,tf.newaxis],tf.constant([1,1,2]))[:,:,:,tf.newaxis]
    def call(self, X):

        box_x,box_y,box_w,box_h,objectness = tf.split(X, (1,1,1,1,1), axis=-1)

        box_x = tf.sigmoid(box_x) + self.cx
        box_y = tf.sigmoid(box_y) + self.cy
        box_w = self.anchors[:,0]*tf.exp(box_w)
        box_h = self.anchors[:,1]*tf.exp(box_h)
        objectness = tf.sigmoid(objectness)

        return (box_x,box_y,box_w,box_h,objectness)

#tf.keras.backend.clear_session()

class TinyYOLOv3(Model):

    def __init__(self,num_classes,bouding_boxes,**kwargs):
        super(TinyYOLOv3,self).__init__()

        self.block1 = BasicBlock(num_filters = 16, kernel_size = 3)
        self.block2 = BasicBlock(num_filters = 32, kernel_size = 3)
        self.block3 = BasicBlock(num_filters = 64, kernel_size = 3)
        self.block4 = BasicBlock(num_filters = 128, kernel_size = 3)
        self.block5 = BasicBlock(num_filters = 256, kernel_size = 3, root = True)
        self.block6 = BasicBlock(num_filters = 512, kernel_size = 3,max_pool_stride=1)
        self.block7 = BasicBlock(num_filters = 1024, kernel_size = 3,max_pooling=False)
        self.block8 = BasicBlock(num_filters = 256, kernel_size = 1,max_pooling=False)
        self.block9 = BasicBlock(num_filters = 512, kernel_size = 3,max_pooling=False)
        self.block10 = BasicBlock(num_filters = 10, kernel_size = 1, batch_norm =False, max_pooling=False, activation = None)
        self.block11 = BasicBlock(num_filters = 128,kernel_size = 1,max_pooling = False)
        self.block12 = BasicBlock(num_filters = 256,kernel_size = 3,max_pooling = False)
        self.block13 = BasicBlock(num_filters = 10, kernel_size = 1, batch_norm =False, max_pooling=False, activation = None)
        self.concat_block = Concatenate(axis=-1)
        self.upsamp = UpSampling2D(size = 2,interpolation = "nearest")
        self.yolo1 = PredictionLayer(np.array([[0.2,0.5],[0.3,0.8]]),conf_thresh=0.5,grid_size=13)
        self.yolo2 = PredictionLayer(np.array([[0.2,0.5],[0.3,0.8]]),conf_thresh=0.5,grid_size=26)

        #self.final_yolo1 = YOLO_Layer(80,torch.tensor(np.array([[81.,82.],[135.,169.],[344.,319.]])),0.5,13)
        #self.final_yolo2 = YOLO_Layer(80,torch.tensor(np.array([[10.,14.],[23.,27.],[37.,58.]])),0.5,26)

    def build(self,batch_input_shape):
        super().build(batch_input_shape)
    @tf.function
    def call(self,inputs):

        yolo1 = self.block1(inputs)
        yolo1 = self.block2(yolo1)
        yolo1 = self.block3(yolo1)
        yolo1 = self.block4(yolo1)
        yolo1,root = self.block5(yolo1)
        yolo1 = self.block6(yolo1)
        yolo1 = self.block7(yolo1)
        yolo1_branch = self.block8(yolo1)
        yolo1 = self.block9(yolo1_branch)
        yolo1 = self.block10(yolo1)
        yolo2 = self.block11(yolo1_branch)
        yolo2 = self.upsamp(yolo2)
        yolo2 = self.concat_block([yolo2,root])
        yolo2 = self.block12(yolo2)
        yolo2 = self.block13(yolo2)
        yolo1 = tf.reshape(yolo1,(-1,13,13,2,5))
        yolo2 = tf.reshape(yolo2,(-1,26,26,2,5))

        output1 = self.yolo1(yolo1)
        output2 = self.yolo2(yolo2)
        return (output1,output2)


a = TinyYOLOv3(num_classes = 1,bouding_boxes="prueba")

sample_image = np.float32(np.random.random(size=(16,416,416,3)))
#test = a(inputs = sample_image)

a.build(batch_input_shape=(None,416,416,3))

print(a.summary())

#@tf.function
def prueba(x):
	return a(x)

import time

tiempo= []
for i in range(1000):
	if i%100==0:
		print(i)
	inicio = time.time()
	aux1,aux2 = a(sample_image)
	fin = time.time()

	tiempo.append(fin-inicio)


import numpy as np
print(aux1.shape)
print(aux2.shape)
print(np.median(tiempo))
print(np.mean(tiempo))


#prueba = PredictionLayer(np.array([[0.2,0.5],[0.3,0.8]]),conf_thresh=0.5,grid_size=26)
#prueba.build(batch_input_shape=(None,26,26,2,5))
#print(prueba.summary)
