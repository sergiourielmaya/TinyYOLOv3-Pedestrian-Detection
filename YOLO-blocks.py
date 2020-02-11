#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:18:29 2020

@author: sergio
"""

#Import Tensorflow sub classes
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model



#Import basic blocks
from tensorflow.keras.layers import Conv2D,BatchNormalization,ZeroPadding2D,MaxPool2D, LeakyReLU,UpSampling2D,Concatenate

import numpy as np
import matplotlib as plt


class BasicBlock(Layer):
    def __init__(self,num_filters=3, 
                 kernel_size=3,
                 max_pooling=True,
                 max_pool_stride=2,
                 activation=LeakyReLU,
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
                           strides=1,
                           padding="same",
                           use_bias = not self.batch_norm)
        
        if self.batch_norm:
            self.bn = BatchNormalization(axis = -1)
            
        if self.max_pool_stride == 1:
            #Padding Order : ((top_pad,botton_pad),(left_pad,right_pad))
            self.fixed_padding = ZeroPadding2D(padding = ((0,1),(0,1)))
            
        if self.max_pooling:
            self.max_pool = MaxPool2D(pool_size =2, strides= self.max_pool_stride)
            
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

        if self.root:
            return x,x_root
        else:
            return x
        
        
    
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
        self.block10 = BasicBlock(num_filters = 255, kernel_size = 1, batch_norm =False, max_pooling=False, activation = None)
        self.block11 = BasicBlock(num_filters = 128,kernel_size = 1,max_pooling = False)
        self.block12 = BasicBlock(num_filters = 256,kernel_size = 3,max_pooling = False)
        self.block13 = BasicBlock(num_filters = 255, kernel_size = 1, batch_norm =False, max_pooling=False, activation = None)
        self.concat_block = Concatenate(axis=1)
        self.upsamp = UpSampling2D(size = 2,interpolation = "nearest")
        
        #self.final_yolo1 = YOLO_Layer(80,torch.tensor(np.array([[81.,82.],[135.,169.],[344.,319.]])),0.5,13)
        #self.final_yolo2 = YOLO_Layer(80,torch.tensor(np.array([[10.,14.],[23.,27.],[37.,58.]])),0.5,26)
        
    def build(self,batch_input_shape):
        super().build(batch_input_shape)
    
    def call(self,inputs):
        
            
        #start = time.time()
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
        yolo2 = self.upsamp(yolo2, scale_factor = 2, mode = "nearest")
        yolo2 = self.concat_block([yolo2,root])
        yolo2 = self.block12(yolo2)
        yolo2 = self.block13(yolo2)
        #finish = time.time()      
        
        return self.concat_block([yolo1,yolo2])
    
a = TinyYOLOv3(num_classes = 1,bouding_boxes="prueba")

sample_image = np.random.random(size=(416,416,16,3))
test = a(inputs = sample_image)

    
        
        
        
        
    

