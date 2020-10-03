#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:18:29 2020

@author: sergio
"""

#Import Tensorflow sub classes
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss,BinaryCrossentropy

import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.image import non_max_suppression_with_scores,combined_non_max_suppression
#from tensorflow.image import non_max_suppression_v2

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8) #Allocate more memory to Tensorflow
#Arregla un bug donde marca un error con CUDA
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#Import bloques basicos
from tensorflow.keras.layers import Conv2D,BatchNormalization,ZeroPadding2D,MaxPool2D, LeakyReLU,UpSampling2D,Concatenate
import numpy as np
import time


class BasicBlock(Layer):
    '''
    Clase que define el Layer "Basic Block", un boque que utiliaz operadores: covolución, Max Pooling,Batch Normalization, Leaky ReLU.
    Argumentos
    num_filter: Entero, número de filtros de la operación de convolución, default 3
    kernel_size: Entero, tamaño de los filtros de la capa de convolución, default 3
    max_pooling: Booleano, Si despupes de la operación de convolución existe un Max pooling, default True
    max_pool_stride: ENtero, EL valor de stride de la operación de max_pooling, si hay.
    activation: TensorFlow Class, FUnción de activación a utilizar en la capa de convolución.
    batch_norm: Booleano, Si la capa tiene la operación Batch Normalization
    root: Booleano, Indica si exite la salida para la piramide de características
    '''

    def __init__(self,num_filters=3,
                 kernel_size=3,
                 max_pooling=True,
                 max_pool_stride=2,
                 activation= LeakyReLU(0.1),
                 batch_norm=True,
                 root=False,
                 name = None,
                 bn_train_state= False,
                 **kwargs):
        super(BasicBlock,self).__init__(name=name,**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.max_pooling = max_pooling
        self.max_pool_stride = max_pool_stride
        self.root = root
        self.bn_train_state= bn_train_state
        self.conv = Conv2D(self.num_filters,
                           kernel_size=self.kernel_size,
                           strides=(np.int64(1),np.int64(1)),
                           padding="same",
                           use_bias = not self.batch_norm)

        if self.batch_norm:
            self.bn = BatchNormalization(axis = -1)
        else:
            self.bn = None

        if self.max_pool_stride == 1:
            #Padding Order : ((top_pad,botton_pad),(left_pad,right_pad))
            self.fixed_padding = ZeroPadding2D(padding = ((0,1),(0,1)))
        else:
            self.fixed_padding = None

        if self.max_pooling:
            self.max_pool = MaxPool2D(pool_size =(2,2), strides= (self.max_pool_stride,self.max_pool_stride))
        else:
            self.max_pool = None

        self.act = activation
    
    def call(self,X,training = None):

        x= self.conv(X)

        if self.batch_norm:
            x_root = self.bn(x,training=False)
        else:
            x_root = x

        if self.act != None:
            x=x_root = self.act(x_root)

        if self.max_pool_stride ==1:
            x = self.fixed_padding(x)

        if self.max_pooling:
            x=self.max_pool(x)

        if self.root:
            return x,x_root
        else:
            return x

        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
            'num_layers': self.num_layers,
            'units': self.units,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
        })
        return config
    def get_config(self):

        base_config = super().get_config().copy()
        base_config.update({"num_filters": self.num_filters,
        "kernel_size":self.kernel_size,
        "batch_norm":self.batch_norm,
        "max_pooling":self.max_pooling,
        "max_pool_stride":self.max_pool_stride,
        "root":self.root,
        "conv":self.conv,
        "bn":self.bn,
        "fixed_padding":self.fixed_padding,
        "max_pool":self.max_pool,
        "act":self.act})

        return base_config

class PredictionLayer(Layer):
    '''
    Clase de la clase Prediction Layer. ESta capa calcula las predicciones de la red, calcula las coordenadas (x,y) y las dimensiones (w,h)
    del Bouding box. Aplica finalemnte Non max supression para eliminar los bouding boxes redundantes.
    Argumentos:
    Anchor_boxes: List, Lista de los Anchorx Boxes para la PredictionLayer, para el caso de TInyYOLOv3-pedestrian son 2 anchor boxes
    conf_thresh: Float [0,1] Umbral para el algoritmo de NMS.
    grid_size: Integer, Tamaño del tensor de que recibe la capa como entrada , (grid_size,grid_size,num_anchors*(5+num_classes))
    num_classes: Entero, número de clases a detectar, default 1
    '''

    def __init__(self,anchor_boxes,grid_size,conf_thresh,name=None,training=True,**kwargs):
        super(PredictionLayer,self).__init__(name=name,**kwargs)
        self.num_anchors = len(anchor_boxes)
        self.training = training
        self.final_conv_length = 5
        self.anchors_boxes = anchor_boxes #a list of list
        self.conf_thresh = conf_thresh
        self.grid_size = grid_size
        #Anchor boxes en forma matricial de tamaño (grid_size*grid_size*anchors,2)
        self.anchors_matrix =tf.cast(tf.tile(anchor_boxes,[self.grid_size*self.grid_size,1]),dtype=tf.float32)
        #print(self.anchors_matrix)
        x = tf.range(self.grid_size, dtype=tf.float32)
        y = tf.range(self.grid_size, dtype=tf.float32)
        x_offset, y_offset = tf.meshgrid(x, y)
        x_offset = tf.reshape(x_offset, (-1, 1))
        y_offset = tf.reshape(y_offset, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.tile(x_y_offset, [1, self.num_anchors])
        x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])     
        #x_y_offset es de tamaño (None,grid_size*grid_size*num_anchors,2) y asi ya es invariante del tamaño del grid.
        self.x_y_offset = x_y_offset

        self.strides = 1./ self.grid_size

    def call(self, X):

        #Se redimensiona la entrada para tener dimensiones (Batch_size,grid_size*grid_size*anchors,()
        X = tf.reshape(X,[-1,self.grid_size*self.grid_size*self.num_anchors,self.final_conv_length])
        #print("Nuevas dimensiones del tensor de entrada: [Batch_size,grid_size*grid_size*anchors, 5]",X.shape)
        #print("Se redimensiona la salida de la CNN a:",X.shape)

        box_xy,box_wh,objectness = tf.split(X, [2,2,1], axis=-1)

        #print("Tensor para cada grid con las coordenadas de x e y",box_xy.shape)
        #print("Tensor para cada grid con las coordendas de w y h",box_wh.shape)
        #print("Tensor de offset en función de la posición de la imagen de cada grid unit", self.x_y_offset.shape)
        #print("")
        #print(box_wh)
        #print(self.anchors_matrix)
        box_xy = tf.sigmoid(box_xy)
        box_xy = (box_xy + self.x_y_offset)*self.strides #Se encontra la coordenada (x,y) global del bouding box
        box_wh = tf.exp(box_wh) * self.anchors_matrix #Se encuenta el ancho (eje X) y alto (eje Y) para cada bouding box
        objectness = tf.sigmoid(objectness) # Se encuentra la probabilidad de cada bouding box de que sea una persona

        output = tf.concat([box_xy,box_wh,objectness],axis=-1)

        return output

    def get_config(self):
        base_config = super().get_config().copy()
        base_config.update({"num_anchors": self.num_anchors,
        "training":self.training,
        "final_conv_length":self.final_conv_length,
        "anchors_boxes":str(self.anchors_boxes),
        "conf_thresh":self.conf_thresh,
        "anchors_matrix":str(self.anchors_matrix.numpy),
        "x_y_offset":str(self.x_y_offset.numpy),
        "strides":self.strides})        
        return base_config

class NMSLayer(Layer):
    '''
    Clase de la capa NMS Layer. Esta capa obtiene las coordenadas de los bouding box esquina superior izquierda y la esquina inferior derecha.
    También aplica el algoritmo de Non Max Supression sobre todos los bouding boxes
    Argumentos:
    num_classes: Entero, número de clases a detectar, default 1
    '''

    def __init__(self,num_classes=1,iou_thresh=0.5,max_output_size=10, name=None,obj_threshold=0.5,**kwargs):
        super(NMSLayer,self).__init__(name=name,**kwargs)  
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.obj_threshold = obj_threshold
        self.max_output_size = max_output_size

    @tf.function
    def call(self,inputs,):

        if self.num_classes==1:
            center_x,center_y,width,height,objectness = tf.split(inputs,[1,1,1,1,1],axis=-1)
            #print("Solo es una clase")
        else: 
            center_x,center_y,width,height,objectness,classes = tf.split(inputs,[1,1,1,1,1,self.num_classes],axis=-1)

        #print("EL tamaño de clases es",classes.shape)

        top_left_x = center_x - (width / 2)
        top_left_y = center_y - (height / 2)
        bottom_right_x = center_x + (width / 2)
        bottom_right_y = center_y + (height / 2)
        
        #boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1)[:,:,tf.newaxis,:]
        #print(boxes.shape)

        boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1)
        return boxes
            
            #boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1)[:,:,tf.newaxis,:]
            #return tf.gather(boxes,tf.where(objectness[0,:,0]>0.5),axis=(1))[0,:,0,0,:]

            #output = combined_non_max_suppression(boxes,objectness,max_output_size_per_class=15,max_total_size=15,iou_threshold=0.6,score_threshold=0.5)
            #return output[0][0,:output[3][0],:]
            

            #selected_boxes = tf.gather(boxes, tf.where(objectness>0.5,axis=1)
            #selected_indices, selected_scores = non_max_suppression_with_scores(boxes[0,:,:],objectness[0,:,0],max_output_size=20,iou_threshold=0.6,score_threshold=0.5)#,soft_nms_sigma=0.5)
            #selected_boxes = tf.gather(boxes, selected_indices,axis=1)

           


    def get_config(self):
        base_config = super().get_config().copy()
        base_config.update({"num_classes": self.num_classes,
        "iou_thresh":self.iou_thresh,
        "obj_threshold":self.obj_threshold,
        "max_output_size":self.max_output_size})        
        return base_config

class TinyYOLOv3(Model):

    def __init__(self,anchor_boxes,**kwargs):
        super(TinyYOLOv3,self).__init__()
        self.num_anchors = len(anchor_boxes)

        self.filter_prediction_layer =5*(len(anchor_boxes)//2)

        self.block1 = BasicBlock(num_filters = 16, kernel_size = 3,name="BasicBlock1")
        self.block2 = BasicBlock(num_filters = 32, kernel_size = 3,name="BasicBlock2")
        self.block3 = BasicBlock(num_filters = 64, kernel_size = 3,name="BasicBlock3")
        self.block4 = BasicBlock(num_filters = 128, kernel_size = 3,name="BasicBlock4")
        self.block5 = BasicBlock(num_filters = 256, kernel_size = 3, root = True,name="BasicBlock5")
        self.block6 = BasicBlock(num_filters = 512, kernel_size = 3,max_pool_stride=1,name="BasicBlock6")
        self.block7 = BasicBlock(num_filters = 1024, kernel_size = 3,max_pooling=False,name="BasicBlock7")
        self.block8 = BasicBlock(num_filters = 256, kernel_size = 1,max_pooling=False,name="BasicBlock8")
        self.block9 = BasicBlock(num_filters = 512, kernel_size = 3,max_pooling=False,name="BasicBlock9")
        self.block10 = BasicBlock(num_filters = self.filter_prediction_layer, kernel_size = 1, batch_norm =False, max_pooling=False, activation = None,name="FinalBlock1")
        self.block11 = BasicBlock(num_filters = 128,kernel_size = 1,max_pooling = False,name="BasicBlock11")
        self.block12 = BasicBlock(num_filters = 256,kernel_size = 3,max_pooling = False,name="BasicBlock12")
        self.block13 = BasicBlock(num_filters = self.filter_prediction_layer, kernel_size = 1, batch_norm =False, max_pooling=False, activation = None,name="FinalBlock2")
        self.concat_block = Concatenate(axis=-1,name="Concatenate")
        self.upsamp = UpSampling2D(size = 2,interpolation = "nearest",name="Upsampling")
        self.yolo1 = PredictionLayer(anchor_boxes[len(anchor_boxes)//2:],grid_size=13,conf_thresh=0.5,name="Prediction1")
        self.yolo2 = PredictionLayer(anchor_boxes[:len(anchor_boxes)//2],grid_size=26,conf_thresh=0.5,name="Prediction2")
        self.concat_bbox = Concatenate(axis=1,name="Concatenate_BBOX")

    def build(self,batch_input_shape):
        super().build(batch_input_shape)

    #@tf.function(input_signature=[tf.TensorSpec(shape=(None,416,416,3), dtype=tf.float32)])
    def call(self,inputs,training=None):

        inicio = time.time()
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

        #print("Tiempo de la CNN",fin-inicio)
        #print("Salida de la CNN",yolo1.shape)
        #print("Salida de la CNN",yolo2.shape)
        inicio = time.time()
        yolo1 = tf.reshape(yolo1,(-1,13,13,self.num_anchors//2,self.filter_prediction_layer//(self.num_anchors//2)))
        yolo2 = tf.reshape(yolo2,(-1,26,26,self.num_anchors//2,self.filter_prediction_layer//(self.num_anchors//2)))
        #print("Reshape de la salida de la CNN",yolo1.shape)
        #print("Reshape de la salida de la CNN",yolo2.shape)
        
        output1 = self.yolo1(yolo1)
        output2 = self.yolo2(yolo2) 
        #print("Salida de la capa YOLO",output1.shape)
        #print("Salida de la capa YOLO",output2.shape) 
        output = self.concat_bbox([output1,output2]) 
        #print(output.shape)

        #print("Tiempo de la Capa YOLO: ",fin-inicio)
        #print(output.shape)

        #final_output = self.nms_layer(output) 
        center_x,center_y,width,height,objectness = tf.split(output,[1,1,1,1,1],axis=-1)
        top_left_x = center_x - (width / 2)
        top_left_y = center_y - (height / 2)
        bottom_right_x = center_x + (width / 2)
        bottom_right_y = center_y + (height / 2)
        boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1)#[:,:,tf.newaxis,:]
        
        #output = combined_non_max_suppression(boxes,objectness,max_output_size_per_class=15,max_total_size=15,iou_threshold=0.6,score_threshold=0.5)
        
        return boxes,objectness
        #return final_output

    def get_config(self):
        #base_config = super().get_config().copy()
        base_config={
        "num_anchors":self.num_anchors,
        "filter_prediction_layer":self.filter_prediction_layer,
        "block1":self.block1,
        "block2":self.block2,
        "block3":self.block3,
        "block4":self.block4,
        "block5":self.block5,
        "block6":self.block6,
        "block7":self.block7,
        "block8":self.block8,
        "block9":self.block9,
        "block10":self.block10,
        "block11":self.block11,
        "block12":self.block12,
        "block13":self.block13,
        "concat_block":self.concat_block,
        "upsamp":self.upsamp,
        "yolo1":self.yolo1,
        "yolo2":self.yolo2,
        "concat_bbox":self.concat_bbox,
        }       
        return base_config
