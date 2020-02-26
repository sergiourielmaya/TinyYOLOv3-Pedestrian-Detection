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
#from tensorflow.compat.v1.image import non_max_suppression_v2

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8) #Allocate more memory to Tensorflow
#Arregla un bug donde marca un error con CUDA
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#Import bloques basicos
from tensorflow.keras.layers import Conv2D,BatchNormalization,ZeroPadding2D,MaxPool2D, LeakyReLU,UpSampling2D,Concatenate
import numpy as np
import matplotlib.pyplot as plt



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
                 activation= LeakyReLU,
                 batch_norm=True,
                 root=False,
                 name = None,
                 **kwargs):
        super(BasicBlock,self).__init__(name=name,**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.max_pooling = max_pooling
        self.max_pool_stride = max_pool_stride
        self.root = root
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
    '''
    Clase de la clase Prediction Layer. ESta capa calcula las predicciones de la red, calcula las coordenadas (x,y) y las dimensiones (w,h)
    del Bouding box. Aplica finalemnte Non max supression para eliminar los bouding boxes redundantes.
    Argumentos:
    Anchor_boxes: List, Lista de los Anchorx Boxes para la PredictionLayer, para el caso de TInyYOLOv3-pedestrian son 2 anchor boxes
    conf_thresh: Float [0,1] Umbral para el algoritmo de NMS.
    grid_size: Integer, Tamaño del tensor de que recibe la capa como entrada , (grid_size,grid_size,num_anchors*(5+num_classes))
    num_classes: Entero, número de clases a detectar, default 1
    '''

    def __init__(self,anchor_boxes,grid_size,conf_thresh,num_classes):
        super(PredictionLayer,self).__init__()
        self.num_anchors = len(anchor_boxes)
        self.num_classes = num_classes

        if self.num_classes==1:
            self.final_conv_length = 5
        else:
            self.final_conv_length = 5+ self.num_classes
        self.anchors_boxes = anchor_boxes #a list of list
        self.conf_thresh = conf_thresh
        self.grid_size = grid_size
        #Anchor boxes en forma matricial de tamaño (grid_size*grid_size*anchors,2)
        self.anchors_matrix =tf.cast(tf.tile(anchor_boxes,[self.grid_size*self.grid_size,1]),dtype=tf.float32)
        
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

        box_xy,box_wh,objectness,classes = tf.split(X, [2,2,1,self.num_classes], axis=-1)
        print("Tensor para cada grid con las coordenadas de x e y",box_xy.shape)
        print("Tensor para cada grid con las coordendas de w y h",box_wh.shape)
        print("Tensor de offset en función de la posición de la imagen de cada grid unit", self.x_y_offset.shape)
        print("")
        box_xy = tf.sigmoid(box_xy)
        box_xy = (box_xy + self.x_y_offset)*self.strides #Se encontra la coordenada (x,y) global del bouding box
        box_wh = tf.exp(box_wh) * self.anchors_matrix #Se encuenta el ancho (eje X) y alto (eje Y) para cada bouding box
        objectness = tf.sigmoid(objectness) # Se encuentra la probabilidad de cada bouding box de que sea una persona
        print(box_xy.shape)
        print(box_wh.shape)
        print(objectness.shape)


        return (box_xy, box_wh,objectness)


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
        self.concat_block = Concatenate(axis=-1)
        self.upsamp = UpSampling2D(size = 2,interpolation = "nearest")
        self.yolo1 = PredictionLayer(np.array([[0.2,0.5],[0.3,0.8],[0.4,0.4]]),conf_thresh=0.5,grid_size=13,num_classes=num_classes)
        self.yolo2 = PredictionLayer(np.array([[0.2,0.5],[0.3,0.8],[0.4,0.4]]),conf_thresh=0.5,grid_size=26,num_classes=num_classes)

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


        yolo1 = tf.reshape(yolo1,(-1,13,13,3,85))
        yolo2 = tf.reshape(yolo2,(-1,26,26,3,85))
        print(yolo1.shape)
        print(yolo2.shape)
        output1 = self.yolo1(yolo1)
        output2 = self.yolo2(yolo2)
        return (output1,output2)
        #return (yolo1,yolo2)


class TinyConvnet(Model):
    def __init__(self,num_classes,bouding_boxes,**kwargs):
        super(TinyConvnet,self).__init__()

        self.block1 = BasicBlock(num_filters = 16, kernel_size = 3,name="BasicBlock1")
        self.block2 = BasicBlock(num_filters = 32, kernel_size = 3,name="BasicBlock2")
        self.block3 = BasicBlock(num_filters = 64, kernel_size = 3,name="BasicBlock3")
        self.block4 = BasicBlock(num_filters = 128, kernel_size = 3,name="BasicBlock4")
        self.block5 = BasicBlock(num_filters = 256, kernel_size = 3, root = True,name="BasicBlock5")
        self.block6 = BasicBlock(num_filters = 512, kernel_size = 3,max_pool_stride=1,name="BasicBlock6")
        self.block7 = BasicBlock(num_filters = 1024, kernel_size = 3,max_pooling=False,name="BasicBlock7")
        self.block8 = BasicBlock(num_filters = 256, kernel_size = 1,max_pooling=False,name="BasicBlock8")
        self.block9 = BasicBlock(num_filters = 512, kernel_size = 3,max_pooling=False,name="BasicBlock9")
        self.block10 = BasicBlock(num_filters = 255, kernel_size = 1, batch_norm =False, max_pooling=False, activation = None,name="FinalBlock1")
        self.block11 = BasicBlock(num_filters = 128,kernel_size = 1,max_pooling = False,name="BasicBlock11")
        self.block12 = BasicBlock(num_filters = 256,kernel_size = 3,max_pooling = False,name="BasicBlock12")
        self.block13 = BasicBlock(num_filters = 255, kernel_size = 1, batch_norm =False, max_pooling=False, activation = None,name="FinalBlock2")
        self.concat_block = Concatenate(axis=-1,name="Concatenate")
        self.upsamp = UpSampling2D(size = 2,interpolation = "nearest",name="Upsampling")
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
        
        return yolo1,yolo2    

    def load_weights(self,weights_file):
        fp = open(weights_file, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values
        #header_info = header
        #seen = header[3] #Número de imágenes totales para el entrenamiento
        #weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        #fp.close()
        for layer in self.layers:
            #Se obtiene una lista de np arrays, el orden en Tensorflow es: 
            #SI la capa tiene BN el orden es: COnv weights,gamma(bn coef), beta(bn bias), moving mean, moving variance (mv*input + mm)
            #Si la capa no tiene BN el orden es : Conv bias,Conv weights.
            layer_weights = layer.get_weights()

            #Tiene batch normalization
            if len(layer_weights)==5:
                num_filters = layer_weights[0].shape[-1]
                size = layer_weights[0].shape[0]
                in_dim = layer_weights[0].shape[2]
                #Darknet order : [beta,gamma,mean,variance]
                bn_weights = np.fromfile(fp,dtype=np.float32,count =4*num_filters)
                print(bn_weights.shape)
                #Tensorflow order: [gamma, beta,mean,variance]
                bn_weights = bn_weights.reshape((4,num_filters))[[1,0,2,3]]
                print(bn_weights.shape)


            elif len(layer_weights)==2:
                num_filters = layer_weights[0].shape[-1]
                size = layer_weights[0].shape[0]
                in_dim =layer_weights[0].shape[2]
                
                bias_weights = np.fromfile(fp,dtype=np.float32,count=num_filters)

            #Darknet conv shape (out_dim,in_dim,height,width)

            conv_shape=(num_filters,in_dim,size,size)

            conv_weights = np.fromfile(fp,dtype= np.float32,count=np.int32(np.prod(conv_shape)))
            print("CONV SHAPE",conv_shape)
            #Tensorflow format (height, width, in_dim, out_dim)
            conv_weights =conv_weights.reshape(np.int32(conv_shape)).transpose([2,3,1,0])

            if len(layer_weights)==5:
                gamma,beta,moving_mean,moving_variance = tf.split(bn_weights,[1,1,1,1],axis=0)
                new_weights = [conv_weights,tf.reshape(gamma,[-1]),tf.reshape(beta,[-1]),tf.reshape(moving_mean,[-1]),tf.reshape(moving_variance,[-1])]
                layer.set_weights(new_weights)

            elif len(layer_weights)==2:
                new_weights = [bias_weights,conv_weights]
                layer.set_weights(new_weights)

        fp.close()



def ReadModelConfig(config_file):
    '''
    Lee el archivo de configuración de la arquitectura YOLO y TINY-YOLO

    Entrada:
    config_file: Archivo con extensión .cfg que contiene la configuración
                 de las arquitecturas hechas en Darknet

    Salida:
    bloques:     Lista de diccionarios donde cada diccionario contiene la
                 configuración de cada bloque de operación

    '''

    #Lee el archivo de configuración
    config = open(config_file, "r")
    #Divide por salto de línea
    lineas = config.read().split("\n")
    #Crea una lista de cada linea que no inicia con #
    lineas = [x for x in lineas if x and not x.startswith('#')]
    #Elimina espacias en blanco por la izq. y der.
    lineas = [x.rstrip().lstrip() for x in lineas]

    blocks = []
    for line in lineas:

        #Guarda el el tipo de operación
        if line.startswith("["):
            blocks.append(dict())
            blocks[-1]["type"] = line[1:-1].rstrip()

        #Configuración de la operación
        else:
            key,value = line.split("=")
            #print(value)
            blocks[-1][key.rstrip()] = value.lstrip()

    return blocks
#prueba = ReadModelConfig("yolov3-tiny.cfg")


#a = TinyYOLOv3(num_classes = 80,bouding_boxes="prueba")

'''
b = TinyConvnet(80,None)
b.build(batch_input_shape=(None,416,416,3))
print(b.summary())


#a.build_graph((32,10,))
print(a.summary)

sample_image = np.float32(np.random.random(size=(1,416,416,3)))
#test = a(inputs = sample_image)

a.build(batch_input_shape=(None,416,416,3))

import time

tiempo= []
for i in range(100):
	if i%10==0:
		print(i)
	inicio = time.time()
	aux1,aux2 = a(sample_image)
	fin = time.time()

	tiempo.append(fin-inicio)


import numpy as np

print(aux1.shape)
print(aux2.shape)

for i in aux1:
    print(i.shape)

for i in aux2:
    print(i.shape)

#print(aux1.shape)
#print(aux2.shape)
print(np.array(tiempo).shape)
print(np.median(tiempo))
print(np.mean(tiempo))
plt.plot(tiempo[1:])
#plt.hist(tiempo[1:],bins = 50)
plt.show()

#prueba = PredictionLayer(np.array([[0.2,0.5],[0.3,0.8]]),conf_thresh=0.5,grid_size=26)
#prueba.build(batch_input_shape=(None,26,26,2,5))
#print(prueba.summary)

'''