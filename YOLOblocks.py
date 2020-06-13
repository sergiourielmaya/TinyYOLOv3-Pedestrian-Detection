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
from tensorflow.compat.v1.image import non_max_suppression_with_scores,combined_non_max_suppression
#from tensorflow.image import non_max_suppression_v2

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8) #Allocate more memory to Tensorflow
#Arregla un bug donde marca un error con CUDA
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#Import bloques basicos
from tensorflow.keras.layers import Conv2D,BatchNormalization,ZeroPadding2D,MaxPool2D, LeakyReLU,UpSampling2D,Concatenate
import numpy as np
import matplotlib.pyplot as plt
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
                 bn_train_state= True,
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
            x_root = self.bn(x,training=self.bn_train_state)
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

    def __init__(self,anchor_boxes,grid_size,conf_thresh,num_classes,name=None,training=True,**kwargs):
        super(PredictionLayer,self).__init__(name=name,**kwargs)
        self.num_anchors = len(anchor_boxes)
        self.num_classes = num_classes
        self.training = training

        if self.num_classes==1:
            self.final_conv_length = 5
        else:
            self.final_conv_length = 5+ self.num_classes
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
        if self.num_classes>1:
            box_xy,box_wh,objectness,classes = tf.split(X, [2,2,1,self.num_classes], axis=-1)
        else:
            box_xy,box_wh,objectness = tf.split(X, [2,2,1], axis=-1)

        #print("Tensor para cada grid con las coordenadas de x e y",box_xy.shape)
        #print("Tensor para cada grid con las coordendas de w y h",box_wh.shape)
        #print("Tensor de offset en función de la posición de la imagen de cada grid unit", self.x_y_offset.shape)
        #print("")
        #print(box_wh)
        #print(self.anchors_matrix)
        if not self.training:
            box_xy = tf.sigmoid(box_xy)
            box_xy = (box_xy + self.x_y_offset)*self.strides #Se encontra la coordenada (x,y) global del bouding box
            box_wh = tf.exp(box_wh) * self.anchors_matrix #Se encuenta el ancho (eje X) y alto (eje Y) para cada bouding box
        else:
            pass

        objectness = tf.sigmoid(objectness) # Se encuentra la probabilidad de cada bouding box de que sea una persona

        if self.num_classes>1:
            classes = tf.nn.sigmoid(classes)
        else:
            pass

        #print(box_xy.shape)
        #print(box_wh.shape)
        #print(objectness.shape)
        '''
        output = tf.concat([box_xy,box_wh,objectness],axis=-1)
        '''
        if self.num_classes==1:
            output = tf.concat([box_xy,box_wh,objectness],axis=-1)
        else:
            output = tf.concat([box_xy,box_wh,objectness,classes],axis=-1)
        
        return output

    def get_config(self):
        base_config = super().get_config().copy()
        base_config.update({"num_anchors": self.num_anchors,
        "num_classes":self.num_classes,
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
            print("Solo es una clase")
        else: 
            center_x,center_y,width,height,objectness,classes = tf.split(inputs,[1,1,1,1,1,self.num_classes],axis=-1)

        #print("EL tamaño de clases es",classes.shape)

        top_left_x = center_x - (width / 2)
        top_left_y = center_y - (height / 2)
        bottom_right_x = center_x + (width / 2)
        bottom_right_y = center_y + (height / 2)
        
        '''
        boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1)#[:,:,tf.newaxis,:]
        #print(boxes.shape)

        aux = tf.zeros(tf.shape(boxes)[1:])
        aux_scores = tf.zeros(tf.shape(2535))
        #output = 0

        aux = boxes[0,:,:]#.reshape([boxes.shape[1],boxes[2].shape])
        aux_scores = objectness[0,:]
        output = non_max_suppression_with_scores(aux,tf.squeeze(aux_scores) ,max_output_size=10)

        for i in range(1,tf.shape(boxes)[0]):
            aux = boxes[i,:,:]#.reshape([boxes.shape[1],boxes[2].shape])
            aux_scores = objectness[i,:]
            output = non_max_suppression_with_scores(aux,tf.squeeze(aux_scores) ,max_output_size=10)

        #output = combined_non_max_suppression(boxes,objectness,max_output_size_per_class=10,max_total_size=10)

        #return output
        '''
        
        #boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1)[:,:,tf.newaxis,:]
        #print(boxes.shape)

        if self.num_classes >1:
            boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1)[:,:,tf.newaxis,:]
            output = combined_non_max_suppression(boxes,classes*objectness,max_output_size_per_class=10,max_total_size=20,iou_threshold=0.6,score_threshold=0.5)
            return output
        else:
            #boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y,objectness], axis=-1)
            #
            
            #output = combined_non_max_suppression(boxes,objectness,max_output_size_per_class=15,max_total_size=15,iou_threshold=0.6,score_threshold=0.3)
            boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1)
            selected_indices, selected_scores = non_max_suppression_with_scores(boxes[0,:,:],objectness[0,:,0],max_output_size=10,iou_threshold=0.6,score_threshold=0.5)#,soft_nms_sigma=0.5)
            selected_boxes = tf.gather(boxes, selected_indices,axis=1)
            return selected_boxes,selected_scores
            #return boxes

class TinyYOLOv3(Model):

    def __init__(self,num_classes,anchor_boxes,train = False,mode = "transfer",obj_threshold=0.5,**kwargs):
        super(TinyYOLOv3,self).__init__()
        self.train = train
        self.num_classes=num_classes
        self.num_anchors = len(anchor_boxes)
        
        if num_classes==1:
            self.filter_prediction_layer =5*(len(anchor_boxes)//2)
        else:
            self.filter_prediction_layer=(5+num_classes)*(len(anchor_boxes)//2)

        if mode == "transfer":
            entrenable = False
        elif mode == "finetuning":
            entrenable = True
        self.block1 = BasicBlock(num_filters = 16, kernel_size = 3,name="BasicBlock1",trainable=entrenable,bn_train_state = self.train)
        self.block2 = BasicBlock(num_filters = 32, kernel_size = 3,name="BasicBlock2",trainable=entrenable,bn_train_state = self.train)
        self.block3 = BasicBlock(num_filters = 64, kernel_size = 3,name="BasicBlock3",trainable=entrenable,bn_train_state = self.train)
        self.block4 = BasicBlock(num_filters = 128, kernel_size = 3,name="BasicBlock4",trainable=entrenable,bn_train_state = self.train)
        self.block5 = BasicBlock(num_filters = 256, kernel_size = 3, root = True,name="BasicBlock5",trainable=entrenable,bn_train_state = self.train)
        self.block6 = BasicBlock(num_filters = 512, kernel_size = 3,max_pool_stride=1,name="BasicBlock6",trainable=entrenable,bn_train_state = self.train)
        self.block7 = BasicBlock(num_filters = 1024, kernel_size = 3,max_pooling=False,name="BasicBlock7",trainable=entrenable,bn_train_state = self.train)
        self.block8 = BasicBlock(num_filters = 256, kernel_size = 1,max_pooling=False,name="BasicBlock8",trainable=entrenable,bn_train_state = self.train)
        self.block9 = BasicBlock(num_filters = 512, kernel_size = 3,max_pooling=False,name="BasicBlock9",trainable=entrenable,bn_train_state = self.train)
        self.block10 = BasicBlock(num_filters = self.filter_prediction_layer, kernel_size = 1, batch_norm =False, max_pooling=False, activation = None,name="FinalBlock1")
        self.block11 = BasicBlock(num_filters = 128,kernel_size = 1,max_pooling = False,name="BasicBlock11",trainable=entrenable,bn_train_state = self.train)
        self.block12 = BasicBlock(num_filters = 256,kernel_size = 3,max_pooling = False,name="BasicBlock12",trainable=entrenable,bn_train_state = self.train)
        self.block13 = BasicBlock(num_filters = self.filter_prediction_layer, kernel_size = 1, batch_norm =False, max_pooling=False, activation = None,name="FinalBlock2")
        self.concat_block = Concatenate(axis=-1,name="Concatenate")
        self.upsamp = UpSampling2D(size = 2,interpolation = "nearest",name="Upsampling")
        self.yolo1 = PredictionLayer(anchor_boxes[len(anchor_boxes)//2:],grid_size=13,conf_thresh=0.5,num_classes=num_classes,name="Prediction1",training=self.train)
        self.yolo2 = PredictionLayer(anchor_boxes[:len(anchor_boxes)//2],grid_size=26,conf_thresh=0.5,num_classes=num_classes,name="Prediction2",training=self.train)
        self.concat_bbox = Concatenate(axis=1,name="Concatenate_BBOX")
        if not self.train:
            self.nms_layer = NMSLayer(num_classes=self.num_classes,obj_threshold=obj_threshold)
        else:
            self.nms_layer = None

    def build(self,batch_input_shape):
        super().build(batch_input_shape)

    def load_weights_darknet(self,weights_file): 

        is_convolution=False
        total_parametros = 0

        fp = open(weights_file, "rb")
        header = np.fromfile(fp,dtype=np.int32,count=5)


        for layer in self.layers:
            '''
            layer es un objeto de la clase "Layer", con .get_weights() se obtiene una lista de np arrays. El orden en Tensorflow es: 
            *Si la capa tiene BN el orden es: Conv weights,gamma(bn coef), beta(bn bias), moving mean, moving variance (mv*input + mm)
            *Si la capa no tiene BN el orden es : Conv bias,Conv weights.
            '''
            #print(layer.name)
            layer_weights = layer.get_weights()
            layer_parametros = 0 #Contamos el número total de parametros de se acargan
            #Tiene batch normalization
            if len(layer_weights)==5:
                num_filters = layer_weights[0].shape[-1]#Obtenemos el número de filtros de la capa de Convolución
                size = layer_weights[0].shape[0] #Tamaño del filtro
                in_dim = layer_weights[0].shape[2] #Dimension del filtro, número de filtros de la capa anterior

                #Con fromfile obtenemos los 4*num_filters float numbers perteneciencias a la capa de BatchNormalization
                #Darknet order : [beta,gamma,mean,variance]
                bn_weights = np.fromfile(fp,dtype=np.float32,count =4*num_filters)

                #print("Pesos del batch normalization",bn_weights.shape)
                #Npumero de parámetros cargados
                layer_parametros += bn_weights.shape[0]
                #print(bn_weights.shape)
                #Ahora usando reshape obtenemos la dimesion correcta pero cambiamos el orden de las filas, debido a la configuracion de TF para la capa BN
                #Tensorflow order: [gamma, beta,mean,variance]
                bn_weights = bn_weights.reshape((4,num_filters))[[1,0,2,3]]
                #print(bn_weights.shape)
                #Se activa la bandera de que fue una capa con la operacion convolucion
                is_convolution = True

            #No tiene batch normalization
            elif len(layer_weights)==2:

                if self.num_classes == 80:
                    num_filters = layer_weights[0].shape[-1]#Obtenemos el número de filtros de la capa de Convolución
                    size = layer_weights[0].shape[0] #Tamaño del filtro
                    in_dim = layer_weights[0].shape[2] #Dimension del filtro, número de filtros de la capa anterior

                else:
                    num_filters = (80+5)*3#Obtenemos el número de filtros de la capa de Convolución
                    size = layer_weights[0].shape[0] #Tamaño del filtro
                    in_dim = layer_weights[0].shape[2] #Dimension del filtro, número de filtros de la capa anterior                   

                #Con fromfile obtenemos num_filters float numbers, que es número de bias que hay en ese capa
                bias_weights = np.fromfile(fp,dtype=np.float32,count=num_filters)
                #print("Bias de la convolucion",bias_weights.shape)
                #Número de parametros cargados
                layer_parametros += bias_weights.shape[0]
                #Se activa la bandera de que fue una capa con la operacion convolucion
                is_convolution=True
            
            #Si la capa analizada tenia una operación de convolucion, cargaremos los pesos correspondientes
            if is_convolution:
                #Se obtienen las dimensiones del tensor correspondiente a los pesos de la operación de COnvolución
                #Darknet conv shape (out_dim,in_dim,height,width)
                conv_shape=(num_filters,in_dim,size,size)
                
                #con frofile se obtienen num_filters*in_dim*size*size float numbers
                conv_weights = np.fromfile(fp,dtype= np.float32,count=np.int32(np.prod(conv_shape)))
                print("Pesos de la convolucion",conv_weights.shape)
                print("CONV SHAPE",conv_shape)
                #Se suman todod estos parametros al número de parametros cargados
                layer_parametros += np.prod(conv_shape)
                #print("Total de parametros",total_parametros)
                #Se obtiene las dimesiones y el ORDEN correcto en el formato de tensorflow para almacenar los pesos
                #Tensorflow format (height, width, in_dim, out_dim)
                conv_weights =conv_weights.reshape(np.int32(conv_shape)).transpose([2,3,1,0])

                #Finalmente se cargan los pesos al objeto layer usando el método set_weights
                if len(layer_weights)==5:
                    gamma,beta,moving_mean,moving_variance = tf.split(bn_weights,[1,1,1,1],axis=0)
                    new_weights = [conv_weights,tf.reshape(gamma,[-1]),tf.reshape(beta,[-1]),tf.reshape(moving_mean,[-1]),tf.reshape(moving_variance,[-1])]
                    layer.set_weights(new_weights)

                elif len(layer_weights)==2:

                    if self.num_classes==80:
                        new_weights = [conv_weights,bias_weights]
                        layer.set_weights(new_weights)
                    #Quiere decir que se hará un re entrenamiento por lo que no se cargaran los pesos en la última capa
                    #COnvolucional pero si se tomaran en cuenta, se jalaránm del archivo ya que si no, causa error
                    else:
                        pass
                #Se regresa la bandera a su valor Falso
                is_convolution=False
            #Se acumula el número de pesos cargados.
            total_parametros += layer_parametros


        fp.close()
    
        return total_parametros

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
        fin=time.time()
        
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
        fin= time.time()
        #print("Tiempo de la Capa YOLO: ",fin-inicio)
        #print(output.shape)
        
        if self.train:
            box_xy,box_wh,objectness = tf.split(output, [2,2,1], axis=-1)
            print("Modo entrenamiento")
            return (box_xy,box_wh,objectness,objectness)
        else:
            #return output
            inicio=time.time()
            final_output = self.nms_layer(output) 
            fin = time.time()
            #print("Tiempo NMS: ",fin-inicio)  
            return final_output

    def get_config(self):
        #base_config = super().get_config().copy()
        base_config={"train": self.train,
        "num_classes":self.num_classes,
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
        "nms_layer":self.nms_layer
        }       
        return base_config


'''
# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
	"category_output": "categorical_crossentropy",
	"color_output": "categorical_crossentropy",
}
lossWeights = {"category_output": 1.0, "color_output": 1.0}
# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])
'''


class YOLOLoss(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)    

    def call(self,y_true,y_pred):
        #[tx_vector,ty_vector,tw_vector,th_vector,obj_mask,noobj_mask]

        bce = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        box_txty_true,box_twth_true,objectness_true,no_objectness_true = tf.split(y_true, [2,2,1,1], axis=-1)
        box_txty_pred,box_twth_pred,objectness_pred = tf.split(y_pred, [2,2,1], axis=-1)

        #print(objectness_true.shape)
        #print(objectness_pred.shape)
        #print(no_objectness_true.shape)

        loss_xy = tf.reduce_sum(tf.multiply(objectness_true,tf.square(box_txty_true-box_txty_pred)))
        loss_wh = tf.reduce_sum(tf.multiply(objectness_true,tf.square(box_twth_true-box_twth_pred)))

        #print(loss_xy.shape)
        #print(loss_wh.shape)

        loss_obj =   tf.reduce_sum(tf.multiply(objectness_true,bce(objectness_true,objectness_pred)))
        loss_noobj = tf.reduce_sum(tf.multiply(no_objectness_true,bce(objectness_true,objectness_pred)))

        #Loss per batch
        loss_xy = tf.reduce_sum(loss_xy)
        loss_wh = tf.reduce_sum(loss_wh)
        loss_obj = tf.reduce_sum(loss_obj)
        loss_noobj = tf.reduce_sum(loss_noobj)

        #return 1
        return loss_xy+loss_wh+loss_obj+loss_noobj
        #return loss_xy,loss_wh,loss_obj,loss_noobj

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