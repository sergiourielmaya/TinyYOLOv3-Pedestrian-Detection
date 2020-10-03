#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:18:29 2020

@author: sergio
"""

#Import Tensorflow sub classes
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model,Input
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

def BasicBlock(x,num_filters = 3,kernel_size = 3,max_pooling = True,max_pool_stride=2,act =True,batch_norm = True,root=False,name=None,train_state=True):

    x = Conv2D(num_filters,
                kernel_size=kernel_size,
                strides=(np.int64(1),np.int64(1)),
                padding="same",
                use_bias = not batch_norm,trainable=train_state)(x)

    if batch_norm:
        x_root = BatchNormalization(axis = -1,trainable = train_state)(x)
    else:
        x_root = x

    if act != None:
        x=x_root = tf.nn.leaky_relu(x_root, alpha=0.1)

    if max_pool_stride ==1:
        x = ZeroPadding2D(padding = ((0,1),(0,1)),trainable = train_state)(x)

    if max_pooling:
        x=tf.nn.max_pool2d(x,ksize =(2,2), strides= (max_pool_stride,max_pool_stride),padding='VALID')

    if root:
        return x,x_root
    else:
        return x

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')

def TinyYOLOv3_functional(anchor_boxes,num_classes=80,mode = "tl",training = True):

    if mode == "tl":
        layers_train_state = False
    else:
        layers_train_state = True
        

    last_filters = 3*(5+num_classes)
    x = inputs = Input([416, 416, 3])
    yolo1 = BasicBlock(x,num_filters = 16, kernel_size = 3,name="BasicBlock1",train_state = layers_train_state)
    yolo1 = BasicBlock(yolo1,num_filters = 32, kernel_size = 3,name="BasicBlock2",train_state = layers_train_state)
    yolo1 = BasicBlock(yolo1,num_filters = 64, kernel_size = 3,name="BasicBlock3",train_state = layers_train_state)
    yolo1 =  BasicBlock(yolo1,num_filters = 128, kernel_size = 3,name="BasicBlock4",train_state = layers_train_state)
    yolo1,root = BasicBlock(yolo1,num_filters = 256, kernel_size = 3, root = True,name="BasicBlock5",train_state = layers_train_state)
    yolo1 = BasicBlock(yolo1,num_filters = 512, kernel_size = 3,max_pool_stride=1,name="BasicBlock6",train_state = layers_train_state)
    yolo1 = BasicBlock(yolo1,num_filters = 1024, kernel_size = 3,max_pooling=False,name="BasicBlock7",train_state = layers_train_state)
    yolo1_branch =  BasicBlock(yolo1,num_filters = 256, kernel_size = 1,max_pooling=False,name="BasicBlock8",train_state = layers_train_state)
    yolo1 = BasicBlock(yolo1_branch,num_filters = 512, kernel_size = 3,max_pooling=False,name="BasicBlock9",train_state = layers_train_state)
    yolo1 = BasicBlock(yolo1,num_filters = last_filters, kernel_size = 1, batch_norm =False, max_pooling=False, act = None,name="FinalBlock1",train_state=True)

    yolo2 = BasicBlock(yolo1_branch,num_filters = 128,kernel_size = 1,max_pooling = False,name="BasicBlock11",train_state = layers_train_state)
    yolo2 = upsample(yolo2)
    yolo2 = tf.concat([yolo2,root],axis=-1,name="Concatenate")
    yolo2 = BasicBlock(yolo2,num_filters = 256,kernel_size = 3,max_pooling = False,name="BasicBlock12",train_state = layers_train_state)
    yolo2 = BasicBlock(yolo2,num_filters = last_filters, kernel_size = 1, batch_norm =False, max_pooling=False, act = None,name="FinalBlock2",train_state=True)
    

    
    if training:
        final_output_1 = decode_trt_training(yolo1,[1/13],anchor_boxes[:3],num_classes=num_classes)
        final_output_2 = decode_trt_training(yolo2,[1/26],anchor_boxes[3:],num_classes=num_classes)
        output = Concatenate(axis=1,name="Concateasdasdanate_boxes")([final_output_1[0],final_output_2[0]])
        #print("output",output.shape)
        objectness =  Concatenate(axis=1,name="Concatenate_obj")([final_output_1[1],final_output_2[1]])
        #print("objt",objectness.shape)
        box_xy,box_wh = tf.split(output, [2,2], axis=-1)
        return tf.keras.Model(inputs, (box_xy, box_wh,objectness,objectness))
    else:
        final_output_1 = decode_trt(yolo1,[1/13],output_size=13,ANCHORS = anchor_boxes[3:],num_classes=num_classes)
        final_output_2 = decode_trt(yolo2,[1/26],output_size=26,ANCHORS = anchor_boxes[:3],num_classes=num_classes)
        output = nms_layer((final_output_1, final_output_2))
        return tf.keras.Model(inputs, output)

    
def decode_trt(conv_output, STRIDES,output_size, ANCHORS,num_classes=80):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5+num_classes))
    #print(conv_output.shape)
    if num_classes == 80:
        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf ,conv_raw_prob= tf.split(conv_output, (2, 2, 1,num_classes), axis=-1)
    else:
        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf = tf.split(conv_output, (2, 2, 1), axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])


    xy_grid = tf.cast(xy_grid, tf.float32)

    # pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
    #           STRIDES[i]
    
    pred_xy = tf.reshape(tf.sigmoid(conv_raw_dxdy), (-1, 2)) + tf.reshape(xy_grid, (-1, 2)) * STRIDES
    pred_xy = tf.reshape(pred_xy, (batch_size, output_size, output_size, 3, 2))
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)


    pred_conf = tf.sigmoid(conv_raw_conf)
    if num_classes == 80:
        pred_prob = tf.sigmoid(conv_raw_prob)
        pred_conf = pred_conf * pred_prob
        pred_conf = tf.reshape(pred_conf, (batch_size, -1, num_classes))
    
    pred_conf = tf.reshape(pred_conf, (batch_size, -1,1))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))
    return pred_xywh, pred_conf


def decode_trt_training(conv_output, STRIDES, ANCHORS,num_classes=80):
    batch_size = tf.shape(conv_output)[0]
    output_size = tf.shape(conv_output)[1]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5+num_classes))

    conv_raw_dxdydwdh, conv_raw_conf = tf.split(conv_output, (4, 1), axis=-1)

    

    pred_conf = tf.sigmoid(conv_raw_conf)

    
    pred_conf = tf.reshape(pred_conf, (batch_size, -1,1))
    pred_xywh = tf.reshape(conv_raw_dxdydwdh, (batch_size, -1, 4))

    return pred_xywh, pred_conf


def nms_layer(outputs):

    output = tf.concat([outputs[0][0],outputs[1][0]],axis=1)
    objectness =  tf.concat([outputs[0][1],outputs[1][1]],axis=1)
    center_x,center_y,width,height = tf.split(output,[1,1,1,1],axis=-1)

    top_left_x = center_x - (width / 2)
    top_left_y = center_y - (height / 2)
    bottom_right_x = center_x + (width / 2)
    bottom_right_y = center_y + (height / 2)
    boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y,objectness], axis=-1)#[:,:,tf.newaxis,:]
    
    return boxes
    
    #boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1)[:,:,tf.newaxis,:]
    #output = combined_non_max_suppression(boxes,objectness,max_output_size_per_class=100,max_total_size=100,iou_threshold=0.6,score_threshold=0.5)
    #return output[0]    

def load_weights_darknet(model,weights_file,num_classes = 1): 

    is_convolution=False
    total_parametros = 0

    fp = open(weights_file, "rb")
    header = np.fromfile(fp,dtype=np.int32,count=5)

    aux = 0
    nombres =[]
    for index,i in enumerate(model.layers[1:45]):
        if index<4:
            #print(i.name)
            nombres.append((i.name+"_0").split("_"))
            #print((i.name+"_0").split("_"))
        else:
            #print(i.name)
            #print((i.name).split("_"))   
            nombres.append((i.name).split("_"))

    nombres_definitivo = []
    for i in nombres:
        try: 
            int(i[-1])
            nombres_definitivo.append(i)
        except:
            pass
        
    nombres_definitivo.sort(key=lambda x: int(x[-1]))

    nombres_definitivo_oficial = []

    for k in nombres_definitivo:
        if k[-1]=='0':
            nombres_definitivo_oficial.append("_".join(k[:-1]))
        else:
            nombres_definitivo_oficial.append("_".join(k))


    lista_definitiva = nombres_definitivo_oficial[:33] +['conv2d_9','conv2d_10','batch_normalization_9',
    'leaky_re_lu_9','conv2d_11','batch_normalization_10',
    'leaky_re_lu_10','conv2d_12']
    #model.get_layer("conv2d")
    for index,name in enumerate(lista_definitiva):
        '''
        layer es un objeto de la clase "Layer", con .get_weights() se obtiene una lista de np arrays. El orden en Tensorflow es: 
        *Si la capa tiene BN el orden es: Conv weights,gamma(bn coef), beta(bn bias), moving mean, moving variance (mv*input + mm)
        *Si la capa no tiene BN el orden es : Conv bias,Conv weights.
        '''
        layer = model.get_layer(name)
        print(layer.name)
        layer_weights = layer.get_weights()
        print(len(layer_weights))
        layer_parametros = 0 #Contamos el número total de parametros de se acargan

        #Tiene batch normalization
        if len(layer_weights)==1:
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

            if num_classes == 80:
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
            if len(layer_weights)==1:
                gamma,beta,moving_mean,moving_variance = tf.split(bn_weights,[1,1,1,1],axis=0)
                new_weights_bn = [tf.reshape(gamma,[-1]),tf.reshape(beta,[-1]),tf.reshape(moving_mean,[-1]),tf.reshape(moving_variance,[-1])]
                new_weights_conv = [conv_weights]

                print(nombres_definitivo_oficial[index])
                print(nombres_definitivo_oficial[index+1])
                model.get_layer(lista_definitiva[index]).set_weights(new_weights_conv)
                model.get_layer(lista_definitiva[index+1]).set_weights(new_weights_bn)

            elif len(layer_weights)==2:

                if num_classes==80:
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

        print(total_parametros)
    fp.close()

    return total_parametros