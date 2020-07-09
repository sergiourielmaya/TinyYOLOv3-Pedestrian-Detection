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

anchors =[[10/416,14/416],[23/416,27/416],[37/416,58/416],[81/416,82/416],[135/416,169/416],[344/416,319/416]]
#anchors =[[10/416,14/416],[23/416,27/416],[37/416,58/416],[81/416,82/416]]

mod = TinyYOLOv3(80,anchor_boxes=anchors)

mod.build(batch_input_shape=(None,416,416,3))
mod.summary()
print(mod.load_weights("yolov3-tiny (1).weights"))

tiempo=[]
with tf.device("GPU:0"):
    for i in range(100):
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