# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import numpy as np
import matplotlib.pyplot as plt

dataDir='/home/sergio/Documentos/COCO_Dataset'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
print(annFile)


from pycocotools.coco import COCO
import numpy as np
#import skimage.io as io
import matplotlib.pyplot as plt
#import pylab

# initialize COCO api for instance annotations
coco=COCO(annFile)


catIds = coco.getCatIds(catNms=['person']); #ID de la categoría persona
imgIds = coco.getImgIds(catIds=catIds ); # IDS de las imagenes con la categoria persona
annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds)
img_metadata = coco.loadImgs(imgIds)
img_annotations = coco.loadAnns(annIds)


print(img_metadata[0])
print(img_annotations[0]["bbox"])
print(len(img_annotations))

# Getting all thw bounding boxes of the dataset



#import os
#os.chdir("/home/sergio/Documentos/COCO_Dataset/pedestrian_dataset_train")



#import numpy as np
#from skimage import io
#from skimage.transform import resize

# Load an color image in grayscale
#img = io.imread(img_metadata[0]["file_name"])

#img_resized = resize(img,(416,416))

info_images = {}

for i in img_metadata:

    info_images[str(i["id"])]={"width":i["width"],
                          "height":i["height"]}
    

print(info_images["262145"])
print(img_annotations[0]["bbox"][2]*(416/640))
print(img_annotations[0]["bbox"][3]*(416/427))

bbox_dim = np.zeros((len(img_annotations),2))

for index,i in enumerate(img_annotations):
    
    key = str(i["image_id"])
    width = info_images[key]["width"]
    height = info_images[key]["height"]
    bbox_dim[index,0] = i["bbox"][2]*(416/width)
    bbox_dim[index,1] = i["bbox"][3]*(416/height)


plt.plot(bbox_dim[:,0],bbox_dim[:,1],'o',markersize=2)
plt.show()

print(bbox_dim[0:10,:])
#print(img.shape)
#io.imshow(img)

#plt.figure()
#io.imshow(img_resized)
#plt.show()