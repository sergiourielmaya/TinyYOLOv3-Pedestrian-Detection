import engine as eng
import inference as inf
import numpy as np
from PIL import Image
import skimage.transform
import tensorrt as trt
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

def rescale_image(image, output_shape, order=1):
   image = skimage.transform.resize(image, output_shape,
               order=order, preserve_range=True, mode='reflect')
   return image


input_file_path =  "pedestrians_example.jpg"
onnx_file = "model.onnx"
serialized_plan_fp32 = "yolo_pedestrian.plan"
HEIGHT = 416
WIDTH = 416

image = np.asarray(Image.open(input_file_path))
img = rescale_image(image, (416, 416),order=1)/255

print(img.shape)

engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float16)

tiempos = []
for i in range(1000):
    inicio = time.time()
    out = inf.do_inference(engine, img, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH);
    fin = time.time()
    tiempos.append(fin-inicio)
print(np.mean(tiempos[10:]))
print(1/np.mean(tiempos[10:]))

print(out)
