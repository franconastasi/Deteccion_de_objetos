
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

###################### ATENCIÓN ##########################
# La sentencia (1) produce error si object_detection.py no se encuentra en la capeta <instalación_tensorflow>/model/research/object_detection
# El problema está relacionado con PYTHONPATH porque aunque se agrega esta carpeta, python devuelve que no existe el modulo object_detection
# Por el momento se arreglo cambiando el working directory a la carpeta mencionada:
wd_old = os.getcwd()
print(os.getcwd())
os.chdir("/home/franco/Documents/detecImage/tensorflow/models/research/object_detection/")
print(os.getcwd())
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops # (1)

os.chdir(wd_old)
print(os.getcwd())

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


# ## Env setup

# In[2]:


# This is needed to display the images.
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

# En caso de que no encuentre el módulo utils, verificar que exista la variable de entorno PYTHONPATH
# y que apunte por lo menos a "carpeta_que_contiene_a_tensorflow"/tensorflow/models/research/object_detection
# Para que sea automático, modificar el achivo .bashrc como dicen las guias que se menciona en Doc_ImageAI,txt

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[13]:


# What model to download.


#Distintos modelos que fueron probados

## COCO

# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17' # línea original
# MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28' # Este modelo parece no andar con este script
# MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03' # No es muy buen modelo pero detectó una persona
# MODEL_NAME = 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03' # Este modelo detecta con errores

# MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28' # Buen modelo pero no detecta tantas cosas en las camaras de la contrucción
# MODEL_NAME = 'mask_rcnn_resnet101_atrous_coco_2018_01_28' # El segundo mejor modelo hasta ahora pero detecta menos de la mitad de lo deseado
# MODEL_NAME = 'rfcn_resnet101_coco_2018_01_28' #No detecta bien personas de las imágenes de la construcción (solo 1 de 4 aprox). Tiene un buen rendimiento para las imágenes de prueba.
# MODEL_NAME = 'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'# El mejor modelo. No detecta al 100% pero en la camara 2 detecta 4 de 6 personas. En la cámara 1 sólo 2 de 4.
# MODEL_NAME = 'mask_rcnn_resnet50_atrous_coco_2018_01_28' # Buen resultado pero un poco peor que mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28


# Kitti

# MODEL_NAME = 'faster_rcnn_resnet101_kitti_2018_01_28' # No detecta bien

# Open Images-trained models
# MODEL_NAME = 'faster_rcnn_inception_resnehttp://download.tensorflow.org/models/object_detection/t_v2_atrous_oid_2018_01_28' # No sirve para detectar personas


MODEL_NAME = 'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS_FOLDER debe apuntar a donde se encuentren los modelos .pbtxt
# En caso de usar un modelos propio, hay que cambiarlo a la carpeta donde este el modelo.

PATH_TO_LABELS_FOLDER = '/home/franco/Documents/detecImage/tensorflow/models/research/object_detection/data'
PATH_TO_LABELS = os.path.join(PATH_TO_LABELS_FOLDER, 'mscoco_label_map.pbtxt')


# ## Download Model

# In[6]:
# Si es módelo ya está descargado, no hace nada y si no está descargado, lo descarga
if not os.path.isfile(MODEL_FILE):
  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[14]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[15]:


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# ## Helper code

# In[9]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[10]:

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,  '{}'.format(filename)) for filename in os.listdir( './' + PATH_TO_TEST_IMAGES_DIR) ]


# Size, in inches, of the output images.
IMAGE_SIZE = (18, 12)


# In[11]:


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


# In[16]:

# Análisis de imágenes

for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      
      ## Permite elegir a parti de que porcentaje empieza a detectar objectos
      min_score_thresh = 0.3,
      
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  #plt.figure(figsize=IMAGE_SIZE)
  
  
  plt.figure()
  plt.imshow(image_np)
  plt.savefig(image_path[:-4] + '_resultado ' + '.jpg' )
  
  
  print('Imagen procesada: ' + image_path)

