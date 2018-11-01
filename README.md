# Deteccion_de_objetos



############################################################################
############################################################################

ImageAI:

github.com/OlafenwaMoses/image/Ai

Si tenés isntalado anaconda y te devuelve el error:

"libopenblas.so.0: cannot open shared object file: No such file or directory"

Ejecuta: python3 -m pip install scipy -I


Si el error es: 

"numpy.core.multiarray failed to import"

Ejecuta:  python3 -m pip install numpy -I

############################################################################
############################################################################

TensorFlow object detection API

En ambos casos tener cuidado que tensorflow soporte  la versión de Python que estás utilizando.

LINUX
	https://medium.com/@karol_majek/10-simple-steps-to-tensorflow-object-detection-api-aa2e9b96dc94
	No es necesario instalar CUDNN o CUDA toolkit

WINDOWS
	https://medium.com/@rohitrpatil/how-to-use-tensorflow-object-detection-api-on-windows-102ec8097699
	https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b


Prueba de modelos:




## COCO

 MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17' # línea original
 MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28' # Este modelo parece no andar con este script
 MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03' # No es muy buen modelo pero detectó una persona
 MODEL_NAME = 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03' # Este modelo detecta con errores

 MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28' # Buen modelo pero no detecta tantas cosas en las cámaras de la construcción
 MODEL_NAME = 'mask_rcnn_resnet101_atrous_coco_2018_01_28' # El mejor modelo hasta ahora pero detecta menos de la mitad de lo deseado
 MODEL_NAME = 'rfcn_resnet101_coco_2018_01_28' #No detecta bien personas de las imágenes de la construcción (solo 1 de 4 aprox). Tiene un buen rendimiento para las imágenes de prueba.
 MODEL_NAME = 'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'# El mejor modelo. No detecta al 100% pero en la cámara 2 detecta 4 de 6 personas. En la cámara 1 sólo 2 de 4.
 MODEL_NAME = 'mask_rcnn_resnet50_atrous_coco_2018_01_28' # Buen resultado pero un poco peor que mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28



# Kitti
 MODEL_NAME = 'faster_rcnn_resnet101_kitti_2018_01_28' # No detecta bien

# Open Images-trained models
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28' # No sirve para detectar personas






# Obtener fecha de las imágenes:

Luego de cortar las porción de la imagen que contiene la fecha, preprocesar las imágenes con :

https://github.com/schollz/python-ocr

usar tesseract para obtener el texto de la imagen procesada.



SCRIPTS:

	obtener_fecha_y_renombrar.py

	process_image.py


Ejecutar ‘script_obtener_fecha_e_insertar_metadato.py’ en la misma carpeta que 	process_image.py de la siguiente forma

python script_obtener_fecha_e_insertar_metadato.py “dirección absoluta de la carpeta donde están las imágenes” “Número de cámara”

(sin comillas) Donde Número de cámara será cam1, cam2, cam3 dependiendo de que cámara sea

Ejemplo:

python script_obtener_fecha_e_insertar_metadato.py /home/franco/Documents/detecImage/object_detection/test_images/extraer_texto_de_image/pruebas/ cam1




# Instalar MONGODB en Linux mint:

http://linuxforever.info/2017/04/13/how-to-install-mongodb-3-4-in-linux-mint-18-ubuntu-16-04/
