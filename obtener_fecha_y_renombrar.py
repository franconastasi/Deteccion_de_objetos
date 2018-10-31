#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:19:28 2018

@author: franco
"""

import os
import sys

import re

import pytesseract
from PIL import Image
import PIL.ImageOps

import subprocess



if len(sys.argv) != 3:
    print("Faltan argumentos. Recuerde que se debe incluir la carpeta donde se encuentran las imagenes y el nombre de la cámara")
    print("%s %s %s " % (sys.argv[0], sys.argv[1], sys.argv[2] ) )
    sys.exit()
else:
    input_dir = sys.argv[1]
    num_cam = sys.argv[2].lower()

if not os.path.isdir(input_dir):
    print("No existe la carpeta especificada: '%s'" % input_dir)
    sys.exit()
    
if num_cam not in  ["cam1","cam2","cam3"]:
    print("Nombre de camara '%s' es incorrecto. \nDebe ser una de las siguientes: \n \t cam1 \n \t cam2 \n \t cam3 " % num_cam)
    sys.exit()        


main_dir = os.getcwd()
os.chdir(input_dir)
if num_cam == "cam1":
    for filename in os.listdir(input_dir):
        im = Image.open(filename)
        w,h = im.size
        
        # Recorto imagen
        im_crop = im.crop((215,464,530,h)).convert(mode='L')
        im_crop.save('im_crop.jpg')
        
        # preproceso la imagen antes de extraer texto
        subprocess.call("python2 " +  main_dir +"/process_image.py " + input_dir +"im_crop.jpg "+ input_dir +"text.jpg" , shell = True)
                
        
        # extraigo texto (tesseract) y reemplazo todos los caracteres que no son número por '-' (re.sub)
        fecha = re.sub( "[^0-9]", "_", pytesseract.image_to_string(Image.open('text.jpg')) )
        
        #elimina duplicados que no son números
        pattern=re.compile(r"([^0-9])\1{1,}",re.DOTALL)
        fecha=pattern.sub(r"\1",fecha)
        
        #subprocess.call("python2 " +  main_dir +"/add_metadata.py " + input_dir + filename + " " + fecha , shell = True)
        
        #Renombro
        nuevo_nombre = num_cam+ '_'+fecha
        os.rename(input_dir + filename, nuevo_nombre + ".jpg")
        
        
elif num_cam == "cam2" or num_cam == "cam3":
    for filename in os.listdir(input_dir):
        im = Image.open(filename)
        w,h = im.size
        
        # Recorto imagen
        im_crop = im.crop((570,h-16,885,h)).convert(mode='L')
        im_crop.save('im_crop.jpg')
        
        # preproceso la imagen antes de extraer texto
        subprocess.call("python2 " +  main_dir +"/process_image.py " + input_dir +"im_crop.jpg "+ input_dir +"text.jpg" , shell = True)
                
        
        # extraigo texto (tesseract) y reemplazo todos los caracteres que no son número por '-' (re.sub)
        fecha = re.sub( "[^0-9]", "_", pytesseract.image_to_string(Image.open('text.jpg')) )
        
        #elimina duplicados que no son números
        pattern=re.compile(r"([^0-9])\1{1,}",re.DOTALL)
        fecha=pattern.sub(r"\1",fecha)
        
        nuevo_nombre = num_cam+ '_'+fecha
        os.rename(input_dir + filename, nuevo_nombre + ".jpg")
        
os.remove('im_crop.jpg')
os.remove('text.jpg')
