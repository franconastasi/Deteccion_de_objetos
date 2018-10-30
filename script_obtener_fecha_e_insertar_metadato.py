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
    print("%s input_file output_file" % (sys.argv[0]))
    sys.exit()
else:
    input_dir = sys.argv[1]
    num_cam = sys.argv[2].lower()

if not os.path.isdir(input_dir):
    print("No such file '%s'" % input_dir)
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
        os.rename(input_dir + filename, num_cam+ '_'+fecha)
        
else:
    for filename in os.listdir(input_dir):
        im = Image.open(filename)
        w,h = im.size
        
        # Recorto imagen
        im_crop = im.crop((215,464,530,h)).convert(mode='L')
        im_crop.save('im_crop.jpg')
        
        # preproceso la imagen antes de extraer texto
        print("\n"+"python2 "+main_dir+"/process_image.py " + input_dir +"im_crop.jpg "+ input_dir +"text.jpg")
        subprocess.call("python2 " +  main_dir +"/process_image.py " + input_dir +"im_crop.jpg "+ input_dir +"text.jpg" , shell = True)
                
        
        # extraigo texto (tesseract) y reemplazo todos los caracteres que no son número por '-' (re.sub)
        fecha = re.sub( "[^0-9]", "_", pytesseract.image_to_string(Image.open('text.jpg')) )
        
        #elimina duplicados que no son números
        pattern=re.compile(r"([^0-9])\1{1,}",re.DOTALL)
        fecha=pattern.sub(r"\1",fecha)
        os.rename(input_dir + filename, num_cam + '_' +fecha)

#print ( re.sub( "[^0-9]", "_", pytesseract.image_to_string(Image.open('text.jpg')) ))

#print ( pytesseract.image_to_string(Image.open('text.jpg')) )
