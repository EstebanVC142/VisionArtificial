# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:55:38 2023

@author: Esteban VC
"""

import cv2
import numpy as np
from PIL import Image
import os

#%% Funciones
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
cv2.destroyAllWindows()
#%%

# Ruta de la carpeta con las imágenes
# input_folder = "D:\Esteban VC\Poli JIC\Semillero Vision Artificial\Vision Artificial\Prueba\Crudas\Malas"
input_folder = "D:\Esteban VC\Poli JIC\Semillero Vision Artificial\Vision Artificial\Prueba\Crudas\Buenas"

# Ruta de la carpeta donde se guardarán las imágenes procesadas
# output_folder = "D:\Esteban VC\Poli JIC\Semillero Vision Artificial\Vision Artificial\Prueba\Recortadas\Malas"
output_folder = "D:\Esteban VC\Poli JIC\Semillero Vision Artificial\Vision Artificial\Prueba\Recortadas\Buenas"

for filename in os.listdir(input_folder):
    # Si el archivo no es una imagen, lo saltamos
    if not filename.endswith(".jpg"):
        continue
    img = Image.open(os.path.join(input_folder, filename))
    img = np.array(img, np.uint8)
    # leer la imagen
    # img = cv2.imread('D:\Esteban VC\Poli JIC\Semillero Vision Artificial\VisionArtificial\Malas\Mala.jpg')
    # img = cv2.imread('D:\Esteban VC\Poli JIC\Semillero Vision Artificial\VisionArtificial\Buenas\Buena.jpg')
    # img = cv2.imread('D:\Esteban VC\Poli JIC\Semillero Vision Artificial\VisionArtificial\F4.jpeg')
    # img = cv2.resize(img,(480,360))
    # convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # aplicar filtros para suavizar la imagen
    imagen_fil = cv2.medianBlur(gray, 7)
    imagen_gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    blure = cv2.addWeighted(imagen_fil, 2, imagen_gauss, -1, 0)
    blur = adjust_gamma(blure,0.95)
    
    # aplicar la transformación de umbral adaptativo para segmentar la imagen
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # encontrar los contornos en la imagen
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # encontrar el contorno más grande
    max_contour = max(contours, key=cv2.contourArea)
    
    # dibujar el contorno más grande en la imagen original
    # cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 2)
    
    # recortar la imagen para que solo se muestre el contorno más grande
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [max_contour], -1, 255, -1)
    res = cv2.bitwise_and(img, img, mask=mask)
    
    # crear una máscara negra para poner el fondo de la imagen recortada en negro
    mask_neg = np.zeros(img.shape[:2], np.uint8)
    mask_neg[:] = (0)
    
    # poner en negro el área fuera del contorno más grande
    mask_neg = cv2.bitwise_not(mask)
    res[mask_neg] = (0)
     
    # mostrar la imagen original y la imagen recortada
    # cv2.imshow('Imagen Original', img)
    # cv2.imshow('Segmentación', thresh)
    # cv2.imshow('Imagen Recortada', res)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    guardada = Image.fromarray(res)
    output_filename = os.path.join(output_folder, filename)
    guardada.save(output_filename)
    # cv2.waitKey(0)
  
