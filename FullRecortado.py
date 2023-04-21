# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:14:28 2023

@author: Esteban VC
"""

import cv2
import numpy as np

cv2.destroyAllWindows()
# leer la imagen
img = cv2.imread('D:\Esteban VC\Poli JIC\Semillero Vision Artificial\VisionArtificial\Prueba\Crudas\Malas\Mala.jpg')
# img = cv2.imread('D:\Esteban VC\Poli JIC\Semillero Vision Artificial\VisionArtificial\Prueba\Crudas\Buenas\Buena.jpg')
# img = cv2.imread('D:\Esteban VC\Poli JIC\Semillero Vision Artificial\VisionArtificial\F4.jpeg')
img = cv2.resize(img,(680,500))

# convertir la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# aplicar un filtro gaussiano para suavizar la imagen
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# aplicar la transformación de umbral adaptativo para segmentar la imagen
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# encontrar los contornos en la imagen
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# encontrar el contorno más grande
max_contour = max(contours, key=cv2.contourArea)

# dibujar el contorno más grande en la imagen original
cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 2)

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
cv2.imshow('Imagen Original', img)
cv2.imshow('Imagen Recortada', res)
# cv2.waitKey(0)
