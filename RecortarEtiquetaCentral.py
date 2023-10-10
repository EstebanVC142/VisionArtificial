# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:22:17 2022

@author: Esteban
"""

import numpy as np
import cv2 as cv2

cv2.destroyAllWindows()

imagen_fuente = cv2.imread('D:/Esteban VC/Poli JIC/Semillero Vision Artificial/Vision Artificial/Dataset/Buenas/Crudas/scene00002.jpg')
imagen  = imagen_fuente.copy()
canny = cv2.Canny(imagen, 120, 255, 1)
cnts, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:3]

maskImage = np.zeros(imagen.shape, dtype=np.uint8)
cv2.drawContours(maskImage, cnts, 2, (255, 255, 255), -1)
cv2.imshow("Etiqueta",cv2.resize(maskImage,(680, 500)))
newImage = cv2.bitwise_and(imagen, maskImage)
cv2.imshow("Etiqueta central Recortada",cv2.resize(newImage,(680, 500)))