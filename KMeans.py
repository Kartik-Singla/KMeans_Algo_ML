import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

im=cv2.imread('scene.jpg') #selecting th required image
im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)# convert image from BGR to RGB
orig=im.shape

all=im.reshape(600*571,3) #array from 3D to 2D
km=KMeans(n_clusters=4) # n_clusters is for extracting the number of dominant colors

km.fit(all)
centers=km.cluster_centers_
centers=np.array(centers, dtype='uint8')

plt.figure(0,figsize=(8,2))

new_img=np.zeros((600*571,3),dtype='uint8')# initializing a new image with 0x0x0 array

for ix in range(new_img.shape[0]):
    new_img[ix]=centers[km.labels_[ix]]#filling the pixels accordingly
new_img=new_img.reshape(orig)
plt.imshow(new_img)
plt.show()

