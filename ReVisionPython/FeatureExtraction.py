from skimage.io import imread,imshow
from skimage.color import rgb2grey
from skimage.transform import resize
import sys
import numpy as np
from matplotlib import pyplot as plt
import pylab 

def show_img(img):
     width = img.shape[1]/75.0
     height = img.shape[0]*width/img.shape[1]
     f = plt.figure(figsize=(width, height))
     plt.imshow(img)
     pylab.show()

def oneImFile(loc):
    size=(256,256)
    im=resize(rgb2grey(imread(loc)),size)
    print type(im),im.shape,im[0][1]
    show_img(im)
    

def main():
    loc=sys.argv[1]
    oneImFile(loc)
    

if __name__ == "__main__":
    main()	

	
    



 
