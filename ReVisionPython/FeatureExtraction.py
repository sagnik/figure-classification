from skimage.io import imread,imshow
from skimage.color import rgb2grey
from skimage.transform import resize
import sys
import numpy as np
from matplotlib import pyplot as plt
import pylab 
import random
from scipy import ndimage

PS=6 #patch size

def show_img(img):
     width = img.shape[1]/75.0
     height = img.shape[0]*width/img.shape[1]
     f = plt.figure(figsize=(width, height))
     plt.imshow(img)
     pylab.show()

def randomPatchExtraction(img):
    points=range(125*125)
    random.shuffle(points)
    
    VR = 0.1 #variance ratio 
    '''
    To filter out frequently occurring constant color regions,
    we reject sample patches with variance less than 10%
    of the maximum pixel value. 	
    '''

    patchlocs=filter(lambda y: y.shape[0]*y.shape[1]==PS*PS, [img[x/125:x/125+PS,x%125:x%125+PS] for x in points[0:400] \
    if ndimage.variance(img[x/125:x/125+PS,x%125:x%125+PS]) > VR*ndimage.variance(img)])
    
    random.shuffle(patchlocs)
    if len(patchlocs)<=100:
        return patchlocs
    else:
        return patchlocs[0:100] 


def standardizePatch(im):
    s1=[(x-ndimage.mean(im))/ndimage.variance(im) for x in im]
    return  np.reshape(np.asarray([item for x in s1 for item in x]),(im.shape[0],im.shape[1]))     

'''
http://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening
original data matrix X= n x d
mean centered data matrix = subtract mean from each element in X
covariance matrix C for mean centered data matrix = X*X.T / n 
SVD(C) =E*D*E.T, wheree E contains eigen vectors and D contains eigen values of C.
'''
def zca_whitening(inputs): 
    #note input is already mean centered.
    sigma = np.dot(inputs,inputs.T)/inputs.shape[0] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    '''
    see http://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python . This code has np.diag(S),
    which should actually be np.diag(np.diag(s)). See the last example in 
    http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.diag.html 
    
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T) 
    '''
    ZCAMatrix=1/(np.sqrt(sigma + epsilon)) #ZCA Whitening matrix, 
    return np.dot(ZCAMatrix, inputs)   #Data whitening	    

def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector

def oneImFile(loc):
    size=(128,128)
    im=(resize(rgb2grey(imread(loc)),size)*255) #resize
    #random patch extraction
    rp=randomPatchExtraction(im)
    #patch standardization
    rps=[list(flatten_matrix(standardizePatch(x))[0]) for x in rp]
    rpsl=[item for sublist in rps for item in sublist]
    rpsa=np.reshape(rpsl,(len(rps),PS*PS))
    print "rpsa",rpsa.shape
    rpsz=zca_whitening(rpsa)
    
def main():
    loc=sys.argv[1]
    oneImFile(loc)
    

if __name__ == "__main__":
    main()	

	
    



 
