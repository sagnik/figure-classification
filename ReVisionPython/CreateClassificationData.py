import sys,os
import numpy as np
from FeatureExtractionOneFile import featureExtractOneFileUnit
import pickle

def main():
    baseDir="/home/sagnik/codes/figure-classification/data-for-fig-classification/"
    lineIms=[baseDir+"lines/"+x for x in os.listdir(baseDir+"lines") if x.endswith(".png")]
    barIms=[baseDir+"bars/"+x for x in os.listdir(baseDir+"bars") if x.endswith(".png")]
    otherIms=[baseDir+"others/"+x for x in os.listdir(baseDir+"others") if x.endswith(".png")]
    
    patchClusterLoc="/home/sagnik/codes/figure-classification/data-for-fig-classification/nonzcapatch-clustered.nparray.pickle"
    classificationDataLoc="/home/sagnik/codes/figure-classification/data-for-fig-classification/\
    imdatawithlabels-allpixels-fast.nparray.pickle" 
    clusterData=pickle.load(open(patchClusterLoc))
    
    #initialize
    a=np.zeros(800).reshape((1,800))

    for i,im in enumerate(lineIms+barIms+otherIms):
        print "processing",i+1,"of",len(lineIms)+len(barIms)+len(otherIms),im
        feat=featureExtractOneFileUnit(im,clusterData)
        a=np.vstack((a,feat.reshape(1,800)))   
    
    labels=np.array([0]*len(lineIms)+[1]*len(barIms)+[2]*len(otherIms)).reshape((len(lineIms)+len(barIms)+len(otherIms),1))
    print a.shape,labels.shape
    data=np.hstack((a[1:,:],labels))

    pickle.dump(data,open(classificationDataLoc,"wb"))

if __name__=="__main__":
    main() 
        
