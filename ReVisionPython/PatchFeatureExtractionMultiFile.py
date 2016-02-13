from PatchFeatureExtractionOneFile import zca_whitening,patchFeatExtractionOneImFile
import sys,os
import pickle
import numpy as np
from collections import Counter

def featExtract(fs,a,labels,classLabel):
    for f in fs:
        r=patchFeatExtractionOneImFile(f)
        if r is not None:
            a=np.vstack((a,r))
            labels.append(classLabel)
    
    return (a,labels)

def main():
   
    #default locations
    patchImLineDir="../data-for-fig-classification/patchimages/lines/"
    patchImBarDir="../data-for-fig-classification/patchimages/bars/"
    patchImOtherDir="../data-for-fig-classification/patchimages/others/"

    pickleLoc="../data-for-fig-classification/nonzcapatch-unclustered-labeled.nparray.pickle"
    
    '''
    if len(sys.argv)==3:
        patchImDir=sys.argv[1]
        pickleLoc=sys.argv[2]
    elif len(sys.argv)==2:
        patchImDir=sys.argv[1]
        pickleLoc="../data-for-fig-classification/zcapatch-unclustered.nparray.pickle"
    '''
    imLines=[patchImLineDir+x for x in os.listdir(patchImLineDir)]
    imBars=[patchImBarDir+x for x in os.listdir(patchImBarDir)]
    imOthers=[patchImOtherDir+x for x in os.listdir(patchImOtherDir)]

    
    a=np.zeros(3600).reshape((1,3600))
    labels=[]
    (a,labels)=featExtract(imLines,a,labels,0)
    (a,labels)=featExtract(imBars,a,labels,1)
    (a,labels)=featExtract(imOthers,a,labels,2)
    
    #labels=np.array(labels).reshape(len(labels),1)
    #print a.shape,labels.shape  

    
    print "feature extraction done"     
    preZca=a[1:,:]
    postZca=zca_whitening(preZca-np.mean(preZca))
    #postZca=preZca    
    
    print "zca whitening done"
      
    data=np.zeros(36).reshape(1,36)
    for i in postZca[:,]:
        data=np.vstack((data,i.reshape(100,36)))
    
    print "data reshaping done"

    data=data[1:,:]
    
    labels=np.array([0]*Counter(labels)[0]*100+[1]*Counter(labels)[1]*100+[2]*Counter(labels)[2]*100)
    #print labels.shape,data.shape
        
    datal=np.hstack((data,labels.reshape((data.shape[0],1))))  
    
    print "writing data"
    pickle.dump(datal,open(pickleLoc,"wb"))    

if __name__ == "__main__":
    main()

