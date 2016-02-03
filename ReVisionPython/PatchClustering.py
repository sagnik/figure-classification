from sklearn.cluster import KMeans
import pickle
import sys
import numpy as np
from scipy import stats

def main():
    #default locations
    dataLoc="/home/sagnik/codes/figure-classification/data-for-fig-classification/patch-unclustered.nparray.pickle"
    saveLoc="/home/sagnik/codes/figure-classification/data-for-fig-classification/patch-clustered.nparray.pickle"

    if len(sys.argv)==3:
        dataLoc=sys.argv[1]
        saveLoc=sys.argv[2]
    elif len(sys.argv)==2:
        dataLoc=sys.argv[1]
        saveLoc="/home/sagnik/codes/figure-classification/data-for-fig-classification/patch-clustered.nparray.pickle"

    data=pickle.load(open(dataLoc))

    NOC=200 #no. of clusters, default 200
    print "clustering ",data.shape[0],"data points in",NOC,"clusters"
    
    km = KMeans(n_clusters=NOC, init='k-means++',random_state=1).fit(data)
    labels = km.labels_
    
    print "clustering done"  
    #TODO: there are some highly frequent clusters, possibly from white pixels. Should we remove them?
    #print stats.itemfreq(labels) 
    
    labels=list(labels)
    
    clusterDict={}
    for i,v in enumerate(labels):
        if v not in clusterDict:
            clusterDict[v]=data[i]
        else:      
            clusterDict[v]=np.vstack((clusterDict[v],data[i]))
    
    clusterMeans=np.zeros(36).reshape((1,36))

    for x in clusterDict.keys():
        clusterMeans=np.vstack((clusterMeans,np.sum(clusterDict[x],axis=0)))
    
    clusterMeans=clusterMeans[1:,:]
    print clusterMeans.shape
    
    pickle.dump(clusterMeans,open(saveLoc,"wb"))

if __name__=="__main__":
    main()    
        
