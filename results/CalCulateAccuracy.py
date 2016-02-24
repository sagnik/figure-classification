from __future__ import division
import pickle
import sys
import numpy as np

cfs=pickle.load(open(sys.argv[1]))
precisions=[]
recalls=[]
F1scores=[]

for cf in cfs:
    #print cf
    if cf.shape[0]!=cf.shape[1]:
        print "wrong conf. matrix"
        continue
    else:
        thiscfprecision=[]
        thiscfrecall=[]
        thiscfF1score=[] 
        for i in range(cf.shape[0]):
            
            retrel=cf[i][i]
            ret=np.sum(cf[:,i])
            rel=np.sum(cf[i,:])
            precision=round((retrel/ret)*100)
            recall=round((retrel/rel)*100)
            f1score=round(2*precision*recall/(precision+recall+0.1))
            thiscfprecision.append(precision)
            thiscfrecall.append(recall)
            thiscfF1score.append(f1score)
    
        precisions.append(thiscfprecision)
        recalls.append(thiscfrecall)
        F1scores.append(thiscfF1score)

#print "precisions",[np.mean(np.array(precisions)[:,x]) for x in range(np.shape[1])]
print "precisions", [round(np.mean(np.array(precisions)[:,x])) for x in range(np.array(precisions).shape[1])] 
print "recalls", [round(np.mean(np.array(recalls)[:,x])) for x in range(np.array(recalls).shape[1])] 
print "F1 scores",[round(np.mean(np.array(F1scores)[:,x])) for x in range(np.array(F1scores).shape[1])]

         
     
