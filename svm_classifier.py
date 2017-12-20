import time
import numpy as np
import matplotlib.pyplot as plt
from svmutil import *
from pandas import DataFrame 
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score, confusion_matrix 

def getKernelSVMSolution(Xtr, Ytr, C, G, Xts, Yts):
	model   = svm_train	 (svm_problem(Ytr, Xtr), svm_parameter('-t 2 -g '+str(G)+' -c '+str(C)+' -b 1 -q'))
	L, A, V = svm_predict(Yts,Xts,model,'-b 1')
	dist = [(L[i]*V[i][int((abs(int(L[i]))-int(L[i]))/2)]) for i in range(len(L))]
	return L, dist

def imagemismatch(X,Y,P,t):
	C=0
	L=[]
	for i in range(len(Y)):
		if(Y[i]!=P[i]):
			C=C+1
			L.append(i)
	w=int(pow(C,0.5))+1
	c=0
	f,a=plt.subplots(w,w)
	for i in range(w):
		for j in range(w):
			if(C<=c):
				a[i,j].axis('off')
				continue
			a[i,j].imshow(X[L[c],:].reshape(16,16), cmap='gray')
			a[i,j].set_xlabel(str(int(Y[L[c]]))+'$\mapsto$'+str(int(P[L[c]])), labelpad=2, fontsize=7)
			a[i,j].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
			plt.subplots_adjust(wspace=1.8)
			c=c+1
	plt.suptitle(t+str(C), fontsize=13)

def onevsone(Xtr, Ytr, C, Xts, Yts):
	Prediction=[]
	Label=[]
	for i in range(10):
		for j in range(10):
			if(i>=j):
				continue
			else:
				X=[]
				Y=[]
				for r in range(len(Ytr)):
					if(i==Ytr[r]):
						X.append(Xtr[r]); Y.append(1)
					elif(j==Ytr[r]):
						X.append(Xtr[r]); Y.append(-1)	
				M=np.median(cdist(DataFrame(X).fillna(0).values, DataFrame(X).fillna(0).values)[np.triu_indices(DataFrame(X).fillna(0).values.shape[0],1)])**2		# median distance
				Prediction.append(getKernelSVMSolution(X,Y,C,3/M,Xts,Yts)[0])
	Prediction=np.matrix(Prediction)

	for x in range(Prediction.shape[1]):
		M=np.zeros((10,10))
		M[np.triu_indices(10,1)]=Prediction[:,x].T 
		mapping={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
		for r in range(M.shape[0]):
			for c in range(M.shape[1]):
				if(M[r,c]>0):
					mapping[r]=mapping[r]+1
				elif(M[r,c]<0):
					mapping[c]=mapping[c]+1 
		Label.append([x for x in mapping.keys() if mapping[x]==max(mapping.values())][0]) #maximum mapping key
	return Label

def onevsrest(Xtr,Ytr,C,Xts,Yts):
	Prediction=[]
	Label=[]
	for i in range(10):
		X=[]
		Y=[]
		for r in range(len(Ytr)):
			if(i==Ytr[r]):
				X.append(Xtr[r]); Y.append(1)
			else:
				X.append(Xtr[r]); Y.append(-1)
		M=np.median(cdist(DataFrame(X).fillna(0).values, DataFrame(X).fillna(0).values)[np.triu_indices(DataFrame(X).fillna(0).values.shape[0],1)])**2 #median
		Prediction.append(getKernelSVMSolution(X,Y,C,3/M,Xts,Yts)[1])
	Prediction=np.matrix(Prediction)

	for x in range(Prediction.shape[1]):
		mapping = {i:Prediction[i,x] for i in range(10)}
		Label.append([x for x in mapping.keys() if mapping[x]==max(mapping.values())][0])
	return Label

def mat2dict(X):
	return [{x+1:y[0,x] for x in range(y.shape[1]) if y[0,x]!=0} for y in X]

Xtr=mat2dict(np.matrix(np.loadtxt('learn_data/train.csv', delimiter=','))); Ytr=np.loadtxt('learn_data/trainlabel.csv', delimiter = ',')
Xts=mat2dict(np.matrix(np.loadtxt('learn_data/test.csv', delimiter=','))); Yts=np.loadtxt('learn_data/testlabel.csv', delimiter = ',')
compTime=time.time()

print "One vs One:"
Yonevsone=onevsone(Xtr,Ytr,100,Xts,Yts)
onevsoneTime=time.time()
print "\nOne vs Rest:"
Yonevsrest=onevsrest(Xtr,Ytr,100,Xts,Yts)
onevsrestTime=time.time()

print "\nOneVsOne Scheme\n",confusion_matrix(Yts,Yonevsone)
print "\nOneVsRest Scheme\n",confusion_matrix(Yts,Yonevsrest)
print "\nRuntime for SVM One Vs One was",onevsoneTime-compTime,"seconds"
print "Runtime for SVM One Vs Rest was",onevsrestTime-onevsoneTime,"seconds"
print "\nf1_score SVM One Vs One is",f1_score(Yts,Yonevsone,average="macro")
print "f1_score SVM One Vs Rest is",f1_score(Yts,Yonevsrest,average="macro")

print '\nLoading Images, Please Wait...'
imagemismatch(np.loadtxt('learn_data/test.csv', delimiter=','),Yts,Yonevsone,'SVM One Vs One Mismatch Count: ')
imagemismatch(np.loadtxt('learn_data/test.csv', delimiter=','),Yts,Yonevsrest,'SVM One Vs Rest Mismatch Count: ')
plt.show()