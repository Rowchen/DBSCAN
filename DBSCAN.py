import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
class DBSCAN2:
    def __init__(self,eps,minpst):
        self.eps=eps
        self.minpst=minpst
        self.label_=[]
    def fit(self,x):
        A=[]
        s=[list() for row in range(x.shape[0])]
        class_ = [0 for i in xrange(x.shape[0])]
        visited=[False for i in xrange(x.shape[0])]
        for i in xrange(x.shape[0]):
            for j in xrange(x.shape[0]):
                if np.sqrt(sum((x[i]-x[j])**2))<self.eps:
                    s[i].append(j)
        for i in xrange(x.shape[0]):
            if len(s[i])>self.minpst:
                A.append(i)
        Q=[]
        top=0
        tail=0
        k=-1
        self.label_=np.full(x.shape[0],-1,dtype=int)
        while(len(A)):
            q=A.pop()
            if visited[q]:
                continue
            k += 1
            Q.append(q)
            tail+=1
            while(top!=tail):
                ls=Q[top]
                top+=1
                self.label_[ls]=k
                class_[ls]=k
                if len(s[ls])>self.minpst:
                    for i in s[ls]:
                        if not visited[i]:
                            Q.append(i)
                            tail+=1
                            visited[i]=True



from sklearn.datasets.samples_generator import make_blobs
centers = [[1, 1], [-1, -1], [1, -1]]
data, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)
# dataset = np.loadtxt("dataset\data4.txt",dtype=float)#num_train*feature_dim
# data=np.zeros((30,2))
# for i in xrange(len(dataset)):
#     if i%3==1:
#         data[i/3,0]=dataset[i]
#     if i%3==2:
#         data[i/3,1]=dataset[i]
s=DBSCAN2(eps=0.3,minpst=10)
db=DBSCAN(eps=0.3,min_samples=10,p=2)
s.fit(data)
db.fit(data)
color=['ro','bo','go','yo','co','ko','mo','wo']
plt.subplot(1, 2, 1)
for i in xrange(data.shape[0]):
    plt.plot(data[i, 0], data[i, 1], color[db.labels_[i]])
plt.subplot(1,2,2)
for i in xrange(data.shape[0]):
    plt.plot(data[i, 0], data[i, 1], color[s.label_[i]])
plt.show()