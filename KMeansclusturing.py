import sklearn
import numpy
from sklearn import preprocessing
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
import time
from time import time

digits=sklearn.datasets.load_digits()
print(digits)
data=sklearn.preprocessing.scale(digits.data)  #Scale values from -1 to 1
#a=sklearn.preprocessing.minmax_scale(digits.data,feature_range=(-1,1))
##y=digits.target   #labels
##k=len(numpy.unique(y))#unique labels to decide number of classifications
##samples, features = data.shape #Gives number of rows and colums
##
##def bench_k_means(estimator, name, data):
##    t0 = time()
##    estimator.fit(data)
##    
##    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
##          % (name, (time() - t0), estimator.inertia_,
##             metrics.homogeneity_score(y, estimator.labels_),
##             metrics.completeness_score(y, estimator.labels_),
##             metrics.v_measure_score(y, estimator.labels_),
##             metrics.adjusted_rand_score(y, estimator.labels_),
##             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
##             metrics.silhouette_score(data, estimator.labels_,
##                                      metric='euclidean',
##                                      sample_size=samples)))
##
##model=KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=10000)
####model.fit(data)
####a=(model.predict(data))
####acc=metrics.accuracy_score(a,y)
####for i in range(len(a)):
####    print("Predicted = ",a[i],"Real = ",y[i])
####print(acc)    
##bench_k_means(model,'Haha',data)
