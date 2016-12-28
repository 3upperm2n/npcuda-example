from sklearn.cluster import KMeans
import numpy as np
import time

#------------------------------------------------------------------------------
# 1 cpu thread
#------------------------------------------------------------------------------

#--------------------
# test case 1
#--------------------
print  '10M x 2d using 1 thread'
npoints = 10000000
nfeatures = 2

X = np.random.rand(npoints, nfeatures)

start = time.time()
kmeans = KMeans(n_clusters = 2, random_state=0, n_jobs=1).fit(X)
end = time.time()

runtime = end - start
iter_time = runtime / kmeans.n_iter_

print str(end - start) +  ' s'
print 'runtime per iter : ' + str(iter_time) + ' s'
print 'niter : ' + str(kmeans.n_iter_)
print 'maxiter : ' + str(kmeans.max_iter)

print '\n\n'

#------------------------------------------------------------------------------
# 4 cpu thread
#------------------------------------------------------------------------------

#--------------------
# test case 2 
#--------------------
print  '10M x 2d using 4 thread'
npoints = 10000000
nfeatures = 2

X = np.random.rand(npoints, nfeatures)

start = time.time()
kmeans = KMeans(n_clusters = 2, random_state=0, n_jobs=4).fit(X)
end = time.time()

runtime = end - start
iter_time = runtime / kmeans.n_iter_

print str(end - start) +  ' s'
print 'runtime per iter : ' + str(iter_time) + ' s'
print 'niter : ' + str(kmeans.n_iter_)
print 'maxiter : ' + str(kmeans.max_iter)


print '\n\n'
#------------------------------------------------------------------------------
# 8 cpu thread
#------------------------------------------------------------------------------

#--------------------
# test case 3 
#--------------------
print  '10M x 2d using 8 thread'
npoints = 10000000
nfeatures = 2

X = np.random.rand(npoints, nfeatures)

start = time.time()
kmeans = KMeans(n_clusters = 2, random_state=0, n_jobs=8).fit(X)
end = time.time()

runtime = end - start
iter_time = runtime / kmeans.n_iter_

print str(end - start) +  ' s'
print 'runtime per iter : ' + str(iter_time) + ' s'
print 'niter : ' + str(kmeans.n_iter_)
print 'maxiter : ' + str(kmeans.max_iter)

print '\n\n'
