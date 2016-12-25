from sklearn.cluster import KMeans
import numpy as np
import time

#------------------------------------------------------------------------------
# 1 cpu thread
#------------------------------------------------------------------------------

#--------------------
# test case 1: 1M x 2
#--------------------
print  '1M x 2d'
npoints = 1000000
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

# test case 1: 1M x 2, 1 cpu thread
npoints = 1000000
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

#--------------------
# test case 2: 1M x 10 
#--------------------
print  '1M x 10d'
npoints = 1000000
nfeatures = 10 

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

# test case 1: 1M x 2, 1 cpu thread
npoints = 1000000
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

#--------------------
# test case 2: 1M x 10 
#--------------------
npoints = 1000000
nfeatures = 10 

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

# test case 1: 1M x 2, 1 cpu thread
npoints = 1000000
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

#--------------------
# test case 3: 10M x 2 
#--------------------
print  '10M x 2d'
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

# test case 1: 1M x 2, 1 cpu thread
npoints = 1000000
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

#--------------------
# test case 2: 1M x 10 
#--------------------
npoints = 1000000
nfeatures = 10 

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

# test case 1: 1M x 2, 1 cpu thread
npoints = 1000000
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

#--------------------
# test case 4: 10M x 10 
#--------------------
print  '10M x 10d'
npoints = 10000000
nfeatures = 10 

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

# test case 1: 1M x 2, 1 cpu thread
npoints = 1000000
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

#--------------------
# test case 2: 1M x 10 
#--------------------
npoints = 1000000
nfeatures = 10 

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

# test case 1: 1M x 2, 1 cpu thread
npoints = 1000000
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
