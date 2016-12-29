import numpy as np
from sklearn.linear_model import LogisticRegression

import time

npoints = 10000
nfeatures =  16384

#------------------------------------------------------------------------------
# 1 cpu thread
#------------------------------------------------------------------------------


print  '10K x 16K using 1 thread'

X = np.random.rand(npoints, nfeatures).astype('f')
y = np.random.randint(2, size=npoints)

logreg = LogisticRegression(n_jobs=1)

start = time.time()
logreg.fit(X, y)
end = time.time()


n_iter = logreg.n_iter_
runtime = end - start
iter_time = runtime / n_iter

print str(runtime) +  ' s'
print 'runtime per iter : ' + str(iter_time) + ' s'
print 'niter : ' + str(n_iter)
print '\n\n'


#------------------------------------------------------------------------------
# 4 cpu thread
#------------------------------------------------------------------------------

print  '10K x 16K using 4 thread'

X = np.random.rand(npoints, nfeatures).astype('f')
y = np.random.randint(2, size=npoints)

logreg = LogisticRegression(n_jobs=4)

start = time.time()
logreg.fit(X, y)
end = time.time()


n_iter = logreg.n_iter_
runtime = end - start
iter_time = runtime / n_iter

print str(runtime) +  ' s'
print 'runtime per iter : ' + str(iter_time) + ' s'
print 'niter : ' + str(n_iter)
print '\n\n'

#------------------------------------------------------------------------------
# 8 cpu thread
#------------------------------------------------------------------------------

print  '10K x 16K using 8 thread'

X = np.random.rand(npoints, nfeatures).astype('f')
y = np.random.randint(2, size=npoints)

logreg = LogisticRegression(n_jobs=8)

start = time.time()
logreg.fit(X, y)
end = time.time()


n_iter = logreg.n_iter_
runtime = end - start
iter_time = runtime / n_iter

print str(runtime) +  ' s'
print 'runtime per iter : ' + str(iter_time) + ' s'
print 'niter : ' + str(n_iter)
print '\n\n'
