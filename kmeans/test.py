#import gpuadder
import numpy as np
from gpuML import KmeansGPU
import time


npoints = 10000000
nfeatures = 10 

#npoints = 100
#nfeatures = 2

X = np.random.rand(npoints, nfeatures).astype('f')


#X = X.astype(np.float32)

#http://stackoverflow.com/questions/4389517/in-place-type-conversion-of-a-numpy-array
#X_view = X.view('float32')
#X_view[:] = X



tol = 0.0001
n_clusters = 2
maxiters = 300

# start timer
start = time.time()

#mydata = KmeansGPU(tol, n_clusters, npoints, nfeatures, maxiters, X_view)
mydata = KmeansGPU(tol, n_clusters, npoints, nfeatures, maxiters, X)

mydata.run()
label, iters, centroids = mydata.retreive()

# end the timer
end = time.time()

#print label 
#print iters 
print centroids

runtime = end - start                                                           
iter_time = runtime / iters 
                                                                                
print str(end - start) +  ' s'                                                  
print 'runtime per iter : ' + str(iter_time) + ' s'                             
print 'niter : ' + str(iters)                                          
print 'maxiter : ' + str(maxiters)

