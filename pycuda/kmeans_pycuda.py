import pycuda.driver as cuda
import pycuda.autoinit
# from pycuda.compiler import SourceModule
from pycuda.driver import module_from_file

import pycuda.gpuarray as gpuarray
import numpy as np

import time

npoints = 10000000
nfeatures = 10 


print str(npoints) + ' x ' + str(nfeatures)


### ------------------------------------------
### start timing the start of the end-to-end processing time 
### ------------------------------------------
start = time.time()

## load precompiled cubin file
mod = module_from_file("kmeans_kernels.cubin")

# link to the kernel function
kernel_kmeans = mod.get_function('kernel_kmeans')

# constant memory
clusters_cnst = mod.get_global('clusters_cnst')[0]


#------------------------------------------------------------------------------
# parameters
#------------------------------------------------------------------------------
tol = 0.0001
nclusters = 2
maxiters = 300

# input data
X = np.random.rand(npoints, nfeatures).astype('f')
#X

# init centroids
clusters = X[:nclusters,]

# init membership to zeros
membership = np.zeros(npoints).astype('i')

# delta
delta = np.zeros(1).astype('f')


###
## allocate memory on device
###
X_gpu = cuda.mem_alloc(X.nbytes)

membership_gpu     = cuda.mem_alloc(membership.nbytes)

new_membership_gpu = cuda.mem_alloc(membership.nbytes)

# new clusters
new_clusters = np.zeros(clusters.shape).astype('f')
new_clusters_gpu = cuda.mem_alloc(clusters.nbytes)

# new clusters membership
new_clusters_member = np.zeros(nclusters).astype('f')
new_clusters_member_gpu = cuda.mem_alloc(new_clusters_member.nbytes)

delta_gpu = cuda.mem_alloc(delta.nbytes)

###
## transfer data to gpu
###
cuda.memcpy_htod(X_gpu, X)

cuda.memcpy_htod(membership_gpu, membership)

# The kernel goes into constant memory via a symbol defined in the kernel
cuda.memcpy_htod(clusters_cnst,  clusters)

###
## define kernel configuration
###
blk_size = 128
grd_size = (npoints + blk_size -1) / blk_size
warps_per_blk = 128 / 32


###---------------------------------------------------------------------------
### Run kmeans on gpu
###---------------------------------------------------------------------------
loop_count = 0

for i in range(0,maxiters):
    
    loop_count = loop_count + 1
    
    # change to zero for each iteration
    delta = np.zeros(1).astype('f')
    cuda.memcpy_htod(delta_gpu, delta)
    
    # start from zero for each iteration
    new_clusters = np.zeros(clusters.shape).astype('f')
    new_clusters_member = np.zeros(nclusters).astype('f')
    
    # copy 
    cuda.memcpy_htod(new_clusters_gpu, new_clusters)
    cuda.memcpy_htod(new_clusters_member_gpu, new_clusters_member)
    
    
    
    ## run kernel
    kernel_kmeans(X_gpu, membership_gpu,\
              np.int32(npoints), np.int32(nfeatures), np.int32(nclusters), np.int32(warps_per_blk), \
              delta_gpu, \
              new_membership_gpu, \
                new_clusters_gpu, 
                  new_clusters_member_gpu, \
                  block = (blk_size, 1, 1), grid = (grd_size, 1, 1))

    # copy back
    cuda.memcpy_dtoh(delta, delta_gpu)
    
    if(delta[0] < tol):
        break
        
    # copy back new_clusters
    cuda.memcpy_dtoh(new_clusters, new_clusters_gpu)
    cuda.memcpy_dtoh(new_clusters_member, new_clusters_member_gpu)
    
    ## re-compute clusters
    for i in xrange(0, clusters.shape[0]):
        for j in xrange(0, clusters.shape[1]):
            clusters[i,j] = new_clusters[i,j] / new_clusters_member[i]
            
    
    ## copy to constant memory
    cuda.memcpy_htod(clusters_cnst,  clusters)
    
    # update membership
    cuda.memcpy_dtod(membership_gpu, new_membership_gpu, membership.nbytes)


###----------------------------------------------------------------------------
## end of gpu kmeans
###----------------------------------------------------------------------------

# copy back new_clusters
cuda.memcpy_dtoh(new_clusters, new_clusters_gpu)
cuda.memcpy_dtoh(new_clusters_member, new_clusters_member_gpu)


### output
## re-compute clusters
for i in xrange(0, clusters.shape[0]):
    for j in xrange(0, clusters.shape[1]):
        clusters[i,j] = new_clusters[i,j] / new_clusters_member[i]
        

### output
cuda.memcpy_dtoh(membership, new_membership_gpu)



### ------------------------------------------
### end timing of the end-to-end processing time 
### ------------------------------------------
end = time.time()
runtime = end - start

iter_time = runtime / loop_count
###----------------------------------------------------------------------------
## dump stat 
###----------------------------------------------------------------------------

#print clusters


print 'runtime : ' + str(runtime)  + ' s'                                                  
print 'runtime per iter : ' + str(iter_time) + ' s'                             
print 'niter : ' + str(loop_count)                                                   
print 'maxiter : ' + str(maxiters) 
