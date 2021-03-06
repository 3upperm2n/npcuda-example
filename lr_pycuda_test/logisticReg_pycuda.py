import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.compiler import compile
from pycuda.driver import module_from_file

import scikits.cuda.cublas as cublas

import pycuda.gpuarray as gpuarray
import numpy as np

import time


#------------------------------------------------------------------------------

#npoints = 10000
#nfeatures = 4096

#npoints = 10000
#nfeatures = 16384


#------------------------------------------------------------------------------
max_iters = 1000000
learning_rate = 0.01

#-----------
# cpu memory
#-----------

#X = np.random.rand(npoints, nfeatures).astype('f')
#y = np.random.randint(2, size=npoints)

X = np.load("PhthalatesV1.npy")
X = np.float32(X)
[npoints, nfeatures] = X.shape

y = np.load("PhthalatesV1_Ana.npy")
y = np.int32(y)

if y.shape[0] <> X.shape[0]:
    print("samples of X and y do not match!\n")
    sys.exit(1)

print str(npoints) + ' x ' + str(nfeatures)

theta = np.full((nfeatures, 1), 0.1, dtype=np.float32)

sigmoid_error_cpu = np.zeros((npoints,1), dtype=np.float32)

# start timing
start = time.time()

#-----------
# gpu memory
#-----------
# X_gpu = gpuarray.to_gpu(X.T.copy())
X_gpu = gpuarray.to_gpu(X)
y_gpu = gpuarray.to_gpu(y)
theta_gpu = gpuarray.to_gpu(theta)

xt_gpu = gpuarray.empty((npoints,1), np.float32)
sigmoid_error_gpu = gpuarray.empty((npoints,1), np.float32)
theta_tmp_gpu = gpuarray.empty((nfeatures,1), np.float32)


#--------------------------------
## load precompiled cubin file
mod = module_from_file("logisticReg_kernels.cubin")

kernel_sigmoid       = mod.get_function('kernel_sigmoid')
kernel_update_weight = mod.get_function('kernel_update_weight')

## kernel configuration : sigmoid
blk_size = 256
grd_size = (npoints + blk_size -1) / blk_size

## kernel configuration : update weight
blk_size_1= 256
grd_size_1 = (nfeatures + blk_size_1 -1) / blk_size_1

alpha = np.float32(1.0)
beta = np.float32(0.0)

# create cublas handle
cublas_handle = cublas.cublasCreate()


#------------------------------------------------------------------------------
# running iteration 
#------------------------------------------------------------------------------
loop_count = 0
for i in range(0,max_iters):
    loop_count = loop_count + 1
    
    #------------------------------------
    #    x * theta = xt
    #------------------------------------
    cublas.cublasSgemv(cublas_handle, 't',  nfeatures, npoints, alpha, \
                       X_gpu.gpudata, nfeatures, \
                       theta_gpu.gpudata, 1, \
                       beta, \
                       xt_gpu.gpudata, 1)

    ## run sigmoid kernel
    kernel_sigmoid(xt_gpu.gpudata, y_gpu.gpudata, np.int32(npoints), sigmoid_error_gpu.gpudata, \
                   block = (blk_size, 1, 1), grid = (grd_size, 1, 1))

    # copy back to host
    #cuda.memcpy_dtoh(sigmoid_error_cpu, sigmoid_error_gpu.gpudata)
    #print sigmoid_error_cpu
    error_cpu = sigmoid_error_gpu.get() 
    #print("error = {}".format(np.sum(np.abs(error_cpu))))

    total_error = np.sum(np.abs(error_cpu))
    if total_error < 1:
        print("found best theta  at iter {}\n".format(i))
        break


    #------------------------------------
    #    x.transpose * xt = theta_tmp
    #------------------------------------
    cublas.cublasSgemv(cublas_handle, 'n',  nfeatures, npoints, alpha, \
                       X_gpu.gpudata, nfeatures, \
                       sigmoid_error_gpu.gpudata, 1, \
                       beta, \
                       theta_tmp_gpu.gpudata, 1)

    ## run update weight kernel
    kernel_update_weight(theta_tmp_gpu.gpudata, np.float32(learning_rate), np.int32(nfeatures), theta_gpu.gpudata, \
                   block = (blk_size_1, 1, 1), grid = (grd_size_1, 1, 1))


    # break


print("final error = {}\n".format(total_error))

# copy theta back to host
coef_ = theta_gpu.get()

cublas.cublasDestroy(cublas_handle)

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
print 'maxiter : ' + str(max_iters) 


print "found theta:\n"
print coef_