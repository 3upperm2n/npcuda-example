import numpy as np
cimport numpy as np
import time

#from cython.view cimport array as cvarray

assert sizeof(int)   == sizeof(np.int32_t)
assert sizeof(float) == sizeof(np.float32_t)

cdef extern from "src/kmeans.hh":
    cdef cppclass C_KmeansGPU "KmeansGPU":
        C_KmeansGPU(float, int, int, int, int, np.float32_t*)
        #define the required functions
        void Run()
        #void getData_extern(np.int32_t*, int &)
        void getData_extern(np.int32_t*, int&, np.float32_t*)

cdef class KmeansGPU:


    cdef C_KmeansGPU* g 

    cdef int data_points 
    cdef int features 
    cdef int cluster_num 
    cdef int iter_num

    # http://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
    def __cinit__(self, float tol, int cluster_num, int npoints, int nfeatures, int maxiters, np.ndarray[ndim=2, dtype=np.float32_t,  mode="c"] data_in):
        start = time.time()

        self.iter_num = 0
        self.data_points = npoints
        self.features = nfeatures
        self.cluster_num = cluster_num 
        
        # dim check ?

        #https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC


        # create class
        #self.g = new C_KmeansGPU(tol, cluster_num, npoints, nfeatures, maxiters, &data_in[0])
        #self.g = new C_KmeansGPU(tol, cluster_num, npoints, nfeatures, maxiters, &data_view[0])

        self.g = new C_KmeansGPU(tol, cluster_num, npoints, nfeatures, maxiters, &data_in[0,0])
        end = time.time()
        print "init time : " + str(end - start) + " s"

    def run(self):
        start = time.time()
        self.g.Run()
        end = time.time()
        print "run time : " + str(end - start) + " s"

    #def retreive_inplace(self):
    #    self.g.getData()

    def retreive(self):
        start = time.time()

        # define output membership
        cdef np.ndarray[dtype=np.int32_t] label = np.zeros(self.data_points, dtype=np.int32)

        # define output centroids
        #cdef np.ndarray[dtype=np.float32_t] centroids = np.zeros((self.data_points,self.features), dtype=np.float32)
        #cdef np.ndarray[dtype=np.float32_t, mode="c"] centroids = np.zeros((self.data_points,self.features), dtype=np.float32)
        #centroids = np.zeros((self.data_points,self.features), dtype=np.float32_t, mode="c")

        #print self.data_points
        #print self.features

        cdef np.ndarray[float, ndim=2, mode="c"] centroids = np.zeros((self.cluster_num, self.features), dtype=np.float32)

        self.g.getData_extern(&label[0], self.iter_num, &centroids[0,0])

        end = time.time()
        print "data retrive time : " + str(end - start) + " s"

        return  label, self.iter_num, centroids
