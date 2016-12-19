import numpy as np
cimport numpy as np

assert sizeof(int)   == sizeof(np.int32_t)
#assert sizeof(float) == sizeof(np.float_t)

cdef extern from "src/GPULearn.hh":
    cdef cppclass C_GPULearn "GPULearn":
        C_GPULearn (np.float_t*, int, np.float_t*, int)
        void vectorAdd()
        void getData()
        void getData_extern(np.float_t*, int)

cdef class GPULearn:
    cdef C_GPULearn* g 
    cdef int dim_a
    cdef int dim_b

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.float_t] arr_a, np.ndarray[ndim=1, dtype=np.float_t] arr_b):

        arr_a.astype(np.float32);
        arr_b.astype(np.float32);

        self.dim_a  = len(arr_a)
        self.dim_b  = len(arr_b)
        # create class
        self.g = new C_GPULearn(&arr_a[0], self.dim_a, &arr_b[0], self.dim_b)

    def vectorAdd(self):
        self.g.vectorAdd()

    def retreive_inplace(self):
        self.g.getData()

    def retreive(self):
        cdef np.ndarray[ndim=1, dtype=np.float_t] a = np.zeros(self.dim1, dtype=np.float32)
        #a.astype(float32);
        self.g.getData_extern(&a[0], self.dim1)
        return a
