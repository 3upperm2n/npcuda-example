#import gpuadder
import numpy as np
import numpy.testing as npt
import GPULearn 

#def #test():
#    arr = np.array([1,2,2,2], dtype=np.int32)
#    adder = gpuadder.GPUAdder(arr)
#    adder.increment()
#    
#    adder.retreive_inplace()
#    results2 = adder.retreive()
#
#    npt.assert_array_equal(arr, [2,3,3,3])
#    npt.assert_array_equal(results2, [2,3,3,3])


arr_a = np.array([1,2,2,2], dtype=np.float32)
arr_b = np.array([1,2,2,2], dtype=np.float32)

mydata = GPULearn.GPULearn(arr_a, arr_b)

mydata.vectorAdd()

result = mydata.retreive()

print "gpu result"
print type(result)

print result

print type(result[0])

#mydata.retreive_inplace();

#print mydata


