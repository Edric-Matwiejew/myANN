import numpy as np

# A "list" of 20 10 x 10 matrices
a = np.ones((10, 20, 20))

b = a[0,:,:]

b[:] = 2

v = np.random.uniform(low= 0, high = 1, size = (1,20))

u = np.dot(b, np.transpose(v))

print((u*u + v+v).shape)
