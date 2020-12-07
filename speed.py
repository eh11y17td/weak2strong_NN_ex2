import numpy as np
import time


a = np.random.randint(0, 10, (2, 5, 10))
b = np.arange(100).reshape(2, 5, 10)

print(a)
print(b)
print(np.where(a > 5, b, a+b))


# start=time.time()
# for i in range(10):
#     for j in range(100):
#         for k in range(10):
#             if a[i][j][k] > 5: 
#                 a[i][j][k] = i+j+k
#             else:
#                 a[i][j][k] = i+j+k
# t=time.time()-start
# print(t)
# start=time.time()
# b = [[[i+j+k if a[i][j][k] > 5 else i+j+k for k in range(10)] for j in range(100)] for i in range(10)]
# t=time.time()-start
# print(t)