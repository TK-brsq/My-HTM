import numpy as np
import time

#A = np.random.choice((True, False), size=(1024, 64))
#B = np.random.choice((True, False), size=(1024, 64))

shape = (2048, 32)
num_true = int(1024*64 * 0.05)  # 1%をTrueに
matrix = np.zeros(1024*64, dtype=bool)
matrix[:num_true] = True
np.random.shuffle(matrix)
A = matrix.reshape(shape)
matrix = np.zeros(1024*64, dtype=bool)
matrix[:64] = True
np.random.shuffle(matrix)
B = matrix.reshape(shape)

A_coords = set(zip(*np.where(A)))
B_coords = set(zip(*np.where(B)))

start = time.perf_counter() 
input_sum1 = np.logical_and(A, B).sum()
end = time.perf_counter()
print(input_sum1)
print((end-start)*1024*64*128)

start = time.perf_counter() 
input_sum2 = len(A_coords & B_coords)
end = time.perf_counter()
print(input_sum2)
print((end-start)*1024*64*128)

