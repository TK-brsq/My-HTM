import time
import numpy as np
from bitarray import bitarray
import random

N = 2048  # 要素数
REPEATS = 11  # 最初の1回を捨てるため +1

def numpy_test():
    a = np.array(np.random.choice([True, False], size=N), dtype=bool)
    b = np.array(np.random.choice([True, False], size=N), dtype=bool)

    start = time.perf_counter()
    c = np.logical_and(a, b)
    count = np.count_nonzero(c)
    end = time.perf_counter()

    return end - start, count

def bitarray_test():
    a = bitarray([random.choice([True, False]) for _ in range(N)])
    b = bitarray([random.choice([True, False]) for _ in range(N)])

    start = time.perf_counter()
    c = a & b
    count = c.count(True)
    end = time.perf_counter()

    return end - start, count

# NumPy Test
print("== NumPy ==")
numpy_times = []
for i in range(REPEATS):
    t, cnt = numpy_test()
    if i > 0:  # 最初の1回はスキップ
        numpy_times.append(t)
        print(f"Run {i}: Time = {t:.8f}s, Count = {cnt}")
    else:
        print(f"Warm-up run: {t:.8f}s")
print(f"Avg NumPy time (excluding warm-up): {sum(numpy_times) / len(numpy_times):.8f}s")

# bitarray Test
print("\n== bitarray ==")
bitarray_times = []
for i in range(REPEATS):
    t, cnt = bitarray_test()
    if i > 0:
        bitarray_times.append(t)
        print(f"Run {i}: Time = {t:.8f}s, Count = {cnt}")
    else:
        print(f"Warm-up run: {t:.8f}s")
print(f"Avg bitarray time (excluding warm-up): {sum(bitarray_times) / len(bitarray_times):.8f}s")
