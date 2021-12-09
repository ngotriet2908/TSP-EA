import numpy as np


def f1(a, b):
    return len(np.bitwise_xor(a, b).nonzero()[0])


def f2(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] != b[i]
    return result


# for _ in range(1000):
#     a = np.random.randint(100, size=100)
#     b = np.random.randint(100, size=100)
#     if f1(a, b) != f2(a, b):
#         print("cc")
# print("end")

arr1 = np.array([1,2,3,4,5])
arr2 = np.ndarray.copy(arr1)

arr1[0] = 2

print(arr1)
print(arr2)