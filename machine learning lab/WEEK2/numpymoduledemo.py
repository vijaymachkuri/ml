import numpy as np
arr = np.array([1, 2, 3])
print(arr)
arr = np.arange(1, 6)
print(arr)
arr = np.zeros(3)
print(arr)
arr = np.ones(3)
print(arr)
arr = np.random.rand(3)
print(arr)
arr = np.random.randint(0, 10, 5)
print(arr)
arr = np.array([1, 2, 3])
max_value = np.max(arr)
print(max_value)
min_value = np.min(arr)
print(min_value)
mean_value = np.mean(arr)
print(mean_value)
median_value = np.median(arr)
print(median_value)
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
dot_product = np.dot(arr1, arr2)
print(dot_product)