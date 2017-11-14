import numpy as np

a = np.array([[3, 0, 0, 8, 3],
              [9, 3, 2, 2, 6],
              [5, 5, 4, 2, 8],
              [3, 8, 7, 1, 2],
              [3, 9, 1, 5, 5]
            ])

y = [1, 4]

result = a[y,:].flatten()
print result
print a[y,:]
