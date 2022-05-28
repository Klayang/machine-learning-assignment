import numpy as np
array = np.array(['Kobe', 'Kobe', 'Jordan'])
players = np.unique(array)
map = {label: index for index, label in enumerate(players)}
print(map)
print('Lebron' in array)
a = {}
a['klay'] = 2
print(1 in a)
b = np.array([[1, 2], [3, 4]])
print(b[1, 1])
k = np.array([1, 2, 3])
k = (k != 0).astype(int)
print(k)

