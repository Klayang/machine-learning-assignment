import numpy as np
#创建矩阵
v = np.array([[1, 2, 3, 4],
             [5, 6, 7, 8]])
v = v + 1
print(v)
num = v[0, 0]
print(num)
b = np.zeros((1,))
print(b.shape[0])
print((v.shape[0]))
print(v / 2)
print(-v)
v[0, 0] = 10
print(v)
print(v.max())
x = []
print(len(v))
listA = [1, 2, 3]
B = np.array(listA)
print(B)
print(np.max(v, axis=0))
v = np.insert(v, 2, [1, 2, 3, 4], axis=0)
print(v.sum())
w = np.array([[1, 2, 3, 4]])
print(w.shape)
l = [1, 2, 3, 4]
x = np.array(l)
l.append(5)
print(l)
x = np.zeros([3, 4])
y = np.zeros([4, 7])
z = np.zeros([7, 2])
w = np.dot(x, y)
print(w)
p = np.zeros([1, 10])
print(p.shape)
#矩阵数乘
