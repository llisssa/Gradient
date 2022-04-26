import numpy as np


def const(A, b, x, e, imax):
    r = b - A.dot(x)
    d = np.transpose(r).dot(r)
    d0 = d
    i = 0
    while i < imax:
        x = x + A.dot(r)
        r = b - A.dot(x)
        d = np.transpose(r).dot(r)
    print(x, r, i)


A = np.array([[1, 1], [1, 1]])
b = np.transpose(np.array([1, 1]))
x = np.transpose(np.array([1, 1]))
